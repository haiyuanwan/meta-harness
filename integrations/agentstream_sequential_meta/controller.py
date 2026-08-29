"""Run continual Meta-Harness search on benchmark-native task streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INTEGRATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTEGRATION_DIR.parents[1]
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


from .benchmark_backends import create_backend
from .candidate_contract import (
    CandidateValidationError,
    validate_candidate,
    write_new_harness_state,
)
from .experiment_protocol import FormalPartition, build_formal_partitions
from .hda_reporting import prepare_hda_review

from .opensandbox_backend import (
    DEFAULT_ASSETS_PATH as DEFAULT_OPENSANDBOX_ASSETS_PATH,
)
from .opensandbox_backend import (
    DEFAULT_CACHE_PATH as DEFAULT_OPENSANDBOX_CACHE_PATH,
)
from .opensandbox_backend import (
    DEFAULT_DOMAIN as DEFAULT_OPENSANDBOX_DOMAIN,
)
from .opensandbox_backend import (
    DEFAULT_IMAGE as DEFAULT_OPENSANDBOX_IMAGE,
)
from .opensandbox_backend import (
    OpenSandboxBackend,
    OpenSandboxSettings,
    sequential_task_order,
)
from .proposer import run_claude_proposer
from .protocol import (
    BenchmarkSplit,
    CandidateResult,
    SplitCounts,
    pareto_frontier,
    select_winner,
    split_task_order,
)
from .sandbox_evaluation import BlockRun
from .sandbox_evaluation import run_block as _run_block
from .transfer_evaluation import run_transfer_matrix

__all__ = ["BlockRun"]

BASE_MODEL = "anthropic/Claude-Opus-4.6-hq"
PROPOSER_MODEL = "Claude-Opus-4.6-hq"

BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "bfcl": {
        "backend_kwargs": {"subset": "multi_turn_base"},
        "agent_kwargs": {},
    },
    "browsecompplus": {
        "backend_kwargs": {
            "searcher_type": "faiss",
            "include_get_document": True,
        },
        "grader_kwargs": {},
        "agent_kwargs": {},
    },
}

CONTRACT_TEXT = """# Sequential Meta-Harness Candidate Contract

The complete running harness is candidate.py plus the controller-managed JSON
checkpoint. Edit candidate.py only. It must export:

    CandidateHarness(CandidateHarnessBase)

The candidate owns the agent loop, prompts, context policy, tool-use policy,
and agent-visible memory updates. The fixed evaluator supplies ModelClient and
benchmark-neutral ToolSpec/ToolResult values.

Hard constraints:

- keep the fixed solver ModelClient and supplied tools;
- do not import AgentStream/Exgentic or benchmark implementations;
- do not import subprocess, sockets, or HTTP clients;
- do not access grader, verifier, solution, private_test, secrets, or ground truth;
- do not hardcode benchmark names, task IDs, answers, or evaluator behavior;
- do not edit incoming_harness_store.json or files under history/;
- the close() state update receives agent-visible trajectories only;
- behavior must fail safely and generalize to unseen tasks.
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.next")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _copy_harness(
    candidate_source: Path,
    state_source: Path,
    destination: Path,
) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    candidate_target = destination / "candidate.py"
    state_target = destination / "harness_store.json"
    shutil.copy2(candidate_source, candidate_target)
    shutil.copy2(state_source, state_target)
    return candidate_target, state_target


def _configure_provider(env_file: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if env_file is not None:
        from dotenv import dotenv_values

        for key, value in dotenv_values(env_file).items():
            if value is not None:
                env[str(key)] = str(value)

    api_key = env.get("JOYROUTER_API_KEY") or env.get("ANTHROPIC_API_KEY")
    base_url = env.get("JOYROUTER_BASE_URL") or env.get("ANTHROPIC_BASE_URL")
    if not api_key:
        raise ValueError("Missing JOYROUTER_API_KEY or ANTHROPIC_API_KEY")
    if not base_url:
        raise ValueError("Missing JOYROUTER_BASE_URL or ANTHROPIC_BASE_URL")
    base_url = base_url.rstrip("/").removesuffix("/v1")
    env["ANTHROPIC_API_KEY"] = api_key
    env["ANTHROPIC_BASE_URL"] = base_url
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ.update(
        {
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_BASE_URL": base_url,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
    )
    return env


def _benchmark_configs(
    benchmarks: list[str], browse_grader_model: str
) -> dict[str, dict[str, Any]]:
    unknown = sorted(set(benchmarks) - set(BENCHMARK_REGISTRY))
    if unknown:
        raise ValueError(f"Unsupported benchmark(s): {', '.join(unknown)}")
    configs: dict[str, dict[str, Any]] = {}
    for slug in benchmarks:
        entry = json.loads(json.dumps(BENCHMARK_REGISTRY[slug]))
        if slug == "browsecompplus":
            entry["grader_kwargs"]["grader_model"] = browse_grader_model
        configs[slug] = entry
    return configs


def _local_task_order(
    configs: dict[str, dict[str, Any]], num_tasks: int, ordering_seed: int
) -> list[tuple[str, str]]:
    task_ids: dict[str, list[str]] = {}
    for slug in sorted(configs):
        backend = create_backend(slug, configs[slug])
        try:
            task_ids[slug] = backend.list_tasks()
        finally:
            backend.close()
    return sequential_task_order(task_ids, num_tasks, ordering_seed)


def _local_task_inventory(
    configs: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    task_ids: dict[str, list[str]] = {}
    for slug in sorted(configs):
        backend = create_backend(slug, configs[slug])
        try:
            task_ids[slug] = backend.list_tasks()
        finally:
            backend.close()
    return task_ids


def _candidate_result(
    *,
    candidate_id: str,
    order: int,
    candidate_path: Path,
    state_path: Path,
    rows: list[dict[str, Any]],
) -> CandidateResult:
    validation = [row for row in rows if row["split"] == "validation"]
    if not validation:
        raise ValueError("candidate evaluation has no validation rows")
    return CandidateResult(
        candidate_id=candidate_id,
        validation_score=sum(row["score"] for row in validation) / len(validation),
        validation_successes=sum(bool(row["success"]) for row in validation),
        validation_tasks=len(validation),
        mean_tokens=sum(
            row["input_tokens"] + row["output_tokens"] for row in validation
        )
        / len(validation),
        mean_cost=sum(row["agent_cost"] for row in validation) / len(validation),
        order=order,
        candidate_path=str(candidate_path.resolve()),
        state_path=str(state_path.resolve()),
        candidate_sha256=_sha256(candidate_path),
        state_sha256=_sha256(state_path),
    )


def _write_frontier(path: Path, candidates: list[CandidateResult]) -> CandidateResult:
    winner = select_winner(candidates)
    _atomic_write_json(
        path,
        {
            "updated_at": _utc_now(),
            "winner": winner.to_dict(),
            "pareto": [item.to_dict() for item in pareto_frontier(candidates)],
            "candidates": [item.to_dict() for item in candidates],
        },
    )
    return winner


def _archive_search_candidate(
    *, output_dir: Path, benchmark_index: int, candidate: CandidateResult, source: Path
) -> None:
    target = (
        output_dir
        / "global_history"
        / "candidates"
        / f"{benchmark_index:03d}_{candidate.candidate_id}"
    )
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(candidate.candidate_path, target / "candidate.py")
    shutil.copy2(candidate.state_path, target / "harness_store.json")
    input_state = source / "input" / "harness_store.json"
    if input_state.is_file():
        shutil.copy2(input_state, target / "incoming_harness_store.json")
    public_source = source / "public"
    if public_source.is_dir():
        shutil.copytree(public_source, target / "public")
    proposal_source = source / "proposal"
    if proposal_source.is_dir():
        proposal_target = target / "proposal"
        proposal_target.mkdir()
        for name in (
            "proposer_stdout.jsonl",
            "proposer_stderr.txt",
            "proposer_meta.json",
            "validation.json",
        ):
            source_file = proposal_source / name
            if source_file.is_file():
                shutil.copy2(source_file, proposal_target / name)
    _atomic_write_json(target / "result.json", candidate.to_dict())


def _prepare_proposer_workspace(
    *,
    workspace: Path,
    parent: CandidateResult,
    incoming_state: Path,
    global_history: Path,
) -> str:
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copy2(parent.candidate_path, workspace / "candidate.py")
    shutil.copy2(incoming_state, workspace / "incoming_harness_store.json")
    incoming_hash = _sha256(workspace / "incoming_harness_store.json")
    if global_history.is_dir():
        shutil.copytree(global_history, workspace / "history")
    else:
        (workspace / "history").mkdir()
    (workspace / "CONTRACT.md").write_text(CONTRACT_TEXT, encoding="utf-8")
    return incoming_hash


def _record_evolution(
    output_dir: Path,
    *,
    benchmark_index: int,
    benchmark: str,
    iteration: int,
    candidate_number: int,
    proposer: dict[str, Any],
    validation: dict[str, Any],
    result: CandidateResult | None,
) -> None:
    _append_jsonl(
        output_dir / "global_history" / "evolution_summary.jsonl",
        {
            "timestamp": _utc_now(),
            "benchmark_index": benchmark_index,
            "benchmark": benchmark,
            "iteration": iteration,
            "candidate_number": candidate_number,
            "proposer": proposer,
            "validation": validation,
            "result": result.to_dict() if result is not None else None,
        },
    )


def _initialize_run(
    *, output_dir: Path, experiment: dict[str, Any], resume: bool
) -> tuple[Path, Path, dict[str, Any]]:
    experiment_path = output_dir / "experiment.json"
    progress_path = output_dir / "progress.json"
    current_dir = output_dir / "current"
    candidate_path = current_dir / "candidate.py"
    state_path = current_dir / "harness_store.json"
    if experiment_path.exists():
        if not resume:
            raise FileExistsError(
                f"Run exists at {output_dir}; pass --resume to continue"
            )
        saved = json.loads(experiment_path.read_text(encoding="utf-8"))
        comparison_keys = (
            "mode",
            "harness_runtime",
            "benchmark_runtime",
            "execution_backend",
            "execution_runtime",
            "benchmarks",
            "benchmark_configs",
            "partition_profile",
            "run_transfer_matrix",
            "bootstrap_samples",
            "bootstrap_seed",
            "task_order",
            "splits",
            "iterations",
            "candidates_per_iteration",
            "base_model",
            "proposer_model",
        )
        mismatches = [
            key for key in comparison_keys if saved.get(key) != experiment.get(key)
        ]
        if mismatches:
            raise ValueError("Resume configuration differs for: " + ", ".join(mismatches))
        validate_candidate(candidate_path, state_path)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        return candidate_path, state_path, progress

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    current_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INTEGRATION_DIR / "candidate.py", candidate_path)
    write_new_harness_state(state_path)
    validate_candidate(candidate_path, state_path)
    _atomic_write_json(experiment_path, experiment)
    progress = {
        "next_benchmark_index": 0,
        "phase": "ready",
        "candidate_sha256": _sha256(candidate_path),
        "state_sha256": _sha256(state_path),
        "updated_at": _utc_now(),
    }
    _atomic_write_json(progress_path, progress)
    return candidate_path, state_path, progress


def _recover_incomplete_benchmark(
    *,
    output_dir: Path,
    benchmark_dir: Path,
    current_candidate: Path,
    current_state: Path,
) -> None:
    incoming_candidate = benchmark_dir / "incoming" / "candidate.py"
    incoming_state = benchmark_dir / "incoming" / "harness_store.json"
    if not incoming_candidate.is_file() or not incoming_state.is_file():
        raise FileNotFoundError(
            f"Cannot recover incomplete benchmark at {benchmark_dir}: missing incoming"
        )
    _atomic_copy(incoming_candidate, current_candidate)
    _atomic_copy(incoming_state, current_state)
    history_snapshot = benchmark_dir / "incoming" / "search_history"
    global_history = output_dir / "global_history"
    if global_history.exists():
        shutil.rmtree(global_history)
    if history_snapshot.is_dir():
        shutil.copytree(history_snapshot, global_history)
    else:
        global_history.mkdir(parents=True, exist_ok=True)

    test_metrics = output_dir / "private_metrics" / "test_metrics.jsonl"
    if test_metrics.is_file():
        benchmark_index = int(benchmark_dir.name.split("_", 1)[0])
        kept = []
        for line in test_metrics.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("benchmark_index", -1)) != benchmark_index:
                kept.append(line)
        test_metrics.write_text(
            "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recovery = output_dir / "recovery_attempts" / f"{benchmark_dir.name}_{stamp}"
    recovery.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(benchmark_dir), str(recovery))


def _snapshot_search_history(output_dir: Path, benchmark_dir: Path) -> None:
    global_history = output_dir / "global_history"
    global_history.mkdir(parents=True, exist_ok=True)
    snapshot = benchmark_dir / "incoming" / "search_history"
    shutil.copytree(global_history, snapshot)


def _split_counts(args: argparse.Namespace) -> SplitCounts:
    supplied = (args.train_tasks, args.validation_tasks, args.test_tasks)
    if all(value is None for value in supplied):
        if args.num_tasks != 50:
            raise ValueError(
                "Non-standard --num-tasks requires --train-tasks, "
                "--validation-tasks, and --test-tasks"
            )
        return SplitCounts(train=30, validation=10, test=10)
    if any(value is None for value in supplied):
        raise ValueError("All three split counts must be provided together")
    counts = SplitCounts(
        train=int(args.train_tasks),
        validation=int(args.validation_tasks),
        test=int(args.test_tasks),
    )
    counts.validate(args.num_tasks)
    return counts


def _search_task_lists(split: BenchmarkSplit) -> tuple[list[str], list[str]]:
    task_ids = [*split.train, *split.validation]
    split_names = ["train"] * len(split.train) + ["validation"] * len(
        split.validation
    )
    return task_ids, split_names


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    env_file = Path(args.env_file).resolve() if args.env_file else None
    provider_env = _configure_provider(env_file)
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    partition_profile = getattr(args, "partition_profile", "legacy")
    run_hidden_transfer = bool(getattr(args, "run_transfer_matrix", False))
    bootstrap_samples = int(getattr(args, "bootstrap_samples", 10_000))
    bootstrap_seed = int(getattr(args, "bootstrap_seed", 2026))
    if bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    if run_hidden_transfer and partition_profile != "transfer-hda":
        raise ValueError(
            "--run-transfer-matrix requires --partition-profile transfer-hda"
        )
    configs = _benchmark_configs(
        benchmarks,
        getattr(args, "browse_grader_model", BASE_MODEL),
    )
    tasks_per_worker = int(getattr(args, "sandbox_tasks_per_worker", 10))
    if tasks_per_worker < 1:
        raise ValueError("--sandbox-tasks-per-worker must be positive")
    for config in configs.values():
        config["sandbox_tasks_per_worker"] = tasks_per_worker
    counts = _split_counts(args) if partition_profile == "legacy" else None
    execution_backend = getattr(args, "execution_backend", "local")
    browse_assets_dir = getattr(args, "browse_assets_dir", None)
    if (
        browse_assets_dir
        and execution_backend == "local"
        and "browsecompplus" in configs
    ):
        configs["browsecompplus"]["backend_kwargs"]["assets_dir"] = str(
            Path(browse_assets_dir).resolve()
        )
    sandbox_backend: OpenSandboxBackend | None = None
    formal_partitions: list[FormalPartition] | None = None
    execution_runtime: dict[str, Any]
    if execution_backend in {"opensandbox", "harbor"}:
        cache_path = Path(
            getattr(
                args,
                "opensandbox_runtime_cache",
                str(DEFAULT_OPENSANDBOX_CACHE_PATH),
            )
        ).expanduser()
        if not cache_path.is_absolute():
            cache_path = REPO_ROOT / cache_path
        assets_root = Path(
            getattr(
                args,
                "opensandbox_assets_root",
                str(DEFAULT_OPENSANDBOX_ASSETS_PATH),
            )
        ).expanduser()
        if not assets_root.is_absolute():
            assets_root = REPO_ROOT / assets_root
        settings = OpenSandboxSettings(
            domain=getattr(args, "opensandbox_domain", None)
            or os.environ.get("OPENSANDBOX_DOMAIN")
            or DEFAULT_OPENSANDBOX_DOMAIN,
            api_key=getattr(args, "opensandbox_api_key", None)
            or os.environ.get("OPENSANDBOX_API_KEY", ""),
            protocol=getattr(args, "opensandbox_protocol", "http"),
            use_server_proxy=getattr(args, "opensandbox_use_server_proxy", True),
            request_timeout_sec=getattr(
                args, "opensandbox_request_timeout", 600
            ),
            ready_timeout_sec=getattr(args, "opensandbox_ready_timeout", 1800),
            sandbox_timeout_sec=getattr(
                args, "opensandbox_sandbox_timeout", 7200
            ),
            command_timeout_sec=getattr(
                args, "opensandbox_command_timeout", 3600
            ),
            snapshot_ready_timeout_sec=getattr(
                args, "opensandbox_snapshot_timeout", 1800
            ),
            image=getattr(args, "opensandbox_image", DEFAULT_OPENSANDBOX_IMAGE),
            runtime_mode=getattr(args, "opensandbox_runtime_mode", "auto"),
            runtime_cache_path=cache_path.resolve(),
            runtime_assets_root=assets_root.resolve(),
            cpus=getattr(args, "opensandbox_cpus", 4),
            memory=getattr(args, "opensandbox_memory", "16Gi"),
        )
        backend_class = OpenSandboxBackend
        if execution_backend == "harbor":
            from .harbor_backend import HarborOpenSandboxBackend

            backend_class = HarborOpenSandboxBackend
        sandbox_backend = backend_class(
            settings=settings,
            meta_harness_root=REPO_ROOT,
            provider_env=provider_env,
        )
        execution_runtime = sandbox_backend.public_config()
        if getattr(args, "prepare_only", False):
            prepared = sandbox_backend.prepare(benchmarks)
            print(
                json.dumps(
                    {
                        "prepared": {
                            slug: {
                                "snapshot_id": snapshot.snapshot_id,
                                "identity": snapshot.identity,
                            }
                            for slug, snapshot in prepared.items()
                        },
                        "runtime": execution_runtime,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if partition_profile == "transfer-hda":
            inventories = sandbox_backend.get_task_inventory(configs)
            formal_partitions = build_formal_partitions(inventories, args.seed)
            splits = [partition.split for partition in formal_partitions]
            task_order = [
                (split.benchmark, task_id)
                for split in splits
                for task_id in split.all_tasks
            ]
        else:
            task_order = sandbox_backend.get_task_order(
                configs, args.num_tasks, args.seed
            )
        block_runner = sandbox_backend.run_block
    elif execution_backend == "local":
        if getattr(args, "prepare_only", False):
            raise ValueError(
                "--prepare-only requires --execution-backend opensandbox or harbor"
            )
        execution_runtime = {"runner": "benchmark-native local"}
        if partition_profile == "transfer-hda":
            inventories = _local_task_inventory(configs)
            formal_partitions = build_formal_partitions(inventories, args.seed)
            splits = [partition.split for partition in formal_partitions]
            task_order = [
                (split.benchmark, task_id)
                for split in splits
                for task_id in split.all_tasks
            ]
        else:
            task_order = _local_task_order(configs, args.num_tasks, args.seed)
        block_runner = _run_block
    else:
        raise ValueError(f"Unsupported execution backend: {execution_backend}")
    if partition_profile == "legacy":
        assert counts is not None
        splits = split_task_order(task_order, counts)
    else:
        assert formal_partitions is not None
    experiment = {
        "created_at": _utc_now(),
        "mode": "sequential_meta_harness",
        "harness_runtime": "benchmark_neutral_candidate_v1",
        "benchmark_runtime": "benchmark_native_backend_v1",
        "execution_backend": execution_backend,
        "execution_runtime": execution_runtime,
        "selection_seed": 42,
        "ordering_seed": args.seed,
        "num_tasks_per_benchmark": args.num_tasks,
        "benchmarks": benchmarks,
        "benchmark_configs": configs,
        "partition_profile": partition_profile,
        "run_transfer_matrix": run_hidden_transfer,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "task_order": [[slug, task_id] for slug, task_id in task_order],
        "splits": [split.to_manifest() for split in splits],
        "iterations": args.iterations,
        "candidates_per_iteration": args.candidates_per_iteration,
        "base_model": args.base_model,
        "proposer_model": args.proposer_model,
        "test_visible_to_proposer": False,
        "evolve_after_every_task": False,
        "stream_protocol_reference": "AgentStream Sequential ordering semantics",
    }
    current_candidate, current_state, progress = _initialize_run(
        output_dir=output_dir, experiment=experiment, resume=args.resume
    )
    if formal_partitions is not None:
        private_manifest_dir = output_dir / "private_manifests"
        private_manifest_dir.mkdir(parents=True, exist_ok=True)
        commitments = {
            partition.benchmark: partition.public_commitment()
            for partition in formal_partitions
        }
        _atomic_write_json(
            output_dir / "public_split_commitment.json", commitments
        )
        for partition in formal_partitions:
            _atomic_write_json(
                private_manifest_dir / f"{partition.benchmark}.json",
                partition.private_manifest(),
            )
        checkpoint_zero = output_dir / "checkpoints" / "H0"
        if not checkpoint_zero.exists():
            checkpoint_candidate, checkpoint_state = _copy_harness(
                current_candidate, current_state, checkpoint_zero
            )
            _atomic_write_json(
                checkpoint_zero / "manifest.json",
                {
                    "checkpoint": "H0",
                    "after_benchmark": None,
                    "candidate_sha256": _sha256(checkpoint_candidate),
                    "state_sha256": _sha256(checkpoint_state),
                },
            )
    next_benchmark = int(progress.get("next_benchmark_index", 0))
    test_metrics_path = output_dir / "private_metrics" / "test_metrics.jsonl"

    print(
        f"Sequential Meta-Harness: benchmarks={len(splits)} "
        f"iterations={args.iterations} candidates/iteration="
        f"{args.candidates_per_iteration} resume_index={next_benchmark}"
    )
    for benchmark_index, split in enumerate(splits):
        if benchmark_index < next_benchmark:
            continue
        benchmark_started = time.monotonic()
        benchmark_dir = (
            output_dir / "benchmarks" / f"{benchmark_index:03d}_{split.benchmark}"
        )
        if benchmark_dir.exists():
            manifest_path = benchmark_dir / "manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {}
            )
            if manifest.get("completed"):
                outgoing = benchmark_dir / "outgoing"
                _atomic_copy(outgoing / "candidate.py", current_candidate)
                _atomic_copy(outgoing / "harness_store.json", current_state)
                next_benchmark = benchmark_index + 1
                _atomic_write_json(
                    output_dir / "progress.json",
                    {
                        "next_benchmark_index": next_benchmark,
                        "phase": "ready",
                        "candidate_sha256": _sha256(current_candidate),
                        "state_sha256": _sha256(current_state),
                        "updated_at": _utc_now(),
                        "repaired_from_completed_manifest": True,
                    },
                )
                continue
            if not args.resume:
                raise FileExistsError(f"Incomplete benchmark exists: {benchmark_dir}")
            _recover_incomplete_benchmark(
                output_dir=output_dir,
                benchmark_dir=benchmark_dir,
                current_candidate=current_candidate,
                current_state=current_state,
            )
        incoming_candidate, incoming_state = _copy_harness(
            current_candidate, current_state, benchmark_dir / "incoming"
        )
        _snapshot_search_history(output_dir, benchmark_dir)
        _atomic_write_json(
            benchmark_dir / "split_manifest.json",
            {
                **split.to_manifest(),
                "selection_seed": 42,
                "ordering_seed": args.seed,
            },
        )
        _atomic_write_json(
            output_dir / "progress.json",
            {
                "next_benchmark_index": benchmark_index,
                "phase": "baseline",
                "benchmark": split.benchmark,
                "candidate_sha256": _sha256(incoming_candidate),
                "state_sha256": _sha256(incoming_state),
                "updated_at": _utc_now(),
            },
        )
        print(
            f"[{benchmark_index + 1}/{len(splits)}] {split.benchmark}: "
            f"train={len(split.train)} val={len(split.validation)} "
            f"test={len(split.test)}"
        )

        search_task_ids, search_split_names = _search_task_lists(split)
        baseline_dir = benchmark_dir / "baseline"
        baseline_candidate, baseline_input_state = _copy_harness(
            incoming_candidate, incoming_state, baseline_dir / "input"
        )
        baseline_output_state = baseline_dir / "output" / "harness_store.json"
        baseline_run = block_runner(
            benchmark_slug=split.benchmark,
            task_ids=search_task_ids,
            split_names=search_split_names,
            candidate_path=baseline_candidate,
            input_state_path=baseline_input_state,
            output_state_path=baseline_output_state,
            evaluation_dir=baseline_dir / "private_evaluation",
            public_dir=baseline_dir / "public",
            config=configs[split.benchmark],
            base_model=args.base_model,
            max_tokens=args.max_tokens,
            embedding_model=args.embedding_model,
        )
        baseline_result = _candidate_result(
            candidate_id="baseline",
            order=0,
            candidate_path=baseline_candidate,
            state_path=baseline_output_state,
            rows=baseline_run.rows,
        )
        candidates = [baseline_result]
        _archive_search_candidate(
            output_dir=output_dir,
            benchmark_index=benchmark_index,
            candidate=baseline_result,
            source=baseline_dir,
        )
        winner = _write_frontier(benchmark_dir / "frontier.json", candidates)
        print(f"  baseline val={baseline_result.validation_score:.3f}")

        evaluation_order = 1
        for iteration in range(1, args.iterations + 1):
            iteration_dir = benchmark_dir / "iterations" / f"{iteration:03d}"
            parent = select_winner(candidates)
            proposal_specs: list[tuple[int, Path, dict[str, Any], dict[str, Any]]] = []
            for candidate_number in range(1, args.candidates_per_iteration + 1):
                workspace = iteration_dir / f"candidate_{candidate_number:03d}" / "proposal"
                incoming_hash = _prepare_proposer_workspace(
                    workspace=workspace,
                    parent=parent,
                    incoming_state=incoming_state,
                    global_history=output_dir / "global_history",
                )
                proposer = run_claude_proposer(
                    workspace=workspace,
                    model=args.proposer_model,
                    base_model=args.base_model,
                    claude_bin=args.claude_bin,
                    timeout_seconds=args.proposer_timeout,
                    benchmark=split.benchmark,
                    iteration=iteration,
                    candidate_number=candidate_number,
                    env=provider_env,
                )
                validation: dict[str, Any]
                if _sha256(workspace / "incoming_harness_store.json") != incoming_hash:
                    validation = {
                        "valid": False,
                        "error": "proposer edited incoming_harness_store.json",
                    }
                elif proposer["exit_code"] != 0:
                    validation = {
                        "valid": False,
                        "error": f"proposer exit code {proposer['exit_code']}",
                    }
                else:
                    try:
                        validation = validate_candidate(
                            workspace / "candidate.py", incoming_state
                        )
                    except (
                        CandidateValidationError,
                        ImportError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        validation = {
                            "valid": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                _atomic_write_json(workspace / "validation.json", validation)
                proposal_specs.append(
                    (candidate_number, workspace, proposer, validation)
                )

            for candidate_number, workspace, proposer, validation in proposal_specs:
                result: CandidateResult | None = None
                if validation.get("valid"):
                    candidate_id = f"iter{iteration:03d}_candidate{candidate_number:03d}"
                    candidate_dir = iteration_dir / f"candidate_{candidate_number:03d}"
                    evaluated_candidate = candidate_dir / "input" / "candidate.py"
                    evaluated_state = candidate_dir / "input" / "harness_store.json"
                    evaluated_candidate.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(workspace / "candidate.py", evaluated_candidate)
                    shutil.copy2(incoming_state, evaluated_state)
                    output_state = candidate_dir / "output" / "harness_store.json"
                    try:
                        block = block_runner(
                            benchmark_slug=split.benchmark,
                            task_ids=search_task_ids,
                            split_names=search_split_names,
                            candidate_path=evaluated_candidate,
                            input_state_path=evaluated_state,
                            output_state_path=output_state,
                            evaluation_dir=candidate_dir / "private_evaluation",
                            public_dir=candidate_dir / "public",
                            config=configs[split.benchmark],
                            base_model=args.base_model,
                            max_tokens=args.max_tokens,
                            embedding_model=args.embedding_model,
                        )
                        result = _candidate_result(
                            candidate_id=candidate_id,
                            order=evaluation_order,
                            candidate_path=evaluated_candidate,
                            state_path=output_state,
                            rows=block.rows,
                        )
                    except Exception as exc:  # noqa: BLE001 - generated candidate failed
                        validation = {
                            **validation,
                            "evaluation_valid": False,
                            "evaluation_error_type": type(exc).__name__,
                            "evaluation_error": str(exc),
                        }
                        _atomic_write_json(
                            candidate_dir / "evaluation_error.json", validation
                        )
                        print(
                            f"  iteration={iteration} candidate={candidate_number} "
                            f"evaluation failed: {type(exc).__name__}: {exc}"
                        )
                    else:
                        evaluation_order += 1
                        candidates.append(result)
                        _archive_search_candidate(
                            output_dir=output_dir,
                            benchmark_index=benchmark_index,
                            candidate=result,
                            source=candidate_dir,
                        )
                        winner = _write_frontier(
                            benchmark_dir / "frontier.json", candidates
                        )
                        print(
                            f"  iteration={iteration} candidate={candidate_number} "
                            f"val={result.validation_score:.3f} "
                            f"winner={winner.candidate_id}"
                        )
                _record_evolution(
                    output_dir,
                    benchmark_index=benchmark_index,
                    benchmark=split.benchmark,
                    iteration=iteration,
                    candidate_number=candidate_number,
                    proposer=proposer,
                    validation=validation,
                    result=result,
                )

        winner = _write_frontier(benchmark_dir / "frontier.json", candidates)
        winner_candidate, winner_state = _copy_harness(
            Path(winner.candidate_path),
            Path(winner.state_path),
            benchmark_dir / "winner",
        )
        _atomic_write_json(
            benchmark_dir / "winner" / "manifest.json", winner.to_dict()
        )

        _atomic_write_json(
            output_dir / "progress.json",
            {
                "next_benchmark_index": benchmark_index,
                "phase": "test",
                "benchmark": split.benchmark,
                "winner": winner.candidate_id,
                "candidate_sha256": _sha256(winner_candidate),
                "state_sha256": _sha256(winner_state),
                "updated_at": _utc_now(),
            },
        )
        test_score: float | None = None
        if partition_profile == "legacy":
            private_test_dir = benchmark_dir / "private_test"
            test_state = private_test_dir / "output" / "harness_store.json"
            test_run = block_runner(
                benchmark_slug=split.benchmark,
                task_ids=list(split.test),
                split_names=["test"] * len(split.test),
                candidate_path=winner_candidate,
                input_state_path=winner_state,
                output_state_path=test_state,
                evaluation_dir=private_test_dir / "evaluation",
                public_dir=None,
                config=configs[split.benchmark],
                base_model=args.base_model,
                max_tokens=args.max_tokens,
                embedding_model=args.embedding_model,
            )
            test_score = sum(row["score"] for row in test_run.rows) / len(
                test_run.rows
            )
            test_record = {
                "timestamp": _utc_now(),
                "benchmark_index": benchmark_index,
                "benchmark": split.benchmark,
                "winner": winner.candidate_id,
                "validation_score": winner.validation_score,
                "test_score": test_score,
                "test_tasks": test_run.rows,
                "candidate_sha256": _sha256(winner_candidate),
                "state_before_test_sha256": _sha256(winner_state),
                "state_after_test_sha256": _sha256(test_state),
            }
            _atomic_write_json(private_test_dir / "metrics.json", test_record)
            _append_jsonl(test_metrics_path, test_record)

        # Hidden test execution is observational only. Its state must not flow
        # into the next benchmark, otherwise transfer is confounded by learning
        # on hidden tasks.
        outgoing_candidate, outgoing_state = _copy_harness(
            winner_candidate, winner_state, benchmark_dir / "outgoing"
        )
        _atomic_copy(outgoing_candidate, current_candidate)
        _atomic_copy(outgoing_state, current_state)
        if formal_partitions is not None:
            checkpoint = output_dir / "checkpoints" / f"H{benchmark_index + 1}"
            checkpoint_candidate, checkpoint_state = _copy_harness(
                outgoing_candidate, outgoing_state, checkpoint
            )
            _atomic_write_json(
                checkpoint / "manifest.json",
                {
                    "checkpoint": f"H{benchmark_index + 1}",
                    "after_benchmark": split.benchmark,
                    "candidate_sha256": _sha256(checkpoint_candidate),
                    "state_sha256": _sha256(checkpoint_state),
                },
            )
        _atomic_write_json(
            benchmark_dir / "manifest.json",
            {
                "completed": True,
                "completed_at": _utc_now(),
                "benchmark": split.benchmark,
                "winner": winner.candidate_id,
                "validation_score": winner.validation_score,
                "candidate_sha256": _sha256(outgoing_candidate),
                "state_sha256": _sha256(outgoing_state),
                "elapsed_seconds": round(time.monotonic() - benchmark_started, 3),
            },
        )
        _atomic_write_json(
            output_dir / "progress.json",
            {
                "next_benchmark_index": benchmark_index + 1,
                "phase": "ready",
                "candidate_sha256": _sha256(current_candidate),
                "state_sha256": _sha256(current_state),
                "updated_at": _utc_now(),
            },
        )
        test_text = f" test={test_score:.3f}" if test_score is not None else ""
        print(
            f"  winner={winner.candidate_id} val={winner.validation_score:.3f}"
            f"{test_text}"
        )

    transfer_result: dict[str, Any] | None = None
    if run_hidden_transfer:
        assert formal_partitions is not None
        matrix_path = output_dir / "transfer_matrix" / "matrix.json"
        if matrix_path.is_file():
            existing_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            if existing_matrix.get("complete"):
                transfer_result = existing_matrix
        if transfer_result is None:
            transfer_result = run_transfer_matrix(
                output_dir=output_dir,
                partitions=formal_partitions,
                block_runner=block_runner,
                configs=configs,
                base_model=args.base_model,
                max_tokens=args.max_tokens,
                embedding_model=args.embedding_model,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        prepare_hda_review(
            output_dir=output_dir, transfer_result=transfer_result
        )

    test_records = []
    if test_metrics_path.is_file():
        test_records = [
            json.loads(line)
            for line in test_metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if partition_profile == "transfer-hda":
        summary = {
            "completed_at": _utc_now(),
            "partition_profile": partition_profile,
            "evolution_benchmarks": len(splits),
            "checkpoints": ["H0", "H1", "H2"],
            "transfer_matrix_complete": transfer_result is not None,
            "transfer_cell_means": (
                {
                    name: cell["mean_score"]
                    for name, cell in transfer_result["cells"].items()
                }
                if transfer_result is not None
                else None
            ),
            "deltas": (
                transfer_result["deltas"]
                if transfer_result is not None
                else None
            ),
        }
    else:
        summary = {
            "completed_at": _utc_now(),
            "benchmarks": len(test_records),
            "mean_test_score": (
                sum(record["test_score"] for record in test_records)
                / len(test_records)
                if test_records
                else None
            ),
            "per_benchmark": {
                record["benchmark"]: {
                    "winner": record["winner"],
                    "validation_score": record["validation_score"],
                    "test_score": record["test_score"],
                }
                for record in test_records
            },
        }
    _atomic_write_json(output_dir / "private_metrics" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run continual Meta-Harness search over benchmark-native task streams"
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmarks", default="bfcl,browsecompplus")
    parser.add_argument(
        "--partition-profile",
        choices=("legacy", "transfer-hda"),
        default="legacy",
    )
    parser.add_argument("--run-transfer-matrix", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--train-tasks", type=int)
    parser.add_argument("--validation-tasks", type=int)
    parser.add_argument("--test-tasks", type=int)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--candidates-per-iteration", type=int, default=1)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--proposer-model", default=PROPOSER_MODEL)
    parser.add_argument("--browse-grader-model", default=BASE_MODEL)
    parser.add_argument("--browse-assets-dir")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--proposer-timeout", type=int, default=2400)
    parser.add_argument(
        "--claude-bin",
        default=(
            "/root/.local/bin/claude"
            if Path("/root/.local/bin/claude").is_file()
            else "claude"
        ),
    )
    parser.add_argument("--env-file")
    parser.add_argument(
        "--execution-backend",
        choices=("opensandbox", "harbor", "local"),
        default="opensandbox",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--opensandbox-domain")
    parser.add_argument("--opensandbox-api-key")
    parser.add_argument(
        "--opensandbox-protocol", choices=("http", "https"), default="http"
    )
    parser.add_argument(
        "--opensandbox-use-server-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--opensandbox-image", default=DEFAULT_OPENSANDBOX_IMAGE)
    parser.add_argument(
        "--opensandbox-runtime-mode",
        choices=("auto", "require", "rebuild"),
        default="auto",
    )
    parser.add_argument(
        "--opensandbox-runtime-cache",
        default=str(DEFAULT_OPENSANDBOX_CACHE_PATH),
    )
    parser.add_argument(
        "--opensandbox-assets-root",
        default=str(DEFAULT_OPENSANDBOX_ASSETS_PATH),
    )
    parser.add_argument("--opensandbox-cpus", type=int, default=4)
    parser.add_argument("--opensandbox-memory", default="16Gi")
    parser.add_argument("--sandbox-tasks-per-worker", type=int, default=10)
    parser.add_argument("--opensandbox-request-timeout", type=int, default=600)
    parser.add_argument("--opensandbox-ready-timeout", type=int, default=1800)
    parser.add_argument("--opensandbox-sandbox-timeout", type=int, default=7200)
    parser.add_argument("--opensandbox-command-timeout", type=int, default=3600)
    parser.add_argument("--opensandbox-snapshot-timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")
    if args.candidates_per_iteration < 1:
        raise SystemExit("--candidates-per-iteration must be positive")
    run(args)


if __name__ == "__main__":
    main()
