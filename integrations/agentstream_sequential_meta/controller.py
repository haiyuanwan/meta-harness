"""Run benchmark-level Meta-Harness search on AgentStream Sequential streams."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INTEGRATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTEGRATION_DIR.parents[1]
DEFAULT_AGENTSTREAM_ROOT = Path(
    os.environ.get(
        "AGENTSTREAM_ROOT", "/mnt/public/users/wanhaiyuan/AgentStream/exgentic"
    )
)
DEFAULT_RUNTIME_CACHE = Path(
    os.environ.get(
        "META_HARNESS_RUNTIME_CACHE", "/tmp/meta-harness-agentstream-sequential"
    )
)

os.environ.setdefault("EXGENTIC_CACHE_DIR", str(DEFAULT_RUNTIME_CACHE / "exgentic"))
os.environ.setdefault(
    "EXGENTIC_LITELLM_CACHE_DIR", str(DEFAULT_RUNTIME_CACHE / "litellm")
)
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _add_agentstream_paths(agentstream_root: Path) -> None:
    for path in (
        agentstream_root / "src",
        agentstream_root / "scripts" / "utils",
        REPO_ROOT,
    ):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


_add_agentstream_paths(DEFAULT_AGENTSTREAM_ROOT)

from exgentic.agents.harness.harness_store import HarnessStore
from exgentic.core.types import ModelSettings
from exgentic.interfaces.lib.api import evaluate
from exgentic.interfaces.registry import AGENTS, RegistryEntry, load_benchmark
from task_ordering import get_unified_task_order

from integrations.agentstream_online.contract import (
    CandidateValidationError,
    load_candidate_module,
    validate_candidate,
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

BASE_MODEL = "anthropic/Claude-Opus-4.8-C"
PROPOSER_MODEL = "Claude-Opus-4.8-C"
STORE_ID = "harness_sequential_global"

BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "appworld": {
        "bm_kwargs": {"subset": "test_challenge"},
        "agent_kwargs": {"enable_tool_shortlisting": True, "max_selected_tools": 30},
    },
    "bfcl": {"bm_kwargs": {"subset": "multi_turn_base"}, "agent_kwargs": {}},
    "browsecompplus": {
        "bm_kwargs": {"searcher_type": "faiss", "include_get_document": True},
        "agent_kwargs": {},
    },
    "hle": {"bm_kwargs": {"runner": "direct"}, "agent_kwargs": {}},
    "swebench": {
        "bm_kwargs": {"subset": "princeton-nlp/SWE-bench_Verified"},
        "agent_kwargs": {},
    },
    "tau2": {"bm_kwargs": {"subset": "telecom"}, "agent_kwargs": {}},
}

CONTRACT_TEXT = """# Sequential Meta-Harness Candidate Contract

The complete running harness is candidate.py plus a controller-managed
HarnessStore checkpoint. Edit candidate.py only.

candidate.py must export:

1. CandidatePolicy(CandidatePolicyBase)
2. AgentHarness(OnlineHarnessAgent)

Allowed changes include prompt/context construction, selection among benchmark
tools, and general memory/skill update logic.

Hard constraints:

- keep the fixed solver model and supplied benchmark tools;
- do not replace OnlineHarnessInstance or call AgentStream run_evolver;
- do not import subprocess, sockets, or HTTP clients;
- do not access grader, verifier, solution, private_test, secrets, or ground truth;
- do not hardcode benchmark names, task IDs, answers, or evaluator behavior;
- do not edit incoming_harness_store.json or files under history/;
- hooks must fail safely and generalize to unseen tasks.
"""


@dataclass(frozen=True)
class BlockRun:
    rows: list[dict[str, Any]]
    state_path: Path


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
    benchmarks: list[str], base_model: str
) -> dict[str, dict[str, Any]]:
    unknown = sorted(set(benchmarks) - set(BENCHMARK_REGISTRY))
    if unknown:
        raise ValueError(f"Unsupported benchmark(s): {', '.join(unknown)}")
    configs: dict[str, dict[str, Any]] = {}
    for slug in benchmarks:
        entry = json.loads(json.dumps(BENCHMARK_REGISTRY[slug]))
        if slug == "tau2":
            entry["bm_kwargs"]["user_simulator_model"] = base_model
        elif slug == "browsecompplus":
            entry["bm_kwargs"]["eval_model_id"] = base_model
        elif slug == "hle":
            entry["bm_kwargs"]["judge_model"] = base_model
        configs[slug] = entry
    return configs


def _register_candidate_agent(module: Any) -> None:
    agent_class = module.AgentHarness
    AGENTS[agent_class.slug_name] = RegistryEntry(
        slug_name=agent_class.slug_name,
        display_name=agent_class.display_name,
        module=module.__name__,
        attr="AgentHarness",
        kind="agent",
    )


def _extract_token_counts(cost_reports: dict[str, Any]) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for report in cost_reports.values():
        if isinstance(report, dict):
            input_tokens += int(report.get("input_tokens", 0) or 0)
            output_tokens += int(report.get("output_tokens", 0) or 0)
        else:
            input_tokens += int(getattr(report, "input_tokens", 0) or 0)
            output_tokens += int(getattr(report, "output_tokens", 0) or 0)
    return input_tokens, output_tokens


def _session_row(
    *, task_id: str, split_name: str, session_result: Any
) -> dict[str, Any]:
    score = session_result.score
    if score is None:
        score = 1.0 if session_result.success else 0.0
    input_tokens, output_tokens = _extract_token_counts(session_result.cost_reports)
    return {
        "task_id": task_id,
        "split": split_name,
        "score": float(score),
        "success": bool(session_result.success),
        "status": (
            session_result.status.value
            if hasattr(session_result.status, "value")
            else str(session_result.status)
        ),
        "steps": session_result.steps,
        "action_count": session_result.action_count,
        "agent_cost": float(session_result.agent_cost or 0.0),
        "execution_time": session_result.execution_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _copy_public_rollouts(evaluation_dir: Path, public_dir: Path) -> None:
    trajectories = sorted(
        evaluation_dir.rglob("trajectory.jsonl"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    for index, trajectory in enumerate(trajectories):
        session_dir = trajectory.parent
        target = public_dir / "rollouts" / f"{index:04d}"
        target.mkdir(parents=True, exist_ok=True)
        safe_files = {
            session_dir / "trajectory.jsonl": target / "trajectory.jsonl",
            session_dir / "agent" / "online_candidate_trace.txt": (
                target / "online_candidate_trace.txt"
            ),
            session_dir / "agent" / "agent.log": target / "agent.log",
            session_dir / "agent" / "harness_state.md": target / "harness_state.md",
        }
        for source, destination in safe_files.items():
            if source.is_file():
                shutil.copy2(source, destination)


def _run_block(
    *,
    benchmark_slug: str,
    task_ids: list[str],
    split_names: list[str],
    candidate_path: Path,
    input_state_path: Path,
    output_state_path: Path,
    evaluation_dir: Path,
    public_dir: Path | None,
    config: dict[str, Any],
    base_model: str,
    max_tokens: int,
    embedding_model: str,
) -> BlockRun:
    if len(task_ids) != len(split_names):
        raise ValueError("task_ids and split_names must have the same length")

    HarnessStore.reset_all()
    store = HarnessStore.get_or_create(shuffle_mode="sequential")
    store.load_checkpoint(str(input_state_path))
    module = load_candidate_module(candidate_path)
    _register_candidate_agent(module)
    model_settings = ModelSettings(max_tokens=max_tokens)
    benchmark = load_benchmark(benchmark_slug)(**config["bm_kwargs"])
    agent = module.AgentHarness(
        candidate_path=str(candidate_path.resolve()),
        model=base_model,
        evolver_model=base_model,
        shuffle_mode="sequential",
        benchmark_id=benchmark_slug,
        embedding_model=embedding_model,
        runner="direct",
        model_settings=model_settings,
        **config.get("agent_kwargs", {}),
    )
    try:
        results = evaluate(
            benchmark=benchmark,
            agent=agent,
            task_ids=task_ids,
            max_workers=1,
            output_dir=str(evaluation_dir),
        )
    finally:
        with contextlib.suppress(Exception):
            benchmark.close()

    if len(results.session_results) != len(task_ids):
        raise RuntimeError(
            f"Expected {len(task_ids)} session results, got "
            f"{len(results.session_results)}"
        )
    live_store = HarnessStore.list_stores().get(STORE_ID)
    if live_store is None:
        raise RuntimeError(f"Missing persistent store {STORE_ID}")
    output_state_path.parent.mkdir(parents=True, exist_ok=True)
    live_store.save_checkpoint(str(output_state_path))
    rows = [
        _session_row(task_id=task_id, split_name=split_name, session_result=result)
        for task_id, split_name, result in zip(
            task_ids, split_names, results.session_results, strict=True
        )
    ]
    if public_dir is not None:
        public_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(public_dir / "metrics.json", {"tasks": rows})
        _copy_public_rollouts(evaluation_dir, public_dir)
    return BlockRun(rows=rows, state_path=output_state_path)


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
            "benchmarks",
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
    HarnessStore.reset_all()
    HarnessStore.get_or_create(shuffle_mode="sequential").save_checkpoint(
        str(state_path)
    )
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
    agentstream_root = Path(args.agentstream_root).resolve()
    _add_agentstream_paths(agentstream_root)
    output_dir = Path(args.output_dir).resolve()
    env_file = Path(args.env_file).resolve() if args.env_file else None
    provider_env = _configure_provider(env_file)
    if args.tau2_data_dir:
        tau2_data = str(Path(args.tau2_data_dir).resolve())
        os.environ["TAU2_DATA_DIR"] = tau2_data
        provider_env["TAU2_DATA_DIR"] = tau2_data

    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    configs = _benchmark_configs(benchmarks, args.base_model)
    counts = _split_counts(args)
    task_order = get_unified_task_order(
        configs, args.num_tasks, args.seed, "sequential"
    )
    splits = split_task_order(task_order, counts)
    experiment = {
        "created_at": _utc_now(),
        "mode": "sequential_meta_harness",
        "selection_seed": 42,
        "ordering_seed": args.seed,
        "num_tasks_per_benchmark": args.num_tasks,
        "benchmarks": benchmarks,
        "task_order": [[slug, task_id] for slug, task_id in task_order],
        "splits": [split.to_manifest() for split in splits],
        "iterations": args.iterations,
        "candidates_per_iteration": args.candidates_per_iteration,
        "base_model": args.base_model,
        "proposer_model": args.proposer_model,
        "test_visible_to_proposer": False,
        "evolve_after_every_task": False,
        "agentstream_root": str(agentstream_root),
    }
    current_candidate, current_state, progress = _initialize_run(
        output_dir=output_dir, experiment=experiment, resume=args.resume
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
        baseline_run = _run_block(
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
                        block = _run_block(
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
        private_test_dir = benchmark_dir / "private_test"
        test_state = private_test_dir / "output" / "harness_store.json"
        test_run = _run_block(
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
        test_score = sum(row["score"] for row in test_run.rows) / len(test_run.rows)
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

        outgoing_candidate, outgoing_state = _copy_harness(
            winner_candidate, test_state, benchmark_dir / "outgoing"
        )
        _atomic_copy(outgoing_candidate, current_candidate)
        _atomic_copy(outgoing_state, current_state)
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
        print(
            f"  winner={winner.candidate_id} val={winner.validation_score:.3f} "
            f"test={test_score:.3f}"
        )

    test_records = []
    if test_metrics_path.is_file():
        test_records = [
            json.loads(line)
            for line in test_metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    summary = {
        "completed_at": _utc_now(),
        "benchmarks": len(test_records),
        "mean_test_score": (
            sum(record["test_score"] for record in test_records) / len(test_records)
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
            "Run Meta-Harness search inside each AgentStream Sequential benchmark"
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmarks", default="hle,bfcl,browsecompplus,appworld,swebench,tau2"
    )
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--train-tasks", type=int)
    parser.add_argument("--validation-tasks", type=int)
    parser.add_argument("--test-tasks", type=int)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--candidates-per-iteration", type=int, default=1)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--proposer-model", default=PROPOSER_MODEL)
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
    parser.add_argument("--agentstream-root", default=str(DEFAULT_AGENTSTREAM_ROOT))
    parser.add_argument("--env-file")
    parser.add_argument("--tau2-data-dir")
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
