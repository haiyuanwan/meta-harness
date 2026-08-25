"""Run AgentStream one task at a time and evolve the harness after each task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
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
        "META_HARNESS_RUNTIME_CACHE", "/tmp/meta-harness-agentstream-online"
    )
)

# Exgentic configures LiteLLM during import. Point its caches at a writable,
# non-repository location before importing any harness modules.
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
from exgentic.interfaces.registry import (
    AGENTS,
    RegistryEntry,
    load_benchmark,
)
from task_ordering import get_unified_task_order

from .contract import (
    CandidateValidationError,
    load_candidate_module,
    validate_candidate,
)
from .proposer import run_claude_proposer

BASE_MODEL = "anthropic/Claude-Opus-4.8-C"
PROPOSER_MODEL = "Claude-Opus-4.8-C"
STORE_ID = "harness_sequential_global"

BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "bfcl": {
        "bm_kwargs": {
            "subset": "multi_turn_base",
            "runner": "direct",
        },
        "agent_kwargs": {},
    },
    "tau2": {
        "bm_kwargs": {
            "subset": "telecom",
            "user_simulator_model": BASE_MODEL,
            "runner": "direct",
        },
        "agent_kwargs": {},
    },
}

CONTRACT_TEXT = """# Online AgentStream Candidate Contract

`candidate.py` must export:

1. `CandidatePolicy(CandidatePolicyBase)`
2. `AgentHarness(OnlineHarnessAgent)`

Allowed evolution surface:

- transform the fixed solver's system message;
- select a subset of tools already supplied by the benchmark;
- update `HarnessStore` memory and skills deterministically at task close;
- edit `harness_store.json` while preserving its schema.

Hard constraints:

- keep the fixed base model and benchmark tools;
- do not replace `OnlineHarnessInstance`;
- do not call AgentStream's `run_evolver`;
- do not access graders, rewards, benchmark result files, or ground truth;
- do not use subprocesses, sockets, HTTP clients, or add dependencies;
- do not fabricate or alter tool schemas;
- policy hooks must fail safely and generalize to unseen tasks.
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_promote(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.next")
    shutil.copy2(source, tmp)
    os.replace(tmp, target)


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _configure_provider(env_file: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if env_file is not None:
        from dotenv import dotenv_values

        for key, value in dotenv_values(env_file).items():
            if value is not None:
                env[str(key)] = str(value)

    api_key = env.get("JOYROUTER_API_KEY") or env.get("ANTHROPIC_API_KEY")
    joyrouter_base = env.get("JOYROUTER_BASE_URL")
    anthropic_base = env.get("ANTHROPIC_BASE_URL")
    if not api_key:
        raise ValueError("Missing JOYROUTER_API_KEY or ANTHROPIC_API_KEY")
    if joyrouter_base:
        anthropic_base = joyrouter_base.rstrip("/")
        anthropic_base = anthropic_base.removesuffix("/v1")
    if not anthropic_base:
        raise ValueError("Missing JOYROUTER_BASE_URL or ANTHROPIC_BASE_URL")

    env["ANTHROPIC_API_KEY"] = api_key
    env["ANTHROPIC_BASE_URL"] = anthropic_base.rstrip("/")
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ.update(
        {
            "ANTHROPIC_API_KEY": env["ANTHROPIC_API_KEY"],
            "ANTHROPIC_BASE_URL": env["ANTHROPIC_BASE_URL"],
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
    )
    return env


def _benchmark_configs(benchmarks: list[str], base_model: str) -> dict[str, dict[str, Any]]:
    unknown = sorted(set(benchmarks) - set(BENCHMARK_REGISTRY))
    if unknown:
        raise ValueError(f"Unsupported benchmark(s): {', '.join(unknown)}")
    configs: dict[str, dict[str, Any]] = {}
    for slug in benchmarks:
        entry = json.loads(json.dumps(BENCHMARK_REGISTRY[slug]))
        if slug == "tau2":
            entry["bm_kwargs"]["user_simulator_model"] = base_model
        configs[slug] = entry
    return configs


def _discover_session_dir(evaluation_dir: Path) -> Path:
    trajectories = sorted(
        evaluation_dir.rglob("trajectory.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not trajectories:
        raise FileNotFoundError(f"No trajectory.jsonl under {evaluation_dir}")
    return trajectories[-1].parent


def _register_candidate_agent(module: Any) -> None:
    """Make the dynamic candidate visible to Exgentic's RunConfig loader."""

    agent_cls = module.AgentHarness
    AGENTS[agent_cls.slug_name] = RegistryEntry(
        slug_name=agent_cls.slug_name,
        display_name=agent_cls.display_name,
        module=module.__name__,
        attr="AgentHarness",
        kind="agent",
    )


def _copy_public_evidence(session_dir: Path, evidence_dir: Path) -> list[str]:
    copied: list[str] = []
    candidates = [
        (session_dir / "trajectory.jsonl", evidence_dir / "trajectory.jsonl"),
        (
            session_dir / "agent" / "online_candidate_trace.txt",
            evidence_dir / "online_candidate_trace.txt",
        ),
        (session_dir / "agent" / "agent.log", evidence_dir / "agent.log"),
        (
            session_dir / "agent" / "litellm" / "trace.jsonl",
            evidence_dir / "litellm_trace.jsonl",
        ),
        (
            session_dir / "agent" / "harness_state.md",
            evidence_dir / "harness_state.md",
        ),
    ]
    for source, target in candidates:
        if _copy_if_exists(source, target):
            copied.append(target.name)
    return copied


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


def _private_metric(
    *,
    index: int,
    benchmark: str,
    task_id: str,
    session_result: Any,
    base_model: str,
    proposer: dict[str, Any],
    validation: dict[str, Any],
    promoted: bool,
) -> dict[str, Any]:
    input_tokens, output_tokens = _extract_token_counts(session_result.cost_reports)
    score = session_result.score
    if score is None:
        score = 1.0 if session_result.success else 0.0
    return {
        "task_index": index,
        "timestamp": _utc_now(),
        "benchmark": benchmark,
        "task_id": task_id,
        "base_model": base_model,
        "score": score,
        "success": session_result.success,
        "status": (
            session_result.status.value
            if hasattr(session_result.status, "value")
            else str(session_result.status)
        ),
        "steps": session_result.steps,
        "action_count": session_result.action_count,
        "agent_cost": session_result.agent_cost,
        "execution_time": session_result.execution_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "proposer": proposer,
        "candidate_validation": validation,
        "promoted": promoted,
    }


def _initialize_run(
    *,
    output_dir: Path,
    experiment: dict[str, Any],
    resume: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    current_dir = output_dir / "current"
    candidate_path = current_dir / "candidate.py"
    state_path = current_dir / "harness_store.json"
    experiment_path = output_dir / "experiment.json"
    progress_path = output_dir / "progress.json"

    if experiment_path.exists():
        if not resume:
            raise FileExistsError(
                f"Run already exists at {output_dir}; pass --resume to continue"
            )
        saved = json.loads(experiment_path.read_text(encoding="utf-8"))
        resume_keys = (
            "mode",
            "seed",
            "num_tasks_per_benchmark",
            "benchmarks",
            "task_order",
            "base_model",
            "proposer_model",
        )
        mismatches = [
            key for key in resume_keys if saved.get(key) != experiment.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Resume configuration differs for: " + ", ".join(mismatches)
            )
        validate_candidate(candidate_path, state_path)
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else {"next_task_index": 0}
        )
        return candidate_path, state_path, progress

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is non-empty and is not a run: {output_dir}"
        )

    current_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INTEGRATION_DIR / "candidate.py", candidate_path)
    HarnessStore.reset_all()
    store = HarnessStore.get_or_create(shuffle_mode="sequential")
    store.save_checkpoint(str(state_path))
    validation = validate_candidate(candidate_path, state_path)
    _atomic_write_json(experiment_path, experiment)
    progress = {
        "next_task_index": 0,
        "candidate_sha256": validation["candidate_sha256"],
        "state_sha256": validation["state_sha256"],
        "updated_at": _utc_now(),
    }
    _atomic_write_json(progress_path, progress)
    return candidate_path, state_path, progress


def _recover_incomplete_episode(
    *,
    output_dir: Path,
    episode_dir: Path,
    episode_name: str,
    index: int,
    candidate_path: Path,
    state_path: Path,
    store: HarnessStore,
) -> None:
    """Archive a partial attempt and restore the exact pre-task version."""

    before_candidate = episode_dir / "before" / "candidate.py"
    before_state = episode_dir / "before" / "harness_store.json"
    if not before_candidate.is_file() or not before_state.is_file():
        raise FileNotFoundError(
            f"Cannot recover {episode_name}: missing before/ candidate or state"
        )

    _atomic_promote(before_candidate, candidate_path)
    _atomic_promote(before_state, state_path)
    store.load_checkpoint(str(state_path))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recovery_dir = output_dir / "recovery_attempts" / f"{episode_name}_{stamp}"
    recovery_dir.mkdir(parents=True, exist_ok=False)
    shutil.move(str(episode_dir), str(recovery_dir / "public_episode"))

    workspace = output_dir / "public_history" / "workspaces" / f"{index:06d}"
    if workspace.exists():
        shutil.move(str(workspace), str(recovery_dir / "proposer_workspace"))
    private_evaluation = output_dir / "private_evaluations" / episode_name
    if private_evaluation.exists():
        shutil.move(
            str(private_evaluation), str(recovery_dir / "private_evaluation")
        )
    print(f"  archived incomplete attempt at {recovery_dir}")


def _prepare_proposer_workspace(
    *,
    output_dir: Path,
    episode_dir: Path,
    index: int,
    candidate_path: Path,
    state_path: Path,
) -> Path:
    workspace = output_dir / "public_history" / "workspaces" / f"{index:06d}"
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copy2(candidate_path, workspace / "candidate.py")
    shutil.copy2(state_path, workspace / "harness_store.json")
    shutil.copytree(episode_dir / "evidence", workspace / "evidence")
    (workspace / "CONTRACT.md").write_text(CONTRACT_TEXT, encoding="utf-8")
    return workspace


def _evolve_after_task(
    *,
    output_dir: Path,
    episode_dir: Path,
    index: int,
    benchmark: str,
    task_id: str,
    candidate_path: Path,
    state_path: Path,
    proposer_model: str,
    claude_bin: str,
    proposer_timeout: int,
    provider_env: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    workspace = _prepare_proposer_workspace(
        output_dir=output_dir,
        episode_dir=episode_dir,
        index=index,
        candidate_path=candidate_path,
        state_path=state_path,
    )
    proposer = run_claude_proposer(
        workspace=workspace,
        model=proposer_model,
        claude_bin=claude_bin,
        timeout_seconds=proposer_timeout,
        task_index=index,
        benchmark=benchmark,
        task_id=task_id,
        env=provider_env,
    )

    staged_candidate = workspace / "candidate.py"
    staged_state = workspace / "harness_store.json"
    validation: dict[str, Any]
    promoted = False
    if proposer["exit_code"] != 0:
        validation = {
            "valid": False,
            "error": f"proposer exit code {proposer['exit_code']}",
        }
    else:
        try:
            validation = validate_candidate(staged_candidate, staged_state)
            _atomic_promote(staged_candidate, candidate_path)
            _atomic_promote(staged_state, state_path)
            promoted = True
        except (CandidateValidationError, ImportError, TypeError, ValueError) as exc:
            validation = {
                "valid": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    _atomic_write_json(workspace / "validation.json", validation)
    proposal_archive = episode_dir / "proposal"
    proposal_archive.mkdir(parents=True, exist_ok=True)
    for name in (
        "candidate.py",
        "harness_store.json",
        "proposer_stdout.jsonl",
        "proposer_stderr.txt",
        "proposer_meta.json",
        "validation.json",
    ):
        _copy_if_exists(workspace / name, proposal_archive / name)
    return proposer, validation, promoted


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
    task_order = get_unified_task_order(
        configs,
        args.num_tasks,
        args.seed,
        "sequential",
    )
    experiment = {
        "created_at": _utc_now(),
        "mode": "sequential",
        "online_only": True,
        "evolve_after_every_task": True,
        "score_visible_to_proposer": False,
        "seed": args.seed,
        "num_tasks_per_benchmark": args.num_tasks,
        "benchmarks": benchmarks,
        "task_order": [[slug, task_id] for slug, task_id in task_order],
        "base_model": args.base_model,
        "proposer_model": args.proposer_model,
        "agentstream_root": str(agentstream_root),
    }
    candidate_path, state_path, progress = _initialize_run(
        output_dir=output_dir,
        experiment=experiment,
        resume=args.resume,
    )

    HarnessStore.reset_all()
    store = HarnessStore.get_or_create(shuffle_mode="sequential")
    store.load_checkpoint(str(state_path))

    next_index = int(progress.get("next_task_index", 0))
    metrics_path = output_dir / "private_metrics" / "online_metrics.jsonl"
    cumulative_scores: list[float] = []
    per_benchmark_scores: defaultdict[str, list[float]] = defaultdict(list)
    completed_metric_indices: set[int] = set()
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed_metric_indices.add(int(row["task_index"]))
            cumulative_scores.append(float(row["score"]))
            per_benchmark_scores[str(row["benchmark"])].append(float(row["score"]))

    repaired_index = next_index
    while repaired_index in completed_metric_indices:
        repaired_index += 1
    if repaired_index != next_index:
        next_index = repaired_index
        progress = {
            "next_task_index": next_index,
            "candidate_sha256": _sha256(candidate_path),
            "state_sha256": _sha256(state_path),
            "updated_at": _utc_now(),
            "repaired_from_private_metrics": True,
        }
        _atomic_write_json(output_dir / "progress.json", progress)

    print(
        f"Online Meta-Harness: {len(task_order)} tasks, resume_index={next_index}, "
        f"base={args.base_model}, proposer={args.proposer_model}"
    )

    for index, (benchmark_slug, task_id) in enumerate(task_order):
        if index < next_index:
            continue

        started = time.monotonic()
        episode_name = f"{index:06d}_{benchmark_slug}"
        episode_dir = output_dir / "public_history" / "episodes" / episode_name
        if episode_dir.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Incomplete episode directory already exists: {episode_dir}"
                )
            _recover_incomplete_episode(
                output_dir=output_dir,
                episode_dir=episode_dir,
                episode_name=episode_name,
                index=index,
                candidate_path=candidate_path,
                state_path=state_path,
                store=store,
            )
        (episode_dir / "before").mkdir(parents=True, exist_ok=False)
        shutil.copy2(candidate_path, episode_dir / "before" / "candidate.py")
        shutil.copy2(state_path, episode_dir / "before" / "harness_store.json")
        _atomic_write_json(
            episode_dir / "task.json",
            {
                "task_index": index,
                "benchmark": benchmark_slug,
                "task_id": task_id,
                "base_model": args.base_model,
                "candidate_sha256": _sha256(candidate_path),
                "state_sha256": _sha256(state_path),
            },
        )

        print(f"[{index + 1}/{len(task_order)}] {benchmark_slug}::{task_id}")
        module = load_candidate_module(candidate_path)
        _register_candidate_agent(module)
        agent_cls = module.AgentHarness
        model_settings = ModelSettings(max_tokens=args.max_tokens)
        bm_kwargs = configs[benchmark_slug]["bm_kwargs"]
        benchmark = load_benchmark(benchmark_slug)(**bm_kwargs)
        agent = agent_cls(
            candidate_path=str(candidate_path),
            model=args.base_model,
            evolver_model=args.base_model,
            shuffle_mode="sequential",
            benchmark_id=benchmark_slug,
            embedding_model=args.embedding_model,
            runner="direct",
            model_settings=model_settings,
            **configs[benchmark_slug].get("agent_kwargs", {}),
        )

        evaluation_dir = output_dir / "private_evaluations" / episode_name
        results = evaluate(
            benchmark=benchmark,
            agent=agent,
            task_ids=[task_id],
            max_workers=1,
            output_dir=str(evaluation_dir),
        )
        session_result = results.session_results[0]

        store = HarnessStore.list_stores().get(STORE_ID)
        if store is None:
            raise RuntimeError(f"Missing persistent store {STORE_ID}")
        store.save_checkpoint(str(state_path))
        (episode_dir / "post_task").mkdir(parents=True, exist_ok=True)
        shutil.copy2(state_path, episode_dir / "post_task" / "harness_store.json")

        session_dir = _discover_session_dir(evaluation_dir)
        evidence_files = _copy_public_evidence(session_dir, episode_dir / "evidence")

        proposer, validation, promoted = _evolve_after_task(
            output_dir=output_dir,
            episode_dir=episode_dir,
            index=index,
            benchmark=benchmark_slug,
            task_id=task_id,
            candidate_path=candidate_path,
            state_path=state_path,
            proposer_model=args.proposer_model,
            claude_bin=args.claude_bin,
            proposer_timeout=args.proposer_timeout,
            provider_env=provider_env,
        )

        if promoted:
            store.load_checkpoint(str(state_path))

        (episode_dir / "after").mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, episode_dir / "after" / "candidate.py")
        shutil.copy2(state_path, episode_dir / "after" / "harness_store.json")
        manifest = {
            "task_index": index,
            "benchmark": benchmark_slug,
            "task_id": task_id,
            "evidence_files": evidence_files,
            "proposer_exit_code": proposer["exit_code"],
            "validation_valid": validation.get("valid", False),
            "promoted": promoted,
            "candidate_before_sha256": _sha256(
                episode_dir / "before" / "candidate.py"
            ),
            "candidate_after_sha256": _sha256(candidate_path),
            "state_before_sha256": _sha256(
                episode_dir / "before" / "harness_store.json"
            ),
            "state_after_sha256": _sha256(state_path),
            "completed_at": _utc_now(),
        }
        _atomic_write_json(episode_dir / "manifest.json", manifest)

        private_metric = _private_metric(
            index=index,
            benchmark=benchmark_slug,
            task_id=task_id,
            session_result=session_result,
            base_model=args.base_model,
            proposer=proposer,
            validation=validation,
            promoted=promoted,
        )
        _append_jsonl(metrics_path, private_metric)
        _atomic_write_json(
            evaluation_dir / "online_private_manifest.json", private_metric
        )

        score = float(private_metric["score"])
        cumulative_scores.append(score)
        per_benchmark_scores[benchmark_slug].append(score)
        progress = {
            "next_task_index": index + 1,
            "candidate_sha256": _sha256(candidate_path),
            "state_sha256": _sha256(state_path),
            "updated_at": _utc_now(),
        }
        _atomic_write_json(output_dir / "progress.json", progress)
        elapsed = time.monotonic() - started
        print(
            f"  score={score:.3f} status={private_metric['status']} "
            f"promoted={promoted} elapsed={elapsed:.1f}s"
        )

    summary = {
        "completed_at": _utc_now(),
        "tasks": len(cumulative_scores),
        "average_score": (
            sum(cumulative_scores) / len(cumulative_scores)
            if cumulative_scores
            else None
        ),
        "per_benchmark": {
            benchmark: {
                "tasks": len(scores),
                "average_score": sum(scores) / len(scores),
            }
            for benchmark, scores in sorted(per_benchmark_scores.items())
        },
    }
    _atomic_write_json(output_dir / "private_metrics" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evolve an AgentStream harness with Claude Code after every task"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmarks", default="bfcl,tau2")
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--proposer-model", default=PROPOSER_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--proposer-timeout", type=int, default=900)
    parser.add_argument(
        "--claude-bin",
        default=(
            "/root/.local/bin/claude"
            if Path("/root/.local/bin/claude").is_file()
            else "claude"
        ),
    )
    parser.add_argument(
        "--agentstream-root",
        default=str(DEFAULT_AGENTSTREAM_ROOT),
    )
    parser.add_argument("--env-file")
    parser.add_argument("--tau2-data-dir")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
