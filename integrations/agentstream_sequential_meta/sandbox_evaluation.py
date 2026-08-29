"""Independent sequential evaluator over benchmark-native environments."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_backends import TaskEnvironment, create_backend
from .candidate_contract import (
    load_candidate_module,
    load_harness_state,
    save_harness_state,
    validate_harness_state,
)
from .harness_protocol import ToolResult
from .model_runtime import LiteLLMModelClient, ModelRuntimeError


class CandidateExecutionError(RuntimeError):
    """A deterministic candidate failure that should not consume infra retries."""


def _candidate_call(callable_: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return callable_(*args, **kwargs)
    except ModelRuntimeError:
        raise
    except Exception as exc:
        raise CandidateExecutionError(f"Candidate execution failed: {exc}") from exc


@dataclass(frozen=True)
class BlockRun:
    rows: list[dict[str, Any]]
    state_path: Path
    grading_artifacts: list[dict[str, Any]] = field(default_factory=list)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in row.items() if key != "error"}
    if "error" in row:
        public["error_type"] = str(row["error"]).split(":", 1)[0]
    return public


def _session_id(task_id: str, task_index: int, attempt: int) -> str:
    return hashlib.sha256(
        f"{task_index}:{task_id}:{attempt}".encode()
    ).hexdigest()[:8]


def _run_harness_task(
    *,
    candidate_class: type,
    state: dict[str, Any],
    environment: TaskEnvironment,
    model_client: Any,
    max_steps: int,
    trajectory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run a candidate and export an unscored verifier artifact."""
    trajectory: list[dict[str, Any]] = [
        {
            "event": "task",
            "task": environment.task,
            "context": environment.context,
        }
    ]
    initial = environment.start()
    for result in initial:
        trajectory.append(
            {
                "event": "tool_result",
                "name": result.name,
                "tool_call_id": result.tool_call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
        )
    harness = _candidate_call(
        candidate_class, model_client=model_client, state=deepcopy(state)
    )
    step = _candidate_call(
        harness.start,
        task=environment.task,
        context=environment.context,
        tools=environment.tools,
        initial_results=initial,
    )
    steps = 0
    action_count = 0
    while not environment.done() and steps < max_steps:
        steps += 1
        for call in step.tool_calls:
            trajectory.append(
                {
                    "event": "tool_call",
                    "name": call.name,
                    "tool_call_id": call.id,
                    "arguments": call.arguments,
                }
            )
        if step.final_text is not None:
            trajectory.append({"event": "assistant_text", "content": step.final_text})
        results, executed = environment.step(step)
        action_count += executed
        if not results:
            results = [
                ToolResult(
                    tool_call_id="no-action",
                    name="environment",
                    content="No valid action was executed; call one supplied tool.",
                    is_error=True,
                )
            ]
        for result in results:
            trajectory.append(
                {
                    "event": "tool_result",
                    "name": result.name,
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
        if environment.done():
            break
        step = _candidate_call(harness.react, results)

    # Deliberately before scoring: candidate state cannot contain score data.
    next_state = _candidate_call(harness.close, deepcopy(trajectory))
    try:
        validate_harness_state(next_state)
    except Exception as exc:
        raise CandidateExecutionError(f"Candidate returned invalid state: {exc}") from exc
    for event in trajectory:
        _append_jsonl(trajectory_path, event)
    artifact = environment.grading_artifact()
    # Reject non-JSON artifacts at the solver boundary, before transport.
    json.dumps(artifact, ensure_ascii=False)
    return next_state, {
        "status": "awaiting_grader",
        "steps": steps,
        "action_count": action_count,
        "agent_cost": float(harness.usage.cost),
        "benchmark_cost": 0.0,
        "execution_time": 0.0,
        "model_calls": int(harness.usage.model_calls),
        "input_tokens": int(harness.usage.input_tokens),
        "output_tokens": int(harness.usage.output_tokens),
        "grader_input_tokens": 0,
        "grader_output_tokens": 0,
    }, artifact


def run_block(
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
    task_attempts: int = 3,
    defer_grading: bool = False,
    grader_attempts: int = 3,
) -> BlockRun:
    # Retained in the controller interface for manifest compatibility.
    del embedding_model
    if len(task_ids) != len(split_names):
        raise ValueError("task_ids and split_names must have the same length")

    state = load_harness_state(input_state_path)
    model_client = LiteLLMModelClient(model=base_model, max_tokens=max_tokens)
    backend = create_backend(benchmark_slug, config)
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    try:
        for task_index, (task_id, split_name) in enumerate(
            zip(task_ids, split_names, strict=True)
        ):
            state_before_task = deepcopy(state)
            last_error: Exception | None = None
            started = time.monotonic()
            if task_attempts < 1:
                raise ValueError("task_attempts must be positive")
            for attempt in range(1, task_attempts + 1):
                session_id = _session_id(task_id, task_index, attempt)
                environment = None
                try:
                    environment = backend.open_task(task_id, attempt_id=session_id)
                    # Reload candidate code for every attempt. Cross-task
                    # persistence is therefore exactly the JSON state.
                    candidate_class = load_candidate_module(
                        candidate_path
                    ).CandidateHarness
                    state, row, artifact = _run_harness_task(
                        candidate_class=candidate_class,
                        state=state_before_task,
                        environment=environment,
                        model_client=model_client,
                        max_steps=int(
                            config.get("agent_kwargs", {}).get("max_steps", 100)
                        ),
                        trajectory_path=(
                            evaluation_dir
                            / "sessions"
                            / session_id
                            / "trajectory.jsonl"
                        ),
                    )
                    row["execution_time"] = time.monotonic() - started
                    row.update(
                        {
                            "task_id": task_id,
                            "split": split_name,
                            "attempts": attempt,
                        }
                    )
                    # Match Harbor's ordering even in local smoke mode: close
                    # the task environment before any verifier is constructed.
                    environment.close()
                    environment = None
                    save_harness_state(output_state_path, state)
                    if defer_grading:
                        artifacts.append(artifact)
                    else:
                        from .grading import grade_artifacts, merge_grade

                        try:
                            grade_result = grade_artifacts(
                                benchmark_slug=benchmark_slug,
                                artifacts=[artifact],
                                config=config,
                                grader_attempts=grader_attempts,
                            )[0]
                        except Exception as exc:
                            grade_result = {
                                "error": f"{type(exc).__name__}: {exc}",
                                "grader_attempts": 0,
                            }
                        row["grader_attempts"] = grade_result["grader_attempts"]
                        if "score" in grade_result:
                            _write_json(
                                evaluation_dir
                                / "private_scores"
                                / f"{session_id}.json",
                                grade_result["score"],
                            )
                            row = merge_grade(row, grade_result["score"])
                        else:
                            row.update(
                                {
                                    "score": 0.0,
                                    "success": False,
                                    "status": "grader_error",
                                    "retryable": True,
                                    "error": grade_result["error"],
                                }
                            )
                    rows.append(row)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    state = deepcopy(state_before_task)
                    if (
                        not isinstance(exc, CandidateExecutionError)
                        and attempt < task_attempts
                    ):
                        continue
                    break
                finally:
                    if environment is not None:
                        with contextlib.suppress(Exception):
                            environment.close()
            if last_error is not None:
                rows.append(
                    {
                        "task_id": task_id,
                        "split": split_name,
                        "score": 0.0,
                        "success": False,
                        "status": "error",
                        "steps": 0,
                        "action_count": 0,
                        "agent_cost": 0.0,
                        "benchmark_cost": 0.0,
                        "execution_time": time.monotonic() - started,
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "grader_input_tokens": 0,
                        "grader_output_tokens": 0,
                        "attempts": attempt,
                        "retryable": not isinstance(
                            last_error, CandidateExecutionError
                        ),
                        "error": f"{type(last_error).__name__}: {last_error}",
                    }
                )
                save_harness_state(output_state_path, state)
    finally:
        with contextlib.suppress(Exception):
            backend.close()

    if not output_state_path.exists():
        save_harness_state(output_state_path, state)
    _write_json(evaluation_dir / "task_rows.json", {"tasks": rows})
    if public_dir is not None:
        public_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            public_dir / "metrics.json",
            {"tasks": [_public_row(row) for row in rows]},
        )
        sessions_dir = evaluation_dir / "sessions"
        if sessions_dir.is_dir():
            shutil.copytree(sessions_dir, public_dir / "rollouts", dirs_exist_ok=True)
    return BlockRun(
        rows=rows, state_path=output_state_path, grading_artifacts=artifacts
    )
