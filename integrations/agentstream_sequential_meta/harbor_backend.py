"""Harbor-backed scored execution while retaining our continual controller."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .grading import merge_grade
from .harbor_executor import HarborExecutorError, HarborTrialExecutor
from .opensandbox_backend import OpenSandboxBackend
from .sandbox_evaluation import BlockRun, _public_row, _write_json


def _memory_mb(value: str) -> int:
    normalized = value.strip().lower()
    units = {
        "kib": 1 / 1024,
        "mib": 1,
        "gib": 1024,
        "ki": 1 / 1024,
        "mi": 1,
        "gi": 1024,
    }
    for suffix, multiplier in units.items():
        if normalized.endswith(suffix):
            amount = float(normalized[: -len(suffix)])
            result = int(amount * multiplier)
            if result > 0:
                return result
    raise ValueError(f"unsupported Harbor memory value: {value!r}")


class HarborOpenSandboxBackend(OpenSandboxBackend):
    """Use Harbor Trial for solver→separate-verifier lifecycle management."""

    def _harbor_executor(self) -> HarborTrialExecutor:
        return HarborTrialExecutor(
            domain=self.settings.domain,
            api_key=self.settings.api_key,
            protocol=self.settings.protocol,
            use_server_proxy=self.settings.use_server_proxy,
            request_timeout_sec=self.settings.request_timeout_sec,
            ready_timeout_sec=self.settings.ready_timeout_sec,
            sandbox_timeout_sec=self.settings.sandbox_timeout_sec,
            cpus=self.settings.cpus,
            memory_mb=_memory_mb(self.settings.memory),
        )

    def run_block(self, **kwargs: Any) -> BlockRun:
        benchmark = str(kwargs["benchmark_slug"])
        output_state = Path(kwargs["output_state_path"])
        evaluation_dir = Path(kwargs["evaluation_dir"])
        public_value = kwargs.get("public_dir")
        public_dir = Path(public_value) if public_value is not None else None
        task_ids = [str(item) for item in kwargs["task_ids"]]
        split_names = [str(item) for item in kwargs["split_names"]]
        if len(task_ids) != len(split_names):
            raise ValueError("task_ids and split_names must have the same length")

        output_state.parent.mkdir(parents=True, exist_ok=True)
        temporary_state = output_state.with_name(f".{output_state.name}.next")
        shutil.copy2(Path(kwargs["input_state_path"]), temporary_state)
        os.replace(temporary_state, output_state)
        if evaluation_dir.exists():
            shutil.rmtree(evaluation_dir)
        evaluation_dir.mkdir(parents=True)
        if public_dir is not None:
            if public_dir.exists():
                shutil.rmtree(public_dir)
            (public_dir / "rollouts").mkdir(parents=True)

        config = dict(kwargs["config"])
        tasks_per_worker = int(config.get("sandbox_tasks_per_worker", 10))
        if tasks_per_worker < 1:
            raise ValueError("sandbox_tasks_per_worker must be positive")
        solver_snapshot = self.ensure_runtime(benchmark, "solver")
        grader_snapshot = self.ensure_runtime(benchmark, "grader")
        executor = self._harbor_executor()
        rows: list[dict[str, Any]] = []

        for chunk_start in range(0, len(task_ids), tasks_per_worker):
            chunk_ids = task_ids[chunk_start : chunk_start + tasks_per_worker]
            chunk_splits = split_names[chunk_start : chunk_start + tasks_per_worker]
            solver_request = {
                "benchmark_slug": benchmark,
                "task_ids": chunk_ids,
                "split_names": chunk_splits,
                "config": {
                    key: value
                    for key, value in config.items()
                    if key != "grader_kwargs"
                },
                "base_model": kwargs["base_model"],
                "max_tokens": kwargs["max_tokens"],
                "embedding_model": kwargs["embedding_model"],
                "task_attempts": 3,
                "public": public_dir is not None,
            }
            grader_request = {
                "benchmark_slug": benchmark,
                "config": config,
                "grader_attempts": 3,
            }
            committed = False

            def commit_solver_output(result_dir: Path) -> None:
                nonlocal committed
                remote_state = result_dir / "harness_store.json"
                remote_evaluation = result_dir / "evaluation"
                if not remote_state.is_file() or not remote_evaluation.is_dir():
                    raise HarborExecutorError(
                        "Harbor solver result is missing state or evaluation"
                    )
                shutil.copytree(remote_evaluation, evaluation_dir, dirs_exist_ok=True)
                shutil.copy2(remote_state, temporary_state)
                os.replace(temporary_state, output_state)
                if public_dir is not None:
                    remote_rollouts = result_dir / "public" / "rollouts"
                    if remote_rollouts.is_dir():
                        shutil.copytree(
                            remote_rollouts,
                            public_dir / "rollouts" / f"chunk_{chunk_start:04d}",
                            dirs_exist_ok=True,
                        )
                committed = True

            result = None
            solver_attempts = 0
            last_error: Exception | None = None
            for attempt in range(1, 4):
                solver_attempts = attempt
                try:
                    result = executor.run_chunk(
                        task_root=(
                            evaluation_dir
                            / "harbor_tasks"
                            / f"chunk_{chunk_start:04d}_attempt_{attempt}"
                        ),
                        trials_dir=evaluation_dir / "harbor_trials",
                        trial_name=f"{benchmark}_{chunk_start:04d}_attempt_{attempt}",
                        solver_snapshot_id=solver_snapshot.snapshot_id,
                        grader_snapshot_id=grader_snapshot.snapshot_id,
                        candidate_path=Path(kwargs["candidate_path"]),
                        state_path=output_state,
                        solver_request=solver_request,
                        grader_request=grader_request,
                        agent_env=self._worker_env("run-solver-block"),
                        verifier_env=self._worker_env("grade-artifacts"),
                        agent_timeout_sec=self.settings.command_timeout_sec,
                        verifier_timeout_sec=self.settings.command_timeout_sec,
                        on_solver_complete=commit_solver_output,
                        solver_volumes=self.runtime_volumes(
                            benchmark, "solver", read_only=True
                        ),
                        grader_volumes=self.runtime_volumes(
                            benchmark, "grader", read_only=True
                        ),
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    # Once state_after has committed, retrying the Harbor trial
                    # would repeat model actions. Preserve the state and stop.
                    if committed:
                        break

            if result is None or result.solver_result_dir is None:
                error = last_error or HarborExecutorError(
                    result.exception_message if result else "Harbor trial failed"
                )
                for task_id, split in zip(chunk_ids, chunk_splits, strict=True):
                    rows.append(self._error_row(task_id, split, solver_attempts, error))
                continue

            solver_payload = json.loads(
                (result.solver_result_dir / "result.json").read_text(encoding="utf-8")
            )
            chunk_rows = [dict(item) for item in solver_payload.get("rows", [])]
            if len(chunk_rows) != len(chunk_ids):
                raise HarborExecutorError("Harbor solver returned invalid row count")
            grader_results: list[dict[str, Any]] | None = None
            if result.grader_result_path is not None:
                grader_payload = json.loads(
                    result.grader_result_path.read_text(encoding="utf-8")
                )
                raw_results = grader_payload.get("grade_results")
                if isinstance(raw_results, list):
                    grader_results = [dict(item) for item in raw_results]

            grade_iter = iter(grader_results or [])
            for row in chunk_rows:
                row["sandbox_attempts"] = solver_attempts
                if row.get("status") != "awaiting_grader":
                    rows.append(row)
                    continue
                grade_result = next(grade_iter, None)
                if grade_result is None:
                    row.update(
                        {
                            "score": 0.0,
                            "success": False,
                            "status": "grader_error",
                            "retryable": True,
                            "error": (
                                f"{result.exception_type}: {result.exception_message}"
                                if result.exception_type
                                else "HarborVerifierError: missing grader result"
                            ),
                        }
                    )
                elif "score" in grade_result:
                    row = merge_grade(row, dict(grade_result["score"]))
                    row["grader_attempts"] = int(
                        grade_result.get("grader_attempts", 0)
                    )
                else:
                    row.update(
                        {
                            "score": 0.0,
                            "success": False,
                            "status": "grader_error",
                            "retryable": True,
                            "error": str(
                                grade_result.get("error", "private grader failed")
                            ),
                        }
                    )
                rows.append(row)

        _write_json(evaluation_dir / "task_rows.json", {"tasks": rows})
        if public_dir is not None:
            _write_json(
                public_dir / "metrics.json",
                {"tasks": [_public_row(row) for row in rows]},
            )
        return BlockRun(rows=rows, state_path=output_state)

    @staticmethod
    def _error_row(
        task_id: str, split: str, attempts: int, error: Exception
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "split": split,
            "score": 0.0,
            "success": False,
            "status": "error",
            "steps": 0,
            "action_count": 0,
            "agent_cost": 0.0,
            "benchmark_cost": 0.0,
            "execution_time": 0.0,
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "grader_input_tokens": 0,
            "grader_output_tokens": 0,
            "attempts": 3,
            "sandbox_attempts": attempts,
            "retryable": True,
            "error": f"{type(error).__name__}: {error}",
        }
