"""Solver-only worker executed inside an OpenSandbox solver runtime."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _write_result(payload: dict[str, Any]) -> None:
    target = Path("/work/result")
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_tasks(request: dict[str, Any]) -> None:
    from integrations.agentstream_sequential_meta.benchmark_backends import (
        create_backend,
    )

    benchmark_slug = str(request["benchmark"])
    backend = create_backend(benchmark_slug, dict(request["config"]))
    try:
        task_ids = [str(item) for item in backend.list_tasks()]
    finally:
        backend.close()
    _write_result({"benchmark": benchmark_slug, "task_ids": task_ids})


def run_solver_block(request: dict[str, Any]) -> None:
    from integrations.agentstream_sequential_meta.sandbox_evaluation import run_block

    result_root = Path("/work/result")
    result_root.mkdir(parents=True, exist_ok=True)
    public_dir = result_root / "public" if request.get("public") else None
    block = run_block(
        benchmark_slug=str(request["benchmark_slug"]),
        task_ids=[str(item) for item in request["task_ids"]],
        split_names=[str(item) for item in request["split_names"]],
        candidate_path=Path("/work/candidate.py"),
        input_state_path=Path("/work/harness_store.json"),
        output_state_path=result_root / "harness_store.json",
        evaluation_dir=result_root / "evaluation",
        public_dir=public_dir,
        config=dict(request["config"]),
        base_model=str(request["base_model"]),
        max_tokens=int(request["max_tokens"]),
        embedding_model=str(request["embedding_model"]),
        task_attempts=int(request.get("task_attempts", 3)),
        defer_grading=True,
    )
    _write_result({"rows": block.rows, "grading_artifacts": block.grading_artifacts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("list-tasks", "run-solver-block"))
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    shutil.rmtree("/work/result", ignore_errors=True)
    if args.operation == "list-tasks":
        list_tasks(request)
    else:
        run_solver_block(request)


if __name__ == "__main__":
    main()
