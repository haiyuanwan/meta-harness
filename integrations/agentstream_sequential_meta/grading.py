"""Late verifier orchestration shared by local and isolated executions."""

from __future__ import annotations

import contextlib
from typing import Any


def merge_grade(row: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    finished = score.get("is_finished")
    success = bool(score.get("success", False))
    merged.update(
        {
            "score": float(score.get("score", 0.0) or 0.0),
            "success": success,
            "status": (
                "success"
                if success
                else "unfinished" if finished is False else "unsuccessful"
            ),
        }
    )
    usage = score.get("grader_usage", {})
    merged["benchmark_cost"] = float(usage.get("cost", 0.0) or 0.0)
    merged["grader_input_tokens"] = int(usage.get("input_tokens", 0) or 0)
    merged["grader_output_tokens"] = int(usage.get("output_tokens", 0) or 0)
    return merged


def grade_artifacts(
    *,
    benchmark_slug: str,
    artifacts: list[dict[str, Any]],
    config: dict[str, Any],
    grader_attempts: int = 3,
) -> list[dict[str, Any]]:
    if grader_attempts < 1:
        raise ValueError("grader_attempts must be positive")
    from .benchmark_graders import create_grader

    grader = create_grader(benchmark_slug, config)
    results: list[dict[str, Any]] = []
    try:
        for artifact in artifacts:
            last_error: Exception | None = None
            for attempt in range(1, grader_attempts + 1):
                try:
                    score = grader.grade(artifact)
                    results.append({"score": score, "grader_attempts": attempt})
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                results.append(
                    {
                        "error": f"{type(last_error).__name__}: {last_error}",
                        "grader_attempts": grader_attempts,
                    }
                )
    finally:
        with contextlib.suppress(Exception):
            grader.close()
    return results
