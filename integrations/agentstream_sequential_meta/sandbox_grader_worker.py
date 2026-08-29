"""Verifier-only worker executed after the solver sandbox has been destroyed."""

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


def grade(request: dict[str, Any]) -> None:
    from integrations.agentstream_sequential_meta.grading import (
        grade_artifacts,
    )

    results = grade_artifacts(
        benchmark_slug=str(request["benchmark_slug"]),
        artifacts=[dict(item) for item in request["grading_artifacts"]],
        config=dict(request["config"]),
        grader_attempts=int(request.get("grader_attempts", 3)),
    )
    _write_result({"grade_results": results})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("grade-artifacts",))
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    shutil.rmtree("/work/result", ignore_errors=True)
    grade(request)


if __name__ == "__main__":
    main()
