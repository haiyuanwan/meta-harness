"""Split decrypted BrowseCompPlus data into public solver/private grader files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def create_split_datasets(source: Path, solver_target: Path, grader_target: Path) -> None:
    solver_target.parent.mkdir(parents=True, exist_ok=True)
    grader_target.parent.mkdir(parents=True, exist_ok=True)
    solver_tmp = solver_target.with_name(f".{solver_target.name}.tmp")
    grader_tmp = grader_target.with_name(f".{grader_target.name}.tmp")
    with source.open(encoding="utf-8") as input_handle, solver_tmp.open(
        "w", encoding="utf-8"
    ) as solver_handle, grader_tmp.open("w", encoding="utf-8") as grader_handle:
        for line in input_handle:
            if not line.strip():
                continue
            item = json.loads(line)
            solver = {
                "query_id": item["query_id"],
                "query": item["query"],
            }
            private = {**solver, "answer": item["answer"]}
            for key in ("gold_docs", "evidence_docs", "negative_docs"):
                private[key] = [
                    document.get("docid")
                    for document in item.get(key, [])
                    if isinstance(document, dict)
                    and document.get("docid") is not None
                ]
            solver_handle.write(json.dumps(solver, ensure_ascii=False) + "\n")
            grader_handle.write(json.dumps(private, ensure_ascii=False) + "\n")
    solver_tmp.replace(solver_target)
    grader_tmp.replace(grader_target)


def create_light_dataset(source: Path, target: Path) -> None:
    """Backward-compatible private-light conversion used by older callers."""
    create_split_datasets(source, target.with_name(target.stem + "_solver.jsonl"), target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--solver-output", required=True, type=Path)
    parser.add_argument("--grader-output", required=True, type=Path)
    args = parser.parse_args()
    create_split_datasets(args.input, args.solver_output, args.grader_output)


if __name__ == "__main__":
    main()
