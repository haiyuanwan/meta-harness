"""Hidden-stream transfer matrix and paired significance calculations."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Callable

from .experiment_protocol import FormalPartition


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    before: list[float],
    after: list[float],
    *,
    samples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    if len(before) != len(after) or not before:
        raise ValueError("Paired scores must be non-empty and have equal length")
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    differences = [right - left for left, right in zip(before, after, strict=True)]
    point = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        bootstrapped.append(
            sum(rng.choice(differences) for _ in differences) / len(differences)
        )
    bootstrapped.sort()
    low = _percentile(bootstrapped, 0.025)
    high = _percentile(bootstrapped, 0.975)
    if point > 0:
        tail = sum(value <= 0 for value in bootstrapped) / len(bootstrapped)
    elif point < 0:
        tail = sum(value >= 0 for value in bootstrapped) / len(bootstrapped)
    else:
        tail = 0.5
    return {
        "delta": point,
        "ci_95": [low, high],
        "p_value": min(1.0, 2.0 * tail),
        "significant": low > 0 or high < 0,
        "paired_streams": len(differences),
        "paired_differences": differences,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def compute_transfer_deltas(
    cells: dict[str, dict[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 2026,
) -> dict[str, dict[str, Any]]:
    def scores(checkpoint: str, benchmark: str) -> list[float]:
        return [
            float(value)
            for value in cells[f"{checkpoint}__{benchmark}"]["stream_scores"]
        ]

    pairs = {
        "bfcl_in_domain_learning": (scores("H0", "bfcl"), scores("H1", "bfcl")),
        "bfcl_to_browse_transfer": (
            scores("H0", "browsecompplus"),
            scores("H1", "browsecompplus"),
        ),
        "browse_in_domain_learning": (
            scores("H1", "browsecompplus"),
            scores("H2", "browsecompplus"),
        ),
        "bfcl_backward_transfer": (
            scores("H1", "bfcl"),
            scores("H2", "bfcl"),
        ),
    }
    deltas = {
        name: {
            **paired_bootstrap(before, after, samples=samples, seed=seed),
            "before": before,
            "after": after,
        }
        for name, (before, after) in pairs.items()
    }
    backward = deltas["bfcl_backward_transfer"]
    deltas["bfcl_forgetting"] = {
        **backward,
        "delta": -float(backward["delta"]),
        "ci_95": [
            -float(backward["ci_95"][1]),
            -float(backward["ci_95"][0]),
        ],
        "paired_differences": [
            -float(value) for value in backward["paired_differences"]
        ],
        "definition": "S_bfcl(H1) - S_bfcl(H2)",
    }
    return deltas


def run_transfer_matrix(
    *,
    output_dir: Path,
    partitions: list[FormalPartition],
    block_runner: Callable[..., Any],
    configs: dict[str, dict[str, Any]],
    base_model: str,
    max_tokens: int,
    embedding_model: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    root = output_dir / "transfer_matrix"
    cells: dict[str, dict[str, Any]] = {}
    for checkpoint_name in ("H0", "H1", "H2"):
        checkpoint = output_dir / "checkpoints" / checkpoint_name
        candidate = checkpoint / "candidate.py"
        state = checkpoint / "harness_store.json"
        if not candidate.is_file() or not state.is_file():
            raise FileNotFoundError(f"Missing transfer checkpoint: {checkpoint}")
        for partition in partitions:
            cell_name = f"{checkpoint_name}__{partition.benchmark}"
            cell_dir = root / "cells" / cell_name
            stream_records = []
            for stream_index, task_ids in enumerate(partition.hidden_streams):
                stream_dir = cell_dir / f"stream_{stream_index:03d}"
                input_dir = stream_dir / "input"
                input_dir.mkdir(parents=True, exist_ok=True)
                stream_candidate = input_dir / "candidate.py"
                stream_state = input_dir / "harness_store.json"
                shutil.copy2(candidate, stream_candidate)
                shutil.copy2(state, stream_state)
                output_state = stream_dir / "output" / "harness_store.json"
                block = block_runner(
                    benchmark_slug=partition.benchmark,
                    task_ids=list(task_ids),
                    split_names=["hidden"] * len(task_ids),
                    candidate_path=stream_candidate,
                    input_state_path=stream_state,
                    output_state_path=output_state,
                    evaluation_dir=stream_dir / "private_evaluation",
                    public_dir=None,
                    config=configs[partition.benchmark],
                    base_model=base_model,
                    max_tokens=max_tokens,
                    embedding_model=embedding_model,
                )
                stream_score = sum(float(row["score"]) for row in block.rows) / len(
                    block.rows
                )
                stream_records.append(
                    {
                        "stream_index": stream_index,
                        "score": stream_score,
                        "tasks": block.rows,
                        "state_before_sha256": _sha256(stream_state),
                        "state_after_sha256": _sha256(output_state),
                    }
                )
                _write_json(stream_dir / "metrics.json", stream_records[-1])
            stream_scores = [record["score"] for record in stream_records]
            cell = {
                "checkpoint": checkpoint_name,
                "benchmark": partition.benchmark,
                "checkpoint_candidate_sha256": _sha256(candidate),
                "checkpoint_state_sha256": _sha256(state),
                "mean_score": sum(stream_scores) / len(stream_scores),
                "stream_scores": stream_scores,
                "streams": stream_records,
            }
            cells[cell_name] = cell
            _write_json(cell_dir / "metrics.json", cell)

    deltas = compute_transfer_deltas(
        cells, samples=bootstrap_samples, seed=bootstrap_seed
    )
    result = {
        "complete": True,
        "cells": cells,
        "deltas": deltas,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    _write_json(root / "matrix.json", result)
    _write_json(output_dir / "significance" / "paired_bootstrap.json", deltas)
    return result
