from __future__ import annotations

import json
from pathlib import Path

from integrations.agentstream_sequential_meta.hda_reporting import (
    prepare_hda_review,
)


def _make_checkpoints(root: Path) -> None:
    for index, name in enumerate(("H0", "H1", "H2")):
        checkpoint = root / "checkpoints" / name
        checkpoint.mkdir(parents=True)
        (checkpoint / "candidate.py").write_text(f"version = {index}\n")
        (checkpoint / "harness_store.json").write_text(
            json.dumps({"version": index})
        )


def _transfer(significant: bool):
    cells = {}
    for checkpoint_index, checkpoint in enumerate(("H0", "H1", "H2")):
        for benchmark in ("bfcl", "browsecompplus"):
            cells[f"{checkpoint}__{benchmark}"] = {
                "checkpoint": checkpoint,
                "benchmark": benchmark,
                "mean_score": checkpoint_index / 2,
                "stream_scores": [checkpoint_index / 2] * 2,
                "streams": [
                    {
                        "tasks": [
                            {
                                "model_calls": checkpoint_index + 1,
                                "input_tokens": 10,
                                "output_tokens": 2,
                                "steps": 1,
                            }
                        ]
                    }
                ],
            }
    delta = {
        "delta": 0.5,
        "ci_95": [0.1, 0.9] if significant else [-0.1, 0.9],
        "p_value": 0.01 if significant else 0.2,
        "significant": significant,
        "paired_streams": 2,
        "paired_differences": [0.5, 0.5],
    }
    return {
        "cells": cells,
        "deltas": {
            "bfcl_to_browse_transfer": dict(delta),
            "bfcl_backward_transfer": {**delta, "significant": False},
            "bfcl_in_domain_learning": {**delta, "significant": False},
            "browse_in_domain_learning": {**delta, "significant": False},
            "bfcl_forgetting": {**delta, "significant": False},
        },
    }


def test_hda_report_stops_when_gate_is_within_noise(tmp_path: Path) -> None:
    _make_checkpoints(tmp_path)
    manifest = prepare_hda_review(
        output_dir=tmp_path, transfer_result=_transfer(False)
    )

    assert manifest["status"] == "within_noise"
    assert manifest["selected_pair"] is None
    report = tmp_path / "report" / "agentstream_transfer_hda.html"
    assert "no T/O/G bar is valid" in report.read_text()


def test_hda_report_selects_priority_pair_but_does_not_approve(tmp_path: Path) -> None:
    _make_checkpoints(tmp_path)
    manifest = prepare_hda_review(
        output_dir=tmp_path, transfer_result=_transfer(True)
    )

    assert manifest["selected_pair"]["delta_name"] == (
        "bfcl_to_browse_transfer"
    )
    assert manifest["status"] == "awaiting_compute_plan_approval"
    assert manifest["approvals"] == {
        "compute_matching_plan": False,
        "neutralization_diff": False,
    }
    assert "H0/candidate.py" in (
        tmp_path / "hda" / "candidate_pair.diff"
    ).read_text()
