from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.agentstream_sequential_meta.experiment_protocol import (
    FormalPartition,
)
from integrations.agentstream_sequential_meta.protocol import BenchmarkSplit
from integrations.agentstream_sequential_meta.sandbox_evaluation import BlockRun
from integrations.agentstream_sequential_meta.transfer_evaluation import (
    compute_transfer_deltas,
    paired_bootstrap,
    run_transfer_matrix,
)


def test_paired_bootstrap_uses_stream_level_differences() -> None:
    result = paired_bootstrap(
        [0.0, 0.2, 0.4], [0.5, 0.7, 0.9], samples=500, seed=1
    )

    assert result["delta"] == 0.5
    assert result["paired_differences"] == pytest.approx([0.5, 0.5, 0.5])
    assert result["ci_95"] == pytest.approx([0.5, 0.5])
    assert result["significant"]


def test_transfer_delta_definitions_include_signed_forgetting() -> None:
    cells = {
        "H0__bfcl": {"stream_scores": [0.0, 0.0]},
        "H1__bfcl": {"stream_scores": [1.0, 0.5]},
        "H2__bfcl": {"stream_scores": [0.5, 0.0]},
        "H0__browsecompplus": {"stream_scores": [0.0, 0.5]},
        "H1__browsecompplus": {"stream_scores": [0.5, 1.0]},
        "H2__browsecompplus": {"stream_scores": [1.0, 1.0]},
    }

    result = compute_transfer_deltas(cells, samples=100, seed=1)

    assert result["bfcl_to_browse_transfer"]["delta"] == 0.5
    assert result["bfcl_backward_transfer"]["delta"] == -0.5
    assert result["bfcl_forgetting"]["delta"] == 0.5


def test_transfer_streams_start_from_independent_checkpoint_copies(
    tmp_path: Path,
) -> None:
    for name, score in (("H0", 0.0), ("H1", 0.5), ("H2", 1.0)):
        checkpoint = tmp_path / "checkpoints" / name
        checkpoint.mkdir(parents=True)
        (checkpoint / "candidate.py").write_text("candidate")
        (checkpoint / "harness_store.json").write_text(
            json.dumps({"count": 0, "score": score})
        )
    partitions = [
        FormalPartition(
            benchmark=benchmark,
            split=BenchmarkSplit(
                benchmark=benchmark,
                train=(),
                validation=(),
                test=(f"{benchmark}-0", f"{benchmark}-1"),
            ),
            audit=(),
            hidden_streams=(
                (f"{benchmark}-0",),
                (f"{benchmark}-1",),
            ),
            pool_fingerprint="fingerprint",
        )
        for benchmark in ("bfcl", "browsecompplus")
    ]

    def fake_runner(**kwargs):
        state = json.loads(Path(kwargs["input_state_path"]).read_text())
        assert state["count"] == 0
        state["count"] = len(kwargs["task_ids"])
        output = Path(kwargs["output_state_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(state))
        rows = [
            {"task_id": task_id, "score": state["score"]}
            for task_id in kwargs["task_ids"]
        ]
        return BlockRun(rows=rows, state_path=output)

    result = run_transfer_matrix(
        output_dir=tmp_path,
        partitions=partitions,
        block_runner=fake_runner,
        configs={"bfcl": {}, "browsecompplus": {}},
        base_model="model",
        max_tokens=10,
        embedding_model="unused",
        bootstrap_samples=100,
    )

    assert len(result["cells"]) == 6
    assert result["cells"]["H1__bfcl"]["stream_scores"] == [0.5, 0.5]
    for name in ("H0", "H1", "H2"):
        state = json.loads(
            (tmp_path / "checkpoints" / name / "harness_store.json").read_text()
        )
        assert state["count"] == 0
