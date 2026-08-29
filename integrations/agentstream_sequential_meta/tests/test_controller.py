from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from integrations.agentstream_sequential_meta import controller
from integrations.agentstream_sequential_meta.controller import (
    BlockRun,
    _recover_incomplete_benchmark,
    _split_counts,
)


def _args(**overrides: int | None) -> argparse.Namespace:
    values = {
        "num_tasks": 50,
        "train_tasks": None,
        "validation_tasks": None,
        "test_tasks": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_official_default_split_is_30_10_10() -> None:
    counts = _split_counts(_args())

    assert (counts.train, counts.validation, counts.test) == (30, 10, 10)


def test_nonstandard_task_count_requires_explicit_split() -> None:
    with pytest.raises(ValueError, match="Non-standard"):
        _split_counts(_args(num_tasks=4))


def test_explicit_split_must_cover_all_tasks() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        _split_counts(
            _args(
                num_tasks=4,
                train_tasks=2,
                validation_tasks=1,
                test_tasks=2,
            )
        )


def test_incomplete_benchmark_restores_complete_incoming_harness(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    benchmark_dir = output_dir / "benchmarks" / "000_bfcl"
    incoming = benchmark_dir / "incoming"
    incoming.mkdir(parents=True)
    (incoming / "candidate.py").write_text("incoming-code", encoding="utf-8")
    (incoming / "harness_store.json").write_text(
        json.dumps({"memory": "incoming"}), encoding="utf-8"
    )
    history_snapshot = incoming / "search_history"
    history_snapshot.mkdir()
    (history_snapshot / "prior.txt").write_text("prior", encoding="utf-8")
    global_history = output_dir / "global_history"
    global_history.mkdir()
    (global_history / "partial.txt").write_text("partial", encoding="utf-8")
    private_metrics = output_dir / "private_metrics"
    private_metrics.mkdir()
    (private_metrics / "test_metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"benchmark_index": 0, "test_score": 1.0}),
                json.dumps({"benchmark_index": 9, "test_score": 0.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    current = output_dir / "current"
    current.mkdir(parents=True)
    current_candidate = current / "candidate.py"
    current_state = current / "harness_store.json"
    current_candidate.write_text("partial-code", encoding="utf-8")
    current_state.write_text(json.dumps({"memory": "partial"}), encoding="utf-8")

    _recover_incomplete_benchmark(
        output_dir=output_dir,
        benchmark_dir=benchmark_dir,
        current_candidate=current_candidate,
        current_state=current_state,
    )

    assert current_candidate.read_text(encoding="utf-8") == "incoming-code"
    assert json.loads(current_state.read_text(encoding="utf-8"))["memory"] == "incoming"
    assert not benchmark_dir.exists()
    assert len(list((output_dir / "recovery_attempts").iterdir())) == 1
    assert (global_history / "prior.txt").read_text(encoding="utf-8") == "prior"
    assert not (global_history / "partial.txt").exists()
    remaining = [
        json.loads(line)
        for line in (private_metrics / "test_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert remaining == [{"benchmark_index": 9, "test_score": 0.5}]


def test_control_run_transfers_state_and_keeps_test_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_order = [
        *(("bfcl", f"b{i}") for i in range(4)),
        *(("browsecompplus", f"w{i}") for i in range(4)),
    ]
    monkeypatch.setattr(
        controller,
        "_local_task_order",
        lambda *args, **kwargs: task_order,
    )
    monkeypatch.setattr(controller, "_configure_provider", lambda env_file: {})

    def fake_run_block(**kwargs: Any) -> BlockRun:
        input_state = json.loads(
            Path(kwargs["input_state_path"]).read_text(encoding="utf-8")
        )
        split_names = list(kwargs["split_names"])
        benchmark = str(kwargs["benchmark_slug"])
        input_state["memory"] += f"|{benchmark}:{','.join(split_names)}"
        input_state["session_count"] += len(split_names)
        output_state = Path(kwargs["output_state_path"])
        output_state.parent.mkdir(parents=True, exist_ok=True)
        output_state.write_text(json.dumps(input_state), encoding="utf-8")
        rows = [
            {
                "task_id": task_id,
                "split": split_name,
                "score": 1.0,
                "success": True,
                "status": "success",
                "steps": 1,
                "action_count": 1,
                "agent_cost": 0.0,
                "execution_time": 0.1,
                "input_tokens": 10,
                "output_tokens": 1,
            }
            for task_id, split_name in zip(
                kwargs["task_ids"], split_names, strict=True
            )
        ]
        public_dir = kwargs["public_dir"]
        if public_dir is not None:
            public_dir = Path(public_dir)
            public_dir.mkdir(parents=True, exist_ok=True)
            (public_dir / "metrics.json").write_text(
                json.dumps({"tasks": rows}), encoding="utf-8"
            )
        return BlockRun(rows=rows, state_path=output_state)

    monkeypatch.setattr(controller, "_run_block", fake_run_block)
    output_dir = tmp_path / "run"
    args = argparse.Namespace(
        output_dir=str(output_dir),
        env_file=None,
        benchmarks="bfcl,browsecompplus",
        num_tasks=4,
        train_tasks=2,
        validation_tasks=1,
        test_tasks=1,
        seed=44,
        iterations=0,
        candidates_per_iteration=1,
        base_model="anthropic/Claude-Opus-4.8-C",
        proposer_model="Claude-Opus-4.8-C",
        max_tokens=128,
        embedding_model="all-MiniLM-L6-v2",
        claude_bin="claude",
        proposer_timeout=1,
        resume=False,
    )

    controller.run(args)

    second_incoming = json.loads(
        (
            output_dir
            / "benchmarks"
            / "001_browsecompplus"
            / "incoming"
            / "harness_store.json"
        ).read_text(encoding="utf-8")
    )
    assert second_incoming["memory"].endswith("|bfcl:train,train,validation")
    assert "|bfcl:test" not in second_incoming["memory"]
    current = json.loads(
        (output_dir / "current" / "harness_store.json").read_text(encoding="utf-8")
    )
    assert current["session_count"] == 6
    assert "|bfcl:test" not in current["memory"]
    assert "|browsecompplus:test" not in current["memory"]
    global_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (output_dir / "global_history").rglob("*")
        if path.is_file()
    )
    assert "test_score" not in global_text
    private_rows = (
        output_dir / "private_metrics" / "test_metrics.jsonl"
    ).read_text(encoding="utf-8")
    assert private_rows.count('"test_score"') == 2


def test_transfer_profile_commits_full_pools_and_writes_h0_h1_h2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        controller,
        "_local_task_inventory",
        lambda configs: {
            "bfcl": [f"b{i}" for i in range(200)],
            "browsecompplus": [f"q{i}" for i in range(830)],
        },
    )
    monkeypatch.setattr(controller, "_configure_provider", lambda env_file: {})
    calls = []

    def fake_run_block(**kwargs: Any) -> BlockRun:
        calls.append((kwargs["benchmark_slug"], list(kwargs["split_names"])))
        state = json.loads(Path(kwargs["input_state_path"]).read_text())
        state["session_count"] += len(kwargs["task_ids"])
        output = Path(kwargs["output_state_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(state))
        rows = [
            {
                "task_id": task_id,
                "split": split,
                "score": 1.0,
                "success": True,
                "status": "success",
                "steps": 1,
                "action_count": 1,
                "agent_cost": 0.0,
                "execution_time": 0.1,
                "input_tokens": 1,
                "output_tokens": 1,
            }
            for task_id, split in zip(
                kwargs["task_ids"], kwargs["split_names"], strict=True
            )
        ]
        public = kwargs["public_dir"]
        if public is not None:
            Path(public).mkdir(parents=True, exist_ok=True)
            (Path(public) / "metrics.json").write_text(json.dumps({"tasks": rows}))
        return BlockRun(rows=rows, state_path=output)

    monkeypatch.setattr(controller, "_run_block", fake_run_block)
    output_dir = tmp_path / "formal"
    controller.run(
        argparse.Namespace(
            output_dir=str(output_dir),
            env_file=None,
            benchmarks="bfcl,browsecompplus",
            partition_profile="transfer-hda",
            num_tasks=50,
            train_tasks=None,
            validation_tasks=None,
            test_tasks=None,
            seed=44,
            iterations=0,
            candidates_per_iteration=1,
            base_model="model",
            proposer_model="proposer",
            max_tokens=128,
            embedding_model="unused",
            claude_bin="claude",
            proposer_timeout=1,
            resume=False,
        )
    )

    assert [len(splits) for _, splits in calls] == [30, 30]
    assert all("test" not in splits for _, splits in calls)
    counts = [
        json.loads(
            (output_dir / "checkpoints" / name / "harness_store.json").read_text()
        )["session_count"]
        for name in ("H0", "H1", "H2")
    ]
    assert counts == [0, 30, 60]
    commitment_text = (output_dir / "public_split_commitment.json").read_text()
    assert '"hidden": [' not in commitment_text
    bfcl_private = json.loads(
        (output_dir / "private_manifests" / "bfcl.json").read_text()
    )
    assert len(bfcl_private["hidden"]) == 50
    assert len(bfcl_private["audit"]) == 120
    assert not (output_dir / "private_metrics" / "test_metrics.jsonl").exists()
