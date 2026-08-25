from __future__ import annotations

from pathlib import Path

from exgentic.agents.harness.harness_store import HarnessStore

from integrations.agentstream_online.controller import _recover_incomplete_episode


def test_incomplete_episode_recovery_restores_before_state(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    episode_name = "000000_bfcl"
    episode_dir = output_dir / "public_history" / "episodes" / episode_name
    before_dir = episode_dir / "before"
    before_dir.mkdir(parents=True)

    current_dir = output_dir / "current"
    current_dir.mkdir(parents=True)
    current_candidate = current_dir / "candidate.py"
    current_state = current_dir / "harness_store.json"
    current_candidate.write_text("changed", encoding="utf-8")

    before_candidate = before_dir / "candidate.py"
    before_state = before_dir / "harness_store.json"
    before_candidate.write_text("before", encoding="utf-8")
    before_store = HarnessStore("harness_sequential_global")
    before_store.edit_memory("before-memory")
    before_store.save_checkpoint(str(before_state))

    changed_store = HarnessStore("changed")
    changed_store.edit_memory("changed-memory")
    changed_store.save_checkpoint(str(current_state))

    workspace = output_dir / "public_history" / "workspaces" / "000000"
    workspace.mkdir(parents=True)
    (workspace / "partial.txt").write_text("partial", encoding="utf-8")
    private_eval = output_dir / "private_evaluations" / episode_name
    private_eval.mkdir(parents=True)

    live_store = HarnessStore("live")
    _recover_incomplete_episode(
        output_dir=output_dir,
        episode_dir=episode_dir,
        episode_name=episode_name,
        index=0,
        candidate_path=current_candidate,
        state_path=current_state,
        store=live_store,
    )

    assert current_candidate.read_text(encoding="utf-8") == "before"
    assert live_store.memory == "before-memory"
    assert not episode_dir.exists()
    recovery_dirs = list((output_dir / "recovery_attempts").iterdir())
    assert len(recovery_dirs) == 1
    assert (recovery_dirs[0] / "public_episode").is_dir()
    assert (recovery_dirs[0] / "proposer_workspace").is_dir()
    assert (recovery_dirs[0] / "private_evaluation").is_dir()
