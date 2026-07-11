import json
from pathlib import Path

import controller
from agents.meta_harness import EvaluationState


VALID_SOURCE = """
from harbor.agents.base import BaseAgent

class AgentHarness(BaseAgent):
    async def run(self, instruction, environment, context):
        return None
"""


def test_command_uses_modal_nano_and_one_attempt(tmp_path: Path) -> None:
    command = controller.command_for(
        controller.TARGET_TASK, tmp_path, tmp_path, "target"
    )

    assert ["-e", "modal"] == command[command.index("-e") : command.index("-e") + 2]
    model_index = command.index("-m")
    assert ["-m", "openai/gpt-5.4-nano"] == command[model_index : model_index + 2]
    assert ["--n-attempts", "1"] == command[
        command.index("--n-attempts") : command.index("--n-attempts") + 2
    ]


def test_validate_source_requires_harness_interface() -> None:
    assert controller.validate_source(VALID_SOURCE) is None
    assert (
        controller.validate_source("class AgentHarness: pass")
        == "AgentHarness must define async run"
    )
    assert (
        controller.validate_source("class Other: pass") == "missing AgentHarness class"
    )


def test_validate_source_rejects_target_and_test_leakage() -> None:
    leaked = VALID_SOURCE + "\nPATH = '/app/ledger.jsonl'\n"

    assert (
        controller.validate_source(leaked)
        == "forbidden benchmark reference: ledger.jsonl"
    )


def test_evaluate_source_rejects_before_any_child_run(
    monkeypatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "harness.py"
    source_path.write_text(VALID_SOURCE + "\nPATH = '/tests/private'\n")

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("leaked source reached a child run")

    monkeypatch.setattr(controller, "run_child", fail)
    result = controller.evaluate_source(source_path, tmp_path / "jobs")

    assert not result.accepted
    assert result.reason == "forbidden benchmark reference: /tests"
    assert result.smoke is None
    assert result.target is None


def test_evaluate_source_requires_smoke_before_target(
    monkeypatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "harness.py"
    source_path.write_text(VALID_SOURCE)
    calls: list[Path] = []

    def fake_run(task: Path, *args: object) -> controller.ChildResult:
        calls.append(task)
        return controller.ChildResult(0, "failed", "job")

    monkeypatch.setattr(controller, "run_child", fake_run)
    result = controller.evaluate_source(source_path, tmp_path / "jobs")

    assert calls == [controller.SMOKE_TASK]
    assert not result.accepted
    assert result.reason == "smoke test failed"
    assert result.target is None


def test_evaluate_source_returns_target_after_smoke(
    monkeypatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "harness.py"
    source_path.write_text(VALID_SOURCE)

    def fake_run(task: Path, *args: object) -> controller.ChildResult:
        return controller.ChildResult(1, task.name, "job")

    monkeypatch.setattr(controller, "run_child", fake_run)
    result = controller.evaluate_source(source_path, tmp_path / "jobs")

    assert result.accepted
    assert result.smoke and result.smoke.summary == "harness-smoke"
    assert result.target and result.target.summary == "reconcile-ledger"


def test_evaluate_source_resolves_job_directory_before_changing_child_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "harness.py").write_text(VALID_SOURCE)
    seen: list[Path] = []

    def fake_run(
        task: Path, candidate_dir: Path, jobs_dir: Path, name: str
    ) -> controller.ChildResult:
        seen.append(jobs_dir)
        return controller.ChildResult(1, name, "job")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(controller, "run_child", fake_run)
    controller.evaluate_source(Path("harness.py"), Path("jobs"))

    assert seen == [tmp_path / "jobs", tmp_path / "jobs"]


def test_run_child_exposes_staged_candidate_only_to_child(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class Completed:
        returncode = 1

    def fake_run(*args: object, **kwargs: object) -> Completed:
        captured.update(kwargs)
        return Completed()

    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    monkeypatch.setattr(controller, "modal_environment", lambda: {})
    candidate_dir = tmp_path / "candidate"
    result = controller.run_child(
        controller.SMOKE_TASK, candidate_dir, tmp_path, "smoke"
    )

    assert result.summary == "Harbor exited 1"
    assert captured["cwd"] == controller.ROOT
    assert (captured["env"] or {})["PYTHONPATH"].startswith(str(candidate_dir))


def test_evaluation_state_is_immutable_and_bounded() -> None:
    initial = EvaluationState()
    next_state = initial.record({"accepted": True})

    assert initial.remaining == 4
    assert initial.history == ()
    assert next_state.remaining == 3
    assert next_state.history == ({"accepted": True},)


def test_child_result_reads_harbor_result(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 1}}})
    )

    assert controller.child_result(tmp_path).reward == 1
