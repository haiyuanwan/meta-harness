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


def _suite(
    tmp_path: Path, tasks: list[Path], aggregate: str = "mean"
) -> controller.SuiteConfig:
    return controller.SuiteConfig(
        tasks=tuple(tasks),
        child_model="openai/gpt-5.4-nano",
        environment="modal",
        n_attempts=1,
        eval_budget=4,
        seed_harness=tmp_path / "harness.py",
        aggregate=aggregate,
        child_timeout_sec=600,
    )


def _task(tmp_path: Path, name: str, forbidden: list[str] | None = None) -> Path:
    task = tmp_path / name
    task.mkdir()
    refs = forbidden or []
    lines = ['schema_version = "1.1"', "[task]", f'name = "{name}"', "[meta_harness]"]
    if refs:
        lines.append("forbidden_references = [")
        lines.extend(f'  "{item}",' for item in refs)
        lines.append("]")
    else:
        lines.append("forbidden_references = []")
    (task / "task.toml").write_text("\n".join(lines) + "\n")
    return task


def test_load_suite_reads_explicit_task_list() -> None:
    suite = controller.load_suite()

    assert suite.tasks == (
        (controller.ROOT / "tasks" / "reconcile-ledger").resolve(),
        (controller.ROOT / "tasks" / "dedupe-events").resolve(),
        (controller.ROOT / "tasks" / "budget-rollups").resolve(),
    )
    assert suite.aggregate == "mean"
    assert suite.child_model == "openai/gpt-5.4-nano"
    assert suite.eval_budget == 4


def test_collection_suites_resolve_from_experiment_root() -> None:
    suites_dir = controller.ROOT / "suites"
    names = sorted(path.name for path in suites_dir.glob("*.toml"))
    assert names == [
        "codepde.toml",
        "humanevalfix-lite.toml",
        "tb2-easy.toml",
    ]
    for path in suites_dir.glob("*.toml"):
        suite = controller.load_suite(path)
        assert suite.tasks
        assert all(task.is_dir() for task in suite.tasks)
        assert controller.suite_forbidden_references(suite)


def test_command_uses_suite_model_and_attempts(tmp_path: Path) -> None:
    task = _task(tmp_path, "demo")
    suite = _suite(tmp_path, [task])
    command = controller.command_for(suite, task, tmp_path, tmp_path, "demo")

    assert ["-e", "modal"] == command[command.index("-e") : command.index("-e") + 2]
    model_index = command.index("-m")
    assert ["-m", "openai/gpt-5.4-nano"] == command[model_index : model_index + 2]
    assert ["--n-attempts", "1"] == command[
        command.index("--n-attempts") : command.index("--n-attempts") + 2
    ]


def test_validate_source_requires_harness_interface() -> None:
    assert controller.validate_source(VALID_SOURCE, ()) is None
    assert (
        controller.validate_source("class AgentHarness: pass", ())
        == "AgentHarness must define async run"
    )
    assert (
        controller.validate_source("class Other: pass", ())
        == "missing AgentHarness class"
    )


def test_suite_forbidden_references_are_per_task_union(tmp_path: Path) -> None:
    first = _task(tmp_path, "one", ["ledger.jsonl"])
    second = _task(tmp_path, "two", ["other.bin", "ledger.jsonl"])
    suite = _suite(tmp_path, [first, second])

    forbidden = controller.suite_forbidden_references(suite)

    assert "ledger.jsonl" in forbidden
    assert "other.bin" in forbidden
    assert "/tests" in forbidden


def test_validate_source_rejects_task_leakage(tmp_path: Path) -> None:
    task = _task(tmp_path, "one", ["ledger.jsonl"])
    suite = _suite(tmp_path, [task])
    leaked = VALID_SOURCE + "\nPATH = '/app/ledger.jsonl'\n"

    assert (
        controller.validate_source(leaked, controller.suite_forbidden_references(suite))
        == "forbidden benchmark reference: ledger.jsonl"
    )


def test_evaluate_source_rejects_before_any_child_run(
    monkeypatch, tmp_path: Path
) -> None:
    task = _task(tmp_path, "one")
    suite = _suite(tmp_path, [task])
    source_path = tmp_path / "harness.py"
    source_path.write_text(VALID_SOURCE + "\nPATH = '/tests/private'\n")

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("leaked source reached a child run")

    monkeypatch.setattr(controller, "run_child", fail)
    result = controller.evaluate_source(source_path, tmp_path / "jobs", suite)

    assert not result.accepted
    assert result.reason == "forbidden benchmark reference: /tests"
    assert result.reward is None
    assert result.tasks == ()


def test_evaluate_source_aggregates_mean_over_tasks(
    monkeypatch, tmp_path: Path
) -> None:
    first = _task(tmp_path, "one")
    second = _task(tmp_path, "two")
    suite = _suite(tmp_path, [first, second], aggregate="mean")
    source_path = tmp_path / "harness.py"
    source_path.write_text(VALID_SOURCE)
    rewards = {first: 1.0, second: 0.0}

    def fake_run(
        suite_arg: controller.SuiteConfig,
        task: Path,
        *args: object,
    ) -> controller.ChildResult:
        return controller.ChildResult(rewards[task], task.name, "job")

    monkeypatch.setattr(controller, "run_child", fake_run)
    result = controller.evaluate_source(source_path, tmp_path / "jobs", suite)

    assert result.accepted
    assert result.reward == 0.5
    assert [item.task for item in result.tasks] == ["one", "two"]


def test_evaluate_source_supports_min_aggregate(monkeypatch, tmp_path: Path) -> None:
    first = _task(tmp_path, "one")
    second = _task(tmp_path, "two")
    suite = _suite(tmp_path, [first, second], aggregate="min")
    source_path = tmp_path / "harness.py"
    source_path.write_text(VALID_SOURCE)
    rewards = {first: 1.0, second: 0.0}

    def fake_run(
        suite_arg: controller.SuiteConfig,
        task: Path,
        *args: object,
    ) -> controller.ChildResult:
        return controller.ChildResult(rewards[task], task.name, "job")

    monkeypatch.setattr(controller, "run_child", fake_run)
    result = controller.evaluate_source(source_path, tmp_path / "jobs", suite)

    assert result.reward == 0.0


def test_evaluate_source_resolves_job_directory_before_changing_child_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    task = _task(tmp_path, "one")
    suite = _suite(tmp_path, [task])
    (tmp_path / "harness.py").write_text(VALID_SOURCE)
    seen: list[Path] = []

    def fake_run(
        suite_arg: controller.SuiteConfig,
        task_arg: Path,
        candidate_dir: Path,
        jobs_dir: Path,
        name: str,
    ) -> controller.ChildResult:
        seen.append(jobs_dir)
        return controller.ChildResult(1, name, "job")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(controller, "run_child", fake_run)
    controller.evaluate_source(Path("harness.py"), Path("jobs"), suite)

    assert seen == [tmp_path / "jobs"]


def test_run_child_exposes_staged_candidate_only_to_child(
    monkeypatch, tmp_path: Path
) -> None:
    task = _task(tmp_path, "one")
    suite = _suite(tmp_path, [task])
    captured: dict[str, object] = {}

    class Completed:
        returncode = 1

    def fake_run(*args: object, **kwargs: object) -> Completed:
        captured.update(kwargs)
        return Completed()

    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    monkeypatch.setattr(controller, "modal_environment", lambda: {})
    candidate_dir = tmp_path / "candidate"
    result = controller.run_child(suite, task, candidate_dir, tmp_path, "one")

    assert result.summary == "Harbor exited 1"
    assert captured["cwd"] == controller.ROOT
    assert (captured["env"] or {})["PYTHONPATH"].startswith(str(candidate_dir))


def test_evaluation_state_is_immutable_and_bounded() -> None:
    initial = EvaluationState(remaining=4)
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
