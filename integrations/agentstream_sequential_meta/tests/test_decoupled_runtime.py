from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from integrations.agentstream_sequential_meta.benchmark_backends.base import (
    TaskEnvironment,
)
from integrations.agentstream_sequential_meta.candidate_contract import (
    load_harness_state,
    validate_candidate,
    write_new_harness_state,
)
from integrations.agentstream_sequential_meta.harness_protocol import (
    CandidateHarnessBase,
    HarnessStep,
    ModelReply,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from integrations.agentstream_sequential_meta.model_runtime import LiteLLMModelClient
from integrations.agentstream_sequential_meta.sandbox_evaluation import (
    CandidateExecutionError,
    _candidate_call,
    _public_row,
    _run_harness_task,
)


class FakeEnvironment(TaskEnvironment):
    def __init__(self) -> None:
        self.finished = False

    @property
    def task_id(self) -> str:
        return "echo-1"

    @property
    def task(self) -> str:
        return "echo once"

    @property
    def context(self) -> dict[str, Any]:
        return {"public": True}

    @property
    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="echo",
                description="Echo text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ]

    def step(self, step: HarnessStep) -> tuple[list[ToolResult], int]:
        call = step.tool_calls[0]
        if call.name != "echo":
            return [ToolResult(call.id, call.name, "Unknown tool", True)], 0
        self.finished = True
        return [ToolResult(call.id, call.name, str(call.arguments["text"]))], 1

    def done(self) -> bool:
        return self.finished

    def grading_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": "fake",
            "task_id": self.task_id,
            "is_finished": self.finished,
        }


def test_native_environment_executes_generic_tool_calls() -> None:
    environment = FakeEnvironment()

    assert [tool.name for tool in environment.tools] == ["echo"]
    results, count = environment.step(
        HarnessStep(tool_calls=(ToolCall("c1", "echo", {"text": "hello"}),))
    )

    assert count == 1
    assert results[0].tool_call_id == "c1"
    assert results[0].content == "hello"


def test_native_environment_returns_invalid_call_as_observation() -> None:
    environment = FakeEnvironment()

    results, count = environment.step(
        HarnessStep(tool_calls=(ToolCall("bad", "missing", {}),))
    )

    assert count == 0
    assert results[0].is_error
    assert "Unknown tool" in results[0].content


class ScriptedModel:
    def complete(self, **_: Any) -> ModelReply:
        return ModelReply(
            tool_calls=(ToolCall("echo-1", "echo", {"text": "ok"}),),
            input_tokens=10,
            output_tokens=3,
        )


class TrackingHarness(CandidateHarnessBase):
    closed = False

    def start(self, **_: Any) -> HarnessStep:
        reply = self.model_client.complete(messages=[], tools=[])
        self.usage.add(reply)
        return HarnessStep(tool_calls=reply.tool_calls)

    def react(self, results) -> HarnessStep:
        del results
        return HarnessStep()

    def close(self, trajectory):
        del trajectory
        type(self).closed = True
        self.state["session_count"] += 1
        return self.state


class ScoreOrderEnvironment(FakeEnvironment):
    def grading_artifact(self) -> dict[str, Any]:
        assert TrackingHarness.closed, "artifact exported before candidate close"
        return super().grading_artifact()


def test_runner_updates_state_before_official_scoring(tmp_path: Path) -> None:
    TrackingHarness.closed = False
    state = {
        "schema_version": 1,
        "session_count": 0,
        "memory": "",
        "skills": {},
        "history": [],
    }

    next_state, row, artifact = _run_harness_task(
        candidate_class=TrackingHarness,
        state=state,
        environment=ScoreOrderEnvironment(),
        model_client=ScriptedModel(),
        max_steps=3,
        trajectory_path=tmp_path / "trajectory.jsonl",
    )

    assert next_state["session_count"] == 1
    assert row["status"] == "awaiting_grader"
    assert row["input_tokens"] == 10
    assert artifact["is_finished"] is True
    events = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert all("score" not in event for event in events)
    assert events[-1] == {
        "event": "tool_result",
        "name": "echo",
        "tool_call_id": "echo-1",
        "content": "ok",
        "is_error": False,
    }
    assert not (tmp_path / "private_score.json").exists()


def test_generation_zero_candidate_matches_decoupled_contract(tmp_path: Path) -> None:
    integration = Path(__file__).parents[1]
    state_path = tmp_path / "harness_store.json"
    write_new_harness_state(state_path)

    result = validate_candidate(integration / "candidate.py", state_path)

    assert result == {"valid": True}
    assert load_harness_state(state_path)["session_count"] == 0
    source = (integration / "candidate.py").read_text(encoding="utf-8")
    assert "exgentic" not in source.lower()


class CapturingModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def complete(self, *, messages, tools) -> ModelReply:
        del tools
        self.messages = messages
        return ModelReply(content="done")


def test_environment_observation_is_not_an_orphan_tool_message() -> None:
    from integrations.agentstream_sequential_meta.candidate import CandidateHarness

    model = CapturingModel()
    harness = CandidateHarness(
        model_client=model,
        state={
            "schema_version": 1,
            "session_count": 0,
            "memory": "",
            "skills": {},
            "history": [],
        },
    )

    harness.start(
        task="next turn",
        context={},
        tools=[],
        initial_results=[ToolResult("turn-2", "environment", "new user input")],
    )

    assert model.messages[-2] == {"role": "user", "content": "new user input"}


def test_fixed_model_client_configuration_is_immutable() -> None:
    client = LiteLLMModelClient(model="fixed", max_tokens=10)

    import pytest

    with pytest.raises(AttributeError, match="immutable"):
        client._model = "changed"  # type: ignore[attr-defined]


def test_candidate_code_failure_is_classified_as_non_infrastructure() -> None:
    import pytest

    def broken_candidate() -> None:
        raise ValueError("deterministic bug")

    with pytest.raises(CandidateExecutionError, match="deterministic bug"):
        _candidate_call(broken_candidate)


def test_public_metrics_strip_private_exception_text() -> None:
    public = _public_row(
        {
            "task_id": "q1",
            "score": 0.0,
            "error": "PrivateGraderError: secret checker detail",
        }
    )

    assert public == {
        "task_id": "q1",
        "score": 0.0,
        "error_type": "PrivateGraderError",
    }
