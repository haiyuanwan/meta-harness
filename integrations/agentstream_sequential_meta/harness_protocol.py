"""Benchmark-neutral protocol exposed to evolvable harness candidates.

This module deliberately has no AgentStream/Exgentic imports. Benchmark
objects are translated into these small value objects by the fixed evaluator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    is_finish: bool = False
    is_message: bool = False

    def as_litellm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ModelReply:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class HarnessStep:
    tool_calls: tuple[ToolCall, ...] = ()
    final_text: str | None = None


@dataclass
class Usage:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def add(self, reply: ModelReply) -> None:
        self.model_calls += 1
        self.input_tokens += reply.input_tokens
        self.output_tokens += reply.output_tokens
        self.cost += reply.cost


class ModelClient(Protocol):
    """Fixed model gateway supplied by the evaluator, not by candidate code."""

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> ModelReply: ...


class CandidateHarnessBase(ABC):
    """Interface implemented by the single evolvable ``candidate.py`` file."""

    def __init__(self, *, model_client: ModelClient, state: dict[str, Any]) -> None:
        self.model_client = model_client
        self.state = state
        self.usage = Usage()

    @abstractmethod
    def start(
        self,
        *,
        task: str,
        context: dict[str, Any],
        tools: list[ToolSpec],
        initial_results: list[ToolResult],
    ) -> HarnessStep:
        """Start one task and return the first action(s)."""

    @abstractmethod
    def react(self, results: list[ToolResult]) -> HarnessStep:
        """Consume official environment observations and choose the next step."""

    @abstractmethod
    def close(self, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
        """Update and return persistent state using agent-visible data only."""
