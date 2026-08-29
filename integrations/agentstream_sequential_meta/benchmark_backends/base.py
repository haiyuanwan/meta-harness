"""Benchmark-neutral solver environment contracts.

This module is safe to ship in a solver image.  Private graders deliberately
live in the sibling ``benchmark_graders`` package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..harness_protocol import HarnessStep, ToolResult, ToolSpec


class TaskEnvironment(ABC):
    """One stateful task containing only data needed to execute the task."""

    @property
    @abstractmethod
    def task_id(self) -> str: ...

    @property
    @abstractmethod
    def task(self) -> str: ...

    @property
    def context(self) -> dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def tools(self) -> list[ToolSpec]: ...

    def start(self) -> list[ToolResult]:
        return []

    @abstractmethod
    def step(self, step: HarnessStep) -> tuple[list[ToolResult], int]: ...

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def grading_artifact(self) -> dict[str, Any]:
        """Export the minimal JSON artifact consumed by a later verifier."""

    def close(self) -> None:
        return None


class BenchmarkBackend(ABC):
    """Task discovery and environment factory for one benchmark configuration."""

    @abstractmethod
    def list_tasks(self) -> list[str]: ...

    @abstractmethod
    def open_task(self, task_id: str, *, attempt_id: str) -> TaskEnvironment: ...

    def close(self) -> None:
        return None
