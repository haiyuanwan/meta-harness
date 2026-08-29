"""Contract for graders that run after the solver environment has stopped."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PrivateGrader(ABC):
    @abstractmethod
    def grade(self, artifact: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None:
        return None


def validate_artifact(artifact: dict[str, Any], benchmark: str) -> None:
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported grading artifact schema")
    if artifact.get("benchmark") != benchmark:
        raise ValueError("grading artifact benchmark mismatch")
    if not isinstance(artifact.get("task_id"), str):
        raise ValueError("grading artifact is missing task_id")
