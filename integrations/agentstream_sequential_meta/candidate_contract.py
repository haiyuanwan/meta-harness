"""Validation and state helpers for benchmark-neutral candidate harnesses."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

from .harness_protocol import CandidateHarnessBase, ModelReply

SCHEMA_VERSION = 1
ALLOWED_IMPORTS = {
    "__future__",
    "collections",
    "copy",
    "dataclasses",
    "json",
    "math",
    "re",
    "statistics",
    "typing",
    "integrations.agentstream_sequential_meta.harness_protocol",
}
FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_TEXT = {
    "grader",
    "ground_truth",
    "possible_answer",
    "private_test",
    "reference_answer",
    "verifier",
}


class CandidateValidationError(ValueError):
    pass


def new_harness_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "session_count": 0,
        "memory": "",
        "skills": {},
        "history": [],
    }


def write_new_harness_state(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(new_harness_state(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_harness_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateValidationError(f"Invalid harness state: {exc}") from exc
    validate_harness_state(state)
    return state


def validate_harness_state(state: Any) -> None:
    if not isinstance(state, dict):
        raise CandidateValidationError("Harness state must be a JSON object")
    checks = {
        "schema_version": (state.get("schema_version") == SCHEMA_VERSION),
        "session_count": isinstance(state.get("session_count"), int),
        "memory": isinstance(state.get("memory"), str),
        "skills": isinstance(state.get("skills"), dict),
        "history": isinstance(state.get("history"), list),
    }
    invalid = [name for name, valid in checks.items() if not valid]
    if invalid:
        raise CandidateValidationError(
            "Invalid harness state fields: " + ", ".join(invalid)
        )
    try:
        json.dumps(state, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(
            f"Harness state must be JSON serializable: {exc}"
        ) from exc


def save_harness_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _validate_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise CandidateValidationError(f"candidate.py has invalid syntax: {exc}") from exc
    for node in ast.walk(tree):
        imports: list[str] = []
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports = [node.module]
        disallowed = sorted(name for name in imports if name not in ALLOWED_IMPORTS)
        if disallowed:
            raise CandidateValidationError(
                "candidate.py imports non-allowlisted module(s): " + ", ".join(disallowed)
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise CandidateValidationError(
                    f"candidate.py calls forbidden builtin: {node.func.id}"
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CandidateValidationError(
                f"candidate.py accesses forbidden dunder attribute: {node.attr}"
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CandidateValidationError(
                f"candidate.py accesses forbidden dunder name: {node.id}"
            )
    lowered = source.lower()
    leaked = sorted(token for token in FORBIDDEN_TEXT if token in lowered)
    if leaked:
        raise CandidateValidationError(
            "candidate.py contains protected evaluator terms: " + ", ".join(leaked)
        )


def load_candidate_module(path: Path) -> ModuleType:
    module_name = f"sequential_candidate_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CandidateValidationError(f"Cannot import candidate at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CandidateValidationError(f"Cannot load candidate.py: {exc}") from exc
    return module


class _SmokeModel:
    def complete(self, **_: Any) -> ModelReply:
        return ModelReply(content="done")


def validate_candidate(candidate_path: Path, state_path: Path) -> dict[str, Any]:
    _validate_source(candidate_path)
    state = load_harness_state(state_path)
    module = load_candidate_module(candidate_path)
    candidate_class = getattr(module, "CandidateHarness", None)
    if not isinstance(candidate_class, type) or not issubclass(
        candidate_class, CandidateHarnessBase
    ):
        raise CandidateValidationError(
            "candidate.py must export CandidateHarness(CandidateHarnessBase)"
        )
    try:
        harness = candidate_class(model_client=_SmokeModel(), state=deepcopy(state))
        harness.start(task="smoke", context={}, tools=[], initial_results=[])
        output = harness.close([])
    except Exception as exc:
        raise CandidateValidationError(f"Candidate smoke validation failed: {exc}") from exc
    if not isinstance(output, dict):
        raise CandidateValidationError("CandidateHarness.close() must return state dict")
    validate_harness_state(output)
    return {"valid": True}
