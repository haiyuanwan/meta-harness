"""Candidate and checkpoint validation for the online AgentStream loop."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from exgentic.agents.harness.harness_store import HarnessStore

from .agent import CandidatePolicyBase, OnlineHarnessAgent, OnlineHarnessInstance

FORBIDDEN_IMPORT_ROOTS = {
    "httpx",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}


class CandidateValidationError(ValueError):
    """Raised when candidate code or state violates the online contract."""


def load_candidate_module(candidate_path: Path):
    path = candidate_path.resolve()
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    module_name = f"_validated_agentstream_candidate_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CandidateValidationError(f"Cannot load candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_source(candidate_path: Path) -> None:
    source = candidate_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(candidate_path))
    except SyntaxError as exc:
        raise CandidateValidationError(f"candidate.py syntax error: {exc}") from exc

    if "run_evolver" in source:
        raise CandidateValidationError(
            "candidate.py may not call AgentStream's built-in run_evolver"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {str(node.module or "").split(".", 1)[0]}
        else:
            continue
        forbidden = roots & FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            raise CandidateValidationError(
                f"candidate.py imports forbidden module(s): {sorted(forbidden)}"
            )


def _validate_state(state_path: Path) -> dict[str, Any]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateValidationError(f"invalid harness checkpoint: {exc}") from exc

    expected_types: dict[str, type] = {
        "store_id": str,
        "session_count": int,
        "system_prompt": str,
        "memory": str,
        "skills": dict,
        "skill_embeddings": dict,
        "history": list,
        "versions": list,
    }
    for key, expected in expected_types.items():
        if key not in state:
            raise CandidateValidationError(f"harness checkpoint missing {key!r}")
        if not isinstance(state[key], expected):
            raise CandidateValidationError(
                f"harness checkpoint field {key!r} must be {expected.__name__}"
            )
    if state["session_count"] < 0:
        raise CandidateValidationError("session_count must be non-negative")

    probe = HarnessStore("online_contract_validation")
    probe.load_checkpoint(str(state_path))
    return state


def validate_candidate(candidate_path: Path, state_path: Path) -> dict[str, Any]:
    """Validate code/state without running a benchmark or model call."""

    candidate_path = candidate_path.resolve()
    state_path = state_path.resolve()
    _validate_source(candidate_path)
    state = _validate_state(state_path)
    module = load_candidate_module(candidate_path)

    policy_cls = getattr(module, "CandidatePolicy", None)
    if not inspect.isclass(policy_cls) or not issubclass(
        policy_cls, CandidatePolicyBase
    ):
        raise CandidateValidationError(
            "candidate.py must export CandidatePolicy(CandidatePolicyBase)"
        )

    agent_cls = getattr(module, "AgentHarness", None)
    if not inspect.isclass(agent_cls) or not issubclass(agent_cls, OnlineHarnessAgent):
        raise CandidateValidationError(
            "candidate.py must export AgentHarness(OnlineHarnessAgent)"
        )
    if agent_cls._get_instance_class() is not OnlineHarnessInstance:
        raise CandidateValidationError(
            "AgentHarness may not replace the stable OnlineHarnessInstance"
        )

    policy = policy_cls()
    probe_store = HarnessStore("online_policy_validation")
    transformed = policy.transform_system_message(
        default_message="default system message",
        task="contract smoke task",
        context={},
        store=probe_store,
        injected_skill_names=[],
    )
    if not isinstance(transformed, str) or not transformed.strip():
        raise CandidateValidationError(
            "transform_system_message() must return a non-empty string"
        )

    selected = policy.select_tools(
        tools=[],
        task="contract smoke task",
        context={},
        messages=[],
    )
    if not isinstance(selected, list):
        raise CandidateValidationError("select_tools() must return a list")

    try:
        inspect.signature(policy.update_state).bind(
            store=probe_store,
            task="contract smoke task",
            context={},
            trajectory="",
            injected_skill_names=[],
        )
    except TypeError as exc:
        raise CandidateValidationError(
            f"update_state() has an incompatible signature: {exc}"
        ) from exc

    agent = agent_cls(
        candidate_path=str(candidate_path),
        model="anthropic/Claude-Opus-4.8-C",
        evolver_model="anthropic/Claude-Opus-4.8-C",
        shuffle_mode="sequential",
        benchmark_id="contract",
        runner="direct",
    )
    if Path(agent.candidate_path).resolve() != candidate_path:
        raise CandidateValidationError("AgentHarness did not preserve candidate_path")

    return {
        "valid": True,
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "agent_class": f"{agent_cls.__module__}:{agent_cls.__name__}",
        "policy_class": f"{policy_cls.__module__}:{policy_cls.__name__}",
        "session_count": state["session_count"],
        "skill_count": len(state["skills"]),
    }
