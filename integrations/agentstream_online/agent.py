"""Stable AgentStream agent layer for online Meta-Harness candidates.

The solver loop, action conversion, logging, and persistent HarnessStore come
from AgentStream.  The built-in AgentStream evolver is deliberately bypassed:
candidate evolution is owned by the outer Claude Code controller.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

from exgentic.agents.harness.harness_agent import HarnessAgent
from exgentic.agents.harness.harness_instance import HarnessAgentInstance
from exgentic.agents.harness.harness_store import HarnessStore
from litellm import ChatCompletionSystemMessage


class CandidatePolicyBase:
    """Small, code-evolvable policy surface around AgentStream's solver loop."""

    def transform_system_message(
        self,
        *,
        default_message: str,
        task: str,
        context: dict[str, Any],
        store: HarnessStore,
        injected_skill_names: list[str],
    ) -> str:
        """Return the system message shown to the fixed base model."""

        return default_message

    def select_tools(
        self,
        *,
        tools: list[dict[str, Any]],
        task: str,
        context: dict[str, Any],
        messages: list[Any],
    ) -> list[dict[str, Any]]:
        """Return a subset of the existing benchmark tools."""

        return tools

    def update_state(
        self,
        *,
        store: HarnessStore,
        task: str,
        context: dict[str, Any],
        trajectory: str,
        injected_skill_names: list[str],
    ) -> list[str]:
        """Optionally update memory/skills without using grader feedback."""

        return []


def _load_candidate_policy(candidate_path: str) -> CandidatePolicyBase:
    path = Path(candidate_path).resolve()
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    module_name = f"_agentstream_online_candidate_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    policy_cls = getattr(module, "CandidatePolicy", None)
    if not isinstance(policy_cls, type) or not issubclass(policy_cls, CandidatePolicyBase):
        raise TypeError("candidate.py must export CandidatePolicy(CandidatePolicyBase)")
    return policy_cls()


def _tool_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return str(name) if name else None


class OnlineHarnessInstance(HarnessAgentInstance):
    """Harness instance whose evolution hooks come from the current candidate."""

    def __init__(self, *args: Any, candidate_path: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._candidate_path = str(Path(candidate_path).resolve())
        self._candidate_policy = _load_candidate_policy(self._candidate_path)

    def start(
        self,
        task: str,
        context: dict[str, Any],
        actions: list[Any],
    ) -> None:
        super().start(task, context, actions)
        if not self.messages or self._store is None:
            return

        first = self.messages[0]
        if isinstance(first, dict):
            raw_content = first.get("content", "")
        else:
            raw_content = getattr(first, "content", "")
        default_message = (
            raw_content if isinstance(raw_content, str) else str(raw_content)
        )
        try:
            transformed = self._candidate_policy.transform_system_message(
                default_message=default_message,
                task=self.task,
                context=dict(self.context),
                store=self._store,
                injected_skill_names=list(self._injected_skill_names),
            )
            if not isinstance(transformed, str) or not transformed.strip():
                raise TypeError("transform_system_message() must return a non-empty string")
            self.messages[0] = ChatCompletionSystemMessage(
                role="system", content=transformed
            )
            self.logger.info(
                "Online candidate transformed system message: %d -> %d chars",
                len(default_message),
                len(transformed),
            )
        except Exception as exc:  # noqa: BLE001 - candidate hooks must fail closed
            self.logger.warning("Online candidate system hook failed; using default: %s", exc)
            self._log_failure("candidate_system_message", exc, {})

        try:
            metadata_path = self.paths.agent_dir / "online_candidate.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "candidate_path": self._candidate_path,
                        "policy_class": type(self._candidate_policy).__name__,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - metadata is non-critical
            self.logger.warning("Failed to save online candidate metadata: %s", exc)

    def _assistant_tools(self) -> list[dict[str, Any]]:
        tools = super()._assistant_tools()
        if not tools:
            return tools

        try:
            proposed = self._candidate_policy.select_tools(
                tools=copy.deepcopy(tools),
                task=getattr(self, "task", ""),
                context=dict(getattr(self, "context", {}) or {}),
                messages=list(self.messages),
            )
            if not isinstance(proposed, list):
                raise TypeError("select_tools() must return a list")

            original_by_name = {
                name: tool for tool in tools if (name := _tool_name(tool)) is not None
            }
            selected_names: list[str] = []
            for tool in proposed:
                if not isinstance(tool, dict):
                    raise TypeError("select_tools() returned a non-dict tool")
                name = _tool_name(tool)
                if name not in original_by_name:
                    raise ValueError(f"select_tools() returned unknown tool: {name!r}")
                if name not in selected_names:
                    selected_names.append(name)

            selected = [original_by_name[name] for name in selected_names]
            self.logger.info(
                "Online candidate selected %d/%d tools", len(selected), len(tools)
            )
            return selected
        except Exception as exc:  # noqa: BLE001 - candidate hooks must fail closed
            self.logger.warning("Online candidate tool hook failed; using all tools: %s", exc)
            self._log_failure("candidate_tool_selection", exc, {})
            return tools

    def close(self) -> None:
        """Persist state without calling AgentStream's built-in run_evolver()."""

        store = self._store
        if store is None:
            return

        trajectory = self._build_session_trace()
        snapshot = store.snapshot()
        ops_summary: list[str] = []
        try:
            raw_summary = self._candidate_policy.update_state(
                store=store,
                task=getattr(self, "task", ""),
                context=dict(getattr(self, "context", {}) or {}),
                trajectory=trajectory,
                injected_skill_names=list(self._injected_skill_names),
            )
            if raw_summary is None:
                ops_summary = []
            elif isinstance(raw_summary, str):
                ops_summary = [raw_summary]
            elif isinstance(raw_summary, list) and all(
                isinstance(item, str) for item in raw_summary
            ):
                ops_summary = list(raw_summary)
            else:
                raise TypeError("update_state() must return list[str], str, or None")
        except Exception as exc:  # noqa: BLE001 - candidate hooks must roll back
            store.rollback(snapshot)
            self.logger.warning("Online candidate state hook failed; rolled back: %s", exc)
            self._log_failure("candidate_state_update", exc, {})
            ops_summary = []

        changed = store.snapshot() != snapshot
        if changed and not ops_summary:
            ops_summary = ["candidate policy updated persistent harness state"]

        ops_applied = len(ops_summary) if changed else 0
        if changed:
            version = store.commit_version(
                session_id=self.session_id,
                ops_summary=ops_summary,
            )
            self.logger.info(
                "Online candidate committed state version=%d ops=%s",
                version,
                ops_summary,
            )

        store.record_learning(
            session_id=self.session_id,
            task_id=str((getattr(self, "context", {}) or {}).get("task_id", "")),
            benchmark_id=self.benchmark_id or "",
            ops_applied=ops_applied,
            ops_summary=ops_summary,
        )

        try:
            (self.paths.agent_dir / "online_candidate_trace.txt").write_text(
                trajectory, encoding="utf-8"
            )
            store.save_checkpoint(
                str(self.paths.agent_dir / "harness_checkpoint.json")
            )
            store.save_harness_text(
                str(self.paths.agent_dir / "harness_state.md")
            )
        except Exception as exc:  # noqa: BLE001 - checkpoint failure is logged
            self.logger.warning("Online candidate failed to save state artifacts: %s", exc)
            self._log_failure("candidate_checkpoint", exc, {})


class OnlineHarnessAgent(HarnessAgent):
    """Agent configuration shared by all online candidate files."""

    display_name: ClassVar[str] = "Online Meta-Harness Agent"
    slug_name: ClassVar[str] = "online_meta_harness"

    candidate_path: str

    @classmethod
    def _get_instance_class(cls) -> type[OnlineHarnessInstance]:
        return OnlineHarnessInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "integrations.agentstream_online.agent:OnlineHarnessInstance"

    def _get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        kwargs = super()._get_instance_kwargs(session_id)
        kwargs["candidate_path"] = self.candidate_path
        return kwargs
