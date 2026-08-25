"""Generation-zero online AgentStream harness.

Claude Code evolves this file after each task.  Keep both exported class names
and the inheritance contract intact.
"""

from __future__ import annotations

from typing import Any

from exgentic.agents.harness.harness_store import HarnessStore

from integrations.agentstream_online.agent import (
    CandidatePolicyBase,
    OnlineHarnessAgent,
)


class CandidatePolicy(CandidatePolicyBase):
    """Initial policy: preserve AgentStream's default prompt, state, and tools."""

    def transform_system_message(
        self,
        *,
        default_message: str,
        task: str,
        context: dict[str, Any],
        store: HarnessStore,
        injected_skill_names: list[str],
    ) -> str:
        return default_message

    def select_tools(
        self,
        *,
        tools: list[dict[str, Any]],
        task: str,
        context: dict[str, Any],
        messages: list[Any],
    ) -> list[dict[str, Any]]:
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
        return []


class AgentHarness(OnlineHarnessAgent):
    """Candidate agent loaded by the per-task controller."""
