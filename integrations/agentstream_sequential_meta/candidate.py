"""Generation-zero, Pi-style harness evolved by Meta-Harness.

The fixed controller owns model credentials and benchmark-native environments;
this benchmark-neutral file owns the agent loop, context policy, and memory.
"""

from __future__ import annotations

import json
from typing import Any

from integrations.agentstream_sequential_meta.harness_protocol import (
    CandidateHarnessBase,
    HarnessStep,
    ToolResult,
    ToolSpec,
)

SYSTEM_PROMPT = """You are a careful general-purpose tool-using agent.
Solve the current task using only the supplied tools. Inspect observations,
correct tool errors, and call a finish/submit tool when the task is complete.
Never invent a tool result. Keep tool arguments faithful to their JSON schema.
"""

MAX_MEMORY_CHARS = 6000
MAX_HISTORY_ITEMS = 20
MAX_TOOL_RESULT_CHARS = 24000


def _json_text(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


class CandidateHarness(CandidateHarnessBase):
    """Small readable loop inspired by pi-agent-core's linear state model."""

    def start(
        self,
        *,
        task: str,
        context: dict[str, Any],
        tools: list[ToolSpec],
        initial_results: list[ToolResult],
    ) -> HarnessStep:
        self.tools = list(tools)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        memory = str(self.state.get("memory", "")).strip()
        if memory:
            self.messages.append(
                {
                    "role": "system",
                    "content": "Agent-visible notes from earlier tasks:\n"
                    + memory[-MAX_MEMORY_CHARS:],
                }
            )
        user_content = task
        if context:
            user_content += "\n\nTask context:\n" + _json_text(context, 16000)
        self.messages.append({"role": "user", "content": user_content})
        self._append_results(initial_results)
        return self._next_step()

    def react(self, results: list[ToolResult]) -> HarnessStep:
        self._append_results(results)
        return self._next_step()

    def _append_results(self, results: list[ToolResult]) -> None:
        for result in results:
            content = result.content
            if len(content) > MAX_TOOL_RESULT_CHARS:
                content = content[:MAX_TOOL_RESULT_CHARS] + "...[truncated]"
            if result.is_error:
                content = "TOOL ERROR: " + content
            if result.name == "environment":
                self.messages.append({"role": "user", "content": content})
                continue
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "name": result.name,
                    "content": content,
                }
            )

    def _next_step(self) -> HarnessStep:
        reply = self.model_client.complete(messages=self.messages, tools=self.tools)
        self.usage.add(reply)
        assistant: dict[str, Any] = {"role": "assistant"}
        if reply.content is not None:
            assistant["content"] = reply.content
        if reply.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in reply.tool_calls
            ]
        self.messages.append(assistant)
        return HarnessStep(tool_calls=reply.tool_calls, final_text=reply.content)

    def close(self, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
        calls = [
            str(item.get("name", ""))
            for item in trajectory
            if item.get("event") == "tool_call"
        ]
        errors = sum(
            bool(item.get("is_error"))
            for item in trajectory
            if item.get("event") == "tool_result"
        )
        record = {
            "task": next(
                (
                    str(item.get("task", ""))[:500]
                    for item in trajectory
                    if item.get("event") == "task"
                ),
                "",
            ),
            "tools": calls[-20:],
            "tool_errors": errors,
        }
        history = list(self.state.get("history", []))
        history.append(record)
        history = history[-MAX_HISTORY_ITEMS:]
        self.state["history"] = history
        self.state["session_count"] = int(self.state.get("session_count", 0)) + 1
        self.state["memory"] = "\n".join(
            f"Task {index + 1}: tools={','.join(item.get('tools', [])) or 'none'}; "
            f"tool_errors={item.get('tool_errors', 0)}"
            for index, item in enumerate(history)
        )[-MAX_MEMORY_CHARS:]
        return self.state
