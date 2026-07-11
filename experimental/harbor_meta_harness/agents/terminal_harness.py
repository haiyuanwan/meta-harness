"""Small tool-using agent scaffold for the spike target task."""

import json
from collections.abc import Sequence
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from litellm import acompletion


class TerminalHarness(BaseAgent):
    prompt: str
    max_turns: int

    @staticmethod
    def name() -> str:
        return "terminal-harness"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": instruction},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command in the task container.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        for _ in range(self.max_turns):
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            tool_calls: Sequence[Any] = message.tool_calls or []
            if not tool_calls:
                return

            for tool_call in tool_calls:
                if tool_call.function.name != "run_command":
                    continue
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    result = await environment.exec(
                        arguments["command"], timeout_sec=120
                    )
                    output = (
                        f"exit={result.return_code}\n"
                        f"stdout={result.stdout or ''}\n"
                        f"stderr={result.stderr or ''}"
                    )
                except Exception as error:
                    output = f"tool error: {error}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output[-12000:],
                    }
                )
