"""Editable starting point for the Meta-Harness task.

Intentionally underpowered: one model turn is not enough for the hidden
scored task under the child model. More turns plus inspect/validate
prompting is enough to clear that target.
"""

import json
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from litellm import acompletion


class AgentHarness(BaseAgent):
    @staticmethod
    def name() -> str:
        return "editable-terminal-harness"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Solve the task using the terminal."},
            {"role": "user", "content": instruction},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        for _ in range(1):
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                result = await environment.exec(arguments["command"], timeout_sec=120)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": (
                            f"exit={result.return_code}\n"
                            f"{result.stdout or ''}\n"
                            f"{result.stderr or ''}"
                        )[-12000:],
                    }
                )
