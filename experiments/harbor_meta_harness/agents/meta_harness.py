"""Outer agent with a bounded `evaluate_harness` tool."""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from litellm import acompletion

ROOT = Path(__file__).parents[1]
MAX_EVALUATIONS = 4


@dataclass(frozen=True)
class EvaluationState:
    remaining: int = MAX_EVALUATIONS
    history: tuple[dict[str, Any], ...] = ()

    def record(self, result: dict[str, Any]) -> "EvaluationState":
        return EvaluationState(self.remaining - 1, (*self.history, result))


class AgentHarness(BaseAgent):
    @staticmethod
    def name() -> str:
        return "meta-harness"

    def version(self) -> str | None:
        return "0.2.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def _evaluate(
        self, environment: BaseEnvironment, state: EvaluationState
    ) -> tuple[EvaluationState, dict[str, Any]]:
        if state.remaining == 0:
            return state, {
                "error": "evaluation budget exhausted",
                "remaining_budget": 0,
            }
        source_path = self.logs_dir / f"harness-{len(state.history) + 1}.py"
        result_path = self.logs_dir / f"evaluation-{len(state.history) + 1}.json"
        await environment.download_file("/app/harness.py", source_path)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "controller.py",
            "--source",
            str(source_path),
            "--jobs-dir",
            str(self.logs_dir / "inner-jobs" / str(len(state.history) + 1)),
            "--result",
            str(result_path),
            cwd=ROOT,
        )
        return_code = await process.wait()
        if return_code != 0:
            result: dict[str, Any] = {
                "accepted": False,
                "reason": f"evaluator exited {return_code}",
            }
        else:
            result = json.loads(result_path.read_text())
        next_state = state.record(result)
        response = {
            "reward": (result.get("target") or {}).get("reward", 0.0),
            "verifier_summary": (result.get("target") or result.get("smoke") or {}).get(
                "summary", result.get("reason")
            ),
            "remaining_budget": next_state.remaining,
        }
        return next_state, response

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        state = EvaluationState()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Improve /app/harness.py as a general Harbor BaseAgent harness. "
                    "Use the terminal to inspect and edit it. evaluate_harness is scarce: "
                    "each call smoke-tests then scores the current file. Do not put benchmark, "
                    "test, verifier, solution, or target-data references in the harness."
                ),
            },
            {"role": "user", "content": instruction},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command in the outer task container.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_harness",
                    "description": "Smoke-test then score the current /app/harness.py. Maximum four calls.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        for _ in range(12):
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
                break
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    if call.function.name == "run_command":
                        execution = await environment.exec(
                            arguments["command"], timeout_sec=120
                        )
                        output: Any = {
                            "exit": execution.return_code,
                            "stdout": (execution.stdout or "")[-12000:],
                            "stderr": (execution.stderr or "")[-12000:],
                        }
                    elif call.function.name == "evaluate_harness":
                        state, output = await self._evaluate(environment, state)
                    else:
                        output = {"error": "unknown tool"}
                except Exception as error:
                    output = {"error": str(error)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(output),
                    }
                )
        report = {
            "budget": MAX_EVALUATIONS,
            "remaining_budget": state.remaining,
            "history": list(state.history),
        }
        report_path = self.logs_dir / "evaluation_history.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        await environment.upload_file(report_path, "/app/evaluation_history.json")
