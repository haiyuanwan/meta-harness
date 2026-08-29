"""BFCL backend built directly on the official ``bfcl_eval`` package."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

from ..harness_protocol import HarnessStep, ToolResult, ToolSpec
from .base import BenchmarkBackend, TaskEnvironment

NATIVE_MODEL_NAME = "meta-harness-native-fc"


@lru_cache(maxsize=1)
def _symbols() -> dict[str, Any]:
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.model_handler.utils import convert_to_tool
    from bfcl_eval.utils import load_dataset_entry

    if NATIVE_MODEL_NAME not in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[NATIVE_MODEL_NAME] = ModelConfig(
            model_name=NATIVE_MODEL_NAME,
            display_name="Meta-Harness Native Tool Calling",
            url="local",
            org="Meta-Harness",
            license="MIT",
            model_handler=OpenAICompletionsHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=True,
        )
    return {
        "ModelStyle": ModelStyle,
        "GORILLA_TO_OPENAPI": GORILLA_TO_OPENAPI,
        "convert_to_tool": convert_to_tool,
        "load_dataset_entry": load_dataset_entry,
        "execute_multi_turn_func_call": execute_multi_turn_func_call,
    }


def _is_multi_turn(subset: str) -> bool:
    return subset.startswith("multi_turn_")


def _is_relevance(subset: str) -> bool:
    return subset in {"irrelevance", "live_irrelevance", "live_relevance"}


def _render_turn(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        prefix = "System: " if message.get("role") == "system" else ""
        parts.append(prefix + content)
    return "\n\n".join(parts)


def _function_call(name: str, arguments: dict[str, Any]) -> str:
    if not arguments:
        return f"{name}()"
    rendered = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{name}({rendered})"


class BFCLBackend(BenchmarkBackend):
    def __init__(self, subset: str = "multi_turn_base") -> None:
        self.subset = subset
        self._loaded = False
        self._tasks: dict[str, dict[str, Any]] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        symbols = _symbols()
        entries = symbols["load_dataset_entry"](self.subset)
        self._tasks = {str(entry["id"]): entry for entry in entries}
        self._loaded = True

    def list_tasks(self) -> list[str]:
        self._ensure_loaded()
        return list(self._tasks)

    def open_task(self, task_id: str, *, attempt_id: str) -> TaskEnvironment:
        self._ensure_loaded()
        if task_id not in self._tasks:
            raise KeyError(f"Unknown BFCL task id: {task_id}")
        return BFCLEnvironment(
            subset=self.subset,
            prompt_entry=deepcopy(self._tasks[task_id]),
            attempt_id=attempt_id,
        )


class BFCLEnvironment(TaskEnvironment):
    def __init__(
        self,
        *,
        subset: str,
        prompt_entry: dict[str, Any],
        attempt_id: str,
    ) -> None:
        self.subset = subset
        self.prompt_entry = prompt_entry
        self.attempt_id = attempt_id
        self._turns = [list(turn) for turn in prompt_entry.get("question", [])]
        self._turn_text = [_render_turn(turn) for turn in self._turns]
        self._turn_index = 0
        self._done = False
        turn_count = max(1, len(self._turns))
        self._turn_calls: list[list[list[str]]] = [[] for _ in range(turn_count)]
        self._turn_actions: list[list[list[dict[str, Any]]]] = [
            [] for _ in range(turn_count)
        ]
        self._tools = self._build_tools()
        self._tool_names = {tool.name for tool in self._tools}

    @property
    def task_id(self) -> str:
        return str(self.prompt_entry["id"])

    @property
    def task(self) -> str:
        return self._turn_text[0] if self._turn_text else ""

    @property
    def context(self) -> dict[str, Any]:
        if _is_multi_turn(self.subset):
            policy = (
                "Complete each turn using the available tools, then call finish. "
                "Finish ends only the current turn; continue when another user "
                "turn is provided. Do not ask clarification questions."
            )
        else:
            policy = (
                "Call the required tools and then call finish. Tool calls are "
                "recorded for evaluation. Do not ask clarification questions."
            )
        return {"policy": policy}

    @property
    def tools(self) -> list[ToolSpec]:
        return list(self._tools)

    def done(self) -> bool:
        return self._done

    def step(self, step: HarnessStep) -> tuple[list[ToolResult], int]:
        if self._done:
            return [], 0
        results: list[ToolResult] = []
        semantic_calls = []
        finish_calls = []
        for call in step.tool_calls:
            if call.name not in self._tool_names:
                results.append(
                    ToolResult(call.id, call.name, f"Unknown tool: {call.name}", True)
                )
            elif call.name == "finish":
                finish_calls.append(call)
            else:
                semantic_calls.append(call)

        call_strings = [
            _function_call(call.name, call.arguments) for call in semantic_calls
        ]
        if call_strings:
            self._turn_calls[self._turn_index].append(call_strings)
            self._turn_actions[self._turn_index].append(
                [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in semantic_calls
                ]
            )
        if semantic_calls:
            if _is_multi_turn(self.subset):
                raw_results, _ = _symbols()["execute_multi_turn_func_call"](
                    func_call_list=call_strings,
                    initial_config=self.prompt_entry["initial_config"],
                    involved_classes=self.prompt_entry["involved_classes"],
                    model_name=f"{NATIVE_MODEL_NAME}_{self.attempt_id}_runtime",
                    test_entry_id=self.task_id,
                    long_context=("long_context" in self.subset),
                    is_evaL_run=False,
                )
            else:
                raw_results = ["Action recorded." for _ in semantic_calls]
            for call, raw_result in zip(
                semantic_calls, raw_results, strict=False
            ):
                results.append(
                    ToolResult(call.id, call.name, str(raw_result), False)
                )

        if finish_calls:
            for call in finish_calls:
                results.append(ToolResult(call.id, call.name, "Turn finished."))
            if self._turn_index >= len(self._turns) - 1:
                self._done = True
            else:
                self._turn_index += 1
                next_text = self._turn_text[self._turn_index]
                if next_text:
                    results.append(
                        ToolResult(
                            f"turn-{self._turn_index}",
                            "environment",
                            next_text,
                        )
                    )
        return results, len(semantic_calls) + len(finish_calls)

    def grading_artifact(self) -> dict[str, Any]:
        flat_actions = [
            {action["name"]: action["arguments"]}
            for turn in self._turn_actions
            for batch in turn
            for action in batch
        ]
        return {
            "schema_version": 1,
            "benchmark": "bfcl",
            "task_id": self.task_id,
            "subset": self.subset,
            "attempt_id": self.attempt_id,
            "is_finished": self._done,
            "completed_turns": self._turn_index + int(self._done),
            "turn_calls": self._turn_calls,
            "flat_actions": flat_actions,
        }

    def _build_tools(self) -> list[ToolSpec]:
        functions = [deepcopy(item) for item in self.prompt_entry.get("function", [])]
        for items in self.prompt_entry.get("missed_function", {}).values():
            functions.extend(deepcopy(items))
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for function in functions:
            name = str(function.get("name", ""))
            if name and name not in seen:
                seen.add(name)
                deduped.append(function)
        converted = _symbols()["convert_to_tool"](
            deduped,
            _symbols()["GORILLA_TO_OPENAPI"],
            _symbols()["ModelStyle"].OPENAI_COMPLETIONS,
        )
        tools: list[ToolSpec] = []
        for raw in converted:
            function = raw.get("function", raw)
            tools.append(
                ToolSpec(
                    name=str(function["name"]),
                    description=str(function.get("description", "")),
                    parameters=dict(
                        function.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        )
                    ),
                )
            )
        tools.append(
            ToolSpec(
                name="finish",
                description="End the current BFCL turn.",
                parameters={"type": "object", "properties": {}},
                is_finish=True,
            )
        )
        return tools
