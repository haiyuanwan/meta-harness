"""Fixed LiteLLM model gateway used by all candidate generations."""

from __future__ import annotations

import json
import time
from typing import Any

from .harness_protocol import ModelReply, ToolCall, ToolSpec


class ModelRuntimeError(RuntimeError):
    """A provider/transport failure eligible for task-attempt retry."""


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class LiteLLMModelClient:
    __slots__ = (
        "_max_tokens",
        "_model",
        "_retries",
        "_retry_delay",
        "_sealed",
        "_timeout",
    )

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        retries: int = 2,
        retry_delay: float = 1.0,
        timeout: float = 300.0,
    ) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_max_tokens", max_tokens)
        object.__setattr__(self, "_retries", retries)
        object.__setattr__(self, "_retry_delay", retry_delay)
        object.__setattr__(self, "_timeout", timeout)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("The evaluator-owned model client is immutable")
        object.__setattr__(self, name, value)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> ModelReply:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "max_retries": 0,
            "timeout": self._timeout,
        }
        if tools:
            kwargs["tools"] = [tool.as_litellm_tool() for tool in tools]
            kwargs["tool_choice"] = "auto"
        response: Any = None
        for attempt in range(self._retries + 1):
            try:
                response = litellm.completion(**kwargs)
                break
            except Exception as exc:
                if attempt >= self._retries:
                    raise ModelRuntimeError(
                        f"Model request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                time.sleep(self._retry_delay * (2**attempt))
        choices = _field(response, "choices", [])
        if not choices:
            raise ModelRuntimeError("Model response contains no choices")
        message = _field(choices[0], "message", {})
        calls: list[ToolCall] = []
        for index, raw_call in enumerate(_field(message, "tool_calls", []) or []):
            function = _field(raw_call, "function", {})
            raw_arguments = _field(function, "arguments", "{}")
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"_malformed_arguments": raw_arguments}
            else:
                arguments = dict(raw_arguments or {})
            calls.append(
                ToolCall(
                    id=str(_field(raw_call, "id", f"call-{index}")),
                    name=str(_field(function, "name", "")),
                    arguments=arguments,
                )
            )
        usage = _field(response, "usage", {}) or {}
        input_tokens = int(
            _field(usage, "prompt_tokens", _field(usage, "input_tokens", 0)) or 0
        )
        output_tokens = int(
            _field(usage, "completion_tokens", _field(usage, "output_tokens", 0)) or 0
        )
        cost = 0.0
        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            pass
        content = _field(message, "content")
        return ModelReply(
            content=None if content is None else str(content),
            tool_calls=tuple(calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
