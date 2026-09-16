"""Provider boundary and a deterministic provider for local experiments."""

from __future__ import annotations

import json
import os
from urllib import error as urlerror
from urllib import request as urlrequest
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from .types import Message, ProviderResponse, ToolCall, ToolSpec, Usage


class Provider(Protocol):
    """Minimal synchronous provider contract used by :class:`AgentRuntime`."""

    model_id: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        settings: dict[str, Any] | None = None,
    ) -> ProviderResponse: ...


class DeepSeekProvider:
    """Small OpenAI-compatible DeepSeek adapter.

    The adapter intentionally keeps the HTTP boundary narrow.  It does not
    decide retries or accept mutations; ``AgentRuntime`` owns that loop.  The
    raw response is preserved so DeepSeek-specific ``reasoning_content`` and
    usage fields remain available to the fingerprint and ledger layers.
    """

    model_id = "deepseek-flash"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "deepseek-flash",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model_id = model
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        settings: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the DeepSeek provider")
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [_message_payload(message) for message in messages],
        }
        if tools:
            payload["tools"] = [_openai_tool_schema(tool) for tool in tools]
        payload.update(settings or {})
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urlrequest.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {detail[:500]}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"DeepSeek API connection failed: {exc.reason}") from exc
        return normalize_response(raw)


class OpenAICompatibleProvider(DeepSeekProvider):
    """Generic OpenAI-compatible chat provider for external gateways.

    This reuses the same normalized response and reasoning preservation path as
    DeepSeek, while allowing a gateway-specific base URL, API key and model.
    ``base_url`` may be either an origin or an origin ending in ``/v1``.
    """

    def __init__(self, *, api_key: str | None = None, base_url: str,
                 model: str, timeout_seconds: float = 120.0) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"
        super().__init__(api_key=api_key, base_url=normalized, model=model,
                         timeout_seconds=timeout_seconds)


class AnthropicMessagesProvider:
    """Anthropic Messages adapter for gateways exposing ``/claude/v1``."""

    def __init__(self, *, api_key: str | None = None, base_url: str,
                 model: str, timeout_seconds: float = 120.0) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                 settings: dict[str, Any] | None = None) -> ProviderResponse:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Anthropic provider")
        system = "\n\n".join(str(m.content) for m in messages if m.role == "system")
        payload_messages = [_anthropic_message(m) for m in messages if m.role != "system"]
        payload: dict[str, Any] = {"model": self.model_id, "messages": payload_messages,
                                   "max_tokens": 512}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": t.name, "description": t.description,
                                 "input_schema": {"type": "object", "properties": dict(t.parameters),
                                                  "required": list(t.required)}} for t in tools]
        payload.update(settings or {})
        request = urlrequest.Request(f"{self.base_url}/claude/v1/messages", data=json.dumps(payload).encode(),
                                     headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                                              "Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API HTTP {exc.code}: {detail[:500]}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"Anthropic API connection failed: {exc.reason}") from exc
        blocks = raw.get("content", [])
        text_parts, calls = [], []
        reasoning = []
        for block in blocks:
            if block.get("type") == "text": text_parts.append(block.get("text", ""))
            elif block.get("type") == "thinking": reasoning.append(block.get("thinking", ""))
            elif block.get("type") == "tool_use": calls.append(ToolCall(str(block.get("id", "")), str(block.get("name", "")), block.get("input", {}), raw=block))
        usage = _usage(raw.get("usage", {}))
        return ProviderResponse("\n".join(text_parts), calls, raw.get("stop_reason"), usage, raw, "\n".join(reasoning) or None)


def _anthropic_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": message.tool_call_id or "",
                                                "content": str(message.content)}]}
    if message.role == "assistant" and message.tool_calls:
        blocks: list[dict[str, Any]] = []
        if message.content: blocks.append({"type": "text", "text": str(message.content)})
        blocks.extend({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in message.tool_calls)
        return {"role": "assistant", "content": blocks}
    return {"role": "user" if message.role == "user" else "assistant", "content": message.content}


def _message_payload(message: Message) -> dict[str, Any]:
    """Serialize an internal message without leaking non-JSON trace fields."""
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    if message.reasoning_content is not None:
        payload["reasoning_content"] = message.reasoning_content
    return payload


def _openai_tool_schema(tool: ToolSpec) -> dict[str, Any]:
    parameters = dict(getattr(tool, "parameters", {}) or {})
    required = list(getattr(tool, "required", ()) or ())
    if parameters and not ("type" in parameters and "properties" in parameters):
        parameters = {"type": "object", "properties": parameters}
    parameters.setdefault("type", "object")
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", ""),
            "parameters": parameters,
        },
    }


def _usage(value: Any) -> Usage:
    if isinstance(value, Usage):
        return value
    if not isinstance(value, dict):
        return Usage(raw=value)
    details = value.get("completion_tokens_details", {})
    if not isinstance(details, dict):
        details = {}
    return Usage(
        input_tokens=int(value.get("input_tokens", value.get("prompt_tokens", 0)) or 0),
        output_tokens=int(value.get("output_tokens", value.get("completion_tokens", 0)) or 0),
        reasoning_tokens=int(value.get("reasoning_tokens", value.get("reasoning", details.get("reasoning_tokens", 0))) or 0),
        total_tokens=int(value.get("total_tokens", 0) or 0),
        raw=value,
    )


def normalize_response(value: Any) -> ProviderResponse:
    """Normalize common OpenAI-like dictionaries while preserving ``raw``."""
    if isinstance(value, ProviderResponse):
        return value
    if not isinstance(value, dict):
        return ProviderResponse(content=value, raw=value)
    # OpenAI-compatible responses commonly wrap the assistant message in
    # ``choices[0]``; accepting both shapes keeps the adapter intentionally
    # small while retaining the untouched payload in ``raw``.
    if isinstance(value.get("choices"), list) and value["choices"]:
        choice = value["choices"][0]
        message = choice.get("message", choice) if isinstance(choice, dict) else choice
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    else:
        message = value.get("message", value)
        finish_reason = value.get("finish_reason")
    calls: list[ToolCall] = []
    for index, call in enumerate(message.get("tool_calls", value.get("tool_calls", [])) or []):
        function = call.get("function", call) if isinstance(call, dict) else call
        if isinstance(function, dict):
            arguments = function.get("arguments", {})
            name = str(function.get("name", ""))
        else:
            arguments, name = {}, str(function)
        calls.append(ToolCall(str(call.get("id", f"call-{index}")), name, arguments, raw=call))
    return ProviderResponse(
        content=message.get("content", value.get("content", "")) if isinstance(message, dict) else message,
        tool_calls=calls,
        finish_reason=finish_reason,
        usage=_usage(value.get("usage", {})),
        raw=value,
        reasoning_content=(message.get("reasoning_content") if isinstance(message, dict) else None)
        or value.get("reasoning_content"),
    )


class DeterministicTestProvider:
    """Scripted provider that makes the runtime testable without network access.

    ``script`` may contain responses/dicts, or callables receiving
    ``(messages, tools, settings)``.  Once exhausted, the provider returns a
    final response with ``default_content``.  Every request is retained in
    ``requests`` for probe and trace assertions.
    """

    def __init__(
        self,
        script: Iterable[Any] = (),
        *,
        model_id: str = "deterministic-test",
        default_content: str = "done",
    ) -> None:
        self.model_id = model_id
        self._script = list(script)
        self.default_content = default_content
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        settings: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        self.requests.append({
            "messages": [m.to_dict() for m in messages],
            "tools": [_tool_schema(tool) for tool in tools],
            "settings": dict(settings or {}),
        })
        if self._script:
            value = self._script.pop(0)
            if callable(value):
                value = value(messages, tools, settings or {})
            response = normalize_response(value)
            if response.usage.total_tokens == 0:
                input_tokens = sum(len(str(message.content).split()) for message in messages)
                output_tokens = max(1, len(str(response.content).split()))
                response.usage = Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    raw={"estimated": True},
                )
            return response
        input_tokens = sum(len(str(message.content).split()) for message in messages)
        output_tokens = max(1, len(self.default_content.split()))
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            raw={"estimated": True},
        )
        return ProviderResponse(
            content=self.default_content,
            finish_reason="stop",
            usage=usage,
            raw={"test": True, "usage": usage.raw},
        )


def _tool_schema(tool: Any) -> dict[str, Any]:
    """Serialize both runtime and evaluator tool specifications."""
    schema = getattr(tool, "schema", None)
    if callable(schema):
        return schema()
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", ""),
        "parameters": dict(getattr(tool, "parameters", {}) or {}),
        "required": list(getattr(tool, "required", ()) or ()),
    }
