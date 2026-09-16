"""Small, observable tool-calling Agent Runtime used as Ouroboros' mother."""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .providers import Provider, normalize_response
from .profile import RuntimeProfile
from .types import (
    AgentGenome,
    Budget,
    Message,
    RunResult,
    ToolCall,
    ToolResult,
    ToolSpec,
    TraceEvent,
    Usage,
)


class AgentRuntime:
    """Execute a genome against a provider and explicit deterministic tools.

    The runtime does not accept or rank mutations.  It only returns a complete
    trace and usage record so an external Ouroboros controller can evaluate a
    candidate safely.
    """

    def __init__(
        self,
        provider: Provider,
        tools: Sequence[ToolSpec] = (),
        *,
        genome: AgentGenome | None = None,
        budget: Budget | None = None,
        runtime_profile: RuntimeProfile | None = None,
    ) -> None:
        self.provider = provider
        self.tools = list(tools)
        self.genome = genome or AgentGenome()
        self.budget = budget or Budget()
        self.runtime_profile = runtime_profile
        if self.genome.runtime_profile_id is not None:
            if runtime_profile is None:
                raise ValueError("a Genome bound to a RuntimeProfile requires an explicit runtime_profile")
            if self.genome.runtime_profile_id != runtime_profile.profile_id:
                raise ValueError("Genome runtime profile does not match the active RuntimeProfile")
        self._tool_map = {tool.name: tool for tool in self.tools}

    def run(self, task: str | Message, *, settings: dict[str, Any] | None = None) -> RunResult:
        messages = [Message("system", self.genome.system_prompt)]
        messages.append(task if isinstance(task, Message) else Message("user", task))
        trace: list[TraceEvent] = []
        tool_results: list[ToolResult] = []
        usage = Usage()
        retries = 0
        final_content: Any = ""
        started = time.monotonic()

        for step in range(1, self.budget.max_steps + 1):
            request = {
                "messages": [m.to_dict() for m in messages],
                "tools": [_tool_schema(tool) for tool in self.tools],
                "settings": dict(settings or {}),
                "runtime_profile_id": self.runtime_profile.profile_id if self.runtime_profile else None,
            }
            request_time = time.monotonic()
            try:
                response = normalize_response(self.provider.complete(messages, self.tools, settings=settings))
            except Exception as exc:  # provider errors remain visible in the trace
                trace.append(TraceEvent("provider_error", step, _ms(request_time), request=request, raw=exc))
                return RunResult(final_content, messages, trace, usage, tool_results, step, False, str(exc))
            usage.add(response.usage)
            trace.append(TraceEvent("provider_response", step, _ms(request_time), request=request, response=response, raw=response.raw))
            messages.append(Message("assistant", response.content, tool_calls=response.tool_calls,
                                    reasoning_content=response.reasoning_content, raw=response.raw))
            final_content = response.content
            if not response.tool_calls:
                if self.genome.completion_marker and self.genome.completion_marker not in str(final_content):
                    return RunResult(final_content, messages, trace, usage, tool_results, step, False, "completion marker missing")
                return RunResult(final_content, messages, trace, usage, tool_results, step, True)

            for call in response.tool_calls:
                if len(tool_results) >= self.budget.max_tool_calls:
                    return RunResult(final_content, messages, trace, usage, tool_results, step, False, "tool call budget exceeded")
                result = self._execute(call)
                tool_results.append(result)
                trace.append(TraceEvent("tool_result", step, _ms(started), tool_call=call, tool_result=result, raw=result.raw))
                messages.append(Message("tool", self._format_tool_result(result), name=call.name, tool_call_id=call.id, raw=result.raw))
                if result.error and (not self.genome.retry_on_tool_error or retries >= self.genome.max_retries):
                    return RunResult(final_content, messages, trace, usage, tool_results, step, False, result.error)
                if result.error:
                    retries += 1
        return RunResult(final_content, messages, trace, usage, tool_results, self.budget.max_steps, False, "step budget exceeded")

    def _execute(self, call: ToolCall) -> ToolResult:
        tool = self._tool_map.get(call.name)
        handler = getattr(tool, "handler", None) if tool is not None else None
        if tool is None or handler is None:
            return ToolResult(call.id, call.name, error=f"unknown tool: {call.name}", raw=call.raw)
        try:
            args = _arguments(call.arguments)
            value = handler(**args) if isinstance(args, Mapping) else handler(args)
            if inspect.isawaitable(value):
                raise TypeError("async tool handlers are not supported by the sync reference runtime")
            return ToolResult(call.id, call.name, output=value, raw=call.raw)
        except Exception as exc:
            return ToolResult(call.id, call.name, error=f"{type(exc).__name__}: {exc}", raw=call.raw)

    def _format_tool_result(self, result: ToolResult) -> Any:
        if self.genome.tool_result_format == "text":
            return result.error or str(result.output)
        value = {"ok": result.ok, "name": result.name}
        value["error" if result.error else "output"] = result.error or result.output
        return json.dumps(value, ensure_ascii=False, default=str)


def _arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
    return value if value is not None else {}


def _tool_schema(tool: Any) -> dict[str, Any]:
    """Serialize both runtime ``ToolSpec`` and lightweight evaluator specs."""
    schema = getattr(tool, "schema", None)
    if callable(schema):
        return schema()
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", ""),
        "parameters": dict(getattr(tool, "parameters", {}) or {}),
        "required": list(getattr(tool, "required", ()) or ()),
    }


def _ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 3)
