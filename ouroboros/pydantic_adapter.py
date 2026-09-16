"""Optional Pydantic AI mother-runtime adapter.

The reference Runtime remains dependency-free.  This module is the first
concrete mother-runtime integration: it delegates the agent loop to Pydantic
AI while translating its result/usage boundary into Ouroboros records.  The
import is lazy so baseline tests do not require the optional dependency.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import asdict
from typing import Any, Callable, Mapping, Sequence

from .profile import RuntimeProfile
from .types import AgentGenome, Message, RunResult, ToolResult, ToolSpec, TraceEvent, Usage


class PydanticAIRuntime:
    """Thin adapter around :class:`pydantic_ai.Agent`.

    ``agent`` or ``agent_factory`` can be injected for tests and for callers
    that construct a provider-specific Pydantic AI model themselves.
    """

    profile = RuntimeProfile(runtime_id="pydantic-ai-harness", runtime_version="adapter-0.1",
                             adapter_id="pydantic-ai-agent", api_format="pydantic-ai")

    def __init__(self, model: Any = None, *, genome: AgentGenome | None = None,
                 tools: Sequence[ToolSpec] = (), agent: Any = None,
                 agent_factory: Callable[..., Any] | None = None) -> None:
        self.genome = genome or AgentGenome()
        self.tools = tuple(tools)
        self.model_id = str(model)
        self._tool_results: list[ToolResult] = []
        if self.genome.runtime_profile_id and self.genome.runtime_profile_id != self.profile.profile_id:
            raise ValueError("Genome runtime profile does not match Pydantic AI Runtime")
        if agent is not None:
            self.agent = agent
            return
        if agent_factory is not None:
            try:
                self.agent = agent_factory(model, system_prompt=self.genome.system_prompt)
            except TypeError:
                self.agent = agent_factory(model=model, system_prompt=self.genome.system_prompt)
            self._register_tools()
            return
        try:
            from pydantic_ai import Agent
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("install ouroboros[pydantic] to use PydanticAIRuntime") from exc
        self.agent = Agent(model, system_prompt=self.genome.system_prompt)
        self._register_tools()

    def _register_tools(self) -> None:
        if not any(spec.handler is not None for spec in self.tools):
            return
        decorator = getattr(self.agent, "tool_plain", None) or getattr(self.agent, "tool", None)
        if decorator is None:
            raise TypeError("Pydantic AI Agent does not expose a tool decorator")
        for spec in self.tools:
            if spec.handler is None:
                continue
            handler = spec.handler
            if inspect.iscoroutinefunction(handler):
                async def wrapped(*args: Any, _handler: Any = handler, _spec: ToolSpec = spec, **kwargs: Any) -> Any:
                    call_id = f"pydantic-call-{len(self._tool_results) + 1}"
                    try:
                        value = await _handler(*args, **kwargs)
                        self._tool_results.append(ToolResult(call_id, _spec.name, output=value))
                        return value
                    except Exception as exc:
                        self._tool_results.append(ToolResult(call_id, _spec.name,
                                                             error=f"{type(exc).__name__}: {exc}"))
                        raise
            else:
                def wrapped(*args: Any, _handler: Any = handler, _spec: ToolSpec = spec, **kwargs: Any) -> Any:
                    call_id = f"pydantic-call-{len(self._tool_results) + 1}"
                    try:
                        value = _handler(*args, **kwargs)
                        self._tool_results.append(ToolResult(call_id, _spec.name, output=value))
                        return value
                    except Exception as exc:
                        self._tool_results.append(ToolResult(call_id, _spec.name,
                                                             error=f"{type(exc).__name__}: {exc}"))
                        raise
            wrapped.__name__ = spec.name
            wrapped.__doc__ = spec.description
            # Keep the original typed signature so Pydantic AI generates the
            # same JSON schema as the source handler (closure bookkeeping
            # parameters stay out of the model-visible dialect).
            try:
                wrapped.__signature__ = inspect.signature(handler)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                pass
            decorator(wrapped)

    def run(self, prompt: str | Message, *, deps: Any = None,
            message_history: Sequence[Any] | None = None,
            model_settings: Mapping[str, Any] | None = None) -> RunResult:
        text = prompt.content if isinstance(prompt, Message) else prompt
        self._tool_results = []
        started = time.monotonic()
        kwargs: dict[str, Any] = {}
        if deps is not None:
            kwargs["deps"] = deps
        if message_history is not None:
            kwargs["message_history"] = list(message_history)
        if model_settings is not None:
            kwargs["model_settings"] = dict(model_settings)
        try:
            result = self.agent.run_sync(text, **kwargs)
            output = getattr(result, "output", getattr(result, "data", result))
            usage_value = result.usage() if callable(getattr(result, "usage", None)) else getattr(result, "usage", {})
            usage = _usage(usage_value)
            messages = _messages(result)
            trace = [TraceEvent("pydantic_ai_result", 1, round((time.monotonic() - started) * 1000, 3),
                                response=output, raw=result)]
            return RunResult(output, messages, trace, usage, list(self._tool_results), 1, True)
        except Exception as exc:
            trace = [TraceEvent("pydantic_ai_error", 1, round((time.monotonic() - started) * 1000, 3), raw=exc)]
            return RunResult("", [], trace, Usage(), list(self._tool_results), 1, False, str(exc))


def _usage(value: Any) -> Usage:
    if value is None:
        return Usage()
    if isinstance(value, Mapping):
        get = value.get
    else:
        get = lambda key, default=0: getattr(value, key, default)
    return Usage(input_tokens=int(get("input_tokens", get("request_tokens", 0)) or 0),
                 output_tokens=int(get("output_tokens", get("response_tokens", 0)) or 0),
                 reasoning_tokens=int(get("reasoning_tokens", 0) or 0),
                 total_tokens=int(get("total_tokens", 0) or 0))


def _messages(result: Any) -> list[Message]:
    history = result.all_messages() if callable(getattr(result, "all_messages", None)) else []
    messages: list[Message] = []
    for item in history:
        role = getattr(item, "role", None) or getattr(item, "kind", None) or "unknown"
        role = {"request": "user", "response": "assistant", "system-prompt": "system",
                "tool-return": "tool"}.get(str(role), role)
        content = getattr(item, "content", getattr(item, "parts", str(item)))
        if isinstance(content, list):
            content = "".join(str(getattr(part, "content", getattr(part, "text", part))) for part in content)
        messages.append(Message(role=str(role), content=content, raw=item,
                                reasoning_content=getattr(item, "reasoning_content", None)))
    return messages


class PydanticAIAgentAdapter(PydanticAIRuntime):
    """Compatibility name emphasizing this object is an Agent adapter."""

    @property
    def runtime_profile(self) -> RuntimeProfile:
        return self.profile

    def __init__(self, model: Any = None, tools: Sequence[ToolSpec] = (), **kwargs: Any) -> None:
        super().__init__(model, tools=tools, **kwargs)

    @staticmethod
    def available() -> bool:
        try:
            import pydantic_ai  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, task: str | Message, *, genome: AgentGenome | None = None,
            settings: dict[str, Any] | None = None) -> RunResult:
        if genome is not None:
            if genome.runtime_profile_id and genome.runtime_profile_id != self.profile.profile_id:
                raise ValueError("Genome runtime profile does not match Pydantic AI Runtime")
            self.genome = genome
        return super().run(task, model_settings=settings)


__all__ = ["PydanticAIRuntime", "PydanticAIAgentAdapter"]
