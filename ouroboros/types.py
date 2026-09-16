"""Shared, dependency-free types for the Ouroboros reference runtime.

The types deliberately keep provider-specific payloads in ``raw`` fields.  The
evolution controller can therefore inspect a model's dialect without coupling
the runtime to one SDK.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping


@dataclass(slots=True)
class Message:
    role: str
    content: Any = ""
    name: str | None = None
    tool_call_id: str | None = None
    raw: Any = None
    tool_calls: list["ToolCall"] | None = None
    reasoning_content: Any = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {k: v for k, v in value.items() if v is not None}


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None
    required: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "required": list(self.required),
        }


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Any = field(default_factory=dict)
    raw: Any = None


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    output: Any = None
    error: str | None = None
    raw: Any = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    raw: Any = None

    def __post_init__(self) -> None:
        if not self.total_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens + self.reasoning_tokens

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.total_tokens += other.total_tokens


@dataclass(slots=True)
class ProviderResponse:
    """Normalized provider response; ``raw`` contains the untouched payload."""

    content: Any = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: Any = None
    reasoning_content: Any = None


@dataclass(slots=True)
class Budget:
    max_steps: int = 12
    max_tool_calls: int = 32


@dataclass(slots=True)
class AgentGenome:
    """Versionable dialect knobs for the reference runtime."""

    system_prompt: str = "You are a helpful agent. Use tools when needed and stop when the task is complete."
    tool_result_format: str = "structured"
    retry_on_tool_error: bool = True
    max_retries: int = 2
    completion_marker: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Explicit ownership prevents a Genome evolved for one Runtime from being
    # accidentally evaluated under another Runtime implementation.
    model_id: str | None = None
    runtime_profile_id: str | None = None
    branch: str | None = None


@dataclass(slots=True)
class TraceEvent:
    kind: str
    step: int
    timestamp_ms: float
    request: Any = None
    response: Any = None
    tool_call: Any = None
    tool_result: Any = None
    raw: Any = None


@dataclass(slots=True)
class RunResult:
    content: Any
    messages: list[Message]
    trace: list[TraceEvent]
    usage: Usage
    tool_results: list[ToolResult]
    steps: int
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly structural record for evaluators/artifacts."""
        return asdict(self)
