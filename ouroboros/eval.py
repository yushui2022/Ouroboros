"""Deterministic evaluation primitives for Ouroboros.

The evaluator deliberately knows nothing about a model's self-reported score.  A
runner returns an execution trace and this module compares the observable result
with a task's external judge.  Keeping this contract small makes it usable with a
local test provider as well as a remote Agent adapter.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


JSON = Any


@dataclass(frozen=True)
class ToolSpec:
    """The externally visible part of a tool contract."""

    name: str
    parameters: Mapping[str, str] = field(default_factory=dict)
    required: tuple[str, ...] = ()


@dataclass(frozen=True)
class JudgeResult:
    passed: bool
    failure_class: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    """A pre-registered, deterministic task and its external judge.

    ``judge`` receives the normalized runner response.  It must not call the
    model or inspect hidden evaluation data.
    """

    id: str
    prompt: str
    capability: str
    judge: Callable[[Mapping[str, Any]], JudgeResult | bool]
    tools: tuple[ToolSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_value(cls, value: Any) -> "Usage":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            # Interoperate with runtime/provider dataclasses without making
            # the evaluator depend on their module.
            value = {key: getattr(value, key, 0) for key in
                     ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens",
                      "reasoning_tokens", "completion_reasoning_tokens", "total_tokens")}
        def integer(*keys: str) -> int:
            for key in keys:
                if key not in value:
                    continue
                try:
                    return max(0, int(value.get(key, 0) or 0))
                except (TypeError, ValueError):
                    pass
            return 0
        inp = integer("input_tokens", "prompt_tokens")
        out = integer("output_tokens", "completion_tokens")
        reasoning = integer("reasoning_tokens", "completion_reasoning_tokens")
        total = integer("total_tokens") or inp + out
        return cls(inp, out, reasoning, total)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    seed: int
    passed: bool
    capability: str
    failure_class: str | None
    reason: str | None
    tool_call_count: int
    tool_error_count: int
    retry_count: int
    usage: Usage
    latency_ms: float
    response: Mapping[str, Any]


class Runner(Protocol):
    def __call__(self, task: TaskSpec, seed: int) -> Any: ...


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    # Provider payloads occasionally contain enums or lightweight value
    # objects.  Their textual representation is deterministic enough for an
    # artifact hash and avoids making the evaluator depend on an SDK.
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_response(value: Any) -> dict[str, Any]:
    """Normalize common provider responses without discarding raw fields."""
    if isinstance(value, Mapping):
        result = dict(value)
    elif isinstance(value, str):
        result = {"answer": value, "content": value}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, Mapping):
                result.update(parsed)
        except json.JSONDecodeError:
            pass
    elif hasattr(value, "content") or hasattr(value, "tool_calls"):
        calls = []
        for call in (getattr(value, "tool_calls", None) or []):
            if isinstance(call, Mapping):
                calls.append(dict(call))
            else:
                calls.append({"name": getattr(call, "name", ""),
                              "arguments": getattr(call, "arguments", {}),
                              "id": getattr(call, "id", "")})
        usage = getattr(value, "usage", {})
        if is_dataclass(usage):
            usage = asdict(usage)
        result = {"content": getattr(value, "content", ""),
                  "answer": getattr(value, "content", ""),
                  "tool_calls": calls,
                  "usage": usage}
    else:
        result = {"answer": value}
    result.setdefault("tool_calls", [])
    result.setdefault("usage", {})
    if is_dataclass(result.get("usage")):
        result["usage"] = asdict(result["usage"])
    return result


def inspect_tool_errors(response: Mapping[str, Any], tools: Sequence[ToolSpec]) -> tuple[int, int]:
    """Return (number of calls, number of malformed calls) from an observable trace."""
    calls = response.get("tool_calls", ())
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        return 0, 1
    by_name = {tool.name: tool for tool in tools}
    errors = 0
    for call in calls:
        if not isinstance(call, Mapping):
            errors += 1
            continue
        name = call.get("name") or call.get("tool_name")
        args = call.get("arguments", call.get("args", {}))
        spec = by_name.get(str(name))
        if spec is None or not isinstance(args, Mapping):
            errors += 1
            continue
        if any(field not in args for field in spec.required):
            errors += 1
        if any(field not in spec.parameters for field in args):
            errors += 1
    return len(calls), errors


def _retry_count(response: Mapping[str, Any]) -> int:
    value = response.get("retry_count", response.get("retries", 0))
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class Evaluator:
    """Run registered tasks and produce replayable per-task records."""

    def __init__(self, tasks: Iterable[TaskSpec], *, evaluator_version: str = "eval-v1") -> None:
        self.tasks = tuple(tasks)
        self.evaluator_version = evaluator_version

    def manifest(self) -> dict[str, Any]:
        return {
            "evaluator_version": self.evaluator_version,
            "tasks": [
                {"id": t.id, "capability": t.capability, "prompt": t.prompt, "tools": [_tool_manifest(x) for x in t.tools], "metadata": _canonical(t.metadata)}
                for t in self.tasks
            ],
        }

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.manifest())

    def run_task(self, task: TaskSpec, runner: Runner, *, seed: int = 0) -> TaskResult:
        started = time.perf_counter()
        try:
            candidate = runner(task, seed)
            if inspect.isawaitable(candidate):
                raise TypeError("async runners must be awaited by the caller")
            response = normalize_response(candidate)
            judged = task.judge(response)
            result = judged if isinstance(judged, JudgeResult) else JudgeResult(bool(judged))
        except Exception as exc:  # evaluation records failures instead of hiding them
            response = {"error": type(exc).__name__, "error_message": str(exc), "tool_calls": [], "usage": {}}
            result = JudgeResult(False, "runner_error", str(exc))
        calls, tool_errors = inspect_tool_errors(response, task.tools)
        # A runtime may know about handler failures that are not visible in the
        # final assistant tool-call list. Prefer this explicit trace aggregate
        # when present, while retaining structural inspection as the fallback.
        if "tool_error_count" in response:
            try:
                tool_errors = max(0, int(response["tool_error_count"] or 0))
            except (TypeError, ValueError):
                pass
        latency = response.get("latency_ms")
        try:
            latency_ms = float(latency) if latency is not None else (time.perf_counter() - started) * 1000
        except (TypeError, ValueError):
            latency_ms = (time.perf_counter() - started) * 1000
        return TaskResult(task.id, seed, result.passed, task.capability, result.failure_class, result.reason,
                          calls, tool_errors, _retry_count(response), Usage.from_value(response.get("usage")), latency_ms, response)

    def run(self, runner: Runner, *, seeds: Iterable[int] = (0,)) -> list[TaskResult]:
        seed_values = tuple(seeds)
        return [self.run_task(task, runner, seed=seed) for task in self.tasks for seed in seed_values]

    def evaluate(self, runner: Runner, *, seeds: Iterable[int] = (0,)) -> list[TaskResult]:
        """Explicit alias for callers that prefer the evaluator vocabulary."""
        return self.run(runner, seeds=seeds)


def summarize(results: Iterable[TaskResult]) -> dict[str, Any]:
    rows = list(results)
    n = len(rows)
    if not n:
        return {"task_count": 0, "task_success_rate": 0.0}
    usage = [row.usage for row in rows]
    return {
        "task_count": n,
        "task_success_rate": sum(row.passed for row in rows) / n,
        "tool_call_success_rate": 1.0 - sum(row.tool_error_count for row in rows) / max(1, sum(row.tool_call_count for row in rows)),
        "tool_error_rate": sum(row.tool_error_count for row in rows) / n,
        "retry_count": sum(row.retry_count for row in rows) / n,
        "average_input_tokens": sum(x.input_tokens for x in usage) / n,
        "average_output_tokens": sum(x.output_tokens for x in usage) / n,
        "average_reasoning_tokens": sum(x.reasoning_tokens for x in usage) / n,
        "average_total_tokens": sum(x.total_tokens for x in usage) / n,
        "p50_latency_ms": _percentile([r.latency_ms for r in rows], 0.50),
        "p95_latency_ms": _percentile([r.latency_ms for r in rows], 0.95),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return float(ordered[index])


def _tool_manifest(tool: Any) -> dict[str, Any]:
    """Serialize both eval.ToolSpec and runtime.types.ToolSpec safely."""
    if isinstance(tool, Mapping):
        value = dict(tool)
    else:
        value = {key: getattr(tool, key, None) for key in ("name", "description", "parameters", "required")}
    return {key: _canonical(value.get(key)) for key in ("name", "description", "parameters", "required") if value.get(key) is not None}


def results_json(results: Iterable[TaskResult]) -> str:
    """Serialize results for an artifact or ledger without custom encoders."""
    return json.dumps([asdict(row) for row in results], ensure_ascii=False, sort_keys=True, indent=2)
