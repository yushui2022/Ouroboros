"""Small deterministic probe suite.

The probes are protocol checks, not model-generated answers.  A real adapter
executes the prompt and returns a trace; the judges below inspect only that trace
and fixed expected values.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from ouroboros.eval import JudgeResult, TaskSpec, ToolSpec


def _tool(name: str, required: tuple[str, ...], **parameters: str) -> ToolSpec:
    return ToolSpec(name=name, parameters=parameters, required=required)


def _has_call(response: Mapping[str, Any], expected_name: str, expected_args: Mapping[str, Any]) -> JudgeResult:
    calls = response.get("tool_calls", [])
    if not isinstance(calls, list):
        return JudgeResult(False, "invalid_arguments", "tool_calls is not a list")
    for call in calls:
        if isinstance(call, Mapping) and call.get("name", call.get("tool_name")) == expected_name:
            args = call.get("arguments", call.get("args", {}))
            if isinstance(args, Mapping) and all(args.get(k) == v for k, v in expected_args.items()):
                return JudgeResult(True)
    return JudgeResult(False, "invalid_arguments", f"expected {expected_name} call")


def _answer(response: Mapping[str, Any], expected: Any) -> JudgeResult:
    answer = response.get("answer", response.get("final", response.get("content")))
    if answer == expected:
        return JudgeResult(True)
    # Models often wrap an otherwise correct JSON answer in a fenced block or
    # a short explanatory sentence. Decode that representation explicitly so
    # the probe measures state retention rather than markdown style.
    if isinstance(expected, Mapping) and isinstance(answer, str):
        candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", answer, flags=re.DOTALL)
        candidates.append(answer.strip())
        for value in candidates:
            try:
                if json.loads(value) == expected:
                    return JudgeResult(True)
            except json.JSONDecodeError:
                continue
    return JudgeResult(False, "wrong_result_interpretation", f"expected answer {expected!r}")


def _finish_answer(response: Mapping[str, Any], expected: Any) -> JudgeResult:
    """Judge the finish protocol separately from answer interpretation.

    A model may know the right answer but append a progress/waiting sentence.
    That is a dialect mismatch in the completion envelope, so the mutation
    controller can adjust the completion prompt without changing result
    interpretation or tool schemas.
    """
    judged = _answer(response, expected)
    if judged.passed:
        return judged
    answer = response.get("answer", response.get("final", response.get("content")))
    return JudgeResult(
        False,
        "finish_format_mismatch",
        f"expected the exact final answer {expected!r}; received {answer!r}",
    )


def default_probe_tasks() -> tuple[TaskSpec, ...]:
    """Return the immutable P0 probe manifest (six tasks, two capabilities)."""
    calculator = _tool("calculator", ("expression",), expression="string")
    lookup = _tool("lookup", ("key",), key="string")
    return (
        TaskSpec("probe-tool-json", "Call calculator for 2 + 2.", "tool_call_protocol",
                 lambda r: _has_call(r, "calculator", {"expression": "2 + 2"}), (calculator,)),
        TaskSpec("probe-schema-required", "Look up the key 'status'.", "schema_following",
                 lambda r: _has_call(r, "lookup", {"key": "status"}), (lookup,)),
        TaskSpec("probe-result-answer", "Use lookup with key 'status', then return exactly ready.", "result_interpretation",
                 lambda r: _answer(r, "ready"), (lookup,), {"expected_tool_result": {"key": "status", "value": "ready"}}),
        TaskSpec("probe-finish-marker", "When done, return exactly the final answer 'done'.", "finish_detection",
                 lambda r: _finish_answer(r, "done")),
        TaskSpec("probe-retry-recovery", "After a transient tool error, retry once with key 'status'.", "retry_recovery",
                 lambda r: _has_call(r, "lookup", {"key": "status"}), (lookup,), {"requires_retry": True}),
        TaskSpec("probe-context-state", "Use the state tool for user and mode, then return JSON {\"user\":\"alice\",\"mode\":\"fast\"}.", "context_retention",
                 lambda r: _answer(r, {"user": "alice", "mode": "fast"}), (ToolSpec("state", parameters={"key": "string"}, required=("key",)),)),
    )
