"""Constrained, atomic Dialect Genome mutations.

This module is intentionally independent from the CLI and orchestration code.  It
provides the small control-plane contract needed by a future evolution loop:
failure observations are turned into a few immutable candidate genomes and a
paired dev evaluation is reduced to an auditable accept/reject decision.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .types import AgentGenome
from .profile import RuntimeProfile, bind_genome


@dataclass(frozen=True)
class DirectionSpec:
    """Objective and hard constraints for one evolution direction."""

    primary: str = "task_success_rate"
    maximize: bool | None = None
    max_success_drop: float = 0.03
    confidence: float = 0.95
    success_floor: float | None = None
    max_tool_error_rate: float | None = None
    max_total_tokens: int | None = None
    max_average_total_tokens: float | None = None
    min_primary_improvement: float = 0.0
    require_safety_zero: bool = True
    require_pollution_zero: bool = True
    budget: int | None = None

    def __post_init__(self) -> None:
        if self.maximize is None:
            object.__setattr__(self, "maximize", self.primary in {"task_success_rate", "tool_call_success_rate"})
        if not 0 <= self.max_success_drop <= 1:
            raise ValueError("max_success_drop must be between 0 and 1")
        if self.success_floor is not None and not 0 <= self.success_floor <= 1:
            raise ValueError("success_floor must be between 0 and 1")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class MutationCandidate:
    """An atomic change to a parent :class:`AgentGenome`."""

    candidate_id: str
    genome: AgentGenome
    layer: str
    rationale: str
    failure_classes: tuple[str, ...] = ()
    patch: Mapping[str, Any] = field(default_factory=dict)

    @property
    def edit_ref(self) -> str:
        return self.candidate_id


@dataclass(frozen=True)
class Decision:
    """Auditable result of comparing a candidate with its paired parent run."""

    accepted: bool
    candidate_id: str | None
    reason: str
    baseline_metrics: Mapping[str, float]
    candidate_metrics: Mapping[str, float]
    violations: tuple[str, ...] = ()
    deltas: Mapping[str, float] = field(default_factory=dict)


_ALIASES = {
    "tool_name_mismatch": "tool_name_mismatch",
    "invalid_arguments": "invalid_arguments",
    "schema_omission": "schema_omission",
    "wrong_result_interpretation": "wrong_result_interpretation",
    "unnecessary_retry": "unnecessary_retry",
    "context_loss": "context_loss",
    "premature_termination": "premature_termination",
    "late_termination": "late_termination",
    "finish_format_mismatch": "finish_format_mismatch",
}


def _failure_class(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return value.get("failure_class") or value.get("failure") or value.get("class")
    return getattr(value, "failure_class", None)


def _clone_genome(genome: AgentGenome, **changes: Any) -> AgentGenome:
    metadata = copy.deepcopy(genome.metadata)
    metadata.update(changes.pop("metadata", {}))
    changes["metadata"] = metadata
    return replace(genome, **changes)


def generate_candidates(
    genome: AgentGenome,
    failures: Iterable[Any],
    *,
    limit: int = 3,
    runtime_profile: RuntimeProfile | None = None,
    model_id: str | None = None,
) -> list[MutationCandidate]:
    """Generate at most ``limit`` single-layer mutations from observed failures.

    Unknown failure classes are ignored.  A candidate changes one genome field
    (or one named metadata knob), making paired experiments attributable.
    """

    if limit < 1:
        return []
    if runtime_profile is not None and model_id is not None:
        genome = bind_genome(genome, model_id=model_id, profile=runtime_profile)
    classes = tuple(dict.fromkeys(c for c in (_failure_class(x) for x in failures) if c in _ALIASES))
    out: list[MutationCandidate] = []

    def add(suffix: str, layer: str, rationale: str, changed: AgentGenome, patch: Mapping[str, Any]) -> None:
        if len(out) < limit:
            out.append(MutationCandidate(f"mutation-{len(out)+1}-{suffix}", changed, layer, rationale, classes, dict(patch)))

    if any(c in classes for c in ("tool_name_mismatch", "invalid_arguments", "schema_omission")):
        prompt = genome.system_prompt.rstrip() + "\nEmit tool calls with the exact registered name and all required JSON fields."
        add("schema", "prompt", "Make tool names and required arguments explicit.", _clone_genome(genome, system_prompt=prompt), {"system_prompt": prompt})
    if "wrong_result_interpretation" in classes and len(out) < limit:
        fmt = "text" if genome.tool_result_format == "structured" else "structured"
        add("result", "tool_result_formatter", "Use a result representation that is easier to interpret.", _clone_genome(genome, tool_result_format=fmt), {"tool_result_format": fmt})
    if "unnecessary_retry" in classes and len(out) < limit:
        retries = max(0, genome.max_retries - 1)
        add("retry", "retry_policy", "Reduce repeated retries after tool failures.", _clone_genome(genome, max_retries=retries), {"max_retries": retries})
    if "context_loss" in classes and len(out) < limit:
        prompt = genome.system_prompt.rstrip() + "\nPreserve task constraints and relevant tool state across turns."
        add("context", "prompt", "Tell the model to retain state during long runs.", _clone_genome(genome, system_prompt=prompt), {"system_prompt": prompt})
    if "finish_format_mismatch" in classes and len(out) < limit:
        prompt = genome.system_prompt.rstrip() + (
            "\nCompletion protocol: when the task is complete, output only the final answer; "
            "do not add waiting, follow-up, or status text."
        )
        add(
            "finish-format",
            "prompt",
            "Require a concise completion envelope without extra status text.",
            _clone_genome(genome, system_prompt=prompt),
            {"system_prompt": prompt, "completion_output": "concise"},
        )
    if "premature_termination" in classes and len(out) < limit:
        marker = genome.completion_marker or "<FINAL>"
        prompt = genome.system_prompt.rstrip() + f"\nOnly finish after the task is complete and emit {marker}."
        add("finish", "finish_detector", "Require an explicit completion signal.", _clone_genome(genome, system_prompt=prompt, completion_marker=marker), {"completion_marker": marker})
    if "late_termination" in classes and len(out) < limit:
        prompt = genome.system_prompt.rstrip() + "\nStop immediately once the requested result is complete."
        add("stop", "finish_detector", "Discourage work after task completion.", _clone_genome(genome, system_prompt=prompt), {"system_prompt": prompt})
    return out


def _row_value(row: Any, name: str, default: Any = 0) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _metric_rows(rows: Sequence[Any] | Mapping[str, Any]) -> dict[str, float]:
    # Accept a pre-aggregated metric mapping as a convenience for controllers
    # that persist only summaries.  Per-task rows remain the preferred input.
    if isinstance(rows, Mapping):
        result = {
            "task_success_rate": float(rows.get("task_success_rate", rows.get("success_rate", 0.0)) or 0.0),
            "success_lower_bound": float(rows.get("success_lower_bound", rows.get("task_success_rate", rows.get("success_rate", 0.0))) or 0.0),
            "tool_error_rate": float(rows.get("tool_error_rate", 0.0) or 0.0),
            "tool_call_success_rate": float(rows.get("tool_call_success_rate", 1.0) or 0.0),
            "retry_count": float(rows.get("retry_count", 0.0) or 0.0),
            "total_tokens": float(rows.get("total_tokens", 0.0) or 0.0),
            "average_total_tokens": float(rows.get("average_total_tokens", rows.get("average_tokens", 0.0)) or 0.0),
            "safety_violations": float(rows.get("safety_violations", 0.0) or 0.0),
            "pollution_violations": float(rows.get("pollution_violations", 0.0) or 0.0),
        }
        return result
    n = len(rows)
    if not n:
        return {"task_success_rate": 0.0, "success_lower_bound": 0.0, "tool_error_rate": 0.0, "tool_call_success_rate": 1.0, "retry_count": 0.0, "total_tokens": 0.0, "average_total_tokens": 0.0, "safety_violations": 0.0, "pollution_violations": 0.0}
    passed = sum(bool(_row_value(r, "passed", False)) for r in rows)
    errors = sum(float(_row_value(r, "tool_error_count", 0) or 0) for r in rows)
    calls = sum(float(_row_value(r, "tool_call_count", 0) or 0) for r in rows)
    retries = sum(float(_row_value(r, "retry_count", 0) or 0) for r in rows)
    tokens = 0.0
    for r in rows:
        usage = _row_value(r, "usage", {})
        tokens += float(_row_value(usage, "total_tokens", 0) if not isinstance(usage, Mapping) else usage.get("total_tokens", 0) or 0)
    safety = sum(_flagged(r, "safety") for r in rows)
    pollution = sum(_flagged(r, "pollution") for r in rows)
    rate = passed / n
    return {"task_success_rate": rate, "success_lower_bound": _wilson_lower(passed, n), "tool_error_rate": errors / n, "tool_call_success_rate": 1.0 - errors / max(1.0, calls), "retry_count": retries / n, "total_tokens": tokens, "average_total_tokens": tokens / n, "safety_violations": float(safety), "pollution_violations": float(pollution)}


def _flagged(row: Any, kind: str) -> int:
    cls = str(_row_value(row, "failure_class", "") or "").lower()
    response = _row_value(row, "response", {})
    flags = response if isinstance(response, Mapping) else {}
    keys = {"safety": ("safety_violation", "security_violation", "sandbox_violation", "policy_violation"), "pollution": ("evaluation_pollution", "benchmark_pollution", "eval_contamination", "heldout_contamination")}[kind]
    return int(any(x in cls for x in keys) or any(bool(flags.get(x)) for x in keys))


def _wilson_lower(successes: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return 0.0
    # z values used by the supported default confidence; approximation keeps
    # this module dependency-free while remaining conservative for small n.
    z = 1.96 if confidence >= 0.95 else 1.645
    p = successes / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, (centre - spread) / denom)


def compare_paired(
    baseline_results: Sequence[Any],
    candidate_results: Sequence[Any],
    direction: DirectionSpec | None = None,
    *,
    candidate_id: str | None = None,
) -> Decision:
    """Compare paired dev rows and apply objective plus hard constraints."""
    spec = direction or DirectionSpec()
    before = _metric_rows(baseline_results)
    after = _metric_rows(candidate_results)
    deltas = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in after}
    violations: list[str] = []
    if not _paired(baseline_results, candidate_results):
        violations.append("baseline and candidate results are not paired")
    if after["success_lower_bound"] < before["success_lower_bound"] - spec.max_success_drop:
        violations.append("success lower bound below allowed floor")
    if spec.success_floor is not None and after["success_lower_bound"] < spec.success_floor:
        violations.append("success lower bound below configured floor")
    if spec.require_safety_zero and after["safety_violations"] > 0:
        violations.append("safety violations must be zero")
    if spec.require_pollution_zero and after["pollution_violations"] > 0:
        violations.append("evaluation pollution must be zero")
    if spec.max_tool_error_rate is not None and after["tool_error_rate"] > spec.max_tool_error_rate:
        violations.append("tool error rate exceeds limit")
    token_budget = spec.max_total_tokens if spec.max_total_tokens is not None else spec.budget
    if token_budget is not None and after["total_tokens"] > token_budget:
        violations.append("token budget exceeded")
    if spec.max_average_total_tokens is not None and after["average_total_tokens"] > spec.max_average_total_tokens:
        violations.append("average token budget exceeded")
    primary = spec.primary
    old, new = before.get(primary, 0.0), after.get(primary, 0.0)
    gain = (new - old) if spec.maximize else (old - new)
    if not violations and gain < spec.min_primary_improvement:
        violations.append(f"primary objective did not improve: {primary}")
    accepted = not violations
    return Decision(accepted, candidate_id, "accepted" if accepted else "; ".join(violations), before, after, tuple(violations), deltas)


def _paired(before: Sequence[Any], after: Sequence[Any]) -> bool:
    """Require equal task/seed identities when those fields are available."""
    if isinstance(before, Mapping) or isinstance(after, Mapping):
        return isinstance(before, Mapping) and isinstance(after, Mapping)
    if len(before) != len(after):
        return False
    keys_before = [(_row_value(r, "task_id", None), _row_value(r, "seed", None)) for r in before]
    keys_after = [(_row_value(r, "task_id", None), _row_value(r, "seed", None)) for r in after]
    # Plain metric mappings may not have identities; callers can still compare
    # them positionally in that case.
    if all(k == (None, None) for k in keys_before + keys_after):
        return True
    return keys_before == keys_after


# Friendly aliases for callers that prefer explicit controller vocabulary.
generate_mutations = generate_candidates
evaluate_candidate = compare_paired
