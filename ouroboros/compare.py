"""Paired parent/candidate evaluation and practical-effect statistics.

The evolution controller should compare two genomes on exactly the same
``(task_id, seed)`` observations.  This module keeps that comparison small and
dependency-free: it aligns rows, exposes per-task deltas, and computes
deterministic paired-bootstrap intervals for the most useful metrics.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from .eval import TaskResult


@dataclass(frozen=True)
class TaskDifference:
    """The observable difference for one task/seed pair (candidate - parent)."""

    task_id: str
    seed: int
    capability: str
    parent_passed: bool
    candidate_passed: bool
    pass_delta: int
    parent_total_tokens: int
    candidate_total_tokens: int
    total_token_delta: int
    parent_input_tokens: int
    candidate_input_tokens: int
    input_token_delta: int
    parent_output_tokens: int
    candidate_output_tokens: int
    output_token_delta: int
    parent_reasoning_tokens: int
    candidate_reasoning_tokens: int
    reasoning_token_delta: int
    parent_tool_error_count: int
    candidate_tool_error_count: int
    tool_error_delta: int
    parent_retry_count: int
    candidate_retry_count: int
    retry_delta: int
    parent_latency_ms: float
    candidate_latency_ms: float
    latency_delta_ms: float
    parent_failure_class: str | None
    candidate_failure_class: str | None

    @property
    def outcome(self) -> str:
        if self.pass_delta > 0:
            return "improved"
        if self.pass_delta < 0:
            return "regressed"
        return "unchanged"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["outcome"] = self.outcome
        return value


@dataclass(frozen=True)
class EffectJudgment:
    """Whether a metric clears a pre-registered minimum practical effect."""

    metric: str
    observed: float
    ci_low: float
    ci_high: float
    threshold: float
    direction: str
    meets: bool
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonReport:
    """Aggregate and paired statistics for one parent/candidate comparison."""

    pairs: tuple[TaskDifference, ...]
    confidence: float
    bootstrap_samples: int
    success_rate_delta: float
    success_rate_ci: tuple[float, float]
    average_total_token_delta: float
    average_total_token_delta_ci: tuple[float, float]
    token_savings_rate: float
    token_savings_rate_ci: tuple[float, float]
    tool_error_delta: float
    retry_delta: float
    latency_delta_ms: float
    success_effect: EffectJudgment
    token_savings_effect: EffectJudgment

    @property
    def count(self) -> int:
        return len(self.pairs)

    @property
    def parent_success_rate(self) -> float:
        return sum(p.parent_passed for p in self.pairs) / self.count if self.pairs else 0.0

    @property
    def candidate_success_rate(self) -> float:
        return sum(p.candidate_passed for p in self.pairs) / self.count if self.pairs else 0.0

    @property
    def parent_average_total_tokens(self) -> float:
        return _mean([p.parent_total_tokens for p in self.pairs])

    @property
    def candidate_average_total_tokens(self) -> float:
        return _mean([p.candidate_total_tokens for p in self.pairs])

    @property
    def minimum_practical_effect_met(self) -> bool:
        return self.success_effect.meets and self.token_savings_effect.meets

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "confidence": self.confidence,
            "bootstrap_samples": self.bootstrap_samples,
            "parent_success_rate": self.parent_success_rate,
            "candidate_success_rate": self.candidate_success_rate,
            "success_rate_delta": self.success_rate_delta,
            "success_rate_ci": list(self.success_rate_ci),
            "parent_average_total_tokens": self.parent_average_total_tokens,
            "candidate_average_total_tokens": self.candidate_average_total_tokens,
            "average_total_token_delta": self.average_total_token_delta,
            "average_total_token_delta_ci": list(self.average_total_token_delta_ci),
            "token_savings_rate": self.token_savings_rate,
            "token_savings_rate_ci": list(self.token_savings_rate_ci),
            "tool_error_delta": self.tool_error_delta,
            "retry_delta": self.retry_delta,
            "latency_delta_ms": self.latency_delta_ms,
            "minimum_practical_effect_met": self.minimum_practical_effect_met,
            "success_effect": self.success_effect.to_dict(),
            "token_savings_effect": self.token_savings_effect.to_dict(),
            "pairs": [p.to_dict() for p in self.pairs],
        }


def pair_results(
    parent: Iterable[TaskResult], candidate: Iterable[TaskResult], *, strict: bool = True
) -> tuple[TaskDifference, ...]:
    """Align result rows by ``(task_id, seed)`` and calculate differences.

    Strict mode rejects duplicate or missing keys.  Non-strict mode compares
    the intersection, which is useful when a runner timed out before emitting
    a row, but the omitted keys remain visible to callers through their counts.
    """
    parent_rows = list(parent)
    candidate_rows = list(candidate)
    pmap = _index(parent_rows, "parent", strict)
    cmap = _index(candidate_rows, "candidate", strict)
    pkeys, ckeys = set(pmap), set(cmap)
    if strict and pkeys != ckeys:
        missing_parent = sorted(ckeys - pkeys)
        missing_candidate = sorted(pkeys - ckeys)
        raise ValueError(f"unpaired results: missing_parent={missing_parent}, missing_candidate={missing_candidate}")
    keys = sorted(pkeys & ckeys, key=lambda value: (str(value[0]), value[1]))
    return tuple(_difference(pmap[key], cmap[key]) for key in keys)


def compare_results(
    parent: Iterable[TaskResult],
    candidate: Iterable[TaskResult],
    *,
    max_success_drop: float = 0.03,
    min_token_savings: float = 0.0,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    strict: bool = True,
) -> ComparisonReport:
    """Return paired deltas, confidence intervals, and effect judgments.

    ``min_token_savings`` is a fraction of parent tokens (``0.05`` means 5%).
    Success is considered acceptable when the lower confidence bound is no
    worse than ``-max_success_drop``.  Token savings are positive when the
    candidate uses fewer tokens.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    pairs = pair_results(parent, candidate, strict=strict)
    success_values = [float(p.pass_delta) for p in pairs]
    token_values = [float(p.total_token_delta) for p in pairs]
    savings_values = [_savings_fraction(p) for p in pairs]
    success_delta = _mean(success_values)
    token_delta = _mean(token_values)
    savings = _mean(savings_values)
    success_ci = _bootstrap_ci(success_values, confidence, bootstrap_samples, seed)
    token_ci = _bootstrap_ci(token_values, confidence, bootstrap_samples, seed + 1)
    savings_ci = _bootstrap_ci(savings_values, confidence, bootstrap_samples, seed + 2)
    success_effect = EffectJudgment(
        "success_rate_delta", success_delta, success_ci[0], success_ci[1], -max_success_drop,
        "lower_bound_at_least", success_ci[0] >= -max_success_drop,
        f"lower CI {success_ci[0]:.4f} >= allowed floor {-max_success_drop:.4f}",
    )
    token_effect = EffectJudgment(
        "token_savings_rate", savings, savings_ci[0], savings_ci[1], min_token_savings,
        "lower_bound_at_least", savings_ci[0] >= min_token_savings,
        f"lower CI {savings_ci[0]:.4f} >= required savings {min_token_savings:.4f}",
    )
    return ComparisonReport(
        pairs=tuple(pairs), confidence=confidence, bootstrap_samples=bootstrap_samples,
        success_rate_delta=success_delta, success_rate_ci=success_ci,
        average_total_token_delta=token_delta, average_total_token_delta_ci=token_ci,
        token_savings_rate=savings, token_savings_rate_ci=savings_ci,
        tool_error_delta=_mean([p.tool_error_delta for p in pairs]),
        retry_delta=_mean([p.retry_delta for p in pairs]),
        latency_delta_ms=_mean([p.latency_delta_ms for p in pairs]),
        success_effect=success_effect, token_savings_effect=token_effect,
    )


def _index(rows: Sequence[TaskResult], label: str, strict: bool) -> dict[tuple[str, int], TaskResult]:
    result: dict[tuple[str, int], TaskResult] = {}
    for row in rows:
        key = (str(row.task_id), int(row.seed))
        if key in result and strict:
            raise ValueError(f"duplicate {label} result for {key!r}")
        result.setdefault(key, row)
    return result


def _difference(parent: TaskResult, candidate: TaskResult) -> TaskDifference:
    pu, cu = parent.usage, candidate.usage
    return TaskDifference(
        task_id=parent.task_id, seed=parent.seed,
        capability=candidate.capability or parent.capability,
        parent_passed=parent.passed, candidate_passed=candidate.passed,
        pass_delta=int(candidate.passed) - int(parent.passed),
        parent_total_tokens=pu.total_tokens, candidate_total_tokens=cu.total_tokens,
        total_token_delta=cu.total_tokens - pu.total_tokens,
        parent_input_tokens=pu.input_tokens, candidate_input_tokens=cu.input_tokens,
        input_token_delta=cu.input_tokens - pu.input_tokens,
        parent_output_tokens=pu.output_tokens, candidate_output_tokens=cu.output_tokens,
        output_token_delta=cu.output_tokens - pu.output_tokens,
        parent_reasoning_tokens=pu.reasoning_tokens, candidate_reasoning_tokens=cu.reasoning_tokens,
        reasoning_token_delta=cu.reasoning_tokens - pu.reasoning_tokens,
        parent_tool_error_count=parent.tool_error_count, candidate_tool_error_count=candidate.tool_error_count,
        tool_error_delta=candidate.tool_error_count - parent.tool_error_count,
        parent_retry_count=parent.retry_count, candidate_retry_count=candidate.retry_count,
        retry_delta=candidate.retry_count - parent.retry_count,
        parent_latency_ms=parent.latency_ms, candidate_latency_ms=candidate.latency_ms,
        latency_delta_ms=candidate.latency_ms - parent.latency_ms,
        parent_failure_class=parent.failure_class, candidate_failure_class=candidate.failure_class,
    )


def _savings_fraction(pair: TaskDifference) -> float:
    if pair.parent_total_tokens > 0:
        return (pair.parent_total_tokens - pair.candidate_total_tokens) / pair.parent_total_tokens
    return 0.0 if pair.candidate_total_tokens == 0 else -1.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_ci(values: Sequence[float], confidence: float, samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if samples <= 0 or len(values) == 1:
        value = _mean(values)
        return (value, value)
    rng = random.Random(seed)
    n = len(values)
    estimates = [_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return (_quantile(estimates, alpha), _quantile(estimates, 1.0 - alpha))


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower, upper = int(position), min(len(ordered) - 1, int(position) + 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


# Descriptive aliases keep the API discoverable for callers that use the
# terminology from the design document ("paired evaluation").
paired_compare = compare_results
compare_paired_results = compare_results

__all__ = [
    "TaskDifference", "EffectJudgment", "ComparisonReport", "pair_results",
    "compare_results", "paired_compare", "compare_paired_results",
]
