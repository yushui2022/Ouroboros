"""Conditional Model Fingerprint calculation from evaluation traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .eval import TaskResult, stable_hash, summarize
from .profile import RuntimeProfile


@dataclass(frozen=True)
class FingerprintCondition:
    model_id: str
    provider: str
    model_version: str
    api_format: str = "unknown"
    dialect_hash: str = "sha256:unknown"
    decode: Mapping[str, Any] = field(default_factory=dict)
    tool_environment_hash: str = "sha256:unknown"
    probe_manifest_hash: str = "sha256:unknown"
    runtime_profile_id: str = "runtime:unknown"
    runtime_id: str = "unknown"
    runtime_version: str = "unknown"


@dataclass(frozen=True)
class ModelFingerprint:
    condition: FingerprintCondition
    metrics: Mapping[str, float]
    failure_counts: Mapping[str, int]
    task_count: int
    trace_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.condition.model_id,
            "provider": self.condition.provider,
            "model_version": self.condition.model_version,
            "api_format": self.condition.api_format,
            "dialect_hash": self.condition.dialect_hash,
            "decode": dict(self.condition.decode),
            "tool_environment_hash": self.condition.tool_environment_hash,
            "probe_manifest_hash": self.condition.probe_manifest_hash,
            "runtime_profile_id": self.condition.runtime_profile_id,
            "runtime_id": self.condition.runtime_id,
            "runtime_version": self.condition.runtime_version,
            "metrics": dict(self.metrics),
            "failure_counts": dict(self.failure_counts),
            "task_count": self.task_count,
            "trace_hash": self.trace_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)


def build_fingerprint(
    results: Iterable[TaskResult],
    *,
    model_id: str,
    provider: str,
    model_version: str,
    api_format: str = "unknown",
    dialect: Any = None,
    decode: Mapping[str, Any] | None = None,
    tool_environment_hash: str = "sha256:unknown",
    probe_manifest_hash: str = "sha256:unknown",
    runtime_profile: RuntimeProfile | None = None,
) -> ModelFingerprint:
    rows = list(results)
    metrics = summarize(rows)
    # Capability-specific rates make the fingerprint useful to a mutation
    # controller while retaining the generic aggregate metrics from evaluate.
    def capability_rate(name: str) -> float:
        selected = [row for row in rows if row.capability == name]
        return sum(row.passed for row in selected) / len(selected) if selected else 0.0

    metrics = dict(metrics)
    metrics.update({
        "tool_call_success_rate": metrics.get("tool_call_success_rate", 0.0),
        "schema_following_rate": capability_rate("schema_following"),
        "argument_error_rate": sum(row.failure_class == "invalid_arguments" for row in rows) / len(rows) if rows else 0.0,
        "retry_recovery_rate": capability_rate("retry_recovery"),
        "context_retention": capability_rate("context_retention"),
        "finish_detection": capability_rate("finish_detection"),
        "average_tokens": metrics.get("average_total_tokens", 0.0),
    })
    failure_counts: dict[str, int] = {}
    for row in rows:
        if not row.passed:
            key = row.failure_class or "unknown"
            failure_counts[key] = failure_counts.get(key, 0) + 1
    condition = FingerprintCondition(
        model_id=model_id,
        provider=provider,
        model_version=model_version,
        api_format=api_format,
        dialect_hash=stable_hash(dialect) if dialect is not None else "sha256:unknown",
        decode=dict(decode or {}),
        tool_environment_hash=tool_environment_hash,
        probe_manifest_hash=probe_manifest_hash,
        runtime_profile_id=runtime_profile.profile_id if runtime_profile else "runtime:unknown",
        runtime_id=runtime_profile.runtime_id if runtime_profile else "unknown",
        runtime_version=runtime_profile.runtime_version if runtime_profile else "unknown",
    )
    trace_payload = [
        {"task_id": r.task_id, "seed": r.seed, "passed": r.passed, "failure_class": r.failure_class,
         "tool_call_count": r.tool_call_count, "tool_error_count": r.tool_error_count,
         "retry_count": r.retry_count, "usage": asdict(r.usage)}
        for r in rows
    ]
    return ModelFingerprint(condition, metrics, failure_counts, len(rows), stable_hash(trace_payload))


def compare_fingerprints(before: ModelFingerprint, after: ModelFingerprint) -> dict[str, float]:
    """Return metric deltas (after - before) for paired fingerprint reports."""
    keys = set(before.metrics) | set(after.metrics)
    return {key: float(after.metrics.get(key, 0.0) - before.metrics.get(key, 0.0)) for key in sorted(keys)}


def recalibration_required(before: ModelFingerprint, after: ModelFingerprint) -> bool:
    """Whether two reports cannot be treated as the same dialect experiment.

    Runtime replacement, model/provider changes, or probe/config changes all
    invalidate direct Genome reuse.  Metric deltas remain inspectable, but a
    controller must start a fresh calibration branch.
    """

    left, right = before.condition, after.condition
    return any((
        left.model_id != right.model_id,
        left.provider != right.provider,
        left.model_version != right.model_version,
        left.api_format != right.api_format,
        left.dialect_hash != right.dialect_hash,
        left.decode != right.decode,
        left.tool_environment_hash != right.tool_environment_hash,
        left.probe_manifest_hash != right.probe_manifest_hash,
        left.runtime_profile_id != right.runtime_profile_id,
    ))


__all__ = [
    "FingerprintCondition", "ModelFingerprint", "build_fingerprint",
    "compare_fingerprints", "recalibration_required",
]
