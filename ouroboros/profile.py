"""Runtime identity and model-specific Genome branch helpers.

Model behaviour is conditional on the Agent Runtime as well as the provider
and model.  A RuntimeProfile gives that condition a stable, serialisable
identity so replacing the mother runtime cannot silently reuse an old Genome.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .types import AgentGenome


@dataclass(frozen=True)
class RuntimeProfile:
    """Versioned contract for one Agent Runtime/adapter implementation."""

    runtime_id: str = "reference-runtime"
    runtime_version: str = "0.1"
    adapter_id: str = "native"
    api_format: str = "openai-compatible"
    tool_schema_version: str = "1"
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def profile_id(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "runtime:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["profile_id"] = self.profile_id
        return value


def branch_key(model_id: str, profile: RuntimeProfile | str) -> str:
    """Return the only safe Genome branch key for a model/runtime pair."""

    profile_id = profile.profile_id if isinstance(profile, RuntimeProfile) else str(profile)
    return f"{model_id}@{profile_id}"


def bind_genome(genome: AgentGenome, *, model_id: str, profile: RuntimeProfile) -> AgentGenome:
    """Bind a Genome to a model/runtime pair, preserving all dialect knobs."""

    return replace(
        genome,
        model_id=model_id,
        runtime_profile_id=profile.profile_id,
        branch=branch_key(model_id, profile),
    )


__all__ = ["RuntimeProfile", "branch_key", "bind_genome"]
