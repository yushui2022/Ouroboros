"""SQLite backed version graph and capability ledger.

The ledger is intentionally append-only at the public API boundary.  A version
node captures the immutable Agent Genome identity and its evaluation metadata;
subsequent runs, metrics, capability observations and debt records are linked
to that node without rewriting its history.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Version:
    version_id: str
    parent_id: str | None
    model_id: str
    branch: str | None
    created_at: str
    layer: str | None
    direction: Mapping[str, Any]
    edit: Mapping[str, Any]
    environment_hash: str | None
    eval_manifest_hash: str | None
    runtime_profile_id: str | None = None


class Ledger:
    """Append-only SQLite store for versions, evaluations and capability debt."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS versions (
              version_id TEXT PRIMARY KEY, parent_id TEXT REFERENCES versions(version_id),
              model_id TEXT NOT NULL, branch TEXT, created_at TEXT NOT NULL,
              layer TEXT, direction_json TEXT NOT NULL, edit_json TEXT NOT NULL,
              environment_hash TEXT, eval_manifest_hash TEXT, runtime_profile_id TEXT
            );
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(version_id),
              split TEXT NOT NULL, seed INTEGER NOT NULL, evaluator_version TEXT,
              manifest_hash TEXT, started_at TEXT NOT NULL, status TEXT NOT NULL,
              metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metrics (
              run_id TEXT NOT NULL REFERENCES runs(run_id), name TEXT NOT NULL,
              value_json TEXT NOT NULL, PRIMARY KEY(run_id, name)
            );
            CREATE TABLE IF NOT EXISTS task_results (
              run_id TEXT NOT NULL REFERENCES runs(run_id), task_id TEXT NOT NULL,
              capability TEXT, passed INTEGER NOT NULL, failure_class TEXT, reason TEXT,
              result_json TEXT NOT NULL, PRIMARY KEY(run_id, task_id)
            );
            CREATE TABLE IF NOT EXISTS capability_deltas (
              version_id TEXT PRIMARY KEY REFERENCES versions(version_id),
              gained_json TEXT NOT NULL, lost_json TEXT NOT NULL, changed_json TEXT NOT NULL,
              confidence REAL, evidence_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS debt (
              debt_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(version_id),
              capability_id TEXT, status TEXT NOT NULL, opened_at TEXT NOT NULL,
              repaid_at TEXT, reason TEXT, evidence_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attribution (
              version_id TEXT PRIMARY KEY REFERENCES versions(version_id),
              evidence TEXT, confidence REAL, edit_ref TEXT, details_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_versions_model ON versions(model_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_runs_version ON runs(version_id);
            CREATE INDEX IF NOT EXISTS idx_debt_version ON debt(version_id);
            """
        )
        # Lightweight migration for ledgers created before RuntimeProfile was
        # introduced.  SQLite has no portable IF NOT EXISTS for columns.
        columns = {row["name"] for row in self._db.execute("PRAGMA table_info(versions)")}
        if "runtime_profile_id" not in columns:
            self._db.execute("ALTER TABLE versions ADD COLUMN runtime_profile_id TEXT")
        self._db.commit()

    def add_version(
        self, version_id: str, *, model_id: str, parent_id: str | None = None,
        branch: str | None = None, layer: str | None = None,
        direction: Mapping[str, Any] | None = None, edit: Mapping[str, Any] | None = None,
        environment_hash: str | None = None, eval_manifest_hash: str | None = None,
        runtime_profile_id: str | None = None,
        created_at: str | None = None,
    ) -> Version:
        """Insert one immutable version node; reject conflicting duplicate IDs."""
        if parent_id:
            parent = self.get_version(parent_id)
            if parent is None:
                raise ValueError(f"parent version does not exist: {parent_id}")
            if runtime_profile_id != parent.runtime_profile_id and (
                runtime_profile_id is not None or parent.runtime_profile_id is not None
            ):
                # A Runtime replacement starts a fresh branch; it must never
                # masquerade as a child in the old runtime's branch.
                raise ValueError("runtime profile changed; create a new root branch")
        row = Version(version_id=version_id, parent_id=parent_id, model_id=model_id,
                      branch=branch, created_at=created_at or _now(), layer=layer,
                      direction=direction or {}, edit=edit or {},
                      environment_hash=environment_hash, eval_manifest_hash=eval_manifest_hash,
                      runtime_profile_id=runtime_profile_id)
        try:
            self._db.execute(
                "INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (row.version_id, row.parent_id, row.model_id, row.branch, row.created_at,
                 row.layer, _json(row.direction), _json(row.edit), row.environment_hash,
                 row.eval_manifest_hash, row.runtime_profile_id),
            )
            self._db.commit()
        except sqlite3.IntegrityError as exc:
            existing = self.get_version(version_id)
            # Replaying the same logical version without an explicit
            # timestamp is idempotent; creation time is ledger-owned metadata.
            same_logical = existing is not None and existing == row
            if existing is not None and created_at is None:
                same_logical = all(getattr(existing, field) == getattr(row, field)
                                    for field in ("version_id", "parent_id", "model_id", "branch",
                                                  "layer", "direction", "edit", "environment_hash",
                                                  "eval_manifest_hash", "runtime_profile_id"))
            if not same_logical:
                raise ValueError(f"version_id already exists with different content: {version_id}") from exc
        return row

    def get_version(self, version_id: str) -> Version | None:
        row = self._db.execute("SELECT * FROM versions WHERE version_id=?", (version_id,)).fetchone()
        if not row:
            return None
        return Version(version_id=row["version_id"], parent_id=row["parent_id"], model_id=row["model_id"],
                       branch=row["branch"], created_at=row["created_at"], layer=row["layer"],
                       direction=_loads(row["direction_json"], {}), edit=_loads(row["edit_json"], {}),
                       environment_hash=row["environment_hash"], eval_manifest_hash=row["eval_manifest_hash"],
                       runtime_profile_id=row["runtime_profile_id"] if "runtime_profile_id" in row.keys() else None)

    def list_versions(self, *, model_id: str | None = None, branch: str | None = None) -> list[Version]:
        query, params = "SELECT * FROM versions", []
        clauses = []
        if model_id is not None: clauses.append("model_id=?"); params.append(model_id)
        if branch is not None: clauses.append("branch=?"); params.append(branch)
        if clauses: query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, version_id"
        return [self.get_version(row["version_id"]) for row in self._db.execute(query, params)]  # type: ignore[misc]

    def record_run(self, run_id: str, version_id: str, *, split: str = "dev", seed: int = 0,
                   evaluator_version: str | None = None, manifest_hash: str | None = None,
                   started_at: str | None = None, status: str = "completed",
                   metadata: Mapping[str, Any] | None = None) -> None:
        self._db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                         (run_id, version_id, split, seed, evaluator_version, manifest_hash,
                          started_at or _now(), status, _json(metadata or {})))
        self._db.commit()

    def record_metrics(self, run_id: str, metrics: Mapping[str, Any]) -> None:
        self._db.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?)",
                             [(run_id, str(k), _json(v)) for k, v in metrics.items()])
        self._db.commit()

    def record_task_result(self, run_id: str, *, task_id: str, capability: str | None,
                           passed: bool, failure_class: str | None = None,
                           reason: str | None = None, result: Mapping[str, Any] | None = None) -> None:
        self._db.execute("INSERT OR REPLACE INTO task_results VALUES (?,?,?,?,?,?,?)",
                         (run_id, task_id, capability, int(passed), failure_class, reason, _json(result or {})))
        self._db.commit()

    def record_capability_delta(self, version_id: str, *, gained: Iterable[str] = (),
                                lost: Iterable[str] = (), changed: Iterable[str] = (),
                                confidence: float | None = None, evidence: Mapping[str, Any] | None = None) -> None:
        self._db.execute("INSERT OR REPLACE INTO capability_deltas VALUES (?,?,?,?,?,?)",
                         (version_id, _json(list(gained)), _json(list(lost)), _json(list(changed)), confidence, _json(evidence or {})))
        self._db.commit()

    def record_debt(self, debt_id: str, version_id: str, *, capability_id: str | None = None,
                    status: str = "potential_debt", reason: str | None = None,
                    evidence: Mapping[str, Any] | None = None, repaid_at: str | None = None) -> None:
        self._db.execute("INSERT INTO debt VALUES (?,?,?,?,?,?,?,?)",
                         (debt_id, version_id, capability_id, status, _now(), repaid_at, reason, _json(evidence or {})))
        self._db.commit()

    def record_attribution(self, version_id: str, *, evidence: str | None = None,
                           confidence: float | None = None, edit_ref: str | None = None,
                           details: Mapping[str, Any] | None = None) -> None:
        self._db.execute("INSERT OR REPLACE INTO attribution VALUES (?,?,?,?,?)",
                         (version_id, evidence, confidence, edit_ref, _json(details or {})))
        self._db.commit()

    def query_version(self, version_id: str) -> dict[str, Any] | None:
        version = self.get_version(version_id)
        if version is None: return None
        runs = []
        for run in self._db.execute("SELECT * FROM runs WHERE version_id=? ORDER BY started_at", (version_id,)):
            metrics = {r["name"]: _loads(r["value_json"]) for r in self._db.execute("SELECT * FROM metrics WHERE run_id=?", (run["run_id"],))}
            tasks = [dict(r, passed=bool(r["passed"]), result=_loads(r["result_json"], {})) for r in self._db.execute("SELECT * FROM task_results WHERE run_id=?", (run["run_id"],))]
            runs.append({"run_id": run["run_id"], "split": run["split"], "seed": run["seed"], "status": run["status"], "metrics": metrics, "task_results": tasks})
        delta = self._db.execute("SELECT * FROM capability_deltas WHERE version_id=?", (version_id,)).fetchone()
        attr = self._db.execute("SELECT * FROM attribution WHERE version_id=?", (version_id,)).fetchone()
        debts = [dict(r, evidence=_loads(r["evidence_json"], {})) for r in self._db.execute("SELECT * FROM debt WHERE version_id=?", (version_id,))]
        return {"version": asdict(version), "runs": runs,
                "capability_delta": None if not delta else {"gained": _loads(delta["gained_json"], []), "lost": _loads(delta["lost_json"], []), "changed": _loads(delta["changed_json"], []), "confidence": delta["confidence"], "evidence": _loads(delta["evidence_json"], {})},
                "attribution": None if not attr else {"evidence": attr["evidence"], "confidence": attr["confidence"], "edit_ref": attr["edit_ref"], "details": _loads(attr["details_json"], {})},
                "debt": debts}

    def graph(self, *, model_id: str | None = None) -> list[dict[str, str | None]]:
        versions = self.list_versions(model_id=model_id)
        return [{"version_id": v.version_id, "parent_id": v.parent_id, "model_id": v.model_id, "branch": v.branch} for v in versions]

    def query_runs(self, version_id: str | None = None) -> list[dict[str, Any]]:
        """Return flattened run records, optionally restricted to a version."""
        query, params = "SELECT * FROM runs", []
        if version_id is not None:
            query += " WHERE version_id=?"
            params.append(version_id)
        query += " ORDER BY started_at, run_id"
        rows: list[dict[str, Any]] = []
        for row in self._db.execute(query, params):
            metrics = {r["name"]: _loads(r["value_json"]) for r in self._db.execute("SELECT * FROM metrics WHERE run_id=?", (row["run_id"],))}
            rows.append({"run_id": row["run_id"], "version_id": row["version_id"], "split": row["split"], "seed": row["seed"], "evaluator_version": row["evaluator_version"], "manifest_hash": row["manifest_hash"], "started_at": row["started_at"], "status": row["status"], "metadata": _loads(row["metadata_json"], {}), "metrics": metrics})
        return rows

    def query_debt(self, *, version_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM debt", []
        clauses = []
        if version_id is not None: clauses.append("version_id=?"); params.append(version_id)
        if status is not None: clauses.append("status=?"); params.append(status)
        if clauses: query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY opened_at, debt_id"
        return [{**dict(row), "evidence": _loads(row["evidence_json"], {})} for row in self._db.execute(query, params)]
