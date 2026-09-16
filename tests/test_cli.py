"""Tests for the offline baseline harness and shared runtime/evaluator loop."""
import json
from pathlib import Path

from ouroboros.cli import run_adaptation, run_baseline, run_probes, run_evolution
from ouroboros.ledger import Ledger
from ouroboros.providers import DeterministicTestProvider
from ouroboros.runtime import AgentRuntime


def test_runtime_closure_with_test_provider():
    runtime = AgentRuntime(DeterministicTestProvider(default_content="ok"))
    result = runtime.run("return a result")
    assert result.success is True
    assert result.content == "ok"
    assert result.usage.total_tokens > 0


def test_baseline_writes_jsonl_and_manifest(tmp_path: Path):
    manifest = run_baseline(output_dir=tmp_path)
    assert manifest["model_id"] == "deterministic-test"
    results_path = tmp_path / manifest["results_file"]
    assert results_path.exists()
    assert (tmp_path / "run_manifest.json").exists()
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == manifest["metrics"]["task_count"]
    assert manifest["metrics"]["task_success_rate"] == 1.0


def test_baseline_accepts_jsonl_tasks(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"id":"custom-1","input":"x","expected":"yes"}\n', encoding="utf-8")
    manifest = run_baseline(output_dir=tmp_path / "out", tasks_path=tasks)
    assert manifest["metrics"]["task_count"] == 1
    row = json.loads((tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8"))
    assert row["task_id"] == "custom-1"


def test_probe_writes_fingerprint_and_all_probes_pass(tmp_path: Path):
    manifest = run_probes(output_dir=tmp_path / "probe")
    assert manifest["run_type"] == "probe"
    fingerprint = json.loads((tmp_path / "probe" / "fingerprint.json").read_text(encoding="utf-8"))
    assert fingerprint["task_count"] == 6
    assert fingerprint["metrics"]["task_success_rate"] == 1.0
    assert fingerprint["metrics"]["average_tokens"] > 0


def test_evolution_writes_decisions_and_version_ledger(tmp_path: Path):
    manifest = run_evolution(output_dir=tmp_path / "evolve", seeds=(0, 1))
    assert manifest["run_type"] == "evolve"
    assert manifest["candidate_count"] == 1
    assert manifest["accepted_candidates"] == ["mutation-1-schema"]
    assert (tmp_path / "evolve" / "decisions.json").exists()
    with Ledger(tmp_path / "evolve" / "ledger.db") as ledger:
        versions = ledger.list_versions(model_id="deterministic-test")
        assert len(versions) == 2
        assert ledger.query_version("mutation-1-schema")["attribution"]["edit_ref"] == "mutation-1-schema"


def test_adaptation_runs_paired_gate_without_persisting_keys(tmp_path: Path):
    manifest = run_adaptation(output_dir=tmp_path / "adapt", failure_classes=("invalid_arguments",),
                              max_candidates=1, seeds=(0,))
    assert manifest["run_type"] == "adapt"
    assert manifest["candidate_count"] == 1
    assert manifest["accepted_candidates"] == []
    text = (tmp_path / "adapt" / "run_manifest.json").read_text(encoding="utf-8")
    assert "api_key" not in text.lower() or "api_key_env_only" in text
    assert (tmp_path / "adapt" / "baseline_fingerprint.json").exists()
    assert (tmp_path / "adapt" / "decisions.json").exists()
