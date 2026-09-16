"""Offline baseline CLI for the Ouroboros reference runtime.

The command intentionally uses the shared ``ouroboros.runtime`` and
``ouroboros.eval`` contracts so later providers and mutation controllers can
reuse the exact same execution and evaluation path.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .eval import Evaluator, JudgeResult, TaskSpec, summarize, stable_hash, TaskResult
from .fingerprint import build_fingerprint
from benchmarks.probes import default_probe_tasks
from .providers import AnthropicMessagesProvider, DeepSeekProvider, DeterministicTestProvider, OpenAICompatibleProvider
from .runtime import AgentRuntime
from .types import AgentGenome
from .profile import RuntimeProfile, bind_genome
from .mutation import DirectionSpec, compare_paired, generate_candidates
from .compare import compare_results
from .ledger import Ledger

DEFAULT_TASKS: tuple[dict[str, Any], ...] = (
    {"id": "task-001", "input": "2 + 2", "expected": "done", "capability": "basic_response"},
    {"id": "task-002", "input": "uppercase: ouroboros", "expected": "done", "capability": "basic_response"},
    {"id": "task-003", "input": "tool: status", "expected": "done", "capability": "basic_response"},
)


def load_tasks(path: Path | None) -> list[dict[str, Any]]:
    """Load JSON/JSONL tasks; each task requires ``id`` and ``expected``."""
    if path is None:
        return [dict(task) for task in DEFAULT_TASKS]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = payload.get("tasks", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(payload, list):
            raise ValueError("tasks JSON must be a list or an object containing 'tasks'")
        tasks = [dict(item) for item in payload]
    else:
        tasks = [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, task in enumerate(tasks, 1):
        if not task.get("id") or "expected" not in task:
            raise ValueError(f"task at index {index} requires id and expected")
        task.setdefault("input", task.get("prompt", ""))
        task.setdefault("capability", "basic_response")
    return tasks


def _task_spec(task: Mapping[str, Any]) -> TaskSpec:
    expected = str(task["expected"])

    def judge(response: Mapping[str, Any]) -> JudgeResult:
        actual = response.get("answer", response.get("content", ""))
        passed = str(actual) == expected
        return JudgeResult(passed, None if passed else "wrong_result", None if passed else f"expected {expected!r}")

    return TaskSpec(id=str(task["id"]), prompt=str(task.get("input", "")), capability=str(task.get("capability", "basic_response")), judge=judge)


def run_baseline(*, output_dir: Path, provider_name: str = "test", tasks_path: Path | None = None) -> dict[str, Any]:
    """Run deterministic tasks and write replayable JSONL results + manifest."""
    if provider_name != "test":
        raise ValueError("only provider 'test' is available in the offline baseline harness")
    raw_tasks = load_tasks(tasks_path)
    task_specs = tuple(_task_spec(task) for task in raw_tasks)
    runtime_profile = RuntimeProfile(runtime_id="reference-runtime", runtime_version="0.1",
                                     adapter_id="deterministic-provider", api_format="offline")
    genome = bind_genome(AgentGenome(), model_id="deterministic-test", profile=runtime_profile)

    def runner(task: TaskSpec, seed: int) -> Mapping[str, Any]:
        # A fresh scripted provider per task makes the expected answer explicit
        # while preserving the same runtime/evaluator path as real providers.
        expected = next(item["expected"] for item in raw_tasks if str(item["id"]) == task.id)
        provider = DeterministicTestProvider(model_id="deterministic-test", default_content=str(expected))
        result = AgentRuntime(provider, genome=genome, runtime_profile=runtime_profile).run(task.prompt)
        return {
            "answer": result.content,
            "content": result.content,
            "tool_calls": [],
            "usage": asdict(result.usage),
            "latency_ms": sum(event.timestamp_ms for event in result.trace[-1:]),
            "runtime_success": result.success,
        }

    evaluator = Evaluator(task_specs)
    results = evaluator.run(runner, seeds=(0,))
    metrics = summarize(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n")

    manifest = {
        "schema_version": "0.1",
        "run_id": f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider_name,
        "model_id": "deterministic-test",
        "model_version": "deterministic-test-v1",
        "api_format": "offline",
        "genome": asdict(genome),
        "genome_hash": stable_hash(asdict(genome)),
        "task_manifest_hash": evaluator.manifest_hash,
        "results_file": results_path.name,
        "metrics": metrics,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _probe_runner(task: TaskSpec, seed: int, *, provider_name: str = "test", model: str = "deepseek-flash",
                  settings: Mapping[str, Any] | None = None, timeout_seconds: float = 120.0,
                  runtime_profile: RuntimeProfile | None = None, base_url: str | None = None,
                  api_key: str | None = None, genome: AgentGenome | None = None) -> Mapping[str, Any]:
    """Run one deterministic probe through the reference AgentRuntime."""
    del seed  # probe reproducibility is provided by the scripted provider
    from .types import ToolSpec

    lookup_state = {"failed": False}

    def lookup(key: str) -> str:
        if task.id == "probe-retry-recovery" and not lookup_state["failed"]:
            lookup_state["failed"] = True
            raise RuntimeError("transient lookup failure")
        return "ready" if key == "status" else "unknown"

    def state(key: str) -> Any:
        return {"user": "alice", "mode": "fast"}.get(key, "unknown")

    tools = [
        ToolSpec("calculator", "Evaluate an expression", {"expression": {"type": "string"}}, handler=lambda expression: "4"),
        ToolSpec("lookup", "Look up a key", {"key": {"type": "string"}}, handler=lookup),
        ToolSpec("state", "Read preserved task state", {"key": {"type": "string"}}, handler=state),
    ]
    script: list[Any]
    if task.id == "probe-tool-json":
        script = [{"message": {"tool_calls": [{"id": "call-1", "function": {"name": "calculator", "arguments": '{"expression":"2 + 2"}'}}]}}, {"content": "done"}]
    elif task.id in {"probe-schema-required", "probe-retry-recovery"}:
        script = [{"message": {"tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": '{"key":"status"}'}}]}}, {"content": "done"}]
    elif task.id == "probe-result-answer":
        script = [{"message": {"tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": '{"key":"status"}'}}]}}, {"content": "ready"}]
    elif task.id == "probe-context-state":
        script = [{"message": {"tool_calls": [{"id": "call-1", "function": {"name": "state", "arguments": '{"key":"user"}'}}, {"id": "call-2", "function": {"name": "state", "arguments": '{"key":"mode"}'}}]}}, {"content": {"user": "alice", "mode": "fast"}}]
    else:
        script = [{"content": "done"}]

    if provider_name == "test":
        provider = DeterministicTestProvider(script=script, model_id="deterministic-probe")
    elif provider_name == "deepseek":
        provider = DeepSeekProvider(model=model, timeout_seconds=timeout_seconds)
    elif provider_name == "gateway":
        if not base_url:
            raise ValueError("--base-url is required for gateway provider")
        provider = OpenAICompatibleProvider(api_key=api_key, base_url=base_url, model=model,
                                            timeout_seconds=timeout_seconds)
    elif provider_name == "anthropic":
        if not base_url:
            raise ValueError("--base-url is required for anthropic provider")
        provider = AnthropicMessagesProvider(api_key=api_key, base_url=base_url, model=model,
                                             timeout_seconds=timeout_seconds)
    else:
        raise ValueError(f"unsupported probe provider: {provider_name}")
    result = AgentRuntime(provider, tools=tools, genome=genome, runtime_profile=runtime_profile).run(task.prompt, settings=dict(settings or {}))
    calls: list[dict[str, Any]] = []
    for message in result.messages:
        for call in message.tool_calls or []:
            arguments = call.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append({"id": call.id, "name": call.name, "arguments": arguments})
    return {
        "answer": result.content,
        "content": result.content,
        "tool_calls": calls,
        "usage": asdict(result.usage),
        "retry_count": sum(1 for row in result.tool_results if row.error),
        "tool_error_count": sum(1 for row in result.tool_results if row.error),
        "latency_ms": sum(event.timestamp_ms for event in result.trace[-1:]),
        "runtime_success": result.success,
        "error": result.error,
    }


def run_probes(*, output_dir: Path, provider_name: str = "test", model: str = "deepseek-flash",
               settings: Mapping[str, Any] | None = None, timeout_seconds: float = 120.0,
               runtime_profile: RuntimeProfile | None = None, base_url: str | None = None,
               api_key: str | None = None, probe_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Run the offline probe suite and persist results plus a fingerprint."""
    tasks = default_probe_tasks()
    if probe_ids is not None:
        selected = set(str(value) for value in probe_ids)
        tasks = tuple(task for task in tasks if task.id in selected)
        if not tasks:
            raise ValueError("--probe-id did not match any registered probe")
    runtime_profile = runtime_profile or RuntimeProfile(
        runtime_id="reference-runtime", runtime_version="0.1",
        adapter_id=("deepseek-provider" if provider_name == "deepseek" else (("anthropic-provider" if provider_name == "anthropic" else "gateway-provider") if provider_name in {"gateway", "anthropic"} else "deterministic-provider")),
        api_format=("offline" if provider_name == "test" else ("deepseek-openai-compatible" if provider_name == "deepseek" else ("anthropic-messages" if provider_name == "anthropic" else "openai-compatible"))),
    )
    evaluator = Evaluator(tasks, evaluator_version="probe-v1")
    def runner(task: TaskSpec, seed: int) -> Mapping[str, Any]:
        return _probe_runner(task, seed, provider_name=provider_name, model=model, settings=settings,
                             timeout_seconds=timeout_seconds, runtime_profile=runtime_profile,
                             base_url=base_url, api_key=api_key)
    results = evaluator.run(runner, seeds=(0,))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n")
    fingerprint = build_fingerprint(
        results,
        model_id=("deterministic-probe" if provider_name == "test" else model),
        provider=provider_name,
        model_version=("deterministic-probe-v1" if provider_name == "test" else model),
        api_format=("offline" if provider_name == "test" else ("deepseek-openai-compatible" if provider_name == "deepseek" else ("anthropic-messages" if provider_name == "anthropic" else "openai-compatible"))),
        probe_manifest_hash=evaluator.manifest_hash,
        runtime_profile=runtime_profile,
    )
    fingerprint_path = output_dir / "fingerprint.json"
    fingerprint_path.write_text(fingerprint.to_json() + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "0.1",
        "run_type": "probe",
        "model_id": ("deterministic-probe" if provider_name == "test" else model),
        "model_version": ("deterministic-probe-v1" if provider_name == "test" else model),
        "provider": provider_name,
        "runtime_profile": runtime_profile.to_dict(),
        "probe_manifest_hash": evaluator.manifest_hash,
        "results_file": results_path.name,
        "fingerprint_file": fingerprint_path.name,
        "metrics": summarize(results),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_adaptation(*, output_dir: Path, provider_name: str = "test", model: str = "deepseek-flash",
                   settings: Mapping[str, Any] | None = None, timeout_seconds: float = 120.0,
                   base_url: str | None = None, api_key: str | None = None,
                   probe_ids: Iterable[str] | None = None,
                   failure_classes: Iterable[str] | None = None, max_candidates: int = 1,
                   seeds: Iterable[int] = (0,)) -> dict[str, Any]:
    """Run one real-provider adaptation generation on the fixed probe suite.

    This command deliberately performs only offline candidate evaluation: the
    winning genome is recorded in the ledger but never silently activated by a
    running process. API keys are passed in memory and are excluded from all
    artifacts.
    """
    if max_candidates < 0:
        raise ValueError("--max-candidates must be non-negative")
    if provider_name in {"gateway", "anthropic"} and not base_url:
        raise ValueError(f"--base-url is required for {provider_name} provider")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    runtime_profile = RuntimeProfile(
        runtime_id="reference-runtime", runtime_version="0.1",
        adapter_id=("deepseek-provider" if provider_name == "deepseek" else
                    ("anthropic-provider" if provider_name == "anthropic" else
                     ("gateway-provider" if provider_name == "gateway" else "deterministic-provider"))),
        api_format=("offline" if provider_name == "test" else
                    ("deepseek-openai-compatible" if provider_name == "deepseek" else
                     ("anthropic-messages" if provider_name == "anthropic" else "openai-compatible"))),
    )
    model_id = "deterministic-probe" if provider_name == "test" else model
    genome = bind_genome(AgentGenome(), model_id=model_id, profile=runtime_profile)
    tasks = default_probe_tasks()
    if probe_ids is not None:
        selected = {str(value) for value in probe_ids}
        tasks = tuple(task for task in tasks if task.id in selected)
        if not tasks:
            raise ValueError("--probe-id did not match any registered probe")
    evaluator = Evaluator(tasks, evaluator_version="adapt-v1")

    def evaluate(candidate_genome: AgentGenome) -> list[TaskResult]:
        def runner(task: TaskSpec, seed: int) -> Mapping[str, Any]:
            return _probe_runner(task, seed, provider_name=provider_name, model=model,
                                 settings=settings, timeout_seconds=timeout_seconds,
                                 runtime_profile=runtime_profile, base_url=base_url,
                                 api_key=api_key, genome=candidate_genome)
        return evaluator.run(runner, seeds=seed_values)

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(genome)
    _write_results(output_dir / "baseline_results.jsonl", baseline)
    fingerprint = build_fingerprint(baseline, model_id=model_id, provider=provider_name,
                                     model_version=model, api_format=runtime_profile.api_format,
                                     probe_manifest_hash=evaluator.manifest_hash,
                                     runtime_profile=runtime_profile)
    (output_dir / "baseline_fingerprint.json").write_text(fingerprint.to_json() + "\n", encoding="utf-8")
    observed = tuple(dict.fromkeys(row.failure_class for row in baseline if row.failure_class))
    requested = tuple(str(value) for value in (failure_classes or ()) if str(value).strip())
    failures = observed or requested
    candidates = generate_candidates(genome, failures, limit=max_candidates,
                                     runtime_profile=runtime_profile, model_id=model_id)
    direction = DirectionSpec(primary="task_success_rate", max_success_drop=0.0,
                              min_primary_improvement=0.0001)
    parent_id = f"genome-{stable_hash(asdict(genome))[:12]}"
    ledger_path = output_dir / "ledger.db"
    decisions: list[dict[str, Any]] = []
    accepted: list[str] = []
    with Ledger(ledger_path) as ledger:
        ledger.add_version(parent_id, model_id=model_id, branch=genome.branch,
                           layer="shared-base", direction=asdict(direction), edit={"type": "calibration"},
                           eval_manifest_hash=evaluator.manifest_hash,
                           runtime_profile_id=runtime_profile.profile_id)
        for seed in seed_values:
            rows = [row for row in baseline if row.seed == seed]
            run_id = f"baseline-{seed}"
            ledger.record_run(run_id, parent_id, split="dev", seed=seed,
                              evaluator_version=evaluator.evaluator_version,
                              manifest_hash=evaluator.manifest_hash)
            ledger.record_metrics(run_id, summarize(rows))
            for row in rows:
                ledger.record_task_result(run_id, task_id=row.task_id, capability=row.capability,
                                          passed=row.passed, failure_class=row.failure_class,
                                          reason=row.reason, result=asdict(row))
        for candidate in candidates:
            candidate_rows = evaluate(candidate.genome)
            _write_results(output_dir / f"{candidate.candidate_id}_results.jsonl", candidate_rows)
            decision = compare_paired(baseline, candidate_rows, direction,
                                      candidate_id=candidate.candidate_id)
            report = compare_results(baseline, candidate_rows, bootstrap_samples=200)
            decisions.append({"candidate": asdict(candidate), "decision": asdict(decision),
                              "paired_report": report.to_dict()})
            if not decision.accepted:
                continue
            accepted.append(candidate.candidate_id)
            ledger.add_version(candidate.candidate_id, model_id=model_id, parent_id=parent_id,
                               branch=candidate.genome.branch, layer=candidate.layer,
                               direction=asdict(direction), edit=dict(candidate.patch),
                               eval_manifest_hash=evaluator.manifest_hash,
                               runtime_profile_id=runtime_profile.profile_id)
            ledger.record_attribution(candidate.candidate_id, evidence="paired_dev_evaluation",
                                      confidence=direction.confidence, edit_ref=candidate.edit_ref,
                                      details={"failure_classes": candidate.failure_classes,
                                               "decision": asdict(decision)})
    (output_dir / "decisions.json").write_text(json.dumps(decisions, ensure_ascii=False,
                                                            indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    manifest = {"schema_version": "0.1", "run_type": "adapt", "provider": provider_name,
                "model_id": model_id, "model_version": model, "runtime_profile": runtime_profile.to_dict(),
                "task_manifest_hash": evaluator.manifest_hash, "seed_count": len(seed_values),
                "baseline_metrics": summarize(baseline), "failure_classes": list(failures),
                "candidate_count": len(candidates), "accepted_candidates": accepted,
                "baseline_fingerprint_file": "baseline_fingerprint.json", "decisions_file": "decisions.json",
                "ledger_file": ledger_path.name, "api_key_env_only": True}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                              indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


# Short alias for programmatic callers and future orchestration code.
run_adapt = run_adaptation


def _write_results(path: Path, results: Iterable[TaskResult]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def run_evolution(*, output_dir: Path, tasks_path: Path | None = None,
                  seeds: Iterable[int] = (0, 1, 2),
                  failure_classes: Iterable[str] = ("invalid_arguments",)) -> dict[str, Any]:
    """Run one auditable offline mutation generation and paired acceptance pass.

    The deterministic provider is deliberately a smoke harness: it proves the
    control-plane plumbing and ledger artifacts without claiming model quality.
    A real provider/evaluator can later be substituted behind the same runner.
    """
    raw_tasks = load_tasks(tasks_path)
    task_specs = tuple(_task_spec(task) for task in raw_tasks)
    evaluator = Evaluator(task_specs, evaluator_version="evolve-v1")
    runtime_profile = RuntimeProfile(runtime_id="reference-runtime", runtime_version="0.1",
                                     adapter_id="deterministic-provider", api_format="offline")
    genome = bind_genome(AgentGenome(), model_id="deterministic-test", profile=runtime_profile)
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    expected_by_id = {str(item["id"]): str(item["expected"]) for item in raw_tasks}

    def evaluate_genome(candidate_genome: AgentGenome) -> list[TaskResult]:
        def runner(task: TaskSpec, seed: int) -> Mapping[str, Any]:
            provider = DeterministicTestProvider(model_id="deterministic-test", default_content=expected_by_id[task.id])
            result = AgentRuntime(provider, genome=candidate_genome, runtime_profile=runtime_profile).run(task.prompt)
            return {"answer": result.content, "content": result.content, "tool_calls": [],
                    "usage": asdict(result.usage), "latency_ms": 0.0,
                    "runtime_success": result.success}
        return evaluator.run(runner, seeds=seed_values)

    baseline_results = evaluate_genome(genome)
    observed = [row.failure_class for row in baseline_results if row.failure_class]
    requested = tuple(str(value) for value in failure_classes if str(value).strip())
    candidates = generate_candidates(genome, observed or requested, limit=3,
                                     runtime_profile=runtime_profile, model_id="deterministic-test")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_results(output_dir / "baseline_results.jsonl", baseline_results)

    ledger_path = output_dir / "ledger.db"
    parent_id = "genome-000"
    direction = DirectionSpec(primary="task_success_rate", min_primary_improvement=0.0)
    baseline_metrics = summarize(baseline_results)
    with Ledger(ledger_path) as ledger:
        ledger.add_version(parent_id, model_id="deterministic-test", branch="deterministic-test",
                           layer="shared-base", direction=asdict(direction), edit={"type": "initial"},
                           eval_manifest_hash=evaluator.manifest_hash,
                           runtime_profile_id=runtime_profile.profile_id)
        for seed in seed_values:
            rows = [row for row in baseline_results if row.seed == seed]
            run_id = f"baseline-{seed}"
            ledger.record_run(run_id, parent_id, split="dev", seed=seed,
                              evaluator_version=evaluator.evaluator_version,
                              manifest_hash=evaluator.manifest_hash)
            ledger.record_metrics(run_id, summarize(rows))
            for row in rows:
                ledger.record_task_result(run_id, task_id=row.task_id, capability=row.capability,
                                          passed=row.passed, failure_class=row.failure_class,
                                          reason=row.reason, result=asdict(row))

        decisions: list[dict[str, Any]] = []
        accepted: list[str] = []
        for candidate in candidates:
            candidate_results = evaluate_genome(candidate.genome)
            _write_results(output_dir / f"{candidate.candidate_id}_results.jsonl", candidate_results)
            decision = compare_paired(baseline_results, candidate_results, direction,
                                      candidate_id=candidate.candidate_id)
            detail = compare_results(baseline_results, candidate_results, bootstrap_samples=200)
            record = {"candidate": asdict(candidate), "decision": asdict(decision),
                      "paired_report": detail.to_dict()}
            decisions.append(record)
            if decision.accepted:
                accepted.append(candidate.candidate_id)
                ledger.add_version(candidate.candidate_id, model_id="deterministic-test",
                                   parent_id=parent_id, branch="deterministic-test",
                                   layer=candidate.layer, direction=asdict(direction), edit=dict(candidate.patch),
                                   eval_manifest_hash=evaluator.manifest_hash,
                                   runtime_profile_id=runtime_profile.profile_id)
                for seed in seed_values:
                    rows = [row for row in candidate_results if row.seed == seed]
                    run_id = f"{candidate.candidate_id}-{seed}"
                    ledger.record_run(run_id, candidate.candidate_id, split="dev", seed=seed,
                                      evaluator_version=evaluator.evaluator_version,
                                      manifest_hash=evaluator.manifest_hash)
                    ledger.record_metrics(run_id, summarize(rows))
                    for row in rows:
                        ledger.record_task_result(run_id, task_id=row.task_id, capability=row.capability,
                                                  passed=row.passed, failure_class=row.failure_class,
                                                  reason=row.reason, result=asdict(row))
                ledger.record_attribution(candidate.candidate_id, evidence="paired_dev_evaluation",
                                          confidence=direction.confidence, edit_ref=candidate.edit_ref,
                                          details={"failure_classes": candidate.failure_classes,
                                                   "decision": asdict(decision)})
                ledger.record_capability_delta(candidate.candidate_id, changed=[candidate.layer],
                                               evidence={"candidate_id": candidate.candidate_id})

        (output_dir / "decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2,
                                                               sort_keys=True, default=str) + "\n", encoding="utf-8")
        manifest = {"schema_version": "0.1", "run_type": "evolve", "model_id": "deterministic-test",
                    "parent_version": parent_id, "candidate_count": len(candidates),
                    "accepted_candidates": accepted, "failure_classes": list(observed or requested),
                    "seed_count": len(seed_values), "task_manifest_hash": evaluator.manifest_hash,
                    "baseline_metrics": baseline_metrics, "ledger_file": ledger_path.name,
                    "decisions_file": "decisions.json", "runtime_profile": runtime_profile.to_dict()}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                             sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ouroboros", description="Ouroboros offline adaptation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="run a deterministic baseline and write JSONL + manifest")
    baseline.add_argument("--provider", default="test", choices=("test",), help="provider implementation")
    baseline.add_argument("--tasks", type=Path, help="JSON or JSONL task manifest")
    baseline.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"), help="artifact directory")
    probe = subparsers.add_parser("probe", help="run model probes and write a fingerprint")
    probe.add_argument("--provider", default="test", choices=("test", "deepseek", "gateway", "anthropic"), help="provider implementation")
    probe.add_argument("--base-url", help="base URL for an OpenAI-compatible gateway")
    probe.add_argument("--api-key-env", default="OPENAI_COMPATIBLE_API_KEY", help="environment variable containing gateway key")
    probe.add_argument("--model", default="deepseek-flash", help="provider model id")
    probe.add_argument("--thinking", action="store_true", help="request DeepSeek thinking mode when supported")
    probe.add_argument("--timeout", type=float, default=120.0, help="provider request timeout in seconds")
    probe.add_argument("--probe-id", action="append", dest="probe_ids", help="run only selected probe (repeatable)")
    probe.add_argument("--output-dir", type=Path, default=Path("artifacts/probe"), help="artifact directory")
    evolve = subparsers.add_parser("evolve", help="run one offline mutation/evaluation/ledger generation")
    evolve.add_argument("--tasks", type=Path, help="JSON or JSONL task manifest")
    evolve.add_argument("--failure-class", action="append", dest="failure_classes",
                        help="failure class to seed when baseline has no failures (repeatable)")
    evolve.add_argument("--seeds", default="0,1,2", help="comma-separated evaluation seeds")
    evolve.add_argument("--output-dir", type=Path, default=Path("artifacts/evolve"), help="artifact directory")
    adapt = subparsers.add_parser("adapt", help="run one real-provider calibration and offline mutation gate")
    adapt.add_argument("--provider", default="test", choices=("test", "deepseek", "gateway", "anthropic"), help="provider implementation")
    adapt.add_argument("--base-url", help="base URL for an OpenAI-compatible or Anthropic gateway")
    adapt.add_argument("--api-key-env", help="environment variable containing the provider key")
    adapt.add_argument("--model", default="deepseek-flash", help="provider model id")
    adapt.add_argument("--thinking", action="store_true", help="request thinking mode when supported")
    adapt.add_argument("--timeout", type=float, default=120.0, help="provider request timeout in seconds")
    adapt.add_argument("--probe-id", action="append", dest="probe_ids", help="adapt only selected probe (repeatable)")
    adapt.add_argument("--failure-class", action="append", dest="failure_classes", help="seed a mutation when probes pass")
    adapt.add_argument("--max-candidates", type=int, default=1, help="maximum candidates to evaluate (default: 1)")
    adapt.add_argument("--seeds", default="0", help="comma-separated paired evaluation seeds")
    adapt.add_argument("--output-dir", type=Path, default=Path("artifacts/adapt"), help="artifact directory")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "baseline":
            manifest = run_baseline(output_dir=args.output_dir, provider_name=args.provider, tasks_path=args.tasks)
            print(json.dumps({"run_id": manifest["run_id"], "metrics": manifest["metrics"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
            return 0
        if args.command == "probe":
            settings = {"thinking": {"type": "enabled"}} if args.thinking else None
            manifest = run_probes(output_dir=args.output_dir, provider_name=args.provider,
                                  model=args.model, settings=settings, timeout_seconds=args.timeout,
                                  base_url=args.base_url,
                                  api_key=(os.environ.get(args.api_key_env) if args.provider in {"gateway", "anthropic"} else None),
                                  probe_ids=args.probe_ids)
            print(json.dumps({"run_type": manifest["run_type"], "metrics": manifest["metrics"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
            return 0
        if args.command == "evolve":
            seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
            manifest = run_evolution(output_dir=args.output_dir, tasks_path=args.tasks, seeds=seeds,
                                     failure_classes=args.failure_classes or ("invalid_arguments",))
            print(json.dumps({"run_type": manifest["run_type"], "candidate_count": manifest["candidate_count"],
                              "accepted_candidates": manifest["accepted_candidates"],
                              "output_dir": str(args.output_dir)}, ensure_ascii=False))
            return 0
        if args.command == "adapt":
            defaults = {"deepseek": "DEEPSEEK_API_KEY", "gateway": "OPENAI_COMPATIBLE_API_KEY",
                        "anthropic": "ANTHROPIC_API_KEY"}
            key_env = args.api_key_env or defaults.get(args.provider)
            key = os.environ.get(key_env) if key_env else None
            settings = {"thinking": {"type": "enabled"}} if args.thinking else None
            seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
            manifest = run_adaptation(output_dir=args.output_dir, provider_name=args.provider,
                                      model=args.model, settings=settings, timeout_seconds=args.timeout,
                                      base_url=args.base_url, api_key=key, probe_ids=args.probe_ids,
                                      failure_classes=args.failure_classes, max_candidates=args.max_candidates,
                                      seeds=seeds)
            print(json.dumps({"run_type": manifest["run_type"], "model_id": manifest["model_id"],
                              "candidate_count": manifest["candidate_count"],
                              "accepted_candidates": manifest["accepted_candidates"],
                              "output_dir": str(args.output_dir)}, ensure_ascii=False))
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
