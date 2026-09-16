from ouroboros.eval import Evaluator, JudgeResult, TaskSpec
from ouroboros.fingerprint import build_fingerprint


def test_fingerprint_is_conditioned_on_model_and_probe_manifest():
    tasks = (
        TaskSpec("probe-1", "return ok", "basic", lambda response: JudgeResult(response.get("answer") == "ok")),
    )
    evaluator = Evaluator(tasks, evaluator_version="eval-test")
    results = evaluator.run(lambda task, seed: {"answer": "ok", "usage": {"input_tokens": 2, "output_tokens": 1}})
    fingerprint = build_fingerprint(
        results,
        model_id="deepseek-flash",
        provider="deepseek",
        model_version="DeepSeek-V4.1-Flash",
        probe_manifest_hash=evaluator.manifest_hash,
    )

    assert fingerprint.task_count == 1
    assert fingerprint.metrics["task_success_rate"] == 1.0
    assert fingerprint.condition.model_id == "deepseek-flash"
    assert fingerprint.condition.probe_manifest_hash == evaluator.manifest_hash
    assert fingerprint.trace_hash.startswith("sha256:")
