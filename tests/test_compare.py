import pytest

from ouroboros.compare import compare_results, pair_results
from ouroboros.eval import TaskResult, Usage


def row(task_id, seed, passed, tokens, *, errors=0, retries=0):
    return TaskResult(
        task_id, seed, passed, "cap", None if passed else "failed", None,
        1, errors, retries, Usage(total_tokens=tokens), 10.0, {},
    )


def test_pair_results_aligns_task_and_seed_and_reports_deltas():
    parent = [row("b", 1, True, 100), row("a", 0, False, 80)]
    candidate = [row("a", 0, True, 70), row("b", 1, False, 110, errors=1)]
    pairs = pair_results(parent, candidate)
    assert [(p.task_id, p.seed) for p in pairs] == [("a", 0), ("b", 1)]
    assert pairs[0].pass_delta == 1
    assert pairs[1].total_token_delta == 10
    assert pairs[1].tool_error_delta == 1


def test_pair_results_rejects_missing_and_duplicate_rows():
    with pytest.raises(ValueError, match="unpaired"):
        pair_results([row("a", 0, True, 1)], [row("b", 0, True, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        pair_results([row("a", 0, True, 1), row("a", 0, True, 1)], [row("a", 0, True, 1)])


def test_compare_results_has_reproducible_bootstrap_and_effect_judgment():
    parent = [row("a", 0, True, 100), row("b", 0, False, 100), row("c", 0, True, 100)]
    candidate = [row("a", 0, True, 90), row("b", 0, True, 80), row("c", 0, True, 95)]
    report = compare_results(parent, candidate, bootstrap_samples=300, seed=7)
    again = compare_results(parent, candidate, bootstrap_samples=300, seed=7)
    assert report.to_dict() == again.to_dict()
    assert report.candidate_success_rate == 1.0
    assert report.success_rate_delta == pytest.approx(1 / 3)
    assert report.average_total_token_delta == pytest.approx(-35 / 3)
    assert report.token_savings_rate == pytest.approx(35 / 300)
    assert report.minimum_practical_effect_met


def test_compare_results_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        compare_results([], [], confidence=1.0)
