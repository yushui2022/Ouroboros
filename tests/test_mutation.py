from benchmarks.probes import default_probe_tasks
from ouroboros.mutation import generate_candidates
from ouroboros.types import AgentGenome


def test_finish_probe_distinguishes_format_from_wrong_answer():
    task = next(task for task in default_probe_tasks() if task.id == "probe-finish-marker")

    assert task.judge({"answer": "done"}).passed
    mismatch = task.judge({"answer": "done. Waiting for the next task."})
    assert not mismatch.passed
    assert mismatch.failure_class == "finish_format_mismatch"


def test_finish_format_failure_generates_concise_prompt_candidate():
    genome = AgentGenome(system_prompt="You are a coding agent.")
    candidates = generate_candidates(
        genome,
        [{"failure_class": "finish_format_mismatch"}],
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.candidate_id == "mutation-1-finish-format"
    assert candidate.layer == "prompt"
    assert "output only the final answer" in candidate.genome.system_prompt
    assert candidate.patch["completion_output"] == "concise"
