from ouroboros.eval import Evaluator, JudgeResult, TaskSpec
from ouroboros.fingerprint import build_fingerprint, recalibration_required
from ouroboros.ledger import Ledger
from ouroboros.profile import RuntimeProfile, bind_genome, branch_key
from ouroboros.types import AgentGenome
from ouroboros.runtime import AgentRuntime
from ouroboros.providers import DeterministicTestProvider


def test_runtime_profile_binds_genome_and_requires_recalibration():
    p1 = RuntimeProfile(runtime_id="mother-a", runtime_version="1")
    p2 = RuntimeProfile(runtime_id="mother-b", runtime_version="1")
    genome = bind_genome(AgentGenome(), model_id="deepseek-flash", profile=p1)
    assert genome.runtime_profile_id == p1.profile_id
    assert genome.branch == branch_key("deepseek-flash", p1)

    task = TaskSpec("p", "ok", "cap", lambda response: JudgeResult(True))
    rows = Evaluator((task,)).run(lambda _task, _seed: {"ok": True})
    f1 = build_fingerprint(rows, model_id="deepseek-flash", provider="deepseek",
                           model_version="DeepSeek-V4.1-Flash", runtime_profile=p1)
    f2 = build_fingerprint(rows, model_id="deepseek-flash", provider="deepseek",
                           model_version="DeepSeek-V4.1-Flash", runtime_profile=p2)
    assert recalibration_required(f1, f2)


def test_ledger_keeps_runtime_branches_isolated():
    p1 = RuntimeProfile(runtime_id="mother-a")
    p2 = RuntimeProfile(runtime_id="mother-b")
    with Ledger() as ledger:
        ledger.add_version("a0", model_id="m", branch=branch_key("m", p1), runtime_profile_id=p1.profile_id)
        try:
            ledger.add_version("b-child", model_id="m", parent_id="a0",
                               branch=branch_key("m", p2), runtime_profile_id=p2.profile_id)
        except ValueError as exc:
            assert "runtime profile" in str(exc)
        else:
            raise AssertionError("runtime replacement must start an isolated branch")
        ledger.add_version("b0", model_id="m", branch=branch_key("m", p2), runtime_profile_id=p2.profile_id)
        assert len(ledger.list_versions(branch=branch_key("m", p1))) == 1
        assert len(ledger.list_versions(branch=branch_key("m", p2))) == 1


def test_runtime_rejects_genome_bound_to_another_runtime():
    profile = RuntimeProfile(runtime_id="mother-a")
    other = RuntimeProfile(runtime_id="mother-b")
    genome = bind_genome(AgentGenome(), model_id="m", profile=profile)
    try:
        AgentRuntime(DeterministicTestProvider(), genome=genome, runtime_profile=other)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("runtime replacement must reject an incompatible Genome")
