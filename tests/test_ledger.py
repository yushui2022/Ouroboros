from ouroboros.ledger import Ledger


def test_append_only_version_and_query(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        ledger.add_version("v0", model_id="m", branch="m-base")
        ledger.add_version("v1", model_id="m", parent_id="v0", layer="tool_protocol",
                           direction={"primary": "total_tokens"}, edit={"changed_component": "ToolSchema"})
        ledger.record_run("run-1", "v1", split="dev", seed=2)
        ledger.record_metrics("run-1", {"task_success_rate": 0.8, "total_tokens": 120})
        ledger.record_task_result("run-1", task_id="t1", capability="cap-a", passed=False,
                                  failure_class="invalid_arguments")
        ledger.record_capability_delta("v1", lost=["cap-a"], confidence=0.7)
        ledger.record_attribution("v1", evidence="paired_dev", confidence=0.7, edit_ref="patch-1")
        ledger.record_debt("debt-1", "v1", capability_id="cap-a")

        result = ledger.query_version("v1")
        assert result["version"]["parent_id"] == "v0"
        assert result["runs"][0]["metrics"]["total_tokens"] == 120
        assert result["runs"][0]["task_results"][0]["passed"] is False
        assert result["capability_delta"]["lost"] == ["cap-a"]
        assert result["attribution"]["edit_ref"] == "patch-1"
        assert result["debt"][0]["status"] == "potential_debt"


def test_duplicate_version_must_be_immutable():
    with Ledger() as ledger:
        ledger.add_version("v0", model_id="m", layer="prompt")
        ledger.add_version("v0", model_id="m", layer="prompt")  # idempotent replay
        try:
            ledger.add_version("v0", model_id="m", layer="tool_protocol")
        except ValueError:
            pass
        else:
            raise AssertionError("conflicting version insertion should fail")
