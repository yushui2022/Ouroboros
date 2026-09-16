from dataclasses import dataclass

from ouroboros.pydantic_adapter import PydanticAIAgentAdapter
from ouroboros.profile import bind_genome
from ouroboros.types import AgentGenome, ToolSpec


@dataclass
class FakeUsage:
    input_tokens: int = 7
    output_tokens: int = 3
    reasoning_tokens: int = 2
    total_tokens: int = 12


class FakeMessage:
    def __init__(self, role, content):
        self.role, self.content = role, content


class FakeResult:
    output = "done"
    def usage(self):
        return FakeUsage()
    def all_messages(self):
        return [FakeMessage("user", "hello"), FakeMessage("assistant", "done")]


class FakeAgent:
    def __init__(self, model, **kwargs):
        self.model, self.kwargs = model, kwargs
        self.tools = []
    def tool_plain(self, fn):
        self.tools.append(fn)
        return fn
    def run_sync(self, prompt, **kwargs):
        return FakeResult()


def test_adapter_uses_injected_agent_factory_and_maps_result():
    adapter = PydanticAIAgentAdapter("deepseek:deepseek-flash", agent_factory=FakeAgent)
    result = adapter.run("hello")
    assert result.success
    assert result.content == "done"
    assert result.usage.total_tokens == 12
    assert [m.role for m in result.messages] == ["user", "assistant"]
    assert adapter.runtime_profile.runtime_id == "pydantic-ai-harness"


def test_adapter_binds_genome_to_runtime_profile():
    adapter = PydanticAIAgentAdapter("m", agent_factory=FakeAgent)
    genome = bind_genome(AgentGenome(), model_id="m", profile=adapter.runtime_profile)
    assert adapter.run("hello", genome=genome).success


def test_adapter_captures_tool_outcome():
    seen = []
    def add(x: int) -> int:
        seen.append(x)
        return x + 1
    class ToolAgent(FakeAgent):
        def run_sync(self, prompt, **kwargs):
            self.tools[0](2)
            return FakeResult()
    adapter = PydanticAIAgentAdapter("m", [ToolSpec("add", handler=add)], agent_factory=ToolAgent)
    result = adapter.run("hello")
    assert seen == [2]
    assert result.tool_results[0].output == 3
