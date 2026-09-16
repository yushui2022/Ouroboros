from ouroboros.providers import DeterministicTestProvider
from ouroboros.runtime import AgentRuntime
from ouroboros.types import ToolSpec


def test_runtime_executes_tool_and_preserves_provider_trace():
    tool = ToolSpec(
        name="echo",
        description="Echo a value",
        parameters={"value": {"type": "string"}},
        handler=lambda value: value,
    )
    provider = DeterministicTestProvider(
        script=[
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "echo", "arguments": '{"value":"hello"}'},
                        }
                    ],
                }
            },
            {"message": {"content": "done"}, "finish_reason": "stop"},
        ]
    )
    result = AgentRuntime(provider, tools=[tool]).run("say hello")

    assert result.success is True
    assert len(result.tool_results) == 1
    assert result.tool_results[0].output == "hello"
    assert any(event.kind == "tool_result" for event in result.trace)
    assert len(provider.requests) == 2
    assert provider.requests[1]["messages"][-1]["role"] == "tool"
