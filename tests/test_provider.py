import json

from ouroboros.providers import AnthropicMessagesProvider, DeepSeekProvider, OpenAICompatibleProvider
from ouroboros.types import Message, ToolSpec


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_deepseek_provider_preserves_reasoning_and_tool_schema(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response({
            "choices": [{"message": {"content": "ok", "reasoning_content": "kept", "tool_calls": []}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 1}, "total_tokens": 6},
        })

    monkeypatch.setattr("ouroboros.providers.urlrequest.urlopen", fake_urlopen)
    provider = DeepSeekProvider(api_key="test-key", base_url="https://example.test")
    response = provider.complete(
        [Message("system", "be precise", reasoning_content="prior")],
        [ToolSpec("lookup", parameters={"key": {"type": "string"}}, required=("key",))],
        settings={"reasoning_effort": "high"},
    )

    assert captured["body"]["model"] == "deepseek-flash"
    assert captured["body"]["messages"][0]["reasoning_content"] == "prior"
    assert captured["body"]["tools"][0]["function"]["parameters"]["required"] == ["key"]
    assert response.content == "ok"
    assert response.reasoning_content == "kept"
    assert response.usage.total_tokens == 6
    assert response.usage.reasoning_tokens == 1


def test_openai_compatible_provider_normalizes_gateway_url(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _Response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("ouroboros.providers.urlrequest.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(api_key="test-key", base_url="https://gateway.test", model="gpt-test")
    provider.complete([Message("user", "ping")], [])
    assert captured["url"] == "https://gateway.test/v1/chat/completions"


def test_anthropic_messages_provider_uses_content_blocks(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        return _Response({"content": [{"type": "text", "text": "ok"}],
                          "stop_reason": "end_turn", "usage": {"input_tokens": 4, "output_tokens": 2}})

    monkeypatch.setattr("ouroboros.providers.urlrequest.urlopen", fake_urlopen)
    provider = AnthropicMessagesProvider(api_key="test-key", base_url="https://gateway.test", model="claude-test")
    response = provider.complete([Message("system", "precise"), Message("user", "ping")], [])
    assert captured["url"] == "https://gateway.test/claude/v1/messages"
    assert captured["body"]["system"] == "precise"
    assert captured["body"]["messages"][0]["content"] == "ping"
    assert response.content == "ok"
