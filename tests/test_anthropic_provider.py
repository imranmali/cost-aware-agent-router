"""AnthropicProvider against a fake client: request shape and response handling, no network."""

import json
from types import SimpleNamespace

from costrouter.models import HAIKU, OPUS, Route
from costrouter.providers.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from costrouter.providers.base import Request


def _response(text=None, stop="end_turn"):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=40, cache_creation_input_tokens=0, cache_read_input_tokens=10
    )
    return SimpleNamespace(content=content, stop_reason=stop, usage=usage)


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)
        self.beta = SimpleNamespace(messages=FakeMessages(response))


def _req(route):
    return Request("t1", "What is 2+2?", route, max_tokens=512)


def test_haiku_uses_plain_endpoint_with_effort_and_schema():
    client = FakeClient(_response(json.dumps({"answer": "4", "confidence": 0.97})))
    c = AnthropicProvider(client).complete(_req(Route(HAIKU, "low")))

    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] == 512
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in call
    assert not client.beta.messages.calls

    assert c.text == "4" and c.self_confidence == 0.97
    assert c.input_tokens == 110 and c.output_tokens == 40


def test_opus_opts_into_server_side_refusal_fallback():
    client = FakeClient(_response(json.dumps({"answer": "4", "confidence": 0.9})))
    AnthropicProvider(client).complete(_req(Route(OPUS, "high")))
    call = client.beta.messages.calls[0]
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"


def test_fallback_can_be_disabled():
    client = FakeClient(_response(json.dumps({"answer": "4", "confidence": 0.9})))
    AnthropicProvider(client, refusal_fallbacks=False).complete(_req(Route(OPUS, "high")))
    assert client.messages.calls and not client.beta.messages.calls


def test_refusal_and_truncation_return_empty_answer():
    for stop in ("refusal", "max_tokens"):
        client = FakeClient(_response(None, stop=stop))
        c = AnthropicProvider(client).complete(_req(Route(HAIKU, "low")))
        assert c.text == "" and c.stop_reason == stop and c.output_tokens == 40


def test_non_json_text_is_kept_without_confidence():
    client = FakeClient(_response("plain text"))
    c = AnthropicProvider(client).complete(_req(Route(HAIKU, "low")))
    assert c.text == "plain text" and c.self_confidence is None


def test_confidence_is_clamped():
    client = FakeClient(_response(json.dumps({"answer": "4", "confidence": 7})))
    c = AnthropicProvider(client).complete(_req(Route(HAIKU, "low")))
    assert c.self_confidence == 1.0
