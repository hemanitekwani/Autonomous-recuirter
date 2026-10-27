"""Gemini adapter request shape and response handling (SDK client replaced by a fake; no network)."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from src.llm_client import GeminiLLMClient, LLMError

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}


class _FakeGeminiModels:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _gemini(settings, outcomes):
    client = GeminiLLMClient(dataclasses.replace(settings, llm_provider="gemini", gemini_api_key="test-key",
                                                 gemini_model="gemini-3.5-flash-lite"))
    fake = _FakeGeminiModels(outcomes)
    client._client = SimpleNamespace(models=fake)
    return client, fake


def _gem_resp(text, finish="STOP", block=None):
    return SimpleNamespace(text=text, prompt_feedback=SimpleNamespace(block_reason=block),
                           candidates=[SimpleNamespace(finish_reason=finish)])


def _api_error(code):
    from google.genai import errors

    body = {"error": {"code": code, "message": "x", "status": "X"}}
    return errors.ClientError(code, body) if code < 500 else errors.ServerError(code, body)


def call(client):
    return client.generate_json(task="t", system="sys", user_content="u", json_schema=SCHEMA, payload={})


def test_gemini_structured_output_request(settings):
    client, fake = _gemini(settings, [_gem_resp('{"a": "x"}')])
    assert call(client) == {"a": "x"}
    cfg = fake.calls[0]["config"]
    assert fake.calls[0]["model"] == "gemini-3.5-flash-lite"
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema == SCHEMA
    assert cfg.system_instruction == "sys"


def test_gemini_retries_transient_errors(settings, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client, fake = _gemini(settings, [_api_error(503), _gem_resp('{"a": "y"}')])
    assert call(client) == {"a": "y"} and len(fake.calls) == 2


@pytest.mark.parametrize("outcome, message", [
    (_gem_resp('{"a": ', finish="MAX_TOKENS"), "truncated"),
    (_gem_resp("", finish="SAFETY"), "declined"),
    (_gem_resp("", block="PROHIBITED_CONTENT"), "declined"),
    (_gem_resp("nope"), "invalid JSON"),
    (_gem_resp("[1, 2]"), "non-object"),
])
def test_gemini_bad_responses(settings, outcome, message):
    client, _ = _gemini(settings, [outcome])
    with pytest.raises(LLMError, match=message):
        call(client)


def test_gemini_timeout_fails_fast_without_retry(settings):
    import httpx

    client, fake = _gemini(settings, [httpx.ReadTimeout("silent server")])
    with pytest.raises(LLMError, match="did not respond"):
        call(client)
    assert len(fake.calls) == 1


def test_gemini_auth_error_is_clear(settings):
    client, _ = _gemini(settings, [_api_error(403)])
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        call(client)
