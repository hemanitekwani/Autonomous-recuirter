"""LLM adapter: business logic depends only on `LLMClient.generate_json`.

* GeminiLLMClient - Google Gemini via `google-genai`, using response_json_schema
  structured output so the reply is schema-shaped JSON. The caller still
  validates it with Pydantic.
* MockLLMClient - deterministic, rule-based stand-in (src/mock_llm.py) so the
  whole workflow runs without paid API access.

To add another provider, implement `generate_json` and register it in
`get_llm_client`.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from src.config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The LLM call failed or returned something unusable (never contains secrets)."""


class LLMClient(Protocol):
    mode: str
    model: str | None

    def generate_json(
        self, *, task: str, system: str, user_content: str, json_schema: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a JSON object for `task`. `payload` is the structured input (used by the mock)."""
        ...


def _parse_json_object(text: str | None, task: str) -> dict[str, Any]:
    if not text:
        raise LLMError(f"LLM returned no text for {task}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM returned invalid JSON for {task}") from exc
    if not isinstance(data, dict):
        raise LLMError(f"LLM returned a non-object JSON value for {task}")
    return data


class GeminiLLMClient:
    """Google Gemini via the official `google-genai` SDK, using JSON-schema structured output."""

    mode = "llm"
    _RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(self, settings: Settings, max_retries: int = 2):
        from google import genai
        from google.genai import errors, types

        self._types, self._errors = types, errors
        # api_key=None lets the SDK read GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
        # Hard timeout (ms) so a silent server can never hang a resume upload.
        self._client = genai.Client(api_key=settings.gemini_api_key or None,
                                    http_options=types.HttpOptions(timeout=settings.llm_timeout_seconds * 1000))
        self.model = settings.gemini_model
        self._max_retries = max_retries

    def generate_json(
        self, *, task: str, system: str, user_content: str, json_schema: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        import time

        import httpx

        config = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=json_schema,
            temperature=0,
        )
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(model=self.model, contents=user_content, config=config)
                break
            except self._errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code in self._RETRYABLE and attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                if code in (401, 403):
                    raise LLMError("LLM authentication failed - check GEMINI_API_KEY") from exc
                if code == 429:
                    raise LLMError("LLM rate limit / quota reached; retry later") from exc
                if code in (400, 404):
                    raise LLMError(f"LLM rejected the request ({task}): {getattr(exc, 'message', '')}"[:300]) from exc
                raise LLMError(f"LLM API error {code} during {task}") from exc
            except httpx.TimeoutException as exc:  # not retried: another full wait helps nobody
                raise LLMError(f"LLM did not respond within the timeout during {task}; retry later") from exc
            except httpx.TransportError as exc:
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError("Could not reach the LLM API") from exc

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMError(f"LLM declined the {task} request")
        candidates = getattr(response, "candidates", None) or []
        finish = str(getattr(candidates[0], "finish_reason", "") or "") if candidates else ""
        if "MAX_TOKENS" in finish:
            raise LLMError(f"LLM output for {task} was truncated")
        if any(reason in finish for reason in ("SAFETY", "RECITATION", "PROHIBITED", "BLOCKLIST", "SPII")):
            raise LLMError(f"LLM declined the {task} request")
        return _parse_json_object(response.text, task)


class MockLLMClient:
    """Deterministic rule-based implementation of the two resume tasks."""

    mode = "mock"
    model = None

    def generate_json(
        self, *, task: str, system: str, user_content: str, json_schema: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        from src import mock_llm

        handlers = {
            "resume_extraction": mock_llm.mock_extract_resume,
            "profile_validation": mock_llm.mock_validate_profile,
        }
        if task not in handlers:
            raise LLMError(f"Mock LLM has no handler for task '{task}'")
        return handlers[task](payload)


def get_llm_client(settings: Settings) -> LLMClient:
    if settings.mock_llm:
        return MockLLMClient()
    if settings.llm_provider == "gemini":
        return GeminiLLMClient(settings)
    raise LLMError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'")
