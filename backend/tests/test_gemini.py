import inspect
import os

import httpx
import pytest

import app.retrieval.gemini as gemini
from app.retrieval.gemini import (
    DEFAULT_MODEL,
    GeminiAuthError,
    GeminiError,
    GeminiTimeoutError,
    generate_answer,
)
from app.retrieval.llm import LLMError
from app.retrieval.llm import generate_answer as ollama_generate_answer
from app.retrieval.query_flow import answer_with_llm
from app.retrieval.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Mocked-transport tests: fast, deterministic, no real Gemini call, no quota
# consumed. `generate_answer` calls httpx.post; we replace it per-test —
# same pattern as test_llm.py.
# ---------------------------------------------------------------------------


def _patch_post(monkeypatch, handler):
    """Replace gemini.httpx.post with `handler(url, json=, headers=, timeout=)`."""
    monkeypatch.setattr(gemini.httpx, "post", handler)


def _candidates_response(text, status=200):
    return httpx.Response(
        status,
        json={"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def test_gemini_error_is_an_llm_error():
    # The core interchangeability property: answer_with_llm's documented
    # "Raises: LLMError (and subclasses)" must stay true regardless of
    # which backend `generate` is bound to.
    assert issubclass(GeminiError, LLMError)
    assert issubclass(GeminiAuthError, GeminiError)
    assert issubclass(GeminiTimeoutError, GeminiError)


def test_generate_answer_signature_is_call_compatible_with_ollamas():
    # answer_with_llm calls `generate(retrieval["prompt"], model=model)` —
    # both backends must accept that exact call shape.
    gemini_params = inspect.signature(generate_answer).parameters
    ollama_params = inspect.signature(ollama_generate_answer).parameters
    assert list(gemini_params)[:2] == list(ollama_params)[:2] == ["prompt", "model"]
    assert gemini_params["prompt"].kind == ollama_params["prompt"].kind
    assert gemini_params["model"].kind == ollama_params["model"].kind


def test_generate_answer_returns_response_text(monkeypatch):
    captured = {}

    def handler(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _candidates_response("  four  ")

    _patch_post(monkeypatch, handler)

    answer = generate_answer("What is 2 + 2?", api_key="test-key")

    assert answer == "four"  # stripped
    assert captured["url"] == (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{DEFAULT_MODEL}:generateContent"
    )
    assert captured["json"] == {"contents": [{"parts": [{"text": "What is 2 + 2?"}]}]}
    # The key must be in a header, never the URL (logs, proxies, referrers).
    assert captured["headers"] == {"x-goog-api-key": "test-key"}
    assert "test-key" not in captured["url"]


def test_generate_answer_honours_custom_model(monkeypatch):
    seen = {}

    def handler(url, json, headers, timeout):
        seen["url"] = url
        return _candidates_response("ok")

    _patch_post(monkeypatch, handler)

    generate_answer("hi", model="gemini-1.5-pro", api_key="test-key")

    assert seen["url"].endswith("/models/gemini-1.5-pro:generateContent")


def test_multiple_parts_are_concatenated(monkeypatch):
    def handler(url, json, headers, timeout):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "Hello, "}, {"text": "world."}]}}
                ]
            },
        )

    _patch_post(monkeypatch, handler)

    assert generate_answer("hi", api_key="test-key") == "Hello, world."


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\n\t"])
def test_generate_answer_rejects_empty_prompt(bad_prompt, monkeypatch):
    def handler(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("httpx.post should not be called for a bad prompt")

    _patch_post(monkeypatch, handler)

    with pytest.raises(ValueError):
        generate_answer(bad_prompt, api_key="test-key")


@pytest.mark.parametrize("bad_prompt", [None, 123, ["list"], {"d": 1}])
def test_non_string_prompt_raises_value_error(bad_prompt, monkeypatch):
    _patch_post(
        monkeypatch,
        lambda *a, **k: pytest.fail("httpx.post should not be called"),
    )

    with pytest.raises(ValueError):
        generate_answer(bad_prompt, api_key="test-key")


def test_missing_api_key_raises_auth_error_without_a_network_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _patch_post(
        monkeypatch,
        lambda *a, **k: pytest.fail("httpx.post should not be called"),
    )

    with pytest.raises(GeminiAuthError):
        generate_answer("hi", api_key=None)


def test_blank_api_key_raises_auth_error(monkeypatch):
    _patch_post(
        monkeypatch,
        lambda *a, **k: pytest.fail("httpx.post should not be called"),
    )

    with pytest.raises(GeminiAuthError):
        generate_answer("hi", api_key="   ")


def test_api_key_falls_back_to_environment_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    seen = {}

    def handler(url, json, headers, timeout):
        seen["headers"] = headers
        return _candidates_response("ok")

    _patch_post(monkeypatch, handler)

    generate_answer("hi")

    assert seen["headers"] == {"x-goog-api-key": "env-key"}


def test_explicit_api_key_overrides_environment_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    seen = {}

    def handler(url, json, headers, timeout):
        seen["headers"] = headers
        return _candidates_response("ok")

    _patch_post(monkeypatch, handler)

    generate_answer("hi", api_key="explicit-key")

    assert seen["headers"] == {"x-goog-api-key": "explicit-key"}


def test_invalid_api_key_400_raises_auth_error(monkeypatch):
    def handler(url, json, headers, timeout):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiAuthError) as excinfo:
        generate_answer("hi", api_key="bad-key")

    assert "api key" in str(excinfo.value).lower()


def test_permission_denied_403_raises_auth_error(monkeypatch):
    def handler(url, json, headers, timeout):
        return httpx.Response(
            403,
            json={
                "error": {"message": "Permission denied", "status": "PERMISSION_DENIED"}
            },
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiAuthError):
        generate_answer("hi", api_key="test-key")


def test_bad_request_not_about_api_key_raises_generic_error(monkeypatch):
    # A 400 that isn't about the key (e.g. a malformed request) must not be
    # misreported as an auth problem.
    def handler(url, json, headers, timeout):
        return httpx.Response(
            400,
            json={
                "error": {"message": "Invalid JSON payload", "status": "INVALID_ARGUMENT"}
            },
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError) as excinfo:
        generate_answer("hi", api_key="test-key")

    assert not isinstance(excinfo.value, GeminiAuthError)


def test_error_field_as_a_string_does_not_crash_error_handling(monkeypatch):
    # A gateway/proxy in front of the real endpoint could plausibly return
    # {"error": "some string"} rather than Gemini's own {"error": {...}}
    # shape — must still raise GeminiError cleanly, not AttributeError.
    def handler(url, json, headers, timeout):
        return httpx.Response(400, json={"error": "Bad Gateway"})

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError) as excinfo:
        generate_answer("hi", api_key="test-key")

    assert not isinstance(excinfo.value, GeminiAuthError)
    assert "Bad Gateway" in str(excinfo.value)


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.PoolTimeout("pool timed out"),
    ],
    ids=["read", "connect", "pool"],
)
def test_any_timeout_exception_raises_gemini_timeout(exc, monkeypatch):
    def handler(*a, **k):
        raise exc

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiTimeoutError):
        generate_answer("hi", api_key="test-key", timeout=1.0)


def test_connection_error_raises_generic_gemini_error(monkeypatch):
    # Not a timeout — e.g. DNS failure or refused connection — falls back
    # to the generic bucket (see module docstring for why).
    def handler(*a, **k):
        raise httpx.ConnectError("Connection refused")

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError) as excinfo:
        generate_answer("hi", api_key="test-key")

    assert not isinstance(excinfo.value, GeminiTimeoutError)
    assert not isinstance(excinfo.value, GeminiAuthError)


def test_other_http_error_raises_generic_gemini_error(monkeypatch):
    def handler(url, json, headers, timeout):
        return httpx.Response(500, text="internal error")

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError) as excinfo:
        generate_answer("hi", api_key="test-key")

    assert "500" in str(excinfo.value)


def test_non_json_body_raises_gemini_error(monkeypatch):
    def handler(url, json, headers, timeout):
        return httpx.Response(200, text="<html>oops</html>")

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError):
        generate_answer("hi", api_key="test-key")


def test_no_candidates_raises_gemini_error(monkeypatch):
    # e.g. the prompt was blocked by Gemini's safety filters.
    def handler(url, json, headers, timeout):
        return httpx.Response(
            200,
            json={"promptFeedback": {"blockReason": "SAFETY"}},
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError) as excinfo:
        generate_answer("hi", api_key="test-key")

    assert "SAFETY" in str(excinfo.value)


def test_candidate_with_no_text_raises_gemini_error(monkeypatch):
    def handler(url, json, headers, timeout):
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": []}}]}
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError):
        generate_answer("hi", api_key="test-key")


def test_candidate_with_explicit_null_content_raises_gemini_error(monkeypatch):
    # A realistic shape for a blocked/truncated candidate: "content" is
    # present but explicitly null, not merely absent.
    def handler(url, json, headers, timeout):
        return httpx.Response(
            200,
            json={"candidates": [{"finishReason": "SAFETY", "content": None}]},
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError):
        generate_answer("hi", api_key="test-key")


def test_candidate_with_explicit_null_part_text_raises_gemini_error(monkeypatch):
    def handler(url, json, headers, timeout):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": None}]}}]},
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(GeminiError):
        generate_answer("hi", api_key="test-key")


# ---------------------------------------------------------------------------
# Real-Gemini integration. Opt-in via `-m gemini`; skips if no API key.
# ---------------------------------------------------------------------------


@pytest.fixture
def gemini_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set in the environment")
    return key


@pytest.mark.gemini
def test_generate_answer_against_real_gemini(gemini_api_key):
    answer = generate_answer(
        "Reply with exactly one lowercase word: the colour of a clear "
        "daytime sky.",
        api_key=gemini_api_key,
        timeout=30,
    )

    assert isinstance(answer, str)
    assert answer.strip()
    assert "blue" in answer.lower()


@pytest.mark.gemini
def test_real_gemini_reports_invalid_key_clearly():
    with pytest.raises(GeminiAuthError):
        generate_answer("hi", api_key="definitely-not-a-real-key", timeout=30)


# ---------------------------------------------------------------------------
# End-to-end interchangeability with answer_with_llm, against a real
# VectorStore (real embeddings) and the real Gemini API — proves
# gemini.generate_answer is a genuine drop-in for llm.generate_answer.
# ---------------------------------------------------------------------------


@pytest.mark.model
@pytest.mark.gemini
def test_answer_with_llm_accepts_gemini_generate_as_a_drop_in(tmp_path, gemini_api_key):
    store = VectorStore(tmp_path)
    store.add_documents(
        [
            "The Willowbrook Lighthouse was built in 1887 by the architect "
            "Edwin Ashcroft.",
        ],
        source="lighthouse.md",
    )

    result = answer_with_llm(
        "When was the Willowbrook Lighthouse built?",
        store,
        generate=generate_answer,
        model=DEFAULT_MODEL,
    )

    assert result["has_context"] is True
    assert "1887" in result["answer"]
