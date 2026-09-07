import httpx
import pytest

import app.retrieval.llm as llm
from app.retrieval.llm import (
    DEFAULT_MODEL,
    LLMError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    generate_answer,
)

# ---------------------------------------------------------------------------
# Mocked-transport tests: fast, deterministic, no real Ollama.
# `generate_answer` calls httpx.post; we replace it per-test.
# ---------------------------------------------------------------------------


def _patch_post(monkeypatch, handler):
    """Replace llm.httpx.post with `handler(url, json=..., timeout=...)`."""
    monkeypatch.setattr(llm.httpx, "post", handler)


def test_generate_answer_returns_response_text(monkeypatch):
    captured = {}

    def handler(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"response": "  four  ", "done": True})

    _patch_post(monkeypatch, handler)

    answer = generate_answer("What is 2 + 2?", model="llama3.2")

    assert answer == "four"  # stripped
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"] == {
        "model": "llama3.2",
        "prompt": "What is 2 + 2?",
        "stream": False,
    }


def test_generate_answer_honours_custom_base_url_and_model(monkeypatch):
    seen = {}

    def handler(url, json, timeout):
        seen["url"] = url
        seen["model"] = json["model"]
        return httpx.Response(200, json={"response": "ok"})

    _patch_post(monkeypatch, handler)

    generate_answer("hi", model="mistral", base_url="http://box:9999")

    assert seen["url"] == "http://box:9999/api/generate"
    assert seen["model"] == "mistral"


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\n\t"])
def test_generate_answer_rejects_empty_prompt(bad_prompt, monkeypatch):
    def handler(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("httpx.post should not be called for a bad prompt")

    _patch_post(monkeypatch, handler)

    with pytest.raises(ValueError):
        generate_answer(bad_prompt)


def test_connection_refused_raises_ollama_unavailable(monkeypatch):
    def handler(*a, **k):
        raise httpx.ConnectError("Connection refused")

    _patch_post(monkeypatch, handler)

    with pytest.raises(OllamaUnavailableError) as excinfo:
        generate_answer("hi")

    assert "ollama serve" in str(excinfo.value).lower()


def test_read_timeout_raises_ollama_timeout(monkeypatch):
    def handler(*a, **k):
        raise httpx.ReadTimeout("timed out")

    _patch_post(monkeypatch, handler)

    with pytest.raises(OllamaTimeoutError):
        generate_answer("hi", timeout=1.0)


def test_connect_timeout_is_unavailable_not_timeout(monkeypatch):
    # ConnectTimeout subclasses both TimeoutException and ConnectError; a
    # server we can't even connect to should fail fast as "unavailable".
    def handler(*a, **k):
        raise httpx.ConnectTimeout("connect timed out")

    _patch_post(monkeypatch, handler)

    with pytest.raises(OllamaUnavailableError):
        generate_answer("hi")


@pytest.mark.parametrize("bad_prompt", [None, 123, ["list"], {"d": 1}])
def test_non_string_prompt_raises_value_error(bad_prompt, monkeypatch):
    _patch_post(
        monkeypatch,
        lambda *a, **k: pytest.fail("httpx.post should not be called"),
    )

    with pytest.raises(ValueError):
        generate_answer(bad_prompt)


def test_trailing_slash_in_base_url_is_normalised(monkeypatch):
    seen = {}

    def handler(url, json, timeout):
        seen["url"] = url
        return httpx.Response(200, json={"response": "ok"})

    _patch_post(monkeypatch, handler)

    generate_answer("hi", base_url="http://localhost:11434/")

    assert seen["url"] == "http://localhost:11434/api/generate"


def test_404_not_about_a_model_raises_generic_llm_error(monkeypatch):
    # e.g. a wrong base_url path — must not be reported as "pull the model".
    def handler(url, json, timeout):
        return httpx.Response(404, json={"error": "404 page not found"})

    _patch_post(monkeypatch, handler)

    with pytest.raises(LLMError) as excinfo:
        generate_answer("hi")

    assert not isinstance(excinfo.value, OllamaModelNotFoundError)
    assert "base_url" in str(excinfo.value)


def test_missing_model_raises_model_not_found(monkeypatch):
    def handler(url, json, timeout):
        return httpx.Response(404, json={"error": "model 'ghost' not found"})

    _patch_post(monkeypatch, handler)

    with pytest.raises(OllamaModelNotFoundError) as excinfo:
        generate_answer("hi", model="ghost")

    msg = str(excinfo.value)
    assert "ghost" in msg
    assert "ollama pull ghost" in msg


def test_other_http_error_raises_generic_llm_error(monkeypatch):
    def handler(url, json, timeout):
        return httpx.Response(500, text="internal error")

    _patch_post(monkeypatch, handler)

    with pytest.raises(LLMError) as excinfo:
        generate_answer("hi")

    assert "500" in str(excinfo.value)


def test_non_json_body_raises_llm_error(monkeypatch):
    def handler(url, json, timeout):
        return httpx.Response(200, text="<html>oops</html>")

    _patch_post(monkeypatch, handler)

    with pytest.raises(LLMError):
        generate_answer("hi")


def test_json_without_response_field_raises_llm_error(monkeypatch):
    def handler(url, json, timeout):
        return httpx.Response(200, json={"done": True})

    _patch_post(monkeypatch, handler)

    with pytest.raises(LLMError):
        generate_answer("hi")


# ---------------------------------------------------------------------------
# Real-Ollama integration. Opt-in via `-m ollama`; skips if unavailable.
# ---------------------------------------------------------------------------

OLLAMA_TEST_MODEL = "llama3.2"


@pytest.fixture
def ollama_model():
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError):
        pytest.skip("Ollama not reachable on localhost:11434")

    names = {m.get("name", "") for m in response.json().get("models", [])}
    if not any(
        n == OLLAMA_TEST_MODEL or n.startswith(f"{OLLAMA_TEST_MODEL}:")
        for n in names
    ):
        pytest.skip(f"Ollama model {OLLAMA_TEST_MODEL!r} is not pulled")
    return OLLAMA_TEST_MODEL


@pytest.mark.ollama
def test_generate_answer_against_real_ollama(ollama_model):
    answer = generate_answer(
        "Reply with exactly one lowercase word: the colour of a clear "
        "daytime sky.",
        model=ollama_model,
        timeout=60,
    )

    assert isinstance(answer, str)
    assert answer.strip()
    assert "blue" in answer.lower()


@pytest.mark.ollama
def test_real_ollama_reports_missing_model_clearly(ollama_model):
    with pytest.raises(OllamaModelNotFoundError):
        generate_answer("hi", model="definitely-not-a-real-model:tag", timeout=30)
