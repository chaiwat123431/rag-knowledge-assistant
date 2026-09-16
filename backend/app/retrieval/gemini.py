"""LLM transport — send a prompt to Google's Gemini API, get text back.

Same contract as `app.retrieval.llm.generate_answer` (Ollama, dev): a
``generate_answer(prompt, model=...) -> str`` callable, deliberately narrow
and RAG-agnostic, so it's a drop-in for `answer_with_llm(..., generate=...)`
in production. See PLANNING.md's Architecture Decisions: Ollama locally for
dev, Gemini Flash (free tier) for prod.

Direct HTTP via `httpx`, not the official `google-generativeai` SDK — same
reasoning PLANNING.md already gives for Ollama: one narrow function over a
stable REST endpoint is easier to reason about and mock deterministically
(a constructed `httpx.Response`, exactly like `test_llm.py`) than an SDK
client object, and it avoids pulling in the SDK's grpc/protobuf/google-auth
dependency chain for a single `generateContent` call.

Gemini's generateContent API (``POST .../v1beta/models/{model}:generateContent``,
API key in the ``x-goog-api-key`` header, never the URL) returns
``{"candidates": [{"content": {"parts": [{"text": "..."}]}}], ...}`` on
success — richer than Ollama's flat ``{"response": "..."}`` because a
candidate can also come back empty (blocked by Gemini's safety filters),
which has no Ollama equivalent and is treated as a `GeminiError`.

Errors: `GeminiError` subclasses `app.retrieval.llm.LLMError`, not a
separate hierarchy — `answer_with_llm`'s documented "Raises: LLMError (and
subclasses)" then stays true regardless of which backend `generate` is
bound to, so callers don't need to catch two unrelated exception families
depending on environment. Only three categories are distinguished (missing/
invalid API key, timeout, everything else) — unlike `llm.py` there's no
separate "unavailable" class for connection-level failures: Ollama's
distinguishes that case because "start the local server" is an actionable
next step, which has no equivalent for a cloud API, so those fold into the
generic `GeminiError` instead.
"""

import os

import httpx
from dotenv import load_dotenv

from app.retrieval.llm import LLMError, _require_nonempty_prompt

# Mirrors main.py's own load_dotenv() call, so this module also works
# standalone (scripts, tests) without depending on main having run first.
load_dotenv()

API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# "gemini-2.0-flash" (the model this module originally shipped with) is no
# longer served by the real API as of 2026-09 — a live call now returns
# HTTP 404 recommending "gemini-3.6-flash" by name. Verified directly
# against the real API (`pytest -m gemini`), not assumed.
DEFAULT_MODEL = "gemini-3.6-flash"
# Cloud inference, no local model to load — a much shorter default than
# Ollama's 120s is enough headroom for real API latency.
DEFAULT_TIMEOUT = 30.0


class GeminiError(LLMError):
    """Base for every failure of a Gemini call."""


class GeminiAuthError(GeminiError):
    """The API key is missing, empty, or rejected by Gemini."""


class GeminiTimeoutError(GeminiError):
    """Gemini accepted the connection but didn't answer in time."""


def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Send `prompt` to Gemini and return the generated text.

    Args:
        prompt: the full prompt to generate from.
        model: Gemini model name.
        api_key: Gemini API key. Defaults to the `GEMINI_API_KEY`
            environment variable (loaded via `.env` through python-dotenv)
            when not given explicitly.
        timeout: read timeout in seconds (connect is fixed short at 5s).

    Returns:
        The model's response text, stripped.

    Raises:
        ValueError: if `prompt` is not a non-empty string.
        GeminiAuthError: no API key is available, or Gemini rejected it.
        GeminiTimeoutError: the connection was made but no response came
            within `timeout`.
        GeminiError: any other failure (network error, non-2xx response,
            unparseable body, or a candidate with no usable text — e.g.
            blocked by Gemini's safety filters).
    """
    _require_nonempty_prompt(prompt)

    key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    if not key or not key.strip():
        raise GeminiAuthError(
            "No Gemini API key provided. Pass api_key explicitly or set "
            "GEMINI_API_KEY in the environment (e.g. in backend/.env)."
        )

    url = f"{API_BASE_URL}/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"x-goog-api-key": key}

    try:
        response = httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
    except httpx.TimeoutException as exc:
        # Covers connect and read timeouts alike — see module docstring
        # for why Gemini doesn't get Ollama's connect/read distinction.
        raise GeminiTimeoutError(
            f"Gemini did not respond within {timeout}s (model {model!r})."
        ) from exc
    except httpx.RequestError as exc:
        raise GeminiError(f"Could not reach Gemini ({exc!s}).") from exc

    if response.status_code >= 400:
        # Parsed once and threaded through both helpers below, instead of
        # each re-parsing the same response body from scratch.
        error_body = _try_json(response)
        if _is_auth_error(response.status_code, error_body):
            raise GeminiAuthError(
                f"Gemini rejected the API key "
                f"({_error_detail(response, error_body)})."
            )
        raise GeminiError(
            f"Gemini returned HTTP {response.status_code}: "
            f"{_error_detail(response, error_body)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise GeminiError(
            f"Gemini returned a non-JSON response: {response.text[:200]!r}"
        ) from exc

    candidates = data.get("candidates")
    if not candidates:
        # e.g. the prompt was blocked by safety filters — no Ollama
        # equivalent, but a real Gemini failure mode.
        raise GeminiError(
            f"Gemini returned no candidates "
            f"(promptFeedback={data.get('promptFeedback')!r}): {data!r}"
        )

    # `or {}` / `or []` / `or ""`, not `.get(key, default)`, throughout:
    # a key can be explicitly `null` (e.g. content of a safety-blocked
    # candidate) as well as absent, and `.get(key, default)` only
    # substitutes the default for the latter.
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text") or "" for part in parts).strip()
    if not text:
        raise GeminiError(f"Gemini candidate had no text: {candidates[0]!r}")

    return text


def _try_json(response: httpx.Response):
    """Parse `response`'s body as JSON, or None if it isn't valid JSON."""
    try:
        return response.json()
    except ValueError:
        return None


def _is_auth_error(status_code: int, body) -> bool:
    """Whether an error response reports a missing/invalid/unauthorized key.

    `body` is the response's already-parsed JSON (see `_try_json`), or
    None if it wasn't valid JSON.
    """
    if status_code in (401, 403):
        return True
    if status_code == 400 and isinstance(body, dict):
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        return "api key" in str(message or "").lower()
    return False


def _error_detail(response: httpx.Response, body=None) -> str:
    """Best-effort human-readable detail from an error response.

    `body` is the response's already-parsed JSON (see `_try_json`); pass
    None (the default) to fall back to the raw response text, which also
    covers a body that parsed but wasn't a dict.
    """
    if not isinstance(body, dict):
        return response.text[:200]
    if isinstance(body.get("error"), dict):
        return str(body["error"].get("message", body["error"]))
    if "error" in body:
        return str(body["error"])
    return str(body)[:200]
