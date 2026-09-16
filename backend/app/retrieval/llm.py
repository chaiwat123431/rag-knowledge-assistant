"""LLM transport — send a prompt to a local Ollama server, get text back.

Deliberately narrow: this module knows nothing about RAG, retrieval or
citations. It's just the HTTP call plus clear, typed errors, so it can be
swapped for a Gemini client (same ``(prompt) -> str`` shape) in production
without touching anything upstream. See PLANNING.md's Architecture
Decisions: Ollama locally for dev, Gemini Flash for prod.

Ollama's generate API (``POST /api/generate`` with ``stream: false``)
returns a single JSON object ``{"response": "...", "done": true, ...}``.
"""

import httpx

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
# Generation on CPU can take a while, especially the first call that loads
# the model into memory — hence a long read timeout but a short connect
# one (a down server should fail fast, not hang).
DEFAULT_TIMEOUT = 120.0


class LLMError(RuntimeError):
    """Base for every failure of an LLM call."""


class OllamaUnavailableError(LLMError):
    """Ollama couldn't be reached (not running, wrong host, DNS, ...)."""


class OllamaTimeoutError(LLMError):
    """Ollama accepted the connection but didn't answer in time."""


class OllamaModelNotFoundError(LLMError):
    """The requested model isn't pulled on the Ollama server."""


def _require_nonempty_prompt(prompt) -> None:
    """Raise ValueError unless `prompt` is a non-empty string.

    Shared with `gemini.py` (imported from here, not copy-pasted) so both
    backends of the same `generate_answer(prompt, model=...) -> str`
    contract validate their input identically.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")


def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Send `prompt` to Ollama and return the generated text.

    Args:
        prompt: the full prompt to generate from.
        model: Ollama model name (must already be pulled).
        base_url: Ollama server base URL. A trailing slash is tolerated.
        timeout: read timeout in seconds (connect is fixed short at 5s).

    Returns:
        The model's response text, stripped.

    Raises:
        ValueError: if `prompt` is not a non-empty string.
        OllamaUnavailableError: the server couldn't be reached (including a
            connect timeout — a down server should fail fast).
        OllamaTimeoutError: the connection was made but no response came
            within `timeout` (e.g. the model is still loading).
        OllamaModelNotFoundError: `model` isn't available on the server.
        LLMError: any other non-success response.
    """
    _require_nonempty_prompt(prompt)

    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        # Can't establish a connection at all (refused, or no SYN-ACK
        # within the 5s connect deadline) — the server is unreachable,
        # not merely slow. ConnectTimeout is a TimeoutException, so it
        # must be caught before the generic timeout branch below.
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {base_url} ({exc!s}). Is it running? "
            f"Start it with `ollama serve`."
        ) from exc
    except httpx.TimeoutException as exc:
        # Connected, but no response in time (read/write/pool timeout) —
        # e.g. the model is still loading.
        raise OllamaTimeoutError(
            f"Ollama at {base_url} did not respond within {timeout}s "
            f"(model {model!r}). It may still be loading the model — retry, "
            f"or raise the timeout."
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {base_url} ({exc!s}). Is it running? "
            f"Start it with `ollama serve`."
        ) from exc

    if response.status_code == 404:
        detail = _error_detail(response)
        if "model" in detail.lower():
            raise OllamaModelNotFoundError(
                f"Ollama model {model!r} is not available ({detail}). "
                f"Pull it with `ollama pull {model}`."
            )
        # 404 that isn't about the model — most likely a wrong base_url.
        raise LLMError(
            f"Ollama returned HTTP 404 for {url} ({detail}). Check base_url."
        )
    if response.status_code >= 400:
        raise LLMError(
            f"Ollama returned HTTP {response.status_code}: "
            f"{_error_detail(response)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError(
            f"Ollama returned a non-JSON response: {response.text[:200]!r}"
        ) from exc

    answer = data.get("response")
    if answer is None:
        raise LLMError(f"Ollama response had no 'response' field: {data!r}")

    return answer.strip()


def _error_detail(response: httpx.Response) -> str:
    """Best-effort human-readable detail from an error response."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "error" in body:
        return str(body["error"])
    return str(body)[:200]
