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
        base_url: Ollama server base URL.
        timeout: read timeout in seconds (connect is fixed short at 5s).

    Returns:
        The model's response text, stripped.

    Raises:
        ValueError: if `prompt` is empty or whitespace-only.
        OllamaUnavailableError: the server couldn't be reached.
        OllamaTimeoutError: the server didn't respond within `timeout`.
        OllamaModelNotFoundError: `model` isn't available on the server.
        LLMError: any other non-success response.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    url = f"{base_url}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
    except httpx.TimeoutException as exc:
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
        raise OllamaModelNotFoundError(
            f"Ollama model {model!r} is not available ({_error_detail(response)}). "
            f"Pull it with `ollama pull {model}`."
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
