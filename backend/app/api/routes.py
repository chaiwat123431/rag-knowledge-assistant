"""HTTP endpoints: ingest documents, ask questions.

Thin layer over the retrieval modules — parse/chunk/store for ingestion,
`answer_with_llm` for queries. All the real work lives in `app.ingestion`
and `app.retrieval`; this module maps it to HTTP and turns the *known*
domain errors into deliberate status codes:

- 400 for bad input (empty question, unreadable/unsupported/empty file)
- 503 when a backend is *transiently* unavailable — Ollama not running or
  timing out, Gemini unreachable or erroring server-side, or the
  embedding model can't be fetched on first ingest; retrying later may
  succeed
- 500 for a non-retryable backend misconfiguration — Ollama is up but the
  requested model was never pulled (`OllamaModelNotFoundError`), or
  Gemini rejected a missing/invalid API key (`GeminiAuthError`); a human
  has to fix the underlying config (`ollama pull ...` / `GEMINI_API_KEY`)

Genuinely unexpected failures still surface as 500 — that's correct.

Dependencies
------------
`get_vector_store` is an `lru_cache`d factory: one `VectorStore` for the
process, shared by every request. `get_llm` returns the `(generate, model)`
pair to use — which backend (Ollama locally, Gemini in production) is
selected by the `LLM_PROVIDER` environment variable ("ollama", the
default, or "gemini"). It's a *pair*, not two separate dependencies for
`generate` and `model`, on purpose: `answer_with_llm`'s docstring warns
that swapping only `generate` without also passing a matching `model`
silently sends one backend's model name to the other — bundling them
into one dependency makes that impossible to get wrong at the call site.
Both dependencies are plain FastAPI dependencies so tests can override
them (`app.dependency_overrides`) to point at a temp Chroma dir and a
stub LLM.

Note: the handlers are sync (`def`), so each request holds a threadpool
worker for the duration of its blocking work (embedding, Chroma, the LLM
HTTP call — up to `llm.DEFAULT_TIMEOUT`). Fine for the single-user local
target; a multi-user deployment would want async handlers + a streaming
LLM call.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable

import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.ingestion.chunker import chunk_text
from app.ingestion.parser import (
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
    extract_text,
)
from app.retrieval import gemini
from app.retrieval import llm as ollama_llm
from app.retrieval.gemini import GeminiAuthError
from app.retrieval.llm import LLMError, OllamaModelNotFoundError
from app.retrieval.query_flow import InvalidHistoryError, answer_with_llm
from app.retrieval.vector_store import VectorStore

# Kept in sync with the parser by importing its constants rather than
# re-listing extensions here.
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

router = APIRouter()


@lru_cache
def get_vector_store() -> VectorStore:
    """The process-wide vector store (default on-disk Chroma dir)."""
    return VectorStore()


# provider name -> (generate callable, its default model). The two always
# travel together — see the module docstring's "Dependencies" section.
_LLM_PROVIDERS: dict[str, tuple[Callable[..., str], str]] = {
    "ollama": (ollama_llm.generate_answer, ollama_llm.DEFAULT_MODEL),
    "gemini": (gemini.generate_answer, gemini.DEFAULT_MODEL),
}


class LLMProviderConfigError(RuntimeError):
    """`LLM_PROVIDER` is set to something other than a known provider.

    Raised by `get_llm()`, a FastAPI dependency — which means it's raised
    during dependency *resolution*, before any handler body (and its own
    try/except) runs. A plain exception raised there would still surface
    as a 500 (FastAPI's default for an unhandled error), but as a bare
    "Internal Server Error" with the carefully-built message below
    discarded. This dedicated type exists so `main.py` can register an
    `app.exception_handler` for it specifically and return that message
    as the response's `detail`, matching every other error in this
    module. A subclass of RuntimeError so `except RuntimeError` still
    catches it too.
    """


def get_llm() -> tuple[Callable[..., str], str]:
    """The `(generate, model)` pair to use, selected by `LLM_PROVIDER`
    ("ollama", the default, or "gemini"; matched case-insensitively,
    surrounding whitespace ignored).

    Deliberately not cached: resolving it is a cheap dict lookup, and
    always re-reading the environment keeps this straightforward to test
    (`monkeypatch.setenv`) without a cache to clear between cases.

    Raises:
        LLMProviderConfigError: `LLM_PROVIDER` is set to something other
            than a known provider name. A misconfiguration, not a
            transient failure — see that class's docstring for how its
            detail message reaches the client despite being raised
            outside `query()`'s own try/except.
    """
    # Strip *before* falling back to "ollama": a whitespace-only value
    # (e.g. "   ") is truthy and would otherwise skip the fallback, then
    # get stripped down to "" and rejected as an unknown provider —
    # inconsistent with the plain-empty-string case, which is meant to
    # behave like unset.
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower() or "ollama"
    try:
        return _LLM_PROVIDERS[provider]
    except KeyError:
        raise LLMProviderConfigError(
            f"Unknown LLM_PROVIDER {provider!r}. Must be one of "
            f"{sorted(_LLM_PROVIDERS)}."
        ) from None


# --- request / response models ---------------------------------------------


class HistoryMessage(BaseModel):
    # Both fields are optional at the schema level on purpose: a missing /
    # null / wrongly-typed role or content should surface as our own 400
    # (from _validate_history), not a 422, per the "malformed history ->
    # 400" contract. Only a structurally wrong `history` (not a list of
    # objects) is left to pydantic (422).
    role: str | None = None
    content: str | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    history: list[HistoryMessage] | None = None


class Citation(BaseModel):
    marker: int
    source: str | None = None
    chunk_index: int | None = None
    text: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    has_context: bool
    citations: list[Citation]


class IngestResponse(BaseModel):
    source: str
    chunks_added: int


# --- endpoints ------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"description": "Empty question or malformed history"},
        500: {
            "description": (
                "LLM misconfigured (Ollama model not pulled, or invalid/"
                "missing Gemini API key)"
            )
        },
        503: {"description": "LLM backend unavailable / timed out"},
    },
)
def query(
    request: QueryRequest,
    store: VectorStore = Depends(get_vector_store),
    llm: tuple[Callable[..., str], str] = Depends(get_llm),
) -> QueryResponse:
    # Validate the client input here so the only ValueError that can reach
    # the `except` below is a genuine internal fault (which should 500, not
    # be relabelled 400). top_k range is enforced by the model (-> 422).
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    history = (
        [m.model_dump() for m in request.history] if request.history else None
    )
    generate, model = llm

    try:
        result = answer_with_llm(
            request.question,
            store,
            top_k=request.top_k,
            history=history,
            generate=generate,
            model=model,
        )
    except InvalidHistoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaModelNotFoundError as exc:
        # Ollama is up but the model was never pulled — not transient,
        # retrying won't help; an operator must fix it.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except GeminiAuthError as exc:
        # Missing or invalid GEMINI_API_KEY — same "non-retryable
        # misconfiguration" bucket as OllamaModelNotFoundError above, not
        # the generic (transient) LLMError case below. Must be caught
        # before it since GeminiAuthError is itself an LLMError subclass.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMError as exc:
        # Backend unreachable, timed out, or erroring server-side — a
        # transient problem, not a client error; retrying later may work.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(
        answer=result["answer"],
        has_context=result["has_context"],
        citations=result["citations"],
    )


@router.post(
    "/documents",
    response_model=IngestResponse,
    responses={
        400: {
            "description": (
                "Unsupported file type, unreadable file, or no extractable text"
            )
        },
    },
)
def ingest_document(
    file: UploadFile,
    store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    # UploadFile.filename is Optional; a part with no filename normally
    # fails FastAPI validation before reaching here, but guard against
    # Path(None) just in case.
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="uploaded file must have a filename"
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported file type '{suffix}'. "
                f"supported: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    # The parser works off a path, so land the upload in a temp file that
    # keeps the extension (the parser dispatches on it). Record the path
    # before writing so a failed write is still cleaned up.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(file.file.read())
        try:
            text = extract_text(tmp_path)
        except ValueError as exc:
            # decoding failure / corrupt PDF / unsupported type
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="no extractable text in the uploaded document",
        )

    try:
        store.add_documents(chunks, source=file.filename)
    except (OSError, httpx.HTTPError) as exc:
        # e.g. the embedding model can't be fetched on first ingest, or a
        # Chroma write fails — backend not ready, not the client's fault.
        # Both exception types matter: `httpx.HTTPError` (the ONNX model
        # download's own failure mode — chromadb's ONNXMiniLM_L6_V2 uses
        # httpx internally) is NOT an OSError subclass, unlike
        # `requests.exceptions.ConnectionError`, which the old
        # sentence-transformers/huggingface_hub download path raised and
        # this handler was originally written to catch. Verified directly
        # (`issubclass(httpx.ConnectError, OSError)` is False) rather than
        # assumed — a network hiccup during that download used to 500
        # bare and undiagnosed here before this fix.
        raise HTTPException(
            status_code=503,
            detail=f"could not index document (backend unavailable): {exc}",
        ) from exc

    return IngestResponse(source=file.filename, chunks_added=len(chunks))
