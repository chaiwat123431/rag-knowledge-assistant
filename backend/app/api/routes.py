"""HTTP endpoints: ingest documents, ask questions.

Thin layer over the retrieval modules — parse/chunk/store for ingestion,
`answer_with_llm` for queries. All the real work lives in `app.ingestion`
and `app.retrieval`; this module maps it to HTTP and turns the *known*
domain errors into deliberate status codes:

- 400 for bad input (empty question, unreadable/unsupported/empty file)
- 503 when a backend is *transiently* unavailable — Ollama not running or
  timing out, or the embedding model can't be fetched on first ingest;
  retrying later may succeed
- 500 for a non-retryable backend misconfiguration — Ollama is up but the
  requested model was never pulled; a human has to run `ollama pull`

Genuinely unexpected failures still surface as 500 — that's correct.

Dependencies
------------
`get_vector_store` is an `lru_cache`d factory: one `VectorStore` for the
process, shared by every request. `get_llm` returns the callable used to
generate answers (`llm.generate_answer` by default). Both are FastAPI
dependencies so tests can override them (`app.dependency_overrides`) to
point at a temp Chroma dir and a stub LLM.

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

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.ingestion.chunker import chunk_text
from app.ingestion.parser import (
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
    extract_text,
)
from app.retrieval.llm import (
    LLMError,
    OllamaModelNotFoundError,
    generate_answer,
)
from app.retrieval.query_flow import answer_with_llm
from app.retrieval.vector_store import VectorStore

# Kept in sync with the parser by importing its constants rather than
# re-listing extensions here.
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS

router = APIRouter()


@lru_cache
def get_vector_store() -> VectorStore:
    """The process-wide vector store (default on-disk Chroma dir)."""
    return VectorStore()


def get_llm() -> Callable[..., str]:
    """The callable used to generate an answer from a prompt."""
    return generate_answer


# --- request / response models ---------------------------------------------


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)


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
        400: {"description": "Empty question"},
        500: {"description": "LLM misconfigured (model not pulled)"},
        503: {"description": "LLM backend unavailable / timed out"},
    },
)
def query(
    request: QueryRequest,
    store: VectorStore = Depends(get_vector_store),
    generate: Callable[..., str] = Depends(get_llm),
) -> QueryResponse:
    # Validate the client input here so the only ValueError that can reach
    # the `except` below is a genuine internal fault (which should 500, not
    # be relabelled 400). top_k range is enforced by the model (-> 422).
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        result = answer_with_llm(
            request.question, store, top_k=request.top_k, generate=generate
        )
    except OllamaModelNotFoundError as exc:
        # Ollama is up but the model was never pulled — not transient,
        # retrying won't help; an operator must fix it.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMError as exc:
        # Ollama unreachable or timed out — a transient backend problem,
        # not a client error; retrying later may work.
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
    except OSError as exc:
        # e.g. the embedding model can't be fetched on first ingest, or a
        # Chroma write fails — backend not ready, not the client's fault.
        raise HTTPException(
            status_code=503,
            detail=f"could not index document (backend unavailable): {exc}",
        ) from exc

    return IngestResponse(source=file.filename, chunks_added=len(chunks))
