"""HTTP endpoints: ingest documents, ask questions.

Thin layer over the retrieval modules — parse/chunk/store for ingestion,
`answer_with_llm` for queries. All the real work lives in `app.ingestion`
and `app.retrieval`; this module only maps it to HTTP and turns the
domain errors into the right status codes (400 for bad input, 503 when
the LLM backend is down — never a bare 500).

Dependencies
------------
`get_vector_store` is an `lru_cache`d factory: one `VectorStore` for the
process, shared by every request. `get_llm` returns the callable used to
generate answers (`llm.generate_answer` by default). Both are FastAPI
dependencies so tests can override them (`app.dependency_overrides`) to
point at a temp Chroma dir and a stub LLM.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestion.chunker import chunk_text
from app.ingestion.parser import extract_text
from app.retrieval.llm import LLMError, generate_answer
from app.retrieval.query_flow import answer_with_llm
from app.retrieval.vector_store import VectorStore

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

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
    top_k: int = 5


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
        400: {"description": "Empty question or top_k < 1"},
        503: {"description": "LLM backend unavailable / timed out"},
    },
)
def query(
    request: QueryRequest,
    store: VectorStore = Depends(get_vector_store),
    generate: Callable[..., str] = Depends(get_llm),
) -> QueryResponse:
    try:
        result = answer_with_llm(
            request.question, store, top_k=request.top_k, generate=generate
        )
    except ValueError as exc:
        # empty question / top_k < 1
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        # Ollama down / timed out / model missing — a backend problem,
        # not a client error.
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
    # keeps the extension (the parser dispatches on it).
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
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

    store.add_documents(chunks, source=file.filename)
    return IngestResponse(source=file.filename, chunks_added=len(chunks))
