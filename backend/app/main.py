"""FastAPI entrypoint: app creation, health check, route wiring, CORS."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import LLMProviderConfigError, router

# Load backend/.env if present (uvicorn doesn't do this itself), so the
# vars documented in .env.example actually take effect.
load_dotenv()

# setdefault, not a plain assignment: an operator-set value (Render
# dashboard env var, or the .env just loaded above) always wins over this
# hardcoded default. Must run AFTER load_dotenv(), not before:
# load_dotenv()'s default `override=False` only fills keys still absent
# from os.environ, so claiming these two first (an earlier version of
# this file did) makes it silently ignore a value set in .env — verified
# in isolation (test_health.py), though not currently reachable through
# this app specifically: app.retrieval.gemini independently calls
# load_dotenv() too (so it also works standalone, e.g. in scripts), which
# means .env is already loaded as a side effect of `from app.api.routes
# import ...` above, before either ordering of the two lines below would
# matter. This fix removes reliance on that unrelated module's side
# effect for correctness rather than leaving it as the only thing making
# this work. No ordering requirement relative to the imports above:
# nothing in this module (or anything it imports) touches
# onnxruntime/tokenizers at import time — embeddings.py loads the model
# lazily, on the first real request — so being anywhere before that
# first request is enough.
#
# Caps onnxruntime/tokenizers' internal thread pools for a small,
# low-vCPU deploy target (e.g. Render's 512MB Starter plan) — see
# .env.example for what this does and, importantly, does NOT do (it does
# not reduce peak memory; see PLANNING.md for the measurements behind
# that and the separate change — embeddings.py's ONNX runtime — that
# actually brought memory usage under budget).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The frontend (Next.js dev server) runs on a different origin, so the
# browser needs these allowed explicitly. Override for other hosts with
# FRONTEND_ORIGINS (comma-separated). `or` (not the getenv default) so a
# blank value still falls back rather than allowing nothing.
_DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in (os.getenv("FRONTEND_ORIGINS") or _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI(title="RAG Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


# get_llm() (a FastAPI dependency, resolved before any handler's own
# try/except runs) raises this for an unrecognized LLM_PROVIDER. Without
# this handler, FastAPI's default unhandled-exception path would still
# return a 500 but discard the message, as a bare "Internal Server
# Error" — this keeps that detail and matches the {"detail": ...} shape
# every other error in the API already returns (see LLMProviderConfigError's
# docstring).
@app.exception_handler(LLMProviderConfigError)
async def llm_provider_config_error_handler(
    request: Request, exc: LLMProviderConfigError
) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
