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
