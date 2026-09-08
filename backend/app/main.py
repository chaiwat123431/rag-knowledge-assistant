"""FastAPI entrypoint: app creation, health check, route wiring, CORS."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# The frontend (Next.js dev server) runs on a different origin, so the
# browser needs these allowed explicitly. Override for other hosts with
# FRONTEND_ORIGINS (comma-separated).
_DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI(title="RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
