"""FastAPI entrypoint: app creation, health check, and route wiring."""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="RAG Assistant")

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
