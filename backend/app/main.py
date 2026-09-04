"""FastAPI entrypoint.

Kept minimal for now: just app creation + a health check so we can verify
the server boots and is reachable. Ingestion/retrieval routes will be wired
in via app/api/routes.py once those modules exist (see PLANNING.md).
"""

from fastapi import FastAPI

app = FastAPI(title="RAG Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
