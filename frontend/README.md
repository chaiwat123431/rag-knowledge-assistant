# RAG Assistant — frontend

Minimal Next.js (App Router) + TypeScript + Tailwind UI for the backend in
`../backend`. One page: upload a document, then ask questions about it with
cited answers and conversation follow-up.

## Prerequisites

The frontend is just a client for the backend, so you need all three running:

1. **Ollama** with the `llama3.2` model — `ollama serve` and `ollama pull llama3.2`
2. **Backend** — from `../backend`:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
3. **Frontend** — from here:
   ```bash
   npm install
   npm run dev
   ```
   Open http://localhost:3000.

The backend allows `http://localhost:3000` via CORS by default (override with
`FRONTEND_ORIGINS` in the backend env).

## Configuration

`NEXT_PUBLIC_API_BASE_URL` sets the backend URL (default `http://localhost:8000`).
Copy `.env.local.example` to `.env.local` to change it.

## Structure

- `app/page.tsx` — the whole UI and its state (single-file scaffold on purpose)
- `lib/api.ts` — the network boundary: typed `ingestDocument()` / `askQuestion()`
  and a normalized `ApiError`
- `app/layout.tsx` — root layout / metadata
