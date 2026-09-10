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

- `app/page.tsx` — chat orchestration: `messages` / `input` / `loading` state,
  `handleSend`, "New conversation", page layout
- `components/` — `DocumentUpload` (owns its own upload state), `MessageBubble`,
  `Citations`, `ChatComposer`; `components/ui/` holds the shadcn/ui primitives
- `lib/api.ts` — the network boundary: typed `ingestDocument()` / `askQuestion()`
  and a normalized `ApiError` (unchanged by the UI refactor)
- `lib/chat.ts` — the `Message` view type and the `errorText` formatter
- `app/globals.css` — Tailwind v4 + shadcn tokens; dark mode via
  `prefers-color-scheme` (no toggle)

## UI

shadcn/ui (`radix-nova` style, `neutral` base). Minimal "developer tool" look,
no custom branding. Dark mode follows the OS setting.
