# Personal AI Knowledge Assistant — Planning

## Problem Statement
Ingest personal documents (PDFs, notes, text files) and answer natural-language
questions about them with cited sources, instead of manually searching through
files.

## Scope (Phase 1 MVP)
- Single-user, local-first (no auth, no multi-tenancy)
- Supported input formats: PDF, plain text, markdown
- Answers must cite which document/chunk they came from
- Conversational follow-up questions (basic memory of prior turns)

### Explicit Non-Goals (for MVP)
- No multi-user support
- No fine-tuning of embedding or LLM models
- No support for scanned/image-only PDFs (OCR) — plain text extraction only
- No production-grade auth/security — this is a portfolio project, not a
  product with real user data yet

## Architecture Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Backend | Python + FastAPI | Best ecosystem for RAG/LLM tooling (LangChain, embeddings libs) |
| Vector DB | Chroma (local) | No external service dependency during dev; easy to swap for Pinecone later if deploying at scale |
| RAG framework | LangChain | Industry-standard, widely referenced in job postings |
| Embeddings | `sentence-transformers` (local, e.g. `all-MiniLM-L6-v2`) | Runs locally, no API cost or key, good baseline quality for a single-user portfolio project. Was OpenAI `text-embedding-3-small`; switched to avoid paid API dependency. |
| LLM (dev) | Ollama (local) | Free local inference during development; no rate limits or keys while iterating |
| LLM (prod) | Google Gemini API — Flash model (free tier) | Free tier covers portfolio-scale usage; no local GPU needed on the deployment host. Replaces the earlier implicit OpenAI assumption. |
| LLM call — direct HTTP, not LangChain | `llm.generate_answer()` calls Ollama's `/api/generate` with `httpx` directly | One narrow function is easier to reason about and test than a LangChain LLM wrapper for a single call; `answer_with_llm` takes an injectable `generate` callable so the Gemini prod backend is a drop-in. `httpx` over `requests`: FastAPI-native, async-ready, constructible `Response` for deterministic test mocking. |
| Chunking | Recursive character splitter, ~500 tokens, 50 token overlap | Balances context completeness vs. retrieval precision. **Implemented as a custom word-boundary splitter instead** (character-based, no token count), not LangChain's `RecursiveCharacterTextSplitter` — revisit if paragraph/sentence-aware splitting turns out to matter once retrieval quality is measured. |
| Frontend | Next.js 16 (App Router) + TypeScript + Tailwind v4 + shadcn/ui | Phase 3. `create-next-app@latest` installed 16 (not the planned "14+"): Turbopack by default, Tailwind v4, React 19. shadcn/ui `radix-nova` style; native `fetch` + `useState` (no TanStack Query yet). Dark mode via `prefers-color-scheme`, no toggle. |

## Data Flow

```
Ingestion:
  Upload file -> Parse text -> Chunk text -> Generate embeddings -> Store in Chroma

Query:
  User question -> Embed question -> Retrieve top-k chunks from Chroma
  -> Build prompt with retrieved context -> LLM generates answer with citations
```

## Project Structure (planned)
```
rag-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── ingestion/
│   │   │   ├── parser.py        # PDF/text extraction [done]
│   │   │   └── chunker.py       # text splitting logic [done]
│   │   ├── retrieval/
│   │   │   ├── embeddings.py    # embedding generation [done]
│   │   │   ├── vector_store.py  # Chroma interface [done]
│   │   │   ├── query_flow.py    # retrieve -> prompt -> answer + citations [done]
│   │   │   └── llm.py           # Ollama HTTP client [done]
│   │   └── api/
│   │       └── routes.py        # HTTP endpoints (/query, /documents)
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/                    # Next.js 16 (App Router) + TS + Tailwind v4 [scaffold done]
│   ├── app/page.tsx             # upload + chat UI (single file)
│   └── lib/api.ts               # typed backend client
├── PLANNING.md
└── .gitignore
```

## Testing Strategy
- Unit tests for parser, chunker, embeddings, vector store — each in isolation
- Integration test: full ingest -> query -> answer flow on a small fixture document
- Tests written alongside each module, not deferred to a later phase

## Progress

- [x] Project scaffold (FastAPI app + health check) — `feature/project-scaffold`, PR #1
- [x] Ingestion pipeline (`parser.py`, `chunker.py` + tests) — `feature/ingestion-pipeline`, PR #2, merged 2026-09-05
- [x] Retrieval core (`embeddings.py` + `vector_store.py` + tests) — `feature/retrieval-core`, PR #3
  - Embeddings via local `sentence-transformers` (`all-MiniLM-L6-v2`) instead of the OpenAI API (see Architecture Decisions)
  - `VectorStore`: persistent Chroma collection (cosine), `add_documents(chunks, source)` / `query(question, top_k)` returning chunks with `source`/`chunk_index` metadata + similarity score
  - Model- and Chroma-loading tests marked `@pytest.mark.model` (deselect offline with `-m "not model"`)
- [x] Query flow (question -> retrieve -> prompt -> answer with citations) — `feature/query-flow` + `feature/llm-integration`
  - LLM: Ollama locally for dev, Gemini Flash (free tier) for prod/deploy
  - [x] Prompt + citation assembly (`query_flow.py` + tests) — `feature/query-flow`, PR #4
    - `answer_question(question, vector_store, top_k=5)` -> dict with `prompt` (numbered `[n]` source markers), `has_context`, `citations`
    - relevance floor (`MIN_RELEVANCE_SCORE`) drops off-topic chunks so an unrelated question falls back to a "no context" prompt
  - [x] LLM call (`llm.py` + `answer_with_llm` + tests) — `feature/llm-integration`, PR #5
    - `llm.generate_answer(prompt, model="llama3.2")` -> Ollama HTTP (`localhost:11434`), typed errors (unavailable / timeout / model-not-found)
    - `query_flow.answer_with_llm(...)` = retrieve -> prompt -> generate -> `{answer, citations, ...}`; `generate` injectable (Gemini swap-in for prod)
    - `httpx` (not `requests`): FastAPI-native, async-ready; new `@pytest.mark.ollama` marker for tests hitting a real local Ollama (skip if absent)
- [x] API routes (`api/routes.py` + tests) — `feature/api-routes`, PR #6
  - Reordered ahead of conversational memory: we need a callable endpoint (curl/Postman, later the frontend) before iterating on conversation state — the API shape makes it clearer where that state should live.
  - `POST /query` `{question, top_k?}` -> `{answer, has_context, citations}`; `POST /documents` (file upload) -> parse/chunk/`add_documents`
  - error mapping: empty question -> 400, out-of-range `top_k` -> 422, unsupported/unreadable/empty file -> 400, Ollama down/timeout & indexing-backend failure -> 503 (transient), `OllamaModelNotFoundError` -> 500 (non-retryable misconfig)
  - `VectorStore` + LLM callable injected via `lru_cache`d FastAPI dependencies, overridable in tests (`app.dependency_overrides`)
- [x] Conversational follow-up memory (`query_flow.py` + `routes.py` + tests) — `feature/conversational-memory`, PR #7
  - Stateless "client sends the transcript": `POST /query` takes optional `history: [{role, content}]`
  - Prompt: "Conversation so far:" block between instructions and Context; last `MAX_HISTORY_MESSAGES` (6) kept; our `[n]` markers stripped from prior assistant turns
  - Retrieval runs two queries (bare question + last-user-turn-augmented) and merges — a follow-up like "and the second part?" still retrieves the subject, a self-contained new question keeps its own chunks
  - `history=None`/`[]` is byte-identical to before; malformed history -> `InvalidHistoryError` -> 400
- [x] Relevance fixes from manual UI testing (`query_flow.py` + tests) — `feature/relevance-fixes`, PR #10
  - **Bug 1**: an unrelated new question after a substantive history topic returned `has_context=True` with stale citations — the augmented (history + question) query alone can't tell "genuine follow-up" from "history carryover", since concatenating any question onto a real prior topic still scores well against that topic's chunks (measured cosine ~0.52 for a totally unrelated question). Fixed with a 3rd reference-only query (prior turn alone): an augmented-only match is trusted only if the current question didn't erode its score against that reference by more than `MAX_AUGMENTED_SCORE_DROP` (0.15) — measured -0.19 to -0.28 drop for genuinely unrelated questions vs -0.03 to -0.07 for real follow-ups, a wide gap.
  - **Bug 2**: `MIN_RELEVANCE_SCORE` raised `0.15` -> `0.25` — measured across 6 documents x 6 questions, true positives land at 0.55-0.78, surface-noise false positives spike as high as 0.21; 0.15 sat inside that noise band.
  - Regression tests include the exact bug-1 repro (Meridian Bridge history + unrelated "capital of France?" question) against a real embedded `VectorStore`, plus a real-embedding check that genuine follow-ups still work.
  - `/code-review` follow-up: the 3rd (reference-only) query is now skipped whenever every augmented result is already confirmed via the bare query — the common case of a self-contained new question — instead of always paying for a third embedding + Chroma round trip.
  - **Bug 1 recurrence** (2026-09-13, `feature/relevance-fixes-followup`, PR #11): the drop check still leaked a citation once a real, longer document was chunked into multiple pieces by the actual chunker (the original repro used one hand-picked chunk). A chunk only weakly anchored to the prior turn (prev_score ~0.44) needs to lose much less in *absolute* terms to reveal it's unrelated to the new question too — its 0.11 drop looked safe under the flat 0.15 threshold even though it lost 26% of its score, comparable to a strongly-anchored chunk (prev_score ~0.77) correctly rejected for losing 20%. `MAX_AUGMENTED_SCORE_DROP` (0.15, absolute) replaced by `MAX_AUGMENTED_SCORE_DROP_RATIO` (0.15, *fraction of the prior-turn score*) — measured 20-37% relative drop for unrelated questions vs at most ~10% for real follow-ups. New regression test drives a real multi-chunk document (via the actual `chunk_text`, not a hand-picked single chunk) through the exact recurrence.
  - **Bug 3 — cross-source surface-similarity noise** (2026-09-13, `feature/citation-noise-fix`, PR #12): with 2+ documents in the store, a question targeted at one document still cited an unrelated one that merely shared a sentence template ("X was constructed by Y in Z") — measured 0.29 for the noise chunk vs 0.67-0.83 for the true positives, all above `MIN_RELEVANCE_SCORE` (0.25). A higher absolute floor doesn't generalize: a harder but still realistic case (two documents on the *same* narrow topic — a rival lighthouse) scored 0.45, above even a 0.40 floor. A plain ratio-to-best-score threshold doesn't work either: two genuinely relevant chunks *from the same document* can score as low as 58% of each other, almost indistinguishable from the 55% ratio measured for the hard rival-document case — the two distributions overlap. Fixed with `MIN_CROSS_SOURCE_RATIO` (0.6) applied **only across different sources** — same-source results are exempt unconditionally, which removes the overlap (same-doc weaker facts are never competing with the cross-source threshold) and lets a stronger ratio reject the hard case (0.55) with real margin.
    - **Known limitation, documented rather than "fixed"**: this is a mitigation, not a guarantee. A cross-document match scoring above 0.6 of the top result's score — a near-duplicate topic phrased very similarly to the right answer — would still leak through. No numeric threshold on a single cosine-similarity vector (MiniLM, 384-dim) can rule this out; it would need re-ranking (a cross-encoder, hybrid BM25 + semantic scoring, or an LLM relevance check), out of scope for this MVP heuristic.
    - `/code-review` caught a real regression risk before merge: `_retrieve`'s history-augmentation used to take the *max* of a bare-relevant chunk's bare and augmented scores, so history matching one topic could inflate that chunk's score — which then became the reference the cross-source ratio measured every *other* source against, wrongly dropping a second, independently-relevant source in a compound question. Fixed by always keeping the bare score for bare-relevant chunks (the augmented query's job is to rescue candidates the bare query missed, not re-score ones it already vouches for). Two smaller edge cases also fixed: a `None`-safe same-source comparison, and skipping the filter when the top score is non-positive (an inverted-ratio risk reachable only via a very low `min_score` override).
- [ ] Frontend (Phase 3)
  - [x] Scaffold (`frontend/`, Next.js 16 App Router + TS + Tailwind v4) — `feature/frontend-scaffold`, PR #8
    - single `app/page.tsx`: file upload -> `POST /documents`; chat (input + Send) -> `POST /query` with `history`; citations under each answer; "Thinking…" indicator; errors shown, not thrown
    - `lib/api.ts`: typed `ingestDocument` / `askQuestion`, `ApiError` normalizing FastAPI `detail`, `AbortSignal.timeout`
    - backend: `CORSMiddleware` for the Next dev origin (`FRONTEND_ORIGINS` env, `load_dotenv()`); `create-next-app` pulled Next **16**, not 14
  - [x] shadcn/ui + component split + "New conversation" — `feature/frontend-polish`, PR #9
    - shadcn/ui via `shadcn init` (radix-nova, neutral); dark mode kept on `prefers-color-scheme` (no toggle) — token overrides moved to a `@media` block, `.dark` custom-variant dropped
    - `components/`: `DocumentUpload` (owns its upload state), `MessageBubble`, `Citations`, `ChatComposer`; `lib/chat.ts` = `Message` view type + `errorText`
    - "New conversation" clears chat state (frontend only); lucide icons on the action buttons
    - `lib/api.ts` unchanged — presentation refactor only
  - [ ] Polish (later): streaming answers, mobile layout, per-message retry
  - **Note on stack:** planned as "Next.js 14+"; `create-next-app@latest` installed 16.3 (Turbopack default, Tailwind v4, React 19).

## Phase 1 MVP: complete
All backend slices done. `POST /documents` to ingest, `POST /query` (with optional
`history`) to ask cited, conversation-aware questions.

## Git Workflow
- `main` branch stays stable/working
- Feature branches per phase: `feature/ingestion-pipeline`, `feature/retrieval-core`, etc.
- Small, meaningful commits — not one giant commit per phase
