# RAG Knowledge Assistant — Planning

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
| LLM call — direct HTTP, not LangChain | `llm.generate_answer()` calls Ollama's `/api/generate` with `httpx` directly; `gemini.generate_answer()` (same contract) calls Gemini's `generateContent` the same way | One narrow function is easier to reason about and test than a LangChain LLM wrapper for a single call; `answer_with_llm` takes an injectable `generate` callable, and `gemini.py` is now the implemented Gemini prod drop-in, not just a planned one. `httpx` over `requests`: FastAPI-native, async-ready, constructible `Response` for deterministic test mocking. |
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
│   │   │   ├── llm.py           # Ollama HTTP client (dev) [done]
│   │   │   └── gemini.py        # Gemini HTTP client (prod) [done]
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
  - **Bug 4 — 2-turn history recurrence of the augmented-drop check** (2026-09-15, `feature/history-relevance-fix`, PR #15): manual testing with TWO real prior turns in history (a genuine follow-up already answered correctly, then an unrelated question) leaked a citation to a weakly-anchored chunk — a museum/visitors chunk from the same single-document store, at a ~14% relative drop against the prior turn, under the `MAX_AUGMENTED_SCORE_DROP_RATIO` (0.15) set by the Bug 1 recurrence. Measured and ruled out before touching the threshold: turn count itself is irrelevant (`_last_user_turn` only ever reads the single most recent user message, so 1-turn and 2-turn history produce identical retrieval behavior), `MAX_HISTORY_MESSAGES` truncation is not implicated (well under the 6-message cap), and the cross-source filter is not implicated (a single-document store exempts every chunk as same-source). A broader sample across two documents and multiple prior-turn anchors found the true margin narrower than the sample behind the 0.15 ratio: genuine follow-ups drop by at most ~8.6%, unrelated questions by at least ~13.2% — a real but narrow (~4.6-point) gap. `MAX_AUGMENTED_SCORE_DROP_RATIO` lowered `0.15` -> `0.10`.
    - **Accepted structural limitation, not "fixed"**: this is the 3rd time this same threshold has been tightened (0.15 absolute -> 0.15 relative -> 0.10 relative), and the margin has narrowed every time (roughly 10 points, then 5, now ~4.6). A single relative-score threshold on one embedding vector cannot perfectly separate "genuinely unrelated" from "topic-agnostic follow-up" in every case; a structurally different mechanism (e.g. an LLM-based query rewrite) would be needed to close the gap for good, out of scope for this MVP heuristic. A combined bare/augmented signal was considered as a next investigation but deliberately not pursued — the cost/benefit no longer justifies it at this stage of the project. Crucially, in every case measured across all three rounds the LLM's *answer* stayed correct ("I don't know") — the prompt's citation/no-context instructions mean a stray chunk here is at most an occasional spurious citation next to a correct answer, never a wrong answer.
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
  - [x] UI fixes: upload card layout + EN/FR toggle — `feature/ui-fixes`, PR #13
    - `DocumentUpload`: native `<input type=file>` (intrinsic width, left a large empty gap) swapped for the shadcn `Input` primitive with `flex-1`, button right-aligned — consistent with `ChatComposer`
    - EN/FR toggle for static UI labels only (not the LLM's answer language, which already adapts to the question): `lib/i18n.ts` (plain object dictionaries, no next-intl — one page, ~18 short strings), `lib/language-context.tsx` (Context + `localStorage`, `<html lang>` synced via effect), `components/LanguageToggle.tsx` in the header
    - Known, documented limitation: a returning user briefly sees English before the stored preference loads (no cookie/middleware — out of scope for a client-only toggle); accepted over a hydration-mismatch warning
    - `Citations`/`MessageBubble` now `"use client"` (they read the language context) — no behavior change, their only caller was already client-side
  - [x] Conversation history (multi-conversation persistence) — `feature/conversation-history`, PR #14
    - `lib/conversations.ts` (pure data) + `lib/conversations-context.tsx` (`ConversationsProvider`/`useConversations()`, same `localStorage` try/catch pattern as `language-context.tsx`): id, title (derived from the first user message, truncated at a word boundary), messages, `updatedAt`; capped at 50, oldest dropped
    - `syncActive(messages)` is the *only* persistence path, called from `page.tsx` right after each `setMessages()` from actually chatting (not from loading a saved conversation into view) — this single mechanism satisfies both "save continuously" and "New conversation saves before clearing" at once, and never bumps `updatedAt` just from viewing a past conversation
    - `components/ConversationHistory.tsx`: a `Popover` (shadcn primitive added — zero new dependency, already covered by the installed unified `radix-ui` package), not a sidebar restructure; lists by recency, active one highlighted, delete on hover as a sibling button (not nested inside the load button — invalid HTML)
    - `/code-review` fix: switching/deleting conversations while a request for the active one was in flight could write a stale response into the wrong conversation — fixed by disabling history browsing while `loading` (same guard "New conversation" already had) rather than resolving the race after the fact; deleting the on-screen conversation now clears the view via an `onDeleteActive` callback
  - [ ] Polish (later): streaming answers, mobile layout, per-message retry
  - **Note on stack:** planned as "Next.js 14+"; `create-next-app@latest` installed 16.3 (Turbopack default, Tailwind v4, React 19).
- [x] Gemini production LLM client (`gemini.py` + tests) — `feature/gemini-integration`, PR #16
  - `gemini.generate_answer(prompt, model=..., *, api_key=None, timeout=...) -> str` — exact same call contract as `llm.generate_answer` (Ollama), so it's a drop-in for `answer_with_llm(..., generate=...)` (see Architecture Decisions)
  - Direct HTTP via `httpx`, not the official `google-generativeai` SDK — same reasoning as the Ollama client: one narrow function over a stable REST endpoint is easier to mock deterministically (a constructed `httpx.Response`) than an SDK client object, and avoids the SDK's grpc/protobuf/google-auth dependency chain
  - `GEMINI_API_KEY` (python-dotenv, already documented in `backend/.env.example`) sent via the `x-goog-api-key` header, never the URL. `GeminiError` subclasses `llm.LLMError` rather than starting a separate hierarchy, so `answer_with_llm`'s documented "Raises: LLMError (and subclasses)" stays true regardless of which backend `generate` is bound to
  - Error taxonomy: `GeminiAuthError` (missing/invalid key — a missing key is caught locally before any network call), `GeminiTimeoutError`, and a generic `GeminiError` for everything else (network errors, non-2xx responses, and a candidate blocked by Gemini's safety filters — a failure mode with no Ollama equivalent). No separate "unavailable" class unlike Ollama: a cloud API has no "start the local server" remediation, so connection-level failures fold into the generic error
  - New `@pytest.mark.gemini` marker (mirrors `@pytest.mark.ollama`): real-API tests skip cleanly without `GEMINI_API_KEY`. A `model` + `gemini` end-to-end test drives `answer_with_llm(..., generate=gemini.generate_answer)` against a real `VectorStore` to prove genuine interchangeability, not just a matching signature
  - Found via the real API, not assumed: `gemini-2.0-flash` (the originally intended default) is no longer served (HTTP 404, "use models/gemini-3.6-flash") — `DEFAULT_MODEL` set to what the live API itself recommends
  - `/code-review` fixes: `_is_auth_error`/`_error_detail` guarded against explicit-`null` JSON shapes (a safety-blocked candidate's `content`, a part's `text`, an `"error"` field as a plain string e.g. from a proxy) that previously raised an uncaught `AttributeError` instead of the documented `GeminiError`; the two helpers now share a single parsed response body instead of each re-parsing it; prompt-emptiness validation deduplicated into a shared `_require_nonempty_prompt` used by both backends
  - Dev/prod backend-selection wiring (was deliberately deferred here) landed in the next entry below
- [x] LLM_PROVIDER: runtime Ollama/Gemini backend selection (`routes.py` + tests) — `feature/llm-provider-selection`, PR #18
  - `get_llm()` (a FastAPI dependency) now returns a paired `(generate, model)` tuple, selected by the `LLM_PROVIDER` env var (`"ollama"`, the default, or `"gemini"`; case-insensitive, whitespace-trimmed, blank treated as unset) — a small `_LLM_PROVIDERS` registry maps each name to its matching pair
  - Bundled into one dependency rather than two, on purpose: `answer_with_llm`'s docstring already warned that swapping only `generate` without also passing a matching `model` silently sends one backend's model name to the other — this closes that footgun at the one call site that matters, `query()`
  - Error mapping extended for Gemini: `GeminiAuthError` (missing/invalid `GEMINI_API_KEY`) now maps to 500 (non-retryable misconfiguration), the same bucket as `OllamaModelNotFoundError` — previously it would have fallen through to the generic `except LLMError` -> 503 (transient), telling clients to retry a problem retrying can never fix
  - `/code-review` fixes (two rounds): a whitespace-only `LLM_PROVIDER` (`"   "`) used to be rejected as an unknown provider instead of falling back to Ollama like the empty-string case; an unrecognized `LLM_PROVIDER` raised during FastAPI's dependency resolution, outside `query()`'s own try/except, so the client got a bare "Internal Server Error" with the actual diagnostic message discarded — fixed with a dedicated `LLMProviderConfigError` and an `app.exception_handler` in `main.py` that returns it as `{"detail": ...}`, matching every other error in the API. Also: `test_full_roundtrip_real_ollama` used to leave `get_llm()` un-overridden and now-env-driven, so it would have silently called Gemini instead of Ollama if `LLM_PROVIDER=gemini` were ever set in the test environment — pinned explicitly
    - **Noted, not fixed here**: `main.py`'s `FRONTEND_ORIGINS` parsing has the same whitespace-only-value inconsistency this PR fixed for `LLM_PROVIDER` (a value of all spaces is truthy, skips the "unset" fallback, and silently produces an empty allow-list) — pre-existing, unrelated to this diff, left as a spotted follow-up
  - Scope deliberately limited to wiring, per explicit instruction: no automatic fallback if the selected provider fails at runtime, and no shared exception base or `NamedTuple` across backends (both considered, declined as premature for a two-backend, single-consumer codebase)

## Phase 1 MVP: complete
All backend slices done. `POST /documents` to ingest, `POST /query` (with optional
`history`) to ask cited, conversation-aware questions.

## Git Workflow
- `main` branch stays stable/working
- Feature branches per phase: `feature/ingestion-pipeline`, `feature/retrieval-core`, etc.
- Small, meaningful commits — not one giant commit per phase
