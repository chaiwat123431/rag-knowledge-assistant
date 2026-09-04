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
| Embeddings | OpenAI `text-embedding-3-small` | Cheap, well-documented, good baseline quality |
| Chunking | Recursive character splitter, ~500 tokens, 50 token overlap | Balances context completeness vs. retrieval precision |
| Frontend | Next.js + TypeScript | Separate phase; not part of Phase 1 |

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
│   │   │   ├── parser.py        # PDF/text extraction
│   │   │   └── chunker.py       # text splitting logic
│   │   ├── retrieval/
│   │   │   ├── embeddings.py    # embedding generation
│   │   │   └── vector_store.py  # Chroma interface
│   │   └── api/
│   │       └── routes.py        # HTTP endpoints
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/                    # Phase 3, not yet created
├── PLANNING.md
└── .gitignore
```

## Testing Strategy
- Unit tests for parser, chunker, embeddings, vector store — each in isolation
- Integration test: full ingest -> query -> answer flow on a small fixture document
- Tests written alongside each module, not deferred to a later phase

## Git Workflow
- `main` branch stays stable/working
- Feature branches per phase: `feature/ingestion-pipeline`, `feature/retrieval-core`, etc.
- Small, meaningful commits — not one giant commit per phase
