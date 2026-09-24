import pytest
from fastapi.testclient import TestClient

from app.api.routes import (
    _LLM_PROVIDERS,
    LLMProviderConfigError,
    get_llm,
    get_vector_store,
)
from app.main import app
from app.retrieval import gemini
from app.retrieval import llm as ollama_llm
from app.retrieval.embeddings import EmbeddingUnavailableError
from app.retrieval.gemini import GeminiAuthError
from app.retrieval.llm import (
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.retrieval.query_flow import MAX_HISTORY_MESSAGES, NO_CONTEXT_ANSWER


class FakeStore:
    """Stand-in for VectorStore: canned query results, records adds.
    Mirrors the bits of the real contract that answer_with_llm touches."""

    def __init__(self, results=None):
        self.results = results or []
        self.added = []

    def query(self, question, top_k=5):
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        return self.results[:top_k]

    def add_documents(self, chunks, source):
        self.added.append((source, list(chunks)))

    def delete_document(self, source):
        if not source.strip():
            raise ValueError("source must be a non-empty string")
        deleted = sum(len(c) for s, c in self.added if s == source)
        self.added = [(s, c) for s, c in self.added if s != source]
        return deleted


def _result(text, source, chunk_index, score):
    return {
        "id": f"{source}::{chunk_index}",
        "text": text,
        "source": source,
        "chunk_index": chunk_index,
        "score": score,
    }


def _stub_llm(answer="A stubbed answer [1]."):
    """Records each call's (prompt, model) on the returned callable's
    `.calls` — most tests ignore it, but it's there for the ones
    asserting on what `query()` actually passed through, e.g. the model
    half of `get_llm`'s (generate, model) pair."""
    calls = []

    def generate(prompt, model="stub"):
        calls.append({"prompt": prompt, "model": model})
        return answer

    generate.calls = calls
    return generate


@pytest.fixture
def client():
    """Yields a factory: configure(store=..., generate=..., model=...) ->
    TestClient. `model` only matters to tests that inspect what was
    actually passed to `generate` (via its `.calls`, see `_stub_llm`) —
    every other test's stub ignores it."""

    def configure(store, generate=None, model="stub-model", use_real_llm=False):
        app.dependency_overrides[get_vector_store] = lambda: store
        if use_real_llm:
            # Pin to the real Ollama backend explicitly, regardless of
            # whatever LLM_PROVIDER happens to be set to in the ambient
            # environment this test suite runs in — "real LLM" has always
            # meant "real Ollama" for this flag, and get_llm() is now
            # env-driven, so leaving it un-overridden would silently call
            # Gemini instead if LLM_PROVIDER=gemini is set.
            app.dependency_overrides[get_llm] = lambda: _LLM_PROVIDERS["ollama"]
        else:
            app.dependency_overrides[get_llm] = lambda: (
                generate or _stub_llm(),
                model,
            )
        return TestClient(app)

    yield configure

    app.dependency_overrides.clear()
    get_vector_store.cache_clear()


# --- get_llm: LLM_PROVIDER selection --------------------------------------


def test_get_llm_defaults_to_ollama_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    generate, model = get_llm()

    assert generate is ollama_llm.generate_answer
    assert model == ollama_llm.DEFAULT_MODEL


def test_get_llm_selects_ollama_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    generate, model = get_llm()

    assert generate is ollama_llm.generate_answer
    assert model == ollama_llm.DEFAULT_MODEL


def test_get_llm_selects_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    generate, model = get_llm()

    assert generate is gemini.generate_answer
    assert model == gemini.DEFAULT_MODEL


@pytest.mark.parametrize("raw", ["Gemini", " GEMINI ", "gemini\n", "  gemini"])
def test_get_llm_provider_is_case_insensitive_and_trimmed(monkeypatch, raw):
    monkeypatch.setenv("LLM_PROVIDER", raw)

    generate, _ = get_llm()

    assert generate is gemini.generate_answer


def test_get_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(LLMProviderConfigError) as excinfo:
        get_llm()

    message = str(excinfo.value)
    assert "openai" in message
    assert "ollama" in message
    assert "gemini" in message


def test_get_llm_empty_string_falls_back_to_ollama(monkeypatch):
    # An env var explicitly set to "" (e.g. an unset deploy-config
    # placeholder) should behave like unset, not like an unknown provider.
    monkeypatch.setenv("LLM_PROVIDER", "")

    generate, model = get_llm()

    assert generate is ollama_llm.generate_answer
    assert model == ollama_llm.DEFAULT_MODEL


@pytest.mark.parametrize("raw", ["   ", "\n", "\t "])
def test_get_llm_whitespace_only_falls_back_to_ollama(monkeypatch, raw):
    # Regression test: "   " is truthy, so it used to skip the "unset"
    # fallback, then get stripped down to "" and rejected as an unknown
    # provider — inconsistent with the plain-empty-string case above.
    monkeypatch.setenv("LLM_PROVIDER", raw)

    generate, model = get_llm()

    assert generate is ollama_llm.generate_answer
    assert model == ollama_llm.DEFAULT_MODEL


def test_query_unknown_llm_provider_returns_500_with_detail(monkeypatch):
    # Regression test: LLMProviderConfigError is raised during get_llm's
    # dependency resolution, before query()'s own try/except runs — this
    # proves main.py's exception_handler actually surfaces the message
    # (via LLMProviderConfigError, not just as a bare "Internal Server
    # Error" that would discard it).
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    store = FakeStore([_result("x", "d.md", 0, 0.9)])
    app.dependency_overrides[get_vector_store] = lambda: store
    # Deliberately not overriding get_llm — exercise the real dependency.
    try:
        response = TestClient(app).post("/query", json={"question": "hello"})
    finally:
        app.dependency_overrides.clear()
        get_vector_store.cache_clear()

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "bogus" in detail
    assert "ollama" in detail
    assert "gemini" in detail


# --- /query: get_llm's (generate, model) pair reaches answer_with_llm ----


def test_query_passes_the_paired_model_to_answer_with_llm(client):
    store = FakeStore([_result("A relevant fact.", "src.md", 0, 0.9)])
    stub = _stub_llm()
    c = client(store, generate=stub, model="custom-model-x")

    response = c.post("/query", json={"question": "What is it?"})

    assert response.status_code == 200
    assert stub.calls
    assert stub.calls[-1]["model"] == "custom-model-x"


# --- /query --------------------------------------------------------------


def test_query_returns_expected_structure(client):
    store = FakeStore(
        [
            _result("Mitochondria make ATP.", "bio.md", 2, 0.82),
            _result("Photosynthesis makes glucose.", "bio.md", 0, 0.55),
        ]
    )
    c = client(store, generate=_stub_llm("Cells use mitochondria [1]."))

    response = c.post("/query", json={"question": "How do cells make energy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Cells use mitochondria [1]."
    assert body["has_context"] is True
    assert len(body["citations"]) == 2
    first = body["citations"][0]
    assert first == {
        "marker": 1,
        "source": "bio.md",
        "chunk_index": 2,
        "text": "Mitochondria make ATP.",
        "score": pytest.approx(0.82),
    }


def test_query_respects_top_k(client):
    store = FakeStore([_result(f"chunk {i}", "d.md", i, 0.9) for i in range(5)])
    c = client(store)

    response = c.post("/query", json={"question": "anything", "top_k": 2})

    assert response.status_code == 200
    assert len(response.json()["citations"]) == 2


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_query_empty_question_returns_400(client, question):
    c = client(
        FakeStore(),
        generate=lambda *a, **k: pytest.fail("LLM should not be called"),
    )

    response = c.post("/query", json={"question": question})

    assert response.status_code == 400


@pytest.mark.parametrize("top_k", [0, -1, 999])
def test_query_out_of_range_top_k_returns_422(client, top_k):
    # top_k bounds are enforced by the request model -> 422, not our 400.
    c = client(FakeStore([_result("x", "d.md", 0, 0.9)]))

    response = c.post("/query", json={"question": "hi", "top_k": top_k})

    assert response.status_code == 422


def test_query_missing_question_field_returns_422(client):
    c = client(FakeStore())

    response = c.post("/query", json={"top_k": 3})

    assert response.status_code == 422  # pydantic validation, not our 400


def test_query_no_relevant_context_returns_200_and_canned_answer(client):
    store = FakeStore([])  # nothing retrieved
    c = client(
        store,
        generate=lambda *a, **k: pytest.fail("LLM must not be called"),
    )

    response = c.post("/query", json={"question": "unrelated question"})

    assert response.status_code == 200
    body = response.json()
    assert body["has_context"] is False
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["citations"] == []


def test_query_llm_unavailable_returns_503(client):
    store = FakeStore([_result("relevant text", "d.md", 0, 0.9)])

    def broken(prompt, model="stub"):
        raise OllamaUnavailableError(
            "Could not reach Ollama at http://localhost:11434. "
            "Start it with `ollama serve`."
        )

    c = client(store, generate=broken)

    response = c.post("/query", json={"question": "a real question"})

    assert response.status_code == 503
    assert "ollama serve" in response.json()["detail"].lower()


def test_query_llm_timeout_returns_503(client):
    store = FakeStore([_result("relevant text", "d.md", 0, 0.9)])

    def slow(prompt, model="stub"):
        raise OllamaTimeoutError("Ollama did not respond within 120.0s")

    c = client(store, generate=slow)

    response = c.post("/query", json={"question": "a real question"})

    assert response.status_code == 503


def test_query_llm_model_not_found_returns_500(client):
    # Non-retryable misconfiguration (model not pulled) — distinct from the
    # transient 503s above.
    store = FakeStore([_result("relevant text", "d.md", 0, 0.9)])

    def missing_model(prompt, model="stub"):
        raise OllamaModelNotFoundError(
            "Ollama model 'llama3.2' is not available. "
            "Pull it with `ollama pull llama3.2`."
        )

    c = client(store, generate=missing_model)

    response = c.post("/query", json={"question": "a real question"})

    assert response.status_code == 500
    assert "ollama pull" in response.json()["detail"].lower()


def test_query_gemini_auth_error_returns_500_not_503(client):
    # Same "non-retryable misconfiguration" bucket as the Ollama
    # model-not-found case above — a bad/missing GEMINI_API_KEY will never
    # resolve itself by retrying, unlike a generic (transient) LLMError.
    store = FakeStore([_result("relevant text", "d.md", 0, 0.9)])

    def bad_key(prompt, model="stub"):
        raise GeminiAuthError("Gemini rejected the API key (...).")

    c = client(store, generate=bad_key)

    response = c.post("/query", json={"question": "a real question"})

    assert response.status_code == 500
    assert "api key" in response.json()["detail"].lower()


def test_query_embedding_download_failure_returns_503(client):
    # Regression test: every /query call embeds the question
    # (VectorStore.query), so a cold-start embedding-model download
    # hiccup is at least as reachable here as on /documents (which this
    # exact fix was already applied to) -- confirmed via /code-review
    # that without it, this fell through to a bare, undiagnosed 500.
    # Raises EmbeddingUnavailableError, not e.g. httpx.ConnectError
    # directly: that's the actual, normalized contract embeddings.py's
    # _get_model() now guarantees regardless of which concrete library
    # exception caused the download to fail (see its module docstring).
    class StoreThatFailsToEmbed:
        def query(self, question, top_k=5):
            raise EmbeddingUnavailableError("simulated network failure")

    c = client(StoreThatFailsToEmbed())

    response = c.post("/query", json={"question": "a real question"})

    assert response.status_code == 503
    assert "backend unavailable" in response.json()["detail"].lower()


# --- /query conversation history --------------------------------------------


def test_query_with_history_passes_it_to_the_llm(client):
    store = FakeStore([_result("The bridge has two spans.", "bridge.md", 4, 0.8)])
    seen = {}

    def capture(prompt, model="stub"):
        seen["prompt"] = prompt
        return "The second span was added in 1961 [1]."

    c = client(store, generate=capture)

    response = c.post(
        "/query",
        json={
            "question": "and the second one?",
            "history": [
                {"role": "user", "content": "When was the first span built?"},
                {"role": "assistant", "content": "The first span opened in 1932."},
            ],
        },
    )

    assert response.status_code == 200
    assert "Conversation so far:" in seen["prompt"]
    assert "User: When was the first span built?" in seen["prompt"]
    assert "Assistant: The first span opened in 1932." in seen["prompt"]


def test_query_without_history_is_unchanged(client):
    store = FakeStore([_result("x", "d.md", 0, 0.9)])
    seen = {}

    def capture(prompt, model="stub"):
        seen["prompt"] = prompt
        return "ok"

    c = client(store, generate=capture)

    r1 = c.post("/query", json={"question": "hello"})
    p1 = seen["prompt"]
    r2 = c.post("/query", json={"question": "hello", "history": None})
    p2 = seen["prompt"]

    assert r1.status_code == r2.status_code == 200
    assert p1 == p2
    assert "Conversation so far:" not in p1


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "system", "content": "nope"}],  # bad role value
        [{"role": "user", "content": ""}],  # empty content
        [{"role": "user", "content": "   "}],  # whitespace content
        [{"role": "user"}],  # missing content
        [{"role": "user", "content": None}],  # null content
        [{"content": "no role"}],  # missing role
    ],
)
def test_query_malformed_history_returns_400(client, history):
    c = client(
        FakeStore([_result("x", "d.md", 0, 0.9)]),
        generate=lambda *a, **k: pytest.fail("LLM must not be called"),
    )

    response = c.post(
        "/query", json={"question": "a question", "history": history}
    )

    assert response.status_code == 400


def test_query_history_wrong_shape_returns_422(client):
    # not even the {role, content} object shape -> pydantic rejects it
    c = client(FakeStore([_result("x", "d.md", 0, 0.9)]))

    response = c.post(
        "/query", json={"question": "q", "history": "not a list"}
    )

    assert response.status_code == 422


def test_query_long_history_is_truncated_in_the_prompt(client):
    store = FakeStore([_result("x", "d.md", 0, 0.9)])
    seen = {}

    def capture(prompt, model="stub"):
        seen["prompt"] = prompt
        return "ok"

    c = client(store, generate=capture)

    n = 30
    history = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"<<h{i:02d}>>",
        }
        for i in range(n)
    ]

    response = c.post(
        "/query", json={"question": "q", "history": history}
    )

    assert response.status_code == 200
    for i in range(n - MAX_HISTORY_MESSAGES, n):  # last N kept
        assert f"<<h{i:02d}>>" in seen["prompt"]
    for i in range(n - MAX_HISTORY_MESSAGES):  # older dropped
        assert f"<<h{i:02d}>>" not in seen["prompt"]


# --- /documents ---------------------------------------------------------


def test_documents_ingests_a_text_file(client):
    store = FakeStore()
    c = client(store)

    response = c.post(
        "/documents",
        files={"file": ("notes.txt", b"one two three four five six", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "notes.txt"
    assert body["chunks_added"] >= 1
    assert store.added and store.added[0][0] == "notes.txt"


def test_documents_unsupported_extension_returns_400(client):
    c = client(FakeStore())

    response = c.post(
        "/documents",
        files={"file": ("report.docx", b"binary junk", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert ".docx" in response.json()["detail"]


def test_documents_empty_text_returns_400(client):
    store = FakeStore()
    c = client(store)

    response = c.post(
        "/documents",
        files={"file": ("blank.txt", b"   \n\t  ", "text/plain")},
    )

    assert response.status_code == 400
    assert not store.added


def test_documents_indexing_backend_failure_returns_503(client):
    class FailingStore(FakeStore):
        def add_documents(self, chunks, source):
            raise OSError("could not fetch the embedding model")

    c = client(FailingStore())

    response = c.post(
        "/documents",
        files={"file": ("notes.txt", b"one two three four five", "text/plain")},
    )

    assert response.status_code == 503


def test_documents_no_filename_part_is_rejected(client):
    # An upload part with no filename is rejected — either by FastAPI's
    # validation (422) or, if a Starlette version still parses it as an
    # UploadFile, by our own `if not file.filename` guard (400).
    c = client(FakeStore())

    response = c.post(
        "/documents",
        files={"file": ("", b"some content", "text/plain")},
    )

    assert response.status_code in (400, 422)


# --- DELETE /documents/{source} (admin) -------------------------------------

ADMIN_TOKEN = "s3cret-token"


def _store_with(*docs):
    store = FakeStore()
    for source, chunks in docs:
        store.add_documents(chunks, source)
    return store


def test_delete_document_removes_only_that_source(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    store = _store_with(("a.txt", ["a1", "a2"]), ("b.txt", ["b1"]))
    c = client(store)

    response = c.delete("/documents/a.txt", headers={"X-Admin-Token": ADMIN_TOKEN})

    assert response.status_code == 200
    assert response.json() == {"source": "a.txt", "chunks_deleted": 2}
    assert store.added == [("b.txt", ["b1"])]


def test_delete_document_handles_url_encoded_source(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    store = _store_with(("my notes.txt", ["x"]))
    c = client(store)

    response = c.delete(
        "/documents/my%20notes.txt", headers={"X-Admin-Token": ADMIN_TOKEN}
    )

    assert response.status_code == 200
    assert store.added == []


def test_delete_unknown_document_returns_404(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    c = client(_store_with(("a.txt", ["a1"])))

    response = c.delete("/documents/nope.txt", headers={"X-Admin-Token": ADMIN_TOKEN})

    assert response.status_code == 404
    assert "nope.txt" in response.json()["detail"]


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Token": "wrong"}, {"X-Admin-Token": ""}])
def test_delete_without_valid_token_returns_401_and_keeps_data(
    client, monkeypatch, headers
):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    store = _store_with(("a.txt", ["a1"]))
    c = client(store)

    response = c.delete("/documents/a.txt", headers=headers)

    assert response.status_code == 401
    assert store.added == [("a.txt", ["a1"])]


@pytest.mark.parametrize("admin_token", [None, "", "   "])
def test_delete_is_disabled_when_admin_token_unset(client, monkeypatch, admin_token):
    # Unset/blank ADMIN_TOKEN: 404 even with a token supplied — and above
    # all, a blank server-side token must never match a blank header.
    if admin_token is None:
        monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("ADMIN_TOKEN", admin_token)
    store = _store_with(("a.txt", ["a1"]))
    c = client(store)

    response = c.delete("/documents/a.txt", headers={"X-Admin-Token": ""})

    assert response.status_code == 404
    assert store.added == [("a.txt", ["a1"])]


def test_delete_whitespace_source_returns_400(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    c = client(FakeStore())

    response = c.delete("/documents/%20", headers={"X-Admin-Token": ADMIN_TOKEN})

    assert response.status_code == 400


def test_delete_store_failure_returns_503(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)

    class FailingStore(FakeStore):
        def delete_document(self, source):
            raise OSError("chroma write failed")

    c = client(FailingStore())

    response = c.delete("/documents/a.txt", headers={"X-Admin-Token": ADMIN_TOKEN})

    assert response.status_code == 503


# --- end-to-end through real retrieval --------------------------------------


@pytest.mark.model
def test_documents_then_query_roundtrip(client, tmp_path):
    from app.retrieval.vector_store import VectorStore

    store = VectorStore(tmp_path)
    c = client(store, generate=_stub_llm("stubbed"))

    ingest = c.post(
        "/documents",
        files={
            "file": (
                "facts.txt",
                b"The Meridian Bridge was completed in 1932 after four years "
                b"of construction across the Halden River.",
                "text/plain",
            )
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_added"] >= 1

    answer = c.post(
        "/query",
        json={"question": "When was the Meridian Bridge finished?", "top_k": 3},
    )

    assert answer.status_code == 200
    body = answer.json()
    assert body["has_context"] is True
    assert body["citations"][0]["source"] == "facts.txt"
    assert "Meridian Bridge" in body["citations"][0]["text"]


@pytest.mark.ollama
@pytest.mark.model
def test_full_roundtrip_real_ollama(client, tmp_path):
    import httpx

    from app.retrieval.vector_store import VectorStore

    try:
        httpx.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        pytest.skip("Ollama not reachable on localhost:11434")

    store = VectorStore(tmp_path)
    c = client(store, use_real_llm=True)

    c.post(
        "/documents",
        files={
            "file": (
                "hours.txt",
                b"Aurelia's Cafe on Pine Street opens at 7am on weekdays.",
                "text/plain",
            )
        },
    )

    answer = c.post(
        "/query",
        json={"question": "What time does Aurelia's Cafe open on weekdays?"},
    )

    assert answer.status_code == 200
    body = answer.json()
    assert body["has_context"] is True
    assert "7" in body["answer"] or "seven" in body["answer"].lower()
