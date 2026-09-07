import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_llm, get_vector_store
from app.main import app
from app.retrieval.llm import OllamaTimeoutError, OllamaUnavailableError
from app.retrieval.query_flow import NO_CONTEXT_ANSWER


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


def _result(text, source, chunk_index, score):
    return {
        "id": f"{source}::{chunk_index}",
        "text": text,
        "source": source,
        "chunk_index": chunk_index,
        "score": score,
    }


def _stub_llm(answer="A stubbed answer [1]."):
    def generate(prompt, model="stub"):
        return answer

    return generate


@pytest.fixture
def client():
    """Yields a factory: configure(store=..., generate=...) -> TestClient."""

    def configure(store, generate=None, use_real_llm=False):
        app.dependency_overrides[get_vector_store] = lambda: store
        if not use_real_llm:
            app.dependency_overrides[get_llm] = lambda: generate or _stub_llm()
        return TestClient(app)

    yield configure

    app.dependency_overrides.clear()
    get_vector_store.cache_clear()


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


def test_query_non_positive_top_k_returns_400(client):
    c = client(FakeStore([_result("x", "d.md", 0, 0.9)]))

    response = c.post("/query", json={"question": "hi", "top_k": 0})

    assert response.status_code == 400


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


def test_documents_no_filename_part_is_rejected(client):
    # An upload part with no filename isn't parsed as a file — FastAPI
    # rejects the request before our handler runs.
    c = client(FakeStore())

    response = c.post(
        "/documents",
        files={"file": ("", b"some content", "text/plain")},
    )

    assert response.status_code == 422


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
