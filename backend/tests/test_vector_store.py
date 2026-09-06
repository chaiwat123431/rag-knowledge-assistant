import subprocess
import sys
from pathlib import Path

import pytest

import app.retrieval.vector_store as vector_store_module
from app.retrieval.vector_store import VectorStore

# Documents used across the retrieval tests: three clearly distinct topics
# so "most relevant chunk first" is unambiguous.
CHUNKS = [
    "The mitochondrion is the powerhouse of the cell, producing ATP through respiration.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The French Revolution began in 1789 and led to the fall of the monarchy.",
    "Compound interest causes savings to grow exponentially over long time horizons.",
    "A binary search tree keeps nodes ordered so lookups run in logarithmic time.",
]


@pytest.mark.model
def test_query_returns_most_relevant_chunk_first(tmp_path):
    store = VectorStore(tmp_path)
    store.add_documents(CHUNKS, source="notes.md")

    results = store.query("How do cells generate energy?", top_k=3)

    assert results
    assert results[0]["text"] == CHUNKS[0]  # the mitochondrion / ATP chunk
    # scores must be sorted nearest-first
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_on_empty_store_returns_empty_list_without_loading_model(
    tmp_path, monkeypatch
):
    # An empty store must answer without ever embedding the question, so
    # replace embed_texts with something that fails the test if called.
    def _fail_if_called(*args, **kwargs):
        pytest.fail("embed_texts was called for a query against an empty store")

    monkeypatch.setattr(vector_store_module, "embed_texts", _fail_if_called)

    store = VectorStore(tmp_path)

    assert store.query("anything at all") == []


@pytest.mark.model
def test_results_carry_source_metadata(tmp_path):
    store = VectorStore(tmp_path)
    store.add_documents(CHUNKS[:2], source="biology.md")
    store.add_documents(CHUNKS[2:3], source="history.md")

    results = store.query("What produces glucose in plants?", top_k=5)

    by_text = {r["text"]: r for r in results}
    assert by_text[CHUNKS[1]]["source"] == "biology.md"
    assert by_text[CHUNKS[2]]["source"] == "history.md"
    # chunk_index is round-tripped too, as an int
    assert all(isinstance(r["chunk_index"], int) for r in results)


@pytest.mark.model
def test_top_k_is_respected(tmp_path):
    store = VectorStore(tmp_path)
    store.add_documents(CHUNKS, source="notes.md")  # 5 chunks

    results = store.query("Tell me about anything", top_k=2)

    assert len(results) == 2


@pytest.mark.model
def test_re_adding_a_source_replaces_its_chunks_including_a_shorter_version(
    tmp_path,
):
    store = VectorStore(tmp_path)
    store.add_documents(
        [
            "first version paragraph about apples",
            "first version paragraph about pears",
            "first version paragraph about plums",
        ],
        source="doc.md",
    )
    # Re-ingest a shorter version — the stale 2nd/3rd chunks must be gone.
    store.add_documents(
        ["second version, now just one paragraph about oranges"],
        source="doc.md",
    )

    results = store.query("fruit", top_k=10)

    texts = [r["text"] for r in results]
    assert texts == ["second version, now just one paragraph about oranges"]


@pytest.mark.model
def test_data_persists_to_disk_across_a_fresh_process(tmp_path):
    VectorStore(tmp_path).add_documents(CHUNKS, source="notes.md")

    # Query from a brand-new interpreter: chromadb caches its client per
    # path *within* a process, so re-opening in-process would read the same
    # live in-memory index and pass even if on-disk persistence were
    # broken. A subprocess is the only faithful "survives a restart" check.
    backend_dir = Path(__file__).resolve().parents[1]
    script = (
        "from app.retrieval.vector_store import VectorStore\n"
        f"r = VectorStore({str(tmp_path)!r}).query('logarithmic time lookups', top_k=1)\n"
        "print(r[0]['text'])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=backend_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == CHUNKS[4]  # the binary search tree chunk


@pytest.mark.parametrize("bad_question", ["", "   ", "\n\t"])
def test_query_rejects_empty_question(tmp_path, bad_question):
    with pytest.raises(ValueError):
        VectorStore(tmp_path).query(bad_question)


@pytest.mark.parametrize("bad_top_k", [0, -1])
def test_query_rejects_non_positive_top_k(tmp_path, bad_top_k):
    with pytest.raises(ValueError):
        VectorStore(tmp_path).query("a real question", top_k=bad_top_k)


def test_add_documents_rejects_empty_source(tmp_path):
    with pytest.raises(ValueError):
        VectorStore(tmp_path).add_documents(["a chunk"], source="  ")


def test_add_documents_empty_chunk_list_is_a_noop(tmp_path, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        pytest.fail("embed_texts was called for an empty chunk list")

    monkeypatch.setattr(vector_store_module, "embed_texts", _fail_if_called)

    VectorStore(tmp_path).add_documents([], source="doc.md")  # must not raise
