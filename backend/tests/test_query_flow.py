import pytest

from app.retrieval.query_flow import (
    MIN_RELEVANCE_SCORE,
    answer_question,
    build_prompt,
)
from app.retrieval.vector_store import VectorStore


class FakeVectorStore:
    """Stands in for VectorStore so these tests exercise prompt/citation
    assembly in isolation, without loading the embedding model or Chroma
    (retrieval itself is covered by test_vector_store.py)."""

    def __init__(self, results: list[dict]):
        self._results = results
        self.received_question = None
        self.received_top_k = None

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.received_question = question
        self.received_top_k = top_k
        return self._results[:top_k]


def _result(text, source, chunk_index, score):
    return {
        "id": f"{source}::{chunk_index}",
        "text": text,
        "source": source,
        "chunk_index": chunk_index,
        "score": score,
    }


FIVE_RESULTS = [
    _result("Mitochondria produce ATP for the cell.", "biology.md", 2, 0.81),
    _result("Photosynthesis makes glucose from sunlight.", "biology.md", 0, 0.74),
    _result("The Bastille was stormed in 1789.", "history.md", 5, 0.31),
    _result("Compound interest compounds over time.", "finance.md", 1, 0.22),
    _result("A binary search tree is ordered.", "cs.md", 3, 0.18),
]


def test_prompt_and_citations_when_relevant_chunks_exist():
    store = FakeVectorStore(FIVE_RESULTS)

    result = answer_question("How do cells make energy?", store, top_k=3)

    prompt = result["prompt"]
    # the question is in the prompt
    assert "How do cells make energy?" in prompt
    # the text of each included chunk is in the prompt
    for r in FIVE_RESULTS[:3]:
        assert r["text"] in prompt
    # excluded chunks are not
    assert FIVE_RESULTS[3]["text"] not in prompt

    # markers are 1..N in relevance order, each next to its source
    assert "[1] (source: biology.md, chunk 2)" in prompt
    assert "[2] (source: biology.md, chunk 0)" in prompt
    assert "[3] (source: history.md, chunk 5)" in prompt

    assert result["has_context"] is True

    citations = result["citations"]
    assert [c["marker"] for c in citations] == [1, 2, 3]
    assert [c["source"] for c in citations] == [
        "biology.md",
        "biology.md",
        "history.md",
    ]
    assert [c["chunk_index"] for c in citations] == [2, 0, 5]
    assert citations[0]["text"] == FIVE_RESULTS[0]["text"]
    assert citations[0]["score"] == pytest.approx(0.81)


def test_all_results_below_relevance_floor_are_treated_as_no_context():
    # Non-empty store, but the question is unrelated to everything in it:
    # Chroma still returns nearest chunks, just with low scores.
    low = [
        _result("The Bastille was stormed in 1789.", "history.md", 5, 0.07),
        _result("Compound interest compounds over time.", "finance.md", 1, 0.04),
    ]
    store = FakeVectorStore(low)

    result = answer_question("How does photosynthesis work?", store)

    assert result["has_context"] is False
    assert result["citations"] == []
    assert "Context:" not in result["prompt"]
    assert "don't have information" in result["prompt"].lower()
    # the unrelated sources must NOT appear anywhere
    assert "history.md" not in result["prompt"]
    assert "finance.md" not in result["prompt"]


def test_only_results_above_the_floor_are_kept_and_renumbered():
    mixed = [
        _result("Mitochondria produce ATP.", "biology.md", 2, 0.80),
        _result("Totally unrelated aside.", "misc.md", 9, 0.05),
        _result("Cells respire to release energy.", "biology.md", 7, 0.42),
    ]
    store = FakeVectorStore(mixed)

    result = answer_question("How do cells make energy?", store)

    assert result["has_context"] is True
    assert [c["marker"] for c in result["citations"]] == [1, 2]
    assert [c["chunk_index"] for c in result["citations"]] == [2, 7]
    assert "misc.md" not in result["prompt"]
    assert "Totally unrelated aside." not in result["prompt"]
    assert "[1] (source: biology.md, chunk 2)" in result["prompt"]
    assert "[2] (source: biology.md, chunk 7)" in result["prompt"]


def test_min_score_can_be_overridden():
    store = FakeVectorStore(FIVE_RESULTS)  # scores 0.81 .. 0.18

    result = answer_question("q", store, min_score=0.5)

    # only 0.81 and 0.74 clear a 0.5 floor
    assert [c["source"] for c in result["citations"]] == ["biology.md", "biology.md"]
    assert [c["chunk_index"] for c in result["citations"]] == [2, 0]


def test_default_floor_matches_the_module_constant():
    # scores straddling MIN_RELEVANCE_SCORE
    straddle = [
        _result("keep me", "a.md", 0, MIN_RELEVANCE_SCORE),
        _result("drop me", "b.md", 0, MIN_RELEVANCE_SCORE - 0.01),
    ]
    store = FakeVectorStore(straddle)

    result = answer_question("q", store)

    assert [c["text"] for c in result["citations"]] == ["keep me"]


def test_missing_source_renders_as_unknown_in_prompt_and_citation():
    result_dict = {
        "id": "x::0",
        "text": "orphan chunk",
        "source": None,
        "chunk_index": 0,
        "score": 0.9,
    }
    store = FakeVectorStore([result_dict])

    result = answer_question("q", store)

    assert "[1] (source: unknown, chunk 0)" in result["prompt"]
    assert result["citations"][0]["source"] == "unknown"


def test_empty_store_returns_clear_no_context_result_without_error():
    store = FakeVectorStore([])

    result = answer_question("Anything about quantum gravity?", store)

    assert result["has_context"] is False
    assert result["citations"] == []
    assert result["question"] == "Anything about quantum gravity?"
    # prompt is a usable message, not empty / not a context prompt
    assert result["prompt"]
    assert "Context:" not in result["prompt"]
    assert "Anything about quantum gravity?" in result["prompt"]
    assert "don't have information" in result["prompt"].lower()


def test_top_k_limits_chunks_in_prompt_and_citations():
    store = FakeVectorStore(FIVE_RESULTS)

    result = answer_question("tell me anything", store, top_k=2)

    assert store.received_top_k == 2
    assert len(result["citations"]) == 2
    # exactly two numbered markers -> "[1]" and "[2]" present, "[3]" not
    assert "[1] (" in result["prompt"]
    assert "[2] (" in result["prompt"]
    assert "[3] (" not in result["prompt"]


def test_default_top_k_is_five():
    store = FakeVectorStore(FIVE_RESULTS)

    answer_question("q", store)

    assert store.received_top_k == 5


def test_invalid_question_propagates_value_error():
    store = FakeVectorStore(FIVE_RESULTS)

    with pytest.raises(ValueError):
        answer_question("   ", store)


def test_build_prompt_directly_with_no_results_is_the_no_context_prompt():
    prompt = build_prompt("what is X?", [])

    assert "what is X?" in prompt
    assert "Context:" not in prompt
    assert prompt.endswith("Answer:")


@pytest.mark.model
def test_end_to_end_with_a_real_vector_store(tmp_path):
    store = VectorStore(tmp_path)
    store.add_documents(
        [
            "The mitochondrion is the powerhouse of the cell, making ATP.",
            "The French Revolution began in 1789.",
        ],
        source="notes.md",
    )

    result = answer_question("How do cells produce energy?", store, top_k=1)

    assert result["has_context"] is True
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source"] == "notes.md"
    assert "mitochondrion" in result["prompt"]
    assert "How do cells produce energy?" in result["prompt"]


def test_end_to_end_empty_real_store(tmp_path):
    # An empty store answers without embedding the question, so this needs
    # neither the model nor a download — keep it in the fast default run.
    result = answer_question("anything", VectorStore(tmp_path))

    assert result["has_context"] is False
    assert result["citations"] == []


def test_citation_markers_match_prompt_markers():
    store = FakeVectorStore(FIVE_RESULTS)

    result = answer_question("q", store, top_k=4)

    for citation in result["citations"]:
        marker = citation["marker"]
        # the passage for [marker] in the prompt is followed by that
        # citation's own text
        block_header = f"[{marker}] (source: {citation['source']}"
        assert block_header in result["prompt"]
