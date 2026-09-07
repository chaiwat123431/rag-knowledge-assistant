import pytest

from app.retrieval.query_flow import (
    MAX_HISTORY_MESSAGES,
    MIN_RELEVANCE_SCORE,
    NO_CONTEXT_ANSWER,
    InvalidHistoryError,
    answer_question,
    answer_with_llm,
    build_prompt,
)
from app.retrieval.vector_store import VectorStore


class FakeVectorStore:
    """Stands in for VectorStore so these tests exercise prompt/citation
    assembly in isolation, without loading the embedding model or Chroma
    (retrieval itself is covered by test_vector_store.py).

    `results` may be a flat list (returned for any query) or a dict mapping
    an exact query string to its result list (for testing dual retrieval).
    """

    def __init__(self, results):
        self._results = results
        self.received_queries: list[str] = []
        self.received_top_k = None

    @property
    def received_question(self):  # back-compat for older tests
        return self.received_queries[-1] if self.received_queries else None

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.received_queries.append(question)
        self.received_top_k = top_k
        if isinstance(self._results, dict):
            return self._results.get(question, [])[:top_k]
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


# --- conversation history -----------------------------------------------

TWO_TURNS = [
    {"role": "user", "content": "How is the Meridian Bridge built?"},
    {"role": "assistant", "content": "It has a lower truss and an upper deck."},
]


def test_history_is_rendered_into_the_prompt():
    store = FakeVectorStore(FIVE_RESULTS)

    result = answer_question(
        "and the second part?", store, top_k=2, history=TWO_TURNS
    )

    prompt = result["prompt"]
    assert "Conversation so far:" in prompt
    assert "User: How is the Meridian Bridge built?" in prompt
    assert "Assistant: It has a lower truss and an upper deck." in prompt
    # history sits between the instructions and the Context block
    assert prompt.index("Conversation so far:") < prompt.index("Context:")
    assert prompt.index("Context:") < prompt.index("Question: and the second part?")


def test_no_history_leaves_the_prompt_byte_identical():
    store_a = FakeVectorStore(FIVE_RESULTS)
    store_b = FakeVectorStore(FIVE_RESULTS)
    store_c = FakeVectorStore(FIVE_RESULTS)

    without = answer_question("What powers a cell?", store_a, top_k=3)
    none_ = answer_question("What powers a cell?", store_b, top_k=3, history=None)
    empty = answer_question("What powers a cell?", store_c, top_k=3, history=[])

    assert without["prompt"] == none_["prompt"] == empty["prompt"]
    assert "Conversation so far:" not in without["prompt"]
    # retrieval query is the bare question when there's no history
    assert store_a.received_question == "What powers a cell?"


def test_retrieval_runs_both_the_bare_and_augmented_queries():
    store = FakeVectorStore(FIVE_RESULTS)

    answer_question("and the second part?", store, history=TWO_TURNS)

    # query 1: the question as-is; query 2: last *user* turn + question
    # (the assistant turn is not used for retrieval)
    assert store.received_queries == [
        "and the second part?",
        "How is the Meridian Bridge built?\nand the second part?",
    ]


def test_no_history_runs_a_single_retrieval_query():
    store = FakeVectorStore(FIVE_RESULTS)

    answer_question("What powers a cell?", store)

    assert store.received_queries == ["What powers a cell?"]


def test_dual_retrieval_merges_results_from_both_queries():
    # A self-contained new question on a different topic: its own chunks
    # come from the bare query, the previous topic's from the augmented
    # one — both should be considered, best score per chunk, top_k overall.
    pto = _result("PTO accrues at 1.5 days per month.", "hr.md", 3, 0.71)
    bridge = _result("The Meridian Bridge has two spans.", "bridge.md", 1, 0.66)
    store = FakeVectorStore(
        {
            "How much PTO do I get?": [pto],
            "How is the Meridian Bridge built?\nHow much PTO do I get?": [
                bridge,
                pto,
            ],
        }
    )

    result = answer_question(
        "How much PTO do I get?", store, top_k=5, history=TWO_TURNS
    )

    texts = {c["text"] for c in result["citations"]}
    assert pto["text"] in texts  # not crowded out by the old topic
    assert bridge["text"] in texts
    # merged, ranked by score, no duplicate of pto
    assert [c["text"] for c in result["citations"]] == [pto["text"], bridge["text"]]


def test_empty_question_with_history_still_raises_value_error():
    store = FakeVectorStore(FIVE_RESULTS)

    with pytest.raises(ValueError):
        answer_question("   ", store, history=TWO_TURNS)


def test_prior_answer_citation_markers_are_stripped_from_history_block():
    store = FakeVectorStore(FIVE_RESULTS)
    history = [
        {"role": "user", "content": "What is the deck made of?"},
        {"role": "assistant", "content": "The deck is steel [2] and was added later [3]."},
    ]

    result = answer_question("and the trusses?", store, top_k=1, history=history)

    hist_block = (
        result["prompt"].split("Conversation so far:")[1].split("Context:")[0]
    )
    assert "Assistant: The deck is steel and was added later." in hist_block
    assert "[2]" not in hist_block
    assert "[3]" not in hist_block


def test_history_is_trimmed_to_the_last_max_messages():
    store = FakeVectorStore(FIVE_RESULTS)
    long_history = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"<<msg-{i:02d}>>",  # not a prefix of one another
        }
        for i in range(40)
    ]

    result = answer_question("q", store, top_k=2, history=long_history)

    prompt = result["prompt"]
    for message in long_history[-MAX_HISTORY_MESSAGES:]:
        assert message["content"] in prompt
    for message in long_history[:-MAX_HISTORY_MESSAGES]:
        assert message["content"] not in prompt


@pytest.mark.parametrize(
    "bad_history",
    [
        [{"role": "system", "content": "nope"}],
        [{"role": "user", "content": "ok"}, {"role": "bot", "content": "x"}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": "   "}],
        [{"role": "user", "content": None}],
        [{"role": "user"}],  # missing content
        [{"content": "no role"}],
        ["not an object"],
        "not a list",
    ],
)
def test_malformed_history_raises_invalid_history_error(bad_history):
    store = FakeVectorStore(FIVE_RESULTS)

    with pytest.raises(InvalidHistoryError):
        answer_question("q", store, history=bad_history)


def test_answer_with_llm_forwards_history_to_the_prompt():
    store = FakeVectorStore(FIVE_RESULTS)
    seen = {}

    def generate(prompt, model="fake"):
        seen["prompt"] = prompt
        return "answer"

    answer_with_llm(
        "and the second part?", store, top_k=2,
        history=TWO_TURNS, generate=generate,
    )

    assert "Conversation so far:" in seen["prompt"]
    assert "User: How is the Meridian Bridge built?" in seen["prompt"]


def test_answer_with_llm_rejects_malformed_history_before_calling_llm():
    store = FakeVectorStore(FIVE_RESULTS)

    with pytest.raises(InvalidHistoryError):
        answer_with_llm(
            "q", store,
            history=[{"role": "nope", "content": "x"}],
            generate=lambda *a, **k: pytest.fail("LLM must not be called"),
        )


# --- answer_with_llm: the full pipeline, with an injected fake LLM --------


def test_answer_with_llm_calls_generate_with_the_built_prompt():
    store = FakeVectorStore(FIVE_RESULTS)
    seen = {}

    def generate(prompt, model="fake"):
        seen["prompt"] = prompt
        seen["model"] = model
        return "Cells make energy in the mitochondria [1]."

    result = answer_with_llm(
        "How do cells make energy?", store, top_k=2,
        generate=generate, model="llama3.2",
    )

    assert result["answer"] == "Cells make energy in the mitochondria [1]."
    assert result["has_context"] is True
    assert seen["model"] == "llama3.2"
    # the prompt handed to the LLM is the one answer_question built
    assert "How do cells make energy?" in seen["prompt"]
    assert FIVE_RESULTS[0]["text"] in seen["prompt"]
    # citations carried through, matching the [n] markers in that prompt
    assert [c["marker"] for c in result["citations"]] == [1, 2]
    assert result["prompt"] == seen["prompt"]


def test_answer_with_llm_short_circuits_when_no_context():
    store = FakeVectorStore([])
    called = False

    def generate(prompt, model="fake"):
        nonlocal called
        called = True
        return "should not happen"

    result = answer_with_llm("unrelated question", store, generate=generate)

    assert called is False
    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["has_context"] is False
    assert result["citations"] == []


def test_answer_with_llm_short_circuits_when_all_below_floor():
    low = [_result("off topic", "x.md", 0, 0.02)]
    store = FakeVectorStore(low)

    result = answer_with_llm(
        "q", store, generate=lambda *a, **k: pytest.fail("LLM called")
    )

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["has_context"] is False


def test_answer_with_llm_propagates_generate_errors():
    store = FakeVectorStore(FIVE_RESULTS)

    def generate(prompt, model="fake"):
        raise RuntimeError("ollama exploded")

    with pytest.raises(RuntimeError, match="ollama exploded"):
        answer_with_llm("q", store, generate=generate)


def test_answer_with_llm_uses_real_generate_answer_by_default():
    # Not calling it — just checking the default wiring points at the llm
    # module, so a missing `generate` arg would hit Ollama, not a stub.
    from app.retrieval import llm

    import app.retrieval.query_flow as qf

    assert qf.generate_answer is llm.generate_answer


# `model` only, not `ollama`: this needs the embeddings model download too,
# so it doesn't belong in the fast HTTP-only `-m ollama` run. Under
# `-m model` it skips cleanly when Ollama isn't up.
@pytest.mark.model
def test_full_pipeline_end_to_end_real_ollama(tmp_path):
    import httpx

    try:
        httpx.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        pytest.skip("Ollama not reachable on localhost:11434")

    store = VectorStore(tmp_path)
    store.add_documents(
        [
            "Aurelia's Cafe on Pine Street opens at 7am on weekdays.",
            "The public library is closed on Mondays.",
        ],
        source="notes.md",
    )

    result = answer_with_llm(
        "What time does Aurelia's Cafe open on weekdays?",
        store,
        top_k=2,
        model="llama3.2",
    )

    assert result["has_context"] is True
    assert result["citations"][0]["source"] == "notes.md"
    assert "7" in result["answer"] or "seven" in result["answer"].lower()
