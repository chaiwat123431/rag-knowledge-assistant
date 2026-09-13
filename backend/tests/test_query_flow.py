import pytest

from app.ingestion.chunker import chunk_text
from app.retrieval.query_flow import (
    MAX_AUGMENTED_SCORE_DROP_RATIO,
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


def test_relevance_constants_match_the_documented_measurements():
    # Pins today's values so a future change is a deliberate diff, not a
    # silent drift. See the module docstring for the measurements behind
    # both (2026-09-12: raised the floor after manual testing found 0.15
    # let surface-similarity noise through as citations. 2026-09-13: the
    # drop check became a ratio of the prior-turn score, not an absolute
    # amount, after a real multi-chunk-document recurrence of bug 1).
    assert MIN_RELEVANCE_SCORE == 0.25
    assert MAX_AUGMENTED_SCORE_DROP_RATIO == 0.15


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


@pytest.mark.model
def test_end_to_end_unrelated_question_after_history_has_no_context(tmp_path):
    """Regression test for the exact bug report: a substantive history
    topic must not leak citations when the new question has nothing to do
    with it (or with anything else in the store)."""
    store = VectorStore(tmp_path)
    store.add_documents(
        [
            "The Meridian Bridge renovation replaced the original steel "
            "trusses and added a pedestrian walkway on the lower deck, "
            "completed in 2019."
        ],
        source="bridge.md",
    )
    history = [
        {
            "role": "user",
            "content": "Tell me about the Meridian Bridge renovation",
        },
        {
            "role": "assistant",
            "content": "It replaced the trusses and added a walkway, "
            "finished in 2019 [1].",
        },
    ]

    result = answer_question(
        "What is the capital of France?", store, history=history
    )

    assert result["has_context"] is False
    assert result["citations"] == []


@pytest.mark.model
def test_end_to_end_multi_chunk_document_unrelated_question_has_no_context(
    tmp_path,
):
    """Regression test for the recurrence found via manual UI testing after
    the first fix: a longer, realistic document that the real chunker
    splits into multiple chunks (unlike the single-chunk repro above) must
    still not leak a citation for an unrelated new question. One of the
    chunks here is only weakly anchored to the prior turn (prev_score
    ~0.44) — the bug this test locks in is that an absolute drop threshold
    let exactly that kind of chunk through."""
    store = VectorStore(tmp_path)
    document = (
        "The Meridian Bridge is a suspension bridge completed in 1958, "
        "spanning the Halden River and carrying both rail and vehicle "
        "traffic between the eastern and western districts of the city. "
        "At the time of its construction it was the longest suspension "
        "span in the region and was designed by the engineering firm "
        "Voss & Ardell.\n\n"
        "The bridge underwent a major renovation in 2003, during which "
        "the original suspension cables were replaced with high-tensile "
        "steel cables rated for a 75-year lifespan. The renovation also "
        "added reinforced concrete piers, updated lighting along the "
        "main span, and a dedicated lane for cyclists on the lower "
        "deck.\n\n"
        "Maintenance inspections occur every two years, and the bridge "
        "authority publishes traffic volume statistics annually. The "
        "most recent inspection in 2024 found no structural issues "
        "requiring immediate repair."
    )
    chunks = chunk_text(document)
    assert len(chunks) > 1  # exercising the real chunker, not a hand-picked single chunk
    store.add_documents(chunks, source="test-document.txt")

    history = [
        {
            "role": "user",
            "content": "When was the Meridian Bridge renovated?",
        },
        {
            "role": "assistant",
            "content": "The Meridian Bridge underwent a major renovation "
            "in 2003, during which the original cables were replaced "
            "with high-tensile steel cables rated for a 75-year "
            "lifespan.",
        },
    ]

    result = answer_question(
        "What is the capital of France?", store, history=history
    )

    assert result["has_context"] is False
    assert result["citations"] == []


@pytest.mark.model
def test_end_to_end_genuine_followup_still_finds_the_history_topic(tmp_path):
    """The fix for the bug above must not regress the feature it's a part
    of: a content-free follow-up should still retrieve the prior topic."""
    store = VectorStore(tmp_path)
    store.add_documents(
        [
            "The Meridian Bridge's first span opened in 1932 carrying "
            "rail traffic across the river.",
            "The second span was added in 1961 to carry automobile "
            "traffic alongside the original rail span.",
        ],
        source="bridge.md",
    )
    history = [
        {
            "role": "user",
            "content": "When did the first span of the Meridian Bridge open?",
        },
        {"role": "assistant", "content": "The first span opened in 1932 [1]."},
    ]

    result = answer_question(
        "and the second part?", store, top_k=2, history=history
    )

    assert result["has_context"] is True
    assert any("1961" in c["text"] for c in result["citations"])


@pytest.mark.model
def test_end_to_end_surface_similar_document_is_not_cited(tmp_path):
    """Regression test for the too-permissive floor: measured at cosine
    0.2056, a document sharing only generic vocabulary with the question
    used to clear the old 0.15 floor and pollute the citations."""
    store = VectorStore(tmp_path)
    store.add_documents(
        [
            "To make a classic carbonara, cook pancetta until crisp, then "
            "toss hot pasta with eggs, pecorino cheese, and black pepper "
            "off the heat."
        ],
        source="recipe.md",
    )
    store.add_documents(
        [
            "Photosynthesis in plant chloroplasts converts light energy "
            "into chemical energy, producing glucose and releasing oxygen "
            "as a byproduct."
        ],
        source="biology.md",
    )

    result = answer_question("How do I make carbonara?", store, top_k=5)

    sources = {c["source"] for c in result["citations"]}
    assert sources == {"recipe.md"}


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


def test_retrieval_runs_prev_turn_query_only_when_something_needs_it():
    # The augmented query surfaces a chunk bare didn't confirm -> the
    # reference (prior-turn-alone) query is needed to check it.
    bridge = _result("The Meridian Bridge has two spans.", "bridge.md", 1, 0.60)
    store = FakeVectorStore(
        {
            "and the second part?": [],
            "How is the Meridian Bridge built?\nand the second part?": [bridge],
            "How is the Meridian Bridge built?": [bridge],
        }
    )

    answer_question("and the second part?", store, history=TWO_TURNS)

    # bare question, then (last user turn + question), then the last user
    # turn alone — the third is a reference score, never shown to the user,
    # used to tell a real complement from pure history carryover (the
    # assistant turn is never used for retrieval).
    assert store.received_queries == [
        "and the second part?",
        "How is the Meridian Bridge built?\nand the second part?",
        "How is the Meridian Bridge built?",
    ]


def test_prev_turn_query_is_skipped_when_augmented_adds_nothing_new():
    # Every augmented result is already bare-relevant -> the third query
    # (an extra embedding + Chroma round trip) would be pure overhead, so
    # it's never issued.
    pto = _result("PTO accrues at 1.5 days per month.", "hr.md", 3, 0.71)
    store = FakeVectorStore(
        {
            "How much PTO do I get?": [pto],
            "How is the Meridian Bridge built?\nHow much PTO do I get?": [pto],
        }
    )

    answer_question("How much PTO do I get?", store, history=TWO_TURNS)

    assert store.received_queries == [
        "How much PTO do I get?",
        "How is the Meridian Bridge built?\nHow much PTO do I get?",
    ]


def test_no_history_runs_a_single_retrieval_query():
    store = FakeVectorStore(FIVE_RESULTS)

    answer_question("What powers a cell?", store)

    assert store.received_queries == ["What powers a cell?"]


def test_bare_query_results_are_always_trusted_alongside_augmented():
    # Bare-query matches need no drop-check — the question itself, with no
    # help from history, already justifies them.
    pto = _result("PTO accrues at 1.5 days per month.", "hr.md", 3, 0.71)
    store = FakeVectorStore(
        {
            "How much PTO do I get?": [pto],
            "How is the Meridian Bridge built?\nHow much PTO do I get?": [pto],
        }
    )

    result = answer_question(
        "How much PTO do I get?", store, top_k=5, history=TWO_TURNS
    )

    assert [c["text"] for c in result["citations"]] == [pto["text"]]


def test_augmented_only_chunk_kept_when_score_barely_drops_from_prev_turn():
    # A real complement: the augmented query surfaces a chunk the bare
    # query missed, and its score is close to what the prior turn alone
    # gets for that same chunk (small drop) -> genuine relevance, kept.
    bridge = _result("The Meridian Bridge has two spans.", "bridge.md", 1, 0.60)
    bridge_prev_alone = _result(bridge["text"], "bridge.md", 1, 0.68)
    store = FakeVectorStore(
        {
            "and the second part?": [],
            "How is the Meridian Bridge built?\nand the second part?": [bridge],
            "How is the Meridian Bridge built?": [bridge_prev_alone],
        }
    )

    result = answer_question(
        "and the second part?", store, top_k=5, history=TWO_TURNS
    )

    assert [c["text"] for c in result["citations"]] == [bridge["text"]]


def test_augmented_only_chunk_dropped_when_score_falls_far_below_prev_turn():
    # The bug: an augmented-only chunk whose score collapses once the new
    # (unrelated) question is added is pure history carryover, not
    # relevance to what was actually asked — must be rejected even though
    # it clears MIN_RELEVANCE_SCORE on its own.
    bridge = _result("The Meridian Bridge has two spans.", "bridge.md", 1, 0.52)
    bridge_prev_alone = _result(bridge["text"], "bridge.md", 1, 0.76)
    store = FakeVectorStore(
        {
            "What is the capital of France?": [],
            "How is the Meridian Bridge built?\nWhat is the capital of France?": [
                bridge
            ],
            "How is the Meridian Bridge built?": [bridge_prev_alone],
        }
    )

    result = answer_question(
        "What is the capital of France?", store, top_k=5, history=TWO_TURNS
    )

    assert result["citations"] == []
    assert result["has_context"] is False


def test_irrelevant_bare_presence_does_not_bypass_the_augmented_drop_check():
    # Chroma always returns nearest neighbours for a non-empty store, so a
    # chunk can appear in bare_results with an irrelevant score (e.g. it's
    # the only document in a small store). That must not let an augmented
    # score for the SAME chunk skip the drop-check just because the id
    # already "existed" — this is what let bug 1 slip past the first fix.
    bridge = _result("The Meridian Bridge has two spans.", "bridge.md", 1, 0.52)
    bridge_bare = _result(bridge["text"], "bridge.md", 1, -0.01)
    bridge_prev_alone = _result(bridge["text"], "bridge.md", 1, 0.76)
    store = FakeVectorStore(
        {
            "What is the capital of France?": [bridge_bare],
            "How is the Meridian Bridge built?\nWhat is the capital of France?": [
                bridge
            ],
            "How is the Meridian Bridge built?": [bridge_prev_alone],
        }
    )

    result = answer_question(
        "What is the capital of France?", store, top_k=5, history=TWO_TURNS
    )

    assert result["citations"] == []
    assert result["has_context"] is False


def test_relative_drop_rejects_a_weakly_anchored_chunk_absolute_drop_would_keep():
    # Regression test for the real recurrence: a chunk only weakly tied to
    # the prior turn (low prev_score, e.g. a peripheral detail in a longer,
    # multi-chunk document) needs to lose much less in absolute terms to
    # reveal it's unrelated to the new question too. A flat absolute-drop
    # threshold of 0.15 would have kept this (drop of 0.11); measuring the
    # drop as a fraction of prev_score correctly rejects it (25% loss).
    weak = _result("Bridge maintenance schedule details.", "bridge.md", 2, 0.33)
    weak_prev_alone = _result(weak["text"], "bridge.md", 2, 0.44)
    store = FakeVectorStore(
        {
            "What is the capital of France?": [],
            "How is the Meridian Bridge built?\nWhat is the capital of France?": [
                weak
            ],
            "How is the Meridian Bridge built?": [weak_prev_alone],
        }
    )

    result = answer_question(
        "What is the capital of France?", store, top_k=5, history=TWO_TURNS
    )

    assert result["citations"] == []
    assert result["has_context"] is False


def test_augmented_only_chunk_with_non_positive_prev_score_is_dropped():
    # No meaningful baseline to measure a relative drop against.
    weird = _result("Some chunk.", "x.md", 0, 0.9)
    weird_prev_alone = _result(weird["text"], "x.md", 0, 0.0)
    store = FakeVectorStore(
        {
            "and the second part?": [],
            "How is the Meridian Bridge built?\nand the second part?": [weird],
            "How is the Meridian Bridge built?": [weird_prev_alone],
        }
    )

    result = answer_question(
        "and the second part?", store, top_k=5, history=TWO_TURNS
    )

    assert result["citations"] == []


def test_augmented_only_chunk_dropped_when_absent_from_prev_turn_results():
    # Can't confirm it against the prior turn at all (didn't make that
    # query's own top_k) -> don't let it through unchecked.
    mystery = _result("Some chunk.", "x.md", 0, 0.9)
    store = FakeVectorStore(
        {
            "and the second part?": [],
            "How is the Meridian Bridge built?\nand the second part?": [mystery],
            "How is the Meridian Bridge built?": [],  # mystery absent here
        }
    )

    result = answer_question(
        "and the second part?", store, top_k=5, history=TWO_TURNS
    )

    assert result["citations"] == []


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
