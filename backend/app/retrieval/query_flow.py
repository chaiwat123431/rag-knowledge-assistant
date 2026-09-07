"""Query flow — retrieved chunks in, LLM-ready prompt + citations out.

This is the "Build prompt with retrieved context" step of PLANNING.md's
query data flow. It deliberately stops short of calling an LLM: the actual
Ollama/Gemini call is a separate task, gated on confirming the prompt and
citation structure here are right.

Prompt shape
------------
Each retrieved chunk is given a 1-based marker (``[1]``, ``[2]``, ...) in
relevance order and printed with its provenance:

    [1] (source: notes.md, chunk 3)
    <chunk text>

The instructions tell the model to answer *only* from the context, cite
inline with those markers, and admit when the context doesn't cover the
question. The marker is the link back to a citation: an answer containing
``[2]`` refers to ``result["citations"][1]`` — `answer_question` builds
the prompt and the citation list from the one filtered, ordered list, so
the numbering can't drift.

Relevance floor
---------------
Chroma always returns *some* nearest chunks for a non-empty store, however
unrelated. `answer_question` drops results scoring below
`MIN_RELEVANCE_SCORE` (cosine similarity) before building anything, so an
off-topic question doesn't get unrelated passages injected as "context"
and cited as "sources". If nothing clears the floor the result is the
same as an empty store: `has_context` False, empty `citations`, and a
"no information" prompt. `top_k` is therefore a retrieval cap, not a
guarantee — you can ask for 5 and get 2 citations, or 0.

The floor is a heuristic; revisit it once retrieval quality is actually
measured (same caveat as the chunker's char-based sizing).
"""

from app.retrieval.vector_store import VectorStore

# Cosine similarity below this is treated as "not really about this".
# MiniLM puts genuinely unrelated short passages well under 0.15;
# related-but-different material sits above it.
MIN_RELEVANCE_SCORE = 0.15

_INSTRUCTIONS = (
    "You are a helpful assistant answering questions about the user's "
    "personal documents.\n"
    "Use ONLY the context passages below to answer. Each passage is "
    "labelled with a number in brackets, like [1]. When something in your "
    "answer comes from a passage, cite it inline with that number, e.g. "
    '"... as the report notes [2]." Cite every claim you make.\n'
    "If the context does not contain the answer, say you don't know — do "
    "not fall back on outside knowledge."
)

_NO_CONTEXT_INSTRUCTIONS = (
    "You are a helpful assistant answering questions about the user's "
    "personal documents.\n"
    "No relevant passages were found in the knowledge base for the "
    "question below. Tell the user you don't have information about this "
    "in their documents. Do not answer from outside knowledge."
)


def _source_label(result: dict) -> str:
    # source is always present from VectorStore.query, but can be None if a
    # chunk was written without it out of band.
    return result.get("source") or "unknown"


def _format_context_block(marker: int, result: dict) -> str:
    """Render one retrieved chunk as a labelled context passage."""
    source = _source_label(result)
    chunk_index = result.get("chunk_index")
    locator = (
        f"source: {source}, chunk {chunk_index}"
        if chunk_index is not None
        else f"source: {source}"
    )
    return f"[{marker}] ({locator})\n{result['text']}"


def build_prompt(question: str, results: list[dict]) -> str:
    """Build the full LLM prompt for `question` given retrieved `results`.

    `results` are chunk dicts (as from `VectorStore.query`) in the order
    they should appear. Markers are assigned ``[1]..[N]`` over that order
    as-is — no reordering or slicing here, so a caller can enumerate the
    same list to get matching citation markers. Empty `results` produces
    the no-context prompt.
    """
    if not results:
        return f"{_NO_CONTEXT_INSTRUCTIONS}\n\nQuestion: {question}\n\nAnswer:"

    blocks = [
        _format_context_block(marker, result)
        for marker, result in enumerate(results, start=1)
    ]
    context = "\n\n".join(blocks)
    return (
        f"{_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def _to_citation(marker: int, result: dict) -> dict:
    return {
        "marker": marker,
        "source": _source_label(result),
        "chunk_index": result.get("chunk_index"),
        "text": result["text"],
        "score": result.get("score"),
    }


def _is_relevant(result: dict, min_score: float) -> bool:
    score = result.get("score")
    return score is not None and score >= min_score


def answer_question(
    question: str,
    vector_store: VectorStore,
    top_k: int = 5,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> dict:
    """Retrieve context for `question` and build an LLM-ready prompt.

    Does NOT call an LLM — that's a later step. This just runs retrieval,
    drops chunks below the relevance floor, and assembles the prompt +
    citation list.

    Args:
        question: the user's natural-language question.
        vector_store: the store to retrieve from.
        top_k: max chunks to retrieve. A cap, not a guarantee — fewer (or
            zero) survive the relevance filter.
        min_score: cosine-similarity floor; results scoring below it are
            ignored. Defaults to `MIN_RELEVANCE_SCORE`.

    Returns:
        A dict with stable keys in both the found and not-found cases:
          - ``question`` (str): echoed back.
          - ``prompt`` (str): the full text to send to the LLM. When
            nothing cleared the relevance floor this is a short "no
            information" message instead of a context prompt.
          - ``has_context`` (bool): True iff at least one chunk cleared the
            floor. Callers can short-circuit the LLM call on this.
          - ``citations`` (list[dict]): one per included chunk, in prompt
            order, each ``{marker, source, chunk_index, text, score}``.
            The ``marker`` matches the ``[n]`` used in the prompt.

    Raises:
        ValueError: if `question` is empty/whitespace-only or `top_k` < 1
            (raised by `vector_store.query`).
    """
    results = vector_store.query(question, top_k=top_k)
    relevant = [r for r in results if _is_relevant(r, min_score)]

    return {
        "question": question,
        "prompt": build_prompt(question, relevant),
        "has_context": bool(relevant),
        "citations": [
            _to_citation(marker, result)
            for marker, result in enumerate(relevant, start=1)
        ],
    }
