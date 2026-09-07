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
``[2]`` refers to ``result["citations"][1]`` (same order as the prompt).

When retrieval returns nothing, no context prompt is built — the returned
prompt is instead a short message telling the model to say it has no
information on the topic, ``has_context`` is False, and ``citations`` is
empty. Callers can either send that prompt as-is or short-circuit on
``has_context``.
"""

from app.retrieval.vector_store import VectorStore

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


def _format_context_block(marker: int, result: dict) -> str:
    """Render one retrieved chunk as a labelled context passage."""
    source = result.get("source", "unknown")
    chunk_index = result.get("chunk_index")
    locator = (
        f"source: {source}, chunk {chunk_index}"
        if chunk_index is not None
        else f"source: {source}"
    )
    return f"[{marker}] ({locator})\n{result['text']}"


def build_prompt(question: str, results: list[dict]) -> str:
    """Build the full LLM prompt for `question` given retrieved `results`.

    `results` are the dicts returned by `VectorStore.query` (already in
    nearest-first order). An empty `results` produces the no-context
    prompt.
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
        "source": result.get("source"),
        "chunk_index": result.get("chunk_index"),
        "text": result["text"],
        "score": result.get("score"),
    }


def answer_question(
    question: str, vector_store: VectorStore, top_k: int = 5
) -> dict:
    """Retrieve context for `question` and build an LLM-ready prompt.

    Does NOT call an LLM — that's a later step. This just runs retrieval
    and assembles the prompt + citation list.

    Args:
        question: the user's natural-language question.
        vector_store: the store to retrieve from.
        top_k: max chunks to retrieve and include.

    Returns:
        A dict with stable keys in both the found and not-found cases:
          - ``question`` (str): echoed back.
          - ``prompt`` (str): the full text to send to the LLM. When no
            context was found this is a short "no information" message
            instead of a context prompt.
          - ``has_context`` (bool): False when retrieval returned nothing
            (empty store, or nothing similar enough is still returned by
            Chroma as long as the store is non-empty — so in practice this
            is False only for an empty store). Callers can short-circuit
            the LLM call on this.
          - ``citations`` (list[dict]): one per included chunk, in prompt
            order, each ``{marker, source, chunk_index, text, score}``.
            The ``marker`` matches the ``[n]`` used in the prompt.

    Raises:
        ValueError: if `question` is empty/whitespace-only or `top_k` < 1
            (raised by `vector_store.query`).
    """
    results = vector_store.query(question, top_k=top_k)

    citations = [
        _to_citation(marker, result)
        for marker, result in enumerate(results, start=1)
    ]

    return {
        "question": question,
        "prompt": build_prompt(question, results),
        "has_context": bool(results),
        "citations": citations,
    }
