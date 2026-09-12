"""Query flow — question in, grounded answer + citations out.

Two entry points:

- `answer_question(...)` runs retrieval and assembles an LLM-ready prompt
  + citation list, but does NOT call an LLM. It stays hermetic and fast,
  and is the base the tests exercise.
- `answer_with_llm(...)` is the full pipeline: it calls `answer_question`,
  then sends the prompt to an LLM (`app.retrieval.llm.generate_answer` by
  default, injectable) and returns the answer text alongside the
  citations. No context found -> a canned answer, no LLM round-trip.

Both take an optional `history` (a list of ``{"role", "content"}`` messages,
stateless "client sends the transcript" style). `history=None` is exactly
the old behaviour.

Conversation history
--------------------
History is placed *after* the instructions and *before* the retrieved
Context, as a "Conversation so far:" block. It's the least authoritative
material in the prompt (the user's earlier questions and the model's own
earlier answers), so Context and the current Question stay closest to the
generation point. Markers still reference only the retrieved Context.

Only the last `MAX_HISTORY_MESSAGES` messages are kept — a follow-up
almost always refers to the immediately preceding turn, and older turns
just inflate the prompt.

Retrieval also uses history. A bare follow-up like "and the second part?"
has no topical content of its own — retrieved alone it finds nothing. So
when there's a prior user turn we also try (that turn + the current
question) and merge in whatever it finds. That alone is not safe, though:
concatenating an unrelated new question onto a substantive prior topic
still scores well against that topic's own chunks, because the combined
embedding is dominated by the (real, long) prior turn — measured at
cosine ~0.52 for a totally unrelated question, versus ~0.0 for the bare
question alone. So an augmented-only match is trusted only if adding the
current question didn't erode its score against the prior-turn-alone
query by more than `MAX_AUGMENTED_SCORE_DROP` — a large drop means the new
question is pulling away from the old topic (measured: -0.19 to -0.28 for
genuinely unrelated follow-ups, versus -0.03 to -0.07 for real
context-dependent ones — a clean, wide gap). Chunks that clear the floor
via the bare query need no such check; the question itself justifies them.
Assistant turns are never used for retrieval (long, carry the model's own
phrasing). No LLM-based query rewriting — this is the MVP heuristic.

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

The floor was revisited (2026-09-12) after manual testing found it too
permissive with several documents in the store: measured across 6
topically-distinct documents x 6 matching questions, correct-document
scores landed at 0.55-0.78 while surface-noise scores from unrelated
documents mostly stayed under 0.12 but spiked as high as 0.21 (e.g. a
"binary search tree" chunk scoring 0.18-0.21 against unrelated biology
questions, just from generic phrasing overlap). 0.15 sat inside that noise
band; 0.25 sits in the wide gap above every measured false positive and
below every measured true positive.
"""

import re

from app.retrieval.llm import DEFAULT_MODEL, generate_answer
from app.retrieval.vector_store import VectorStore

# Cosine similarity below this is treated as "not really about this". See
# "Relevance floor" above for the measurements behind 0.25 (raised from an
# earlier 0.15 that let surface-similarity noise through as citations).
MIN_RELEVANCE_SCORE = 0.25

# How much an augmented-query chunk's score is allowed to drop relative to
# the previous-user-turn-alone score before it's treated as pure history
# carryover rather than a real match to the current question. See
# "Retrieval also uses history" above for the measurements behind 0.15.
MAX_AUGMENTED_SCORE_DROP = 0.15

# Returned as the answer when retrieval found nothing relevant — no point
# spending an LLM round-trip to have it say the same thing.
NO_CONTEXT_ANSWER = (
    "I don't have information about that in your documents."
)

# Keep the last ~3 exchanges of conversation history. Older turns rarely
# matter for a follow-up and just inflate the prompt / latency.
MAX_HISTORY_MESSAGES = 6

_VALID_ROLES = ("user", "assistant")

_INSTRUCTIONS = (
    "You are a helpful assistant answering questions about the user's "
    "personal documents.\n"
    "Use ONLY the context passages below to answer. Each passage is "
    "labelled with a number in brackets, like [1]. When something in your "
    "answer comes from a passage, cite it inline with that number, e.g. "
    '"... as the report notes [2]." Cite every claim you make.\n'
    "If earlier conversation turns are shown, use them only to understand "
    "what a follow-up question refers to — still answer only from the "
    "context passages.\n"
    "If the context does not contain the answer, say you don't know — do "
    "not fall back on outside knowledge."
)


class InvalidHistoryError(ValueError):
    """Raised when a conversation-history entry is malformed."""

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


def _validate_history(history) -> list[dict]:
    """Return `history` unchanged (or [] for None), or raise.

    Every entry must be an object with a ``role`` of "user"/"assistant"
    and a non-empty string ``content``.
    """
    if history is None:
        return []
    if not isinstance(history, list):
        raise InvalidHistoryError("history must be a list of messages")

    for i, message in enumerate(history):
        if not isinstance(message, dict):
            raise InvalidHistoryError(
                f"history[{i}] must be an object with 'role' and 'content'"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in _VALID_ROLES:
            raise InvalidHistoryError(
                f"history[{i}].role must be one of {list(_VALID_ROLES)}, "
                f"got {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise InvalidHistoryError(
                f"history[{i}].content must be a non-empty string"
            )
    return history


def _last_user_turn(history: list[dict]) -> str | None:
    """The most recent *user* message in `history`, or None.

    Assistant turns are never used to drive retrieval (long, carry the
    model's own phrasing).
    """
    return next(
        (m["content"] for m in reversed(history) if m["role"] == "user"),
        None,
    )


_CITATION_MARKER_RE = re.compile(r"\s*\[\d+\]")


def _format_history(history: list[dict]) -> str:
    lines = []
    for message in history:
        if message["role"] == "user":
            lines.append(f"User: {message['content']}")
        else:
            # Strip our own [n] citation markers from prior answers: they
            # refer to that turn's context, not this one's, so leaving them
            # in invites the model to reuse numbers that no longer map to
            # anything in `citations`.
            cleaned = _CITATION_MARKER_RE.sub("", message["content"])
            lines.append(f"Assistant: {cleaned}")
    return "\n".join(lines)


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


def build_prompt(
    question: str, results: list[dict], history: list[dict] | None = None
) -> str:
    """Build the full LLM prompt for `question` given retrieved `results`.

    `results` are chunk dicts (as from `VectorStore.query`) in the order
    they should appear. Markers are assigned ``[1]..[N]`` over that order
    as-is — no reordering or slicing here, so a caller can enumerate the
    same list to get matching citation markers. Empty `results` produces
    the no-context prompt.

    `history` (already validated and trimmed by the caller) is rendered as
    a "Conversation so far:" block between the instructions and the
    context. Falsy history adds nothing.
    """
    if not results:
        return f"{_NO_CONTEXT_INSTRUCTIONS}\n\nQuestion: {question}\n\nAnswer:"

    history_block = (
        f"Conversation so far:\n{_format_history(history)}\n\n"
        if history
        else ""
    )
    blocks = [
        _format_context_block(marker, result)
        for marker, result in enumerate(results, start=1)
    ]
    context = "\n\n".join(blocks)
    return (
        f"{_INSTRUCTIONS}\n\n"
        f"{history_block}"
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
    history: list[dict] | None = None,
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
        history: optional prior conversation as ``{"role", "content"}``
            messages. Used two ways: it drives a second retrieval query
            (last user turn + question) whose results are merged in, and
            the last `MAX_HISTORY_MESSAGES` messages go into the prompt.
            `None` / `[]` behaves as before.

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
        InvalidHistoryError: if `history` is malformed.
        ValueError: if `question` is empty/whitespace-only or `top_k` < 1.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    trimmed = _validate_history(history)[-MAX_HISTORY_MESSAGES:]

    results = _retrieve(vector_store, question, trimmed, top_k, min_score)
    relevant = [r for r in results if _is_relevant(r, min_score)]

    return {
        "question": question,
        "prompt": build_prompt(question, relevant, history=trimmed),
        "has_context": bool(relevant),
        "citations": [
            _to_citation(marker, result)
            for marker, result in enumerate(relevant, start=1)
        ],
    }


def _retrieve(
    vector_store: VectorStore,
    question: str,
    history: list[dict],
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Retrieve for `question`, folding in a history-augmented query
    without letting it manufacture false relevance from the prior topic
    alone (see "Retrieval also uses history" in the module docstring).

    A chunk is trusted at face value if the *bare* question alone clears
    `min_score` for it. Note this is stricter than merely "present in
    `bare_results`": Chroma always returns its nearest neighbours for a
    non-empty store, so a chunk can appear there with an irrelevant score
    (e.g. a lone unrelated document, matched by default) — that's not
    bare-question relevance, and must not let an augmented-query score for
    the same chunk skip the check below. Everything else — augmented-only,
    or bare-present but below the floor — is trusted only if the current
    question didn't erode its score against the prior-turn-alone query by
    more than `MAX_AUGMENTED_SCORE_DROP`: a large drop means the question
    is pulling away from the old topic, not continuing it.
    """
    bare_results = vector_store.query(question, top_k=top_k)

    last_user = _last_user_turn(history)
    if last_user is None:
        return bare_results

    bare_relevant_ids = {
        r["id"] for r in bare_results if _is_relevant(r, min_score)
    }
    augmented_results = vector_store.query(
        f"{last_user}\n{question}", top_k=top_k
    )
    prev_turn_scores = {
        r["id"]: r["score"]
        for r in vector_store.query(last_user, top_k=top_k)
    }

    merged: dict[str, dict] = {r["id"]: r for r in bare_results}
    for result in augmented_results:
        if result["id"] in bare_relevant_ids:
            if result["score"] > merged[result["id"]]["score"]:
                merged[result["id"]] = result
            continue

        prev_score = prev_turn_scores.get(result["id"])
        if prev_score is None:
            # Not confirmed against the prior turn either -> can't tell
            # whether it's carryover; don't let it in unchecked.
            continue
        if prev_score - result["score"] > MAX_AUGMENTED_SCORE_DROP:
            continue
        merged[result["id"]] = result

    ranked = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:top_k]


def answer_with_llm(
    question: str,
    vector_store: VectorStore,
    top_k: int = 5,
    min_score: float = MIN_RELEVANCE_SCORE,
    history: list[dict] | None = None,
    *,
    generate=generate_answer,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Full pipeline: retrieve, build prompt, generate a grounded answer.

    Args:
        question, vector_store, top_k, min_score, history: passed to
            `answer_question`.
        generate: callable ``(prompt, model=...) -> str`` used to produce
            the answer. Defaults to `app.retrieval.llm.generate_answer`.
            This is the single seam for configuring or replacing the LLM:
            for a non-default Ollama host/timeout, pass
            ``functools.partial(generate_answer, base_url=..., timeout=...)``;
            for a different backend (e.g. Gemini in production) or in tests,
            pass any callable with the same shape.
        model: model name forwarded to `generate`.

    Returns:
        A dict:
          - ``question`` (str)
          - ``answer`` (str): the generated answer, or `NO_CONTEXT_ANSWER`
            when nothing relevant was retrieved (no `generate` call made).
          - ``citations`` (list[dict]): same as `answer_question`.
          - ``has_context`` (bool)
          - ``prompt`` (str): the prompt that was (or would have been) sent,
            kept for observability/debugging.

    Raises:
        InvalidHistoryError / ValueError: propagated from `answer_question`.
        LLMError (and subclasses): propagated from `generate` when context
            was found and the call failed.
    """
    retrieval = answer_question(
        question,
        vector_store,
        top_k=top_k,
        min_score=min_score,
        history=history,
    )

    if retrieval["has_context"]:
        answer = generate(retrieval["prompt"], model=model)
    else:
        answer = NO_CONTEXT_ANSWER

    return {
        "question": question,
        "answer": answer,
        "citations": retrieval["citations"],
        "has_context": retrieval["has_context"],
        "prompt": retrieval["prompt"],
    }
