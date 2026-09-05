"""Splits extracted text into overlapping chunks for embedding/retrieval.

`chunk_size` and `overlap` are both character counts (matching the
"~500 tokens, 50 token overlap" target in PLANNING.md, approximated here by
characters rather than a real tokenizer, which is out of scope until the
embeddings step). Chunks are built word-by-word so a chunk boundary always
falls on whitespace — a word is never split across two chunks, even if that
means a chunk goes slightly over `chunk_size` (e.g. a single word longer than
`chunk_size` still becomes its own chunk).
"""


def chunk_text(
    text: str, chunk_size: int = 500, overlap: int | None = None
) -> list[str]:
    """Split `text` into overlapping, word-boundary-respecting chunks.

    Args:
        text: the text to split.
        chunk_size: target max characters per chunk.
        overlap: target characters of overlap carried from the end of one
            chunk into the start of the next. Defaults to `min(50,
            chunk_size // 2)` — plain `50` for the documented default
            chunk_size (500), scaled down automatically for a smaller
            chunk_size so the defaults never conflict. Pass an explicit
            value to opt out of that scaling.

    Returns:
        A list of chunk strings (words rejoined with single spaces, so
        original whitespace/newlines are not preserved verbatim). Empty
        (or whitespace-only) input unconditionally yields an empty list,
        even if `chunk_size`/`overlap` are themselves invalid — there's
        nothing to chunk, so there's nothing to validate parameters
        against.

    Raises:
        ValueError: if `text` is non-empty and `chunk_size` isn't
            positive, or `overlap` is negative or more than half of
            `chunk_size`. The half-of-chunk_size cap (not just
            `overlap < chunk_size`) rules out the *typical* way overlap
            can swallow a whole chunk, but isn't sufficient on its own —
            see the Note below for the case that still needs a runtime
            fallback.

    Note:
        Overlap is word-boundary-respecting and thus best-effort: it can
        end up smaller than requested, including 0, whenever honoring it
        exactly would produce a chunk with no new content — e.g. a chunk
        that's a single word longer than `overlap` (nothing short enough
        to carry over), or an overlap window immediately followed by a
        word too long to fit alongside it in the same chunk_size budget.
        Either way, we never split a word to force an exact overlap.
    """
    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap is None:
        overlap = min(50, chunk_size // 2)
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap > chunk_size // 2:
        raise ValueError(
            f"overlap ({overlap}) must be at most half of chunk_size "
            f"({chunk_size} // 2 = {chunk_size // 2})"
        )

    n = len(words)
    chunks: list[str] = []
    start = 0
    prev_end = 0  # end of the previous chunk; 0 can never trigger the
    # guard below on the first iteration, since _build_chunk always
    # includes at least one word (idx ends up >= 1 > 0).

    while start < n:
        chunk_words, idx = _build_chunk(words, start, chunk_size)

        # If this chunk doesn't extend past where the previous chunk
        # ended, every word in it was already covered — the overlap
        # window ate the whole budget and left no room for the word that
        # follows it (e.g. a long word right after the boundary). Fall
        # back to zero overlap here instead of emitting a chunk with no
        # new content.
        if idx <= prev_end:
            start = prev_end
            chunk_words, idx = _build_chunk(words, start, chunk_size)

        chunks.append(" ".join(chunk_words))
        prev_end = idx

        if idx >= n:
            break

        # Walk backwards from the end of this chunk to find how many whole
        # words fit within `overlap` characters — that's the start of the
        # next chunk. overlap=0 means no words fit, so overlap_start == idx.
        overlap_len = 0
        overlap_start = idx
        while overlap_start > start:
            word = words[overlap_start - 1]
            added_len = len(word) + (1 if overlap_len else 0)
            if overlap_len + added_len > overlap:
                break
            overlap_len += added_len
            overlap_start -= 1

        # Guarantee forward progress even if overlap >= chunk_size.
        start = max(overlap_start, start + 1)

    return chunks


def _build_chunk(
    words: list[str], start: int, chunk_size: int
) -> tuple[list[str], int]:
    """Greedily accumulate words[start:] up to chunk_size characters.

    The first word is always accepted even if it alone exceeds
    chunk_size, since a word is never split.

    Returns:
        (words included in the chunk, index just past the last one).
    """
    current_words: list[str] = []
    current_len = 0
    idx = start
    n = len(words)
    while idx < n:
        word = words[idx]
        added_len = len(word) + (1 if current_words else 0)  # + space
        if current_words and current_len + added_len > chunk_size:
            break
        current_words.append(word)
        current_len += added_len
        idx += 1
    return current_words, idx
