"""Splits extracted text into overlapping chunks for embedding/retrieval.

`chunk_size` and `overlap` are both character counts (matching the
"~500 tokens, 50 token overlap" target in PLANNING.md, approximated here by
characters rather than a real tokenizer, which is out of scope until the
embeddings step). Chunks are built word-by-word so a chunk boundary always
falls on whitespace — a word is never split across two chunks, even if that
means a chunk goes slightly over `chunk_size` (e.g. a single word longer than
`chunk_size` still becomes its own chunk).
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split `text` into overlapping, word-boundary-respecting chunks.

    Args:
        text: the text to split.
        chunk_size: target max characters per chunk.
        overlap: target characters of overlap carried from the end of one
            chunk into the start of the next.

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
            `overlap < chunk_size`) is what it takes to actually prevent
            degenerate chunks: an overlap merely *close* to chunk_size
            can still re-swallow an entire chunk's worth of words into
            the next one, producing a chunk with no new content.

    Note:
        Overlap is itself word-boundary-respecting and thus best-effort:
        if a chunk ends up being a single word longer than `overlap` (or
        longer than `chunk_size` itself, which single words are allowed
        to be — see above), there's no whole word short enough to carry
        over, so the next chunk starts fresh with zero overlap rather
        than splitting that word.
    """
    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap > chunk_size // 2:
        raise ValueError(
            f"overlap ({overlap}) must be at most half of chunk_size "
            f"({chunk_size} // 2 = {chunk_size // 2})"
        )

    chunks: list[str] = []
    start = 0
    n = len(words)

    while start < n:
        # Grow the chunk word by word until the next word would push it
        # over chunk_size. The first word is always accepted, even if it
        # alone exceeds chunk_size, since we never split a word.
        current_words: list[str] = []
        current_len = 0
        idx = start
        while idx < n:
            word = words[idx]
            added_len = len(word) + (1 if current_words else 0)  # + space
            if current_words and current_len + added_len > chunk_size:
                break
            current_words.append(word)
            current_len += added_len
            idx += 1

        chunks.append(" ".join(current_words))

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
