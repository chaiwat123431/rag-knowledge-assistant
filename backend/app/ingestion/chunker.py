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
        input yields an empty list.

    Raises:
        ValueError: if `chunk_size` isn't positive, or `overlap` is
            negative or >= `chunk_size` (an overlap that large or larger
            would make consecutive chunks near-duplicates of each other,
            one word apart, instead of a meaningful sliding window).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    words = text.split()
    if not words:
        return []

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
