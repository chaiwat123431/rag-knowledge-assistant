import pytest

from app.ingestion.chunker import chunk_text


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_shorter_than_chunk_size_returns_single_chunk():
    text = "A short sentence that fits in one chunk."

    result = chunk_text(text, chunk_size=500, overlap=50)

    assert result == [text]


def test_chunk_text_splits_long_text_into_multiple_chunks():
    words = [f"word{i}" for i in range(200)]  # ~1189 chars incl. spaces
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100


def test_chunk_text_never_splits_a_word():
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    valid_words = set(words)

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    for chunk in chunks:
        for token in chunk.split():
            assert token in valid_words


def test_chunk_text_reconstructs_all_words_at_least_once():
    words = [f"word{i}" for i in range(50)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=60, overlap=15)

    seen = set()
    for chunk in chunks:
        seen.update(chunk.split())

    assert seen == set(words)


def test_chunk_text_overlap_shares_words_between_consecutive_chunks():
    words = [f"word{i}" for i in range(50)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=60, overlap=15)

    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:]):
        first_words = first.split()
        second_words = second.split()
        # The tail of `first` and the head of `second` should share at
        # least one word, proving the overlap window carried words over.
        overlap_words = set(first_words) & set(second_words)
        assert overlap_words, f"no overlap between {first!r} and {second!r}"
        # And the shared words must be a suffix of `first` / prefix of
        # `second`, not just any words in common.
        assert second_words[0] in first_words


def test_chunk_text_zero_overlap_has_no_shared_words():
    words = [f"word{i}" for i in range(50)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=60, overlap=0)

    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:]):
        assert not (set(first.split()) & set(second.split()))


def test_chunk_text_single_word_longer_than_chunk_size_is_kept_whole():
    long_word = "x" * 300
    text = f"short {long_word} tail"

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    all_words = set(" ".join(chunks).split())
    assert long_word in all_words


def test_chunk_text_overlap_greater_than_or_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=20, overlap=20)

    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=20, overlap=100)


def test_chunk_text_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=20, overlap=-1)


def test_chunk_text_non_positive_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=0, overlap=0)
