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


def test_chunk_text_default_overlap_scales_down_with_small_chunk_size():
    # Regression test: the default overlap used to be a flat 50, which
    # violated the overlap<=chunk_size//2 rule for any chunk_size<100
    # (e.g. chunk_size=80 with the implicit default overlap=50 used to
    # raise ValueError). The default now scales down automatically.
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=80)  # overlap left at default

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 80


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


def test_chunk_text_overlap_around_an_oversized_word_is_best_effort():
    # Known, documented limitation: when a chunk is a single word longer
    # than `overlap` (here, longer than `chunk_size` itself), there's no
    # whole word short enough to carry over without splitting it — so
    # overlap silently becomes 0 at that boundary instead of erroring or
    # violating the "never split a word" contract.
    long_word = "x" * 300
    text = f"short {long_word} tail"

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert chunks == ["short", long_word, "tail"]


def test_chunk_text_overlap_falls_back_to_zero_when_next_word_is_too_long():
    # Regression test: overlap<=chunk_size//2 alone doesn't guarantee
    # progress — a long word landing right after the overlap boundary
    # can still leave no room for anything new, so the chunk built from
    # the overlap window ('bbbbb', a subset of the previous chunk) must
    # be discarded in favor of restarting from zero overlap.
    text = "aaaaa bbbbb " + "c" * 16

    chunks = chunk_text(text, chunk_size=11, overlap=5)

    assert chunks == ["aaaaa bbbbb", "c" * 16]
    for first, second in zip(chunks, chunks[1:]):
        first_words = first.split()
        second_words = second.split()
        assert set(second_words) - set(first_words), (
            f"{second!r} added no new words over {first!r}"
        )


def test_chunk_text_overlap_greater_than_or_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=20, overlap=20)

    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=20, overlap=100)


def test_chunk_text_overlap_more_than_half_chunk_size_raises():
    # overlap < chunk_size alone isn't enough: an overlap merely *close* to
    # chunk_size can still swallow an entire chunk's worth of words into
    # the next one (see test_chunk_text_overlap_never_reproduces_a_whole_chunk
    # for the concrete degenerate case this prevents).
    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=20, overlap=15)

    # Exactly half is allowed.
    chunk_text("some words here", chunk_size=20, overlap=10)


def test_chunk_text_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=20, overlap=-1)


def test_chunk_text_non_positive_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=0, overlap=0)


def test_chunk_text_empty_text_returns_empty_list_even_with_invalid_params():
    # Nothing to chunk means nothing to validate parameters against —
    # empty input always short-circuits before chunk_size/overlap checks.
    assert chunk_text("", chunk_size=0, overlap=0) == []
    assert chunk_text("   ", chunk_size=-5, overlap=-5) == []


def test_chunk_text_overlap_never_reproduces_a_whole_chunk():
    # Regression test: chunk_size=5, overlap=4 used to make the previous
    # chunk's overlap window swallow the *entire* previous chunk, so the
    # next chunk ('bb') was a strict subset of the one before it ('a bb')
    # with zero new words. The overlap<=chunk_size//2 rule now rejects
    # this configuration outright rather than silently producing it.
    with pytest.raises(ValueError):
        chunk_text("a bb ccc dddd eeeee ffffff", chunk_size=5, overlap=4)

    words = [f"w{i}" for i in range(30)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=10, overlap=5)

    for first, second in zip(chunks, chunks[1:]):
        first_words = first.split()
        second_words = second.split()
        assert set(second_words) - set(first_words), (
            f"{second!r} added no new words over {first!r}"
        )
