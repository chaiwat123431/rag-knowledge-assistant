import math

import pytest

import app.retrieval.embeddings as embeddings_module
from app.retrieval.embeddings import EMBEDDING_DIM, embed_texts


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


@pytest.mark.model
def test_embed_single_text_returns_one_vector_of_model_dimension():
    result = embed_texts(["a single chunk of text"])

    assert len(result) == 1
    assert len(result[0]) == EMBEDDING_DIM == 384
    assert all(isinstance(x, float) for x in result[0])


@pytest.mark.model
def test_embed_multiple_texts_returns_one_vector_per_input():
    texts = ["first chunk", "second chunk", "third chunk", "fourth chunk"]

    result = embed_texts(texts)

    assert len(result) == len(texts)
    assert all(len(vector) == EMBEDDING_DIM for vector in result)


def test_embed_empty_list_returns_empty_list_without_loading_model(monkeypatch):
    # Verify the "don't load the model for nothing" contract directly:
    # replace the loader with one that fails the test if it's ever called,
    # so this holds regardless of whether an earlier test already warmed
    # the process-wide model.
    def _fail_if_loaded():
        pytest.fail("the model was loaded for an empty input")

    monkeypatch.setattr(embeddings_module, "_get_model", _fail_if_loaded)

    assert embed_texts([]) == []


@pytest.mark.model
def test_semantically_similar_texts_are_closer_than_dissimilar_ones():
    vectors = embed_texts(
        [
            "The cat sat quietly on the warm mat.",
            "A kitten was resting on the cozy rug.",
            "Quarterly revenue exceeded every analyst forecast.",
        ]
    )

    similar = _cosine_similarity(vectors[0], vectors[1])
    dissimilar = _cosine_similarity(vectors[0], vectors[2])

    # The two sentences about a cat resting must land closer together than
    # a cat sentence and an unrelated finance sentence — this is what
    # proves the vectors are semantically meaningful, not just non-crashing.
    # The relative check is the real assertion; the absolute margin is a
    # loose sanity floor, kept wide so a future model/library revision
    # nudging the exact numbers doesn't fail an unrelated build.
    assert similar > dissimilar
    assert similar - dissimilar > 0.05


@pytest.mark.model
def test_embed_logs_warning_when_input_exceeds_model_token_limit(caplog):
    long_text = "word " * 400  # ~400 tokens, well over the 256-token limit

    with caplog.at_level("WARNING", logger="app.retrieval.embeddings"):
        result = embed_texts(["a short one", long_text])

    assert len(result) == 2  # still returns a vector, just a truncated one
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("texts[1]" in m and "truncated" in m for m in warnings)
    assert not any("texts[0]" in m for m in warnings)


def test_embed_rejects_non_list_input():
    # A bare string is iterable, so without the explicit list check it
    # would be silently embedded character by character.
    with pytest.raises(TypeError):
        embed_texts("not a list")


@pytest.mark.parametrize(
    "texts",
    [
        ["valid", 123],
        ["valid", None],
        ["valid", ["still", "not", "a", "string"]],
        [b"bytes are not str"],
    ],
)
def test_embed_rejects_non_string_element(texts):
    with pytest.raises(TypeError):
        embed_texts(texts)


@pytest.mark.parametrize(
    "texts",
    [
        [""],
        ["valid chunk", ""],
        ["   "],
        ["\n\t  \n"],
    ],
)
def test_embed_rejects_empty_or_whitespace_element(texts):
    with pytest.raises(ValueError):
        embed_texts(texts)
