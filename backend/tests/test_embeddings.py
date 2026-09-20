import math
import threading

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


@pytest.mark.model
def test_get_model_is_fully_warm_before_being_returned(monkeypatch):
    """Regression test: `ONNXMiniLM_L6_V2()` itself is cheap (no I/O) --
    unlike the old `SentenceTransformer(...)` it replaced, which did the
    download *and* the load in one blocking call. If `_get_model()`
    returned before forcing a throwaway embed, the first real caller's
    access to `model.tokenizer` (e.g. `_warn_on_truncation`, called before
    the model is ever embedded with) could hit the ONNX model's files
    before they were ever downloaded. Reproduced directly against a
    cleared cache before this fix: `Exception: No such file or directory`.
    """
    monkeypatch.setattr(embeddings_module, "_model", None)

    model = embeddings_module._get_model()

    # Must not raise -- the tokenizer's files must already be on disk and
    # its cached_property already populated.
    encoding = model.tokenizer.encode("a short sentence")
    assert encoding is not None


def test_get_model_normalizes_any_warm_up_failure(monkeypatch):
    """Regression test: three /code-review rounds each found one more
    concrete exception type routes.py's except-clauses missed for a
    model-download failure (a plain OSError, then httpx.HTTPError, then a
    ValueError from chromadb's own SHA256-mismatch check) -- catching
    them one at a time there wasn't exhaustive. `_get_model()` now
    catches *any* exception from the forced warm-up call and normalizes
    it into `EmbeddingUnavailableError`, so callers only ever need to
    catch one type regardless of which concrete library failure caused
    it. Uses a `ValueError` here specifically -- not an `OSError` or
    `httpx.HTTPError`, the two types already found -- to prove this
    covers the whole exception surface, not just those two.

    No @pytest.mark.model / real network needed: ONNXMiniLM_L6_V2()
    itself does no I/O (see test_get_model_is_fully_warm_before_being_
    returned above), so patching its __call__ to fail is enough to
    exercise _get_model()'s wrapping without ever downloading anything.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    def broken_call(self, texts):
        raise ValueError("simulated SHA256 mismatch")

    monkeypatch.setattr(embeddings_module, "_model", None)
    monkeypatch.setattr(ONNXMiniLM_L6_V2, "__call__", broken_call)

    with pytest.raises(embeddings_module.EmbeddingUnavailableError) as excinfo:
        embeddings_module._get_model()

    assert isinstance(excinfo.value, OSError)
    assert "simulated SHA256 mismatch" in str(excinfo.value)


@pytest.mark.model
def test_get_model_concurrent_first_calls_do_not_race(monkeypatch, tmp_path):
    """Regression test: with only the cheap object construction inside
    `_model_lock` (not the deferred download/tokenizer/session-build),
    two threads racing in on a cold model would both hold the same
    `_model` object and call it concurrently, *outside* the lock --
    racing on the same on-disk download/extract path. Reproduced
    directly, cache cleared first: one thread got a valid model, another
    an onnxruntime `InvalidProtobuf` error from a torn concurrent write.

    Needs a genuinely cold on-disk cache to be a meaningful regression
    test -- with a warm cache there's nothing to race on, and this would
    pass even against the bug it's meant to catch. Points
    `ONNXMiniLM_L6_V2.DOWNLOAD_PATH` (a class attribute) at `tmp_path`
    instead of clearing the real one -- an earlier version of this test
    `shutil.rmtree`'d the real, shared, machine-wide
    `~/.cache/chroma/onnx_models` directory, destroying any legitimately
    cached model for every other process/CI run on the machine. Patching
    the class attribute forces the same genuinely-cold-cache download
    race, isolated to a throwaway directory `monkeypatch` cleans up
    automatically.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    monkeypatch.setattr(embeddings_module, "_model", None)
    monkeypatch.setattr(
        ONNXMiniLM_L6_V2,
        "DOWNLOAD_PATH",
        tmp_path / "onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME,
    )

    errors = []
    results = []
    results_lock = threading.Lock()

    def worker(i):
        try:
            vectors = embed_texts([f"thread {i} text"])
            with results_lock:
                results.append(len(vectors[0]))
        except Exception as exc:  # the whole point: nothing may raise here
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert results == [EMBEDDING_DIM] * 8


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
