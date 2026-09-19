"""Embedding generation for text chunks.

Turns chunks of text (produced by `app.ingestion.chunker`) into dense
vectors for similarity search. Per the updated Architecture Decisions in
PLANNING.md, embeddings run locally via the `all-MiniLM-L6-v2` model —
no key, no network call per request, no per-token cost.

Runtime: ONNX, not PyTorch (2026-09)
-------------------------------------
Originally loaded via `sentence-transformers` (PyTorch). Measured on
Render's 512MB Starter plan: a real add-document-then-query cycle peaked
at ~560-650MB resident, over the container's entire memory budget by
itself — the PyTorch runtime import alone costs ~400MB regardless of the
~90MB model weights (a smaller sentence-transformers model wouldn't have
helped; the runtime, not the weights, was the cost). Switched to
`chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2` — the *same*
all-MiniLM-L6-v2 weights, exported to ONNX and run via ONNX Runtime
instead of PyTorch. Same measurement: ~305-354MB, comfortably inside the
budget. No new dependency: `onnxruntime` already ships as a transitive
dependency of `chromadb`, which this project already depends on.

This is a runtime swap, not a different or quantized model: verified
directly (not assumed) that the two runtimes produce numerically
equivalent vectors for the same input — max absolute difference
0.000000 and cosine similarity 1.0 across a sample of the project's own
test texts (short/long, related/unrelated pairs), full backend suite
(including every `@pytest.mark.model` test exercising the
`MIN_RELEVANCE_SCORE` / `MAX_AUGMENTED_SCORE_DROP_RATIO` /
`MIN_CROSS_SOURCE_RATIO` thresholds against real embeddings) re-run and
passing unchanged. No threshold recalibration was needed as a result.

Model loading
-------------
The model is ~90MB and downloads on first ever use — from Chroma's own
hosted artifact (`chroma-onnx-models.s3.amazonaws.com`), not the Hugging
Face hub (that was specific to the old sentence-transformers path). It is
loaded exactly once per process, lazily, on the first `embed_texts` call
that actually has something to embed:

- Lazy (not at import time) so that merely importing this module — during
  pytest collection, or from code that only needs the function signature —
  stays cheap, and so `embed_texts([])` can short-circuit without ever
  touching the model.
- Once, via a module-level `_model` global populated with double-checked
  locking. A bare `if _model is None` guard is not enough on its own —
  FastAPI runs sync routes in a threadpool, so two requests arriving
  before the model is warm would both run `ONNXMiniLM_L6_V2()`, racing on
  the shared download-cache directory and briefly holding two full model
  copies in RAM. The lock makes the cold path strictly one-at-a-time; the
  warm path stays a plain unlocked read.

Sequence length
---------------
`all-MiniLM-L6-v2` has a max input of 256 word-piece tokens. The
tokenizer *silently truncates* anything longer (verified: an over-limit
input does not raise, it just loses its tail) — a ~500-character chunk of
CJK text or whitespace-free content can tokenize well past 256. Keeping
inputs under the limit is really the chunker's job, but `embed_texts`
logs a WARNING (per offending index) when it sees an over-limit input, so
a truncated embedding isn't completely invisible.
"""

import logging
from threading import Lock

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # fixed by the model architecture; asserted by the tests

_model = None
_model_lock = Lock()


def _get_model():
    """Return the process-wide model instance, loading it once on demand.

    Double-checked locking: the warm path is a plain unlocked read of
    `_model`; only the first callers (while it's still None) take the lock,
    and the inner re-check means exactly one of them builds the model.

    `ONNXMiniLM_L6_V2()` itself is cheap (no I/O) — unlike the old
    `SentenceTransformer(MODEL_NAME)` it replaced, which did the download
    *and* the load in that one blocking call. Here, the expensive part
    (downloading the ~90MB model if it isn't cached yet, parsing
    tokenizer.json, building the ONNX Runtime session) is deferred to
    first *use* — the `.tokenizer` / `.model` cached properties and
    `_download_model_if_not_exists()`, all triggered by calling the
    object, not by constructing it. If that work were left to happen on
    the caller's first real `model(texts)` call, two threads racing in on
    a cold cache (FastAPI runs sync routes in a threadpool) would both
    already hold the same cached `_model` object and call it concurrently
    — outside this lock, since by then `_model` is no longer `None` for
    either of them — racing on the same on-disk download/extract path.
    Reproduced directly: one thread got a valid model, the other an
    `InvalidProtobuf` error from a torn concurrent write. Forcing a
    throwaway embed here, still holding the lock, makes the model fully
    warm (downloaded, tokenizer loaded, ONNX session built) before it's
    ever assigned to `_model` — so no caller, however many threads arrive
    concurrently afterward, can observe or trigger that first-use cost
    again.

    `chromadb.utils.embedding_functions` is imported here rather than at
    module top level so that importing this module doesn't pull in
    onnxruntime or trigger a model download until an embedding is actually
    requested.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

                model = ONNXMiniLM_L6_V2()
                model(["warm-up"])  # force the download/tokenizer/session
                _model = model
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of text chunks into dense vectors.

    Args:
        texts: the chunks to embed. An empty list returns immediately with
            an empty list and never loads the model.

    Returns:
        One vector (list of floats, length `EMBEDDING_DIM`) per input
        string, in the same order as `texts`. Inputs longer than the
        model's 256-token limit are truncated before embedding, with a
        WARNING logged per offending index (see "Sequence length" in the
        module docstring).

    Raises:
        TypeError: if `texts` is not a list, or any element is not a str.
        ValueError: if any element is empty or whitespace-only. An empty
            string carries no meaning to embed and almost always signals a
            bug upstream in parsing/chunking, so it's rejected rather than
            silently embedded.
    """
    if not isinstance(texts, list):
        raise TypeError(
            f"texts must be a list, got {type(texts).__name__}"
        )

    if not texts:
        return []

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"texts[{i}] must be a str, got {type(text).__name__}"
            )
        if not text.strip():
            raise ValueError(
                f"texts[{i}] is empty or whitespace-only; nothing to embed"
            )

    model = _get_model()
    _warn_on_truncation(model, texts)
    embeddings = model(texts)
    return [vector.tolist() for vector in embeddings]


def _warn_on_truncation(model, texts: list[str]) -> None:
    """Log a WARNING for any text that `model`'s tokenizer will truncate.

    Truncation happens silently in the embedding call itself; this makes
    the loss visible without changing behaviour. The whole batch is
    tokenised in one call (`encode_batch`, the same work the embedding
    call does internally), and the check is skipped when WARNING is
    disabled. A `tokenizers.Encoding`'s `overflowing` list is non-empty
    exactly when its input didn't fit and got cut — a direct signal, not
    inferred from the (always-256, due to padding) encoded length.

    Unlike the old sentence-transformers-based version, this doesn't
    report *how many* tokens a text was over the limit by — two ways of
    recovering that were tried and both gave wrong numbers, verified
    empirically, not assumed: (1) summing `overflowing`'s pieces plateaus
    at a fixed size regardless of true input length (a 400-word and a
    1000-word input both reported the same total); (2) loading a second,
    non-truncating `Tokenizer` from the same `tokenizer.json` didn't
    actually come back untruncated — the file has its own baked-in
    length limit, so it reported the same fixed count for every input
    length, including short ones well under any limit. Getting an
    accurate count would mean re-tokenising with a library-internal
    detail (the raw pre-processor config) this project doesn't otherwise
    need — not worth it for a log message's nice-to-have magnitude when
    "it was truncated" (which chunk, which document) is the actionable
    part.
    """
    if not logger.isEnabledFor(logging.WARNING):
        return

    encodings = model.tokenizer.encode_batch(texts)
    max_tokens = model.max_tokens()
    for i, encoding in enumerate(encodings):
        if encoding.overflowing:
            logger.warning(
                "texts[%d] is over the %s %d-token limit; it will be "
                "truncated and its tail won't affect the embedding",
                i,
                MODEL_NAME,
                max_tokens,
            )
