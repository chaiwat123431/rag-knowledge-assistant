"""Embedding generation for text chunks.

Turns chunks of text (produced by `app.ingestion.chunker`) into dense
vectors for similarity search. Per the updated Architecture Decisions in
PLANNING.md, embeddings run locally via `sentence-transformers` (model
`all-MiniLM-L6-v2`) instead of a paid embedding API — no key, no network
call per request, no per-token cost.

Model loading
-------------
`all-MiniLM-L6-v2` is ~90 MB, takes a noticeable moment to load into
memory, and downloads from the Hugging Face hub on first ever use. It is
loaded exactly once per process, lazily, on the first `embed_texts` call
that actually has something to embed:

- Lazy (not at import time) so that merely importing this module — during
  pytest collection, or from code that only needs the function signature —
  stays cheap, and so `embed_texts([])` can short-circuit without ever
  touching the model.
- Once, via a module-level `_model` global populated with double-checked
  locking. A bare `if _model is None` guard is not enough on its own —
  FastAPI runs sync routes in a threadpool, so two requests arriving
  before the model is warm would both run `SentenceTransformer(...)`,
  racing on the shared Hugging Face cache directory (concurrent downloads
  to the same path) and briefly holding two full model copies in RAM. The
  lock makes the cold path strictly one-at-a-time; the warm path stays a
  plain unlocked read.

Sequence length
---------------
`all-MiniLM-L6-v2` has a max input of 256 word-piece tokens. `encode()`
*silently truncates* anything longer, so text past that point doesn't
influence the vector — a ~500-character chunk of CJK text or whitespace-
free content can tokenize well past 256. Keeping inputs under the limit is
really the chunker's job, but `embed_texts` logs a WARNING (per offending
index) when it sees an over-limit input, so a truncated embedding isn't
completely invisible.
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

    `sentence_transformers` is imported here rather than at module top
    level so that importing this module doesn't pull in torch/transformers
    or trigger a model download until an embedding is actually requested.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME)
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
    embeddings = model.encode(texts)
    return embeddings.tolist()


def _warn_on_truncation(model, texts: list[str]) -> None:
    """Log a WARNING for any text that will be truncated by `model.encode`.

    `encode` truncates silently; this makes the loss visible without
    changing behaviour. Skipped entirely when WARNING isn't enabled, so
    the extra tokenisation pass costs nothing in the common case.
    """
    if not logger.isEnabledFor(logging.WARNING):
        return

    max_tokens = model.max_seq_length
    if not max_tokens:
        return

    for i, text in enumerate(texts):
        token_count = len(
            model.tokenizer(text, truncation=False)["input_ids"]
        )
        if token_count > max_tokens:
            logger.warning(
                "texts[%d] is %d tokens, over the %s %d-token limit; "
                "it will be truncated and its tail won't affect the embedding",
                i,
                token_count,
                MODEL_NAME,
                max_tokens,
            )
