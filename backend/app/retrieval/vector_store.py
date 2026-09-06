"""Persistent vector store — the Chroma side of retrieval.

Ties together the two modules already written for this phase:

    chunks (from app.ingestion.chunker)
      -> embeddings (from app.retrieval.embeddings.embed_texts)
      -> stored in a persistent Chroma collection on disk

and, for queries, embeds the question the same way and returns the nearest
chunks with the metadata needed to cite them later.

Chunking
--------
`add_documents` does NOT re-chunk. Its input is named `chunks` and is
treated as already split — PLANNING.md's data flow keeps "Chunk text" and
"Store in Chroma" as separate stages, and the caller owns the chunk
parameters. `embed_texts` still validates that every chunk is a non-empty
string.

Metadata
--------
Chroma metadata is a flat dict of scalars (str/int/float/bool — no
nesting). Each chunk is stored with:

- ``source`` (str): the document identifier (filename / path) the chunk
  came from. Required so answers can cite their source.
- ``chunk_index`` (int): 0-based position of the chunk within its source
  document. Enables precise citations, a deterministic id
  (``f"{source}::{chunk_index}"``), fetching neighbouring chunks to widen
  context later, and retrieval debugging.

Char offsets, file hashes, timestamps and page numbers are deliberately
left out until a feature needs them (the chunker rejoins words on single
spaces, so exact source offsets aren't recoverable anyway).

Client initialisation
---------------------
`chromadb.PersistentClient(path=...)` persists to disk and reloads on
restart. Clients are cached per resolved path in a module-level dict
guarded by a `Lock` (the same double-checked-locking pattern as the
embeddings model singleton). Keying on the path — rather than a bare
process singleton — lets each test point at its own `tmp_path` while
production always resolves to the one default directory and thus one
client. `chromadb` is imported inside the loader, not at module top, to
keep importing this module cheap (chromadb pulls in onnxruntime,
opentelemetry, the kubernetes client, ...).
"""

from pathlib import Path
from threading import Lock

from app.retrieval.embeddings import embed_texts

# Anchored to backend/chroma_db via __file__ so it doesn't depend on the
# process's working directory. backend/.gitignore already ignores
# chroma_db/.
_DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

COLLECTION_NAME = "knowledge_base"

_collections: dict[str, object] = {}
_collections_lock = Lock()


def _get_collection(persist_dir: Path):
    """Return the Chroma collection for `persist_dir`, creating the client
    and collection once per path.

    Double-checked locking: the warm path is an unlocked dict lookup; only
    the first caller for a given path takes the lock and builds the client.
    """
    key = str(persist_dir)
    collection = _collections.get(key)
    if collection is None:
        with _collections_lock:
            collection = _collections.get(key)
            if collection is None:
                import chromadb
                from chromadb.config import Settings

                client = chromadb.PersistentClient(
                    path=key,
                    settings=Settings(anonymized_telemetry=False),
                )
                collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    # Cosine matches how sentence-transformers embeddings
                    # are meant to be compared; query() turns the distance
                    # into a 1 - d similarity score.
                    metadata={"hnsw:space": "cosine"},
                )
                _collections[key] = collection
    return collection


class VectorStore:
    """A persistent Chroma-backed store of embedded document chunks.

    Args:
        persist_dir: directory Chroma writes to. Defaults to
            ``backend/chroma_db``. Pass a throwaway path (e.g. pytest's
            ``tmp_path``) in tests so runs don't pollute each other.

    Instances sharing a `persist_dir` share one underlying Chroma client
    (see module docstring), so constructing a `VectorStore` is cheap.
    """

    def __init__(self, persist_dir=None):
        base = Path(persist_dir) if persist_dir is not None else _DEFAULT_PERSIST_DIR
        self._persist_dir = base.resolve()

    @property
    def _collection(self):
        return _get_collection(self._persist_dir)

    def add_documents(self, chunks: list[str], source: str) -> None:
        """Embed `chunks` and store them under `source`.

        Re-ingesting a changed document replaces its chunks: the new
        chunks are upserted first, then any leftover higher-index chunks
        from a previous, longer version are deleted. Writing before
        deleting means a failure partway through degrades to "some stale
        tail chunks remain" rather than losing the document entirely.

        Args:
            chunks: already-split text chunks (not re-chunked here). An
                empty list is a no-op.
            source: document identifier stored as metadata for citations.

        Raises:
            TypeError: if `chunks` is not a list.
            ValueError: if `source` is empty, or any chunk is empty or
                whitespace-only (raised by `embed_texts`).
        """
        if not isinstance(chunks, list):
            raise TypeError(
                f"chunks must be a list, got {type(chunks).__name__}"
            )
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

        if not chunks:
            return

        embeddings = embed_texts(chunks)
        ids = [f"{source}::{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_index": i} for i in range(len(chunks))
        ]

        collection = self._collection
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        # Drop chunks left over from a previous, longer version of this
        # source (ids source::len(chunks), source::len(chunks)+1, ...).
        collection.delete(
            where={
                "$and": [
                    {"source": source},
                    {"chunk_index": {"$gte": len(chunks)}},
                ]
            }
        )

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        """Return the `top_k` stored chunks nearest to `question`.

        Args:
            question: the natural-language query.
            top_k: max number of results (fewer are returned if the store
                holds fewer chunks).

        Returns:
            A list of dicts, nearest first, each with:
              - ``id`` (str): the chunk's stored id
              - ``text`` (str): the chunk text
              - ``source`` (str): originating document
              - ``chunk_index`` (int): position within that document
              - ``score`` (float): cosine similarity, ``1 - distance``,
                in [-1.0, 1.0] — 1.0 identical direction, 0.0 orthogonal
                (unrelated), negative when the vectors point apart.
                sentence-transformer pairs in practice land roughly in
                [0.0, 0.9].
            An empty store returns ``[]`` without embedding the question
            (so it never loads the embedding model just to answer nothing).

        Raises:
            ValueError: if `question` is empty/whitespace-only, or
                `top_k` < 1.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        collection = self._collection
        if collection.count() == 0:
            return []

        question_embedding = embed_texts([question])[0]
        result = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
        )

        # Chroma nests each field one level deep (one entry per query).
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            {
                "id": id_,
                "text": document,
                "source": metadata.get("source"),
                "chunk_index": metadata.get("chunk_index"),
                "score": 1.0 - distance,
            }
            for id_, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]
