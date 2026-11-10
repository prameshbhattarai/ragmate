"""RAG concept: embeddings and the vector store, the "search" half of RAG.

    ingest:  chunk text ──embed_documents──> vector ──> saved in Chroma, next to the text and metadata
    search:  question   ──embed_query──────> vector ──> the k stored vectors closest to it ──> chunks

"Closest" here is cosine similarity: how much two vectors point the same way, from 1 (same
meaning) down to 0 (unrelated). Chroma keeps everything on disk under ``RAGMATE_HOME/chroma``, so
the index survives restarts: the vector database is just a folder.

Two retrieval strategies, both one line in :func:`build_retriever`:

- **similarity**: the k closest chunks. Simple, but they are often near-duplicates of each other.
- **MMR** (maximal marginal relevance): fetch more candidates, then pick k that are relevant *and*
  different from each other, so the prompt gets more distinct information.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

COLLECTION = "ragmate"


# --- 1. The vector store ----------------------------------------------------------------------------

def open_store(settings, embedding: Embeddings | None = None):
    """Open (or create) the persistent Chroma collection.

    LangChain: ``Chroma`` is a VectorStore. Every vector store (Chroma, FAISS, pgvector, Pinecone
    ...) has the same methods: ``add_documents``, ``similarity_search``, ``as_retriever``, so
    swapping the database doesn't touch the rest of the app.
    """
    import chromadb
    from langchain_chroma import Chroma

    from .config import embeddings

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=chromadb.config.Settings(anonymized_telemetry=False),  # stays fully local
    )
    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embedding or embeddings(),
        collection_metadata={"hnsw:space": "cosine"},  # compare by direction, not length
    )


def add_chunks(store, chunks: list[Document]) -> None:
    """Embed and save chunks. Ids are the chunk_ids, so re-adding a chunk overwrites it."""
    if chunks:
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])


def delete_source(store, path: str) -> None:
    """Remove every chunk that came from one file (by its path, see ingest.source_key)."""
    ids = store.get(where={"path": path})["ids"]
    if ids:
        store.delete(ids=ids)


def reset_store(store) -> None:
    """Delete every chunk (the collection itself stays)."""
    ids = store.get()["ids"]
    if ids:
        store.delete(ids=ids)


def stats(store) -> dict[str, int]:
    """``{file path: number of chunks}`` without embedding anything (no model call)."""
    counts: dict[str, int] = {}
    for meta in store.get(include=["metadatas"])["metadatas"]:
        key = meta.get("path", meta["source"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def topics(store) -> list[str]:
    """One line per file: its path and its Markdown headings, e.g.
    ``data/sample/nimbus-handbook.md: Nimbus Coffee Co. — Employee Handbook; Working hours; ...``.

    The router (graph.py) can't search before deciding, so this is all it knows about the
    documents. File names alone are too thin a hint for a small model; headings are a cheap
    summary. They are stored on each file's first chunk at ingest, so this reads one row per file,
    not the whole corpus. Files without headings (PDFs, plain text) show just their path.
    """
    first_chunks = store.get(where={"$and": [{"page": 0}, {"chunk_index": 0}]}, include=["metadatas"])
    lines = []
    for meta in sorted(first_chunks["metadatas"], key=lambda m: m.get("path", m["source"])):
        path, headings = meta.get("path", meta["source"]), meta.get("headings", "")
        lines.append(f"{path}: {headings}" if headings else path)
    return lines


# --- 2. Searching -------------------------------------------------------------------------------

def search_with_scores(store, query: str, k: int) -> list[tuple[Document, float]]:
    """The k closest chunks with a relevance score in [0, 1]: 1 means "same meaning".

    This is what ``ragmate inspect`` shows: retrieval on its own, before any chat model is involved.
    When an answer is wrong, look here first: if the right chunk isn't retrieved, no prompt can fix it.
    """
    return store.similarity_search_with_relevance_scores(query, k=k)


def build_retriever(store, k: int, mmr: bool = False):
    """A Retriever: ``retriever.invoke("question") -> list[Document]``.

    LangChain: a Retriever is a Runnable that takes a string and returns Documents, so it plugs
    straight into an LCEL chain (chain.py) or a graph node (graph.py).
    """
    if mmr:
        # fetch_k candidates, then choose k of them; lambda_mult: 1 = pure relevance, 0 = pure variety
        return store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.5})
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


# --- 3. An embedding with no model ------------------------------------------------------------------

class HashingEmbeddings(Embeddings):
    """A tiny, model-free embedding: each word is hashed into one of ``dim`` buckets.

    Used by the tests and ``evals/run_evals.py --check`` so they never call Ollama. It also makes a
    good contrast with a real embedding model: it only matches *the same words* ("vacation" never
    finds "holiday"), while nomic-embed-text matches *meaning*. That difference is why RAG uses a
    learned embedding model at all.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) < 3:
                continue  # skip "a", "of", "is": they would match everything
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
