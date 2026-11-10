"""LangChain concept: the models. Plus settings, read from the environment (and a local .env).

RAG needs two different models:

- a **chat model** (``gemma3:4b``) that reads text and writes an answer, and
- an **embedding model** (``nomic-embed-text``) that turns text into a vector of numbers, so that
  "texts that mean similar things" become "vectors that point in similar directions".

Both are built lazily by :func:`chat_model` and :func:`embeddings`, so importing ragmate never
contacts Ollama: the tests run without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The repository root (this file is ragmate/config.py). Runtime data lives here, in `.ragmate/`,
# which is gitignored: the vector store is local state, never something to commit.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_home(value: str) -> Path:
    """``~`` is expanded; a relative path is taken from the project root, not the current folder,
    so ``ragmate`` finds the same store whichever directory you run it from."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    model: str = "gemma3:4b"
    embed_model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    num_ctx: int = 8192
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    max_retries: int = 2
    home: Path = PROJECT_ROOT / ".ragmate"

    @property
    def chroma_dir(self) -> Path:
        return self.home / "chroma"

    @property
    def manifest_path(self) -> Path:
        return self.home / "manifest.json"

    @property
    def uploads_dir(self) -> Path:
        return self.home / "uploads"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            model=env("OLLAMA_MODEL", cls.model),
            embed_model=env("OLLAMA_EMBED_MODEL", cls.embed_model),
            base_url=env("OLLAMA_BASE_URL", cls.base_url),
            num_ctx=int(env("OLLAMA_NUM_CTX", cls.num_ctx)),
            chunk_size=int(env("RAGMATE_CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(env("RAGMATE_CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(env("RAGMATE_TOP_K", cls.top_k)),
            max_retries=int(env("RAGMATE_MAX_RETRIES", cls.max_retries)),
            home=_resolve_home(env("RAGMATE_HOME", str(cls.home))),
        )


def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=4)
def chat_model(model: str | None = None):
    """Return the chat model, building it on first use. Not called at import time.

    LangChain: every provider has a chat model class with the same interface (ChatOllama,
    ChatAnthropic, ChatOpenAI ...), so switching providers is a one-line change here. Try:
    ``chat_model().invoke("hello")`` returns an AIMessage; ``.stream("hello")`` yields it in chunks.
    """
    from langchain_ollama import ChatOllama

    settings = get_settings()
    return ChatOllama(
        model=model or settings.model,
        base_url=settings.base_url,
        num_ctx=settings.num_ctx,  # context window: question + retrieved chunks must fit in it
        temperature=0,  # RAG wants answers grounded in the documents, not creative ones
    )


@lru_cache(maxsize=4)
def embeddings(model: str | None = None):
    """Return the embedding model, building it on first use.

    LangChain: an ``Embeddings`` object has two methods: ``embed_documents(texts)`` for what you
    store, and ``embed_query(text)`` for what you search with. Try:
    ``len(embeddings().embed_query("hello"))`` prints the vector size (768 for nomic-embed-text).

    The same embedding model must be used to ingest and to search: vectors from different models
    live in different "spaces" and cannot be compared. Change OLLAMA_EMBED_MODEL, and re-ingest
    with ``ragmate ingest --reset``.
    """
    from langchain_ollama import OllamaEmbeddings

    settings = get_settings()
    return OllamaEmbeddings(model=model or settings.embed_model, base_url=settings.base_url)
