"""RAG concept: ingestion, the "indexing" half of RAG. Load files, split them, label every chunk.

    file ──load──> Documents (one per file, or one per PDF page)
         ──split─> chunks of ~chunk_size characters, overlapping by chunk_overlap
         ──label─> metadata: source, page, chunk_id    (store.py then embeds and saves them)

Why split at all? An embedding is one vector for a whole text, so a 40-page PDF as one vector
would be a blurry average of everything in it. Small chunks give sharp vectors, and only the few
chunks that matter go into the prompt. Too small, and a chunk loses the context it needs to make
sense: chunk size is the first knob worth turning (try ``RAGMATE_CHUNK_SIZE=300``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .config import PROJECT_ROOT

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")
MAX_HEADINGS = 8


# --- 1. Loading ---------------------------------------------------------------------------------

def source_key(path: Path) -> str:
    """A file's unique id in the store: its path from the project root (or absolute, if outside).

    The file *name* is what people read (``metadata["source"]``), but two folders can both hold a
    ``notes.md``; the path (``metadata["path"]``) is what chunk ids, the manifest and deletes use.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_file(path: Path) -> list[Document]:
    """Load one file into LangChain Documents: ``page_content`` (text) plus ``metadata`` (a dict).

    LangChain: a Document is the unit everything else works on: splitters split Documents, vector
    stores store them, retrievers return them. A "document loader" is just code that turns a
    source (a PDF, a web page, a Notion export ...) into Documents; this one is small enough to read.
    """
    suffix = path.suffix.lower()
    meta = {"source": path.name, "path": source_key(path)}
    if suffix == ".pdf":
        from pypdf import PdfReader

        # one Document per page, so an answer can cite "page 3", not just the file
        return [
            Document(page_content=page.extract_text() or "", metadata={**meta, "page": i})
            for i, page in enumerate(PdfReader(str(path)).pages)
        ]
    if suffix in (".md", ".txt"):
        return [Document(page_content=path.read_text(encoding="utf-8"), metadata={**meta, "page": 0})]
    raise ValueError(f"Unsupported file type: {path.name} (supported: {', '.join(SUPPORTED_SUFFIXES)})")


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def find_files(paths: list[Path]) -> list[Path]:
    """Expand folders into the supported files inside them, sorted so runs are repeatable."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("*") if p.is_file() and is_supported(p)))
        elif is_supported(path):
            found.append(path)
    return found


def extract_headings(docs: list[Document], limit: int = MAX_HEADINGS) -> list[str]:
    """The file's Markdown headings, in order: a free summary for the router (see store.topics)."""
    headings: list[str] = []
    for doc in docs:
        for heading in re.findall(r"^#{1,3}\s+(.+?)\s*$", doc.page_content, flags=re.MULTILINE):
            if heading not in headings and len(headings) < limit:
                headings.append(heading)
    return headings


# --- 2. Splitting -------------------------------------------------------------------------------

def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split Documents into chunks and give each a stable ``chunk_id``.

    LangChain: RecursiveCharacterTextSplitter tries to split on paragraphs first, then lines, then
    sentences, then words, so a chunk breaks at the most natural place that keeps it under
    ``chunk_size``. The overlap repeats the end of one chunk at the start of the next, so a sentence
    that straddles a boundary is still whole in at least one chunk.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],  # Markdown headings first
    )
    chunks = splitter.split_documents(docs)
    counters: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        meta = chunk.metadata
        key = (meta.get("path", meta["source"]), int(meta.get("page", 0)))
        index = counters.get(key, 0)
        counters[key] = index + 1
        meta["chunk_index"] = index
        # A stable id means re-ingesting a file overwrites its chunks instead of duplicating them.
        meta["chunk_id"] = f"{key[0]}:p{key[1]}:c{index}"
    return chunks


# --- 3. Incremental ingest ------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    """``{source path: content hash}`` of every file already in the store."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)  # files named explicitly that can't be read
    chunks: int = 0


def ingest_paths(paths: list[Path], store, settings) -> IngestReport:
    """Load, split and store every supported file under ``paths``; unchanged files are skipped.

    Embedding is the slow part of ingestion (one model call per chunk), so ragmate remembers each
    file's hash and only re-embeds files that are new or changed.
    """
    from .store import add_chunks, delete_source

    report = IngestReport()
    report.unsupported = [str(p) for p in paths if p.is_file() and not is_supported(p)]
    manifest = load_manifest(settings.manifest_path)
    for path in find_files(paths):
        key = source_key(path)
        digest = file_hash(path)
        known = manifest.get(key)
        if known == digest:
            report.skipped.append(key)
            continue
        if known is not None:
            delete_source(store, key)  # the file changed: drop its old chunks first
        docs = load_file(path)
        chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
        if chunks:
            # The file's headings ride on its first chunk, so the router's topic list can be read
            # with one small metadata query instead of loading every chunk (store.topics).
            chunks[0].metadata["headings"] = "; ".join(extract_headings(docs))
        add_chunks(store, chunks)
        manifest[key] = digest
        (report.updated if known is not None else report.added).append(key)
        report.chunks += len(chunks)
    save_manifest(settings.manifest_path, manifest)
    return report
