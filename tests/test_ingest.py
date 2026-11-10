from conftest import SAMPLE_DIR

from ragmate.ingest import find_files, ingest_paths, load_file, load_manifest, source_key, split_documents
from ragmate.store import stats


def test_find_files_expands_folders_and_skips_unsupported(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.csv").write_text("x,y")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("world")
    names = [p.name for p in find_files([tmp_path])]
    assert names == ["a.md", "c.txt"]


def test_load_file_sets_source_and_page():
    [doc] = load_file(SAMPLE_DIR / "rag-glossary.md")
    assert doc.metadata["source"] == "rag-glossary.md"  # what people read
    assert doc.metadata["path"] == "data/sample/rag-glossary.md"  # the unique id
    assert doc.metadata["page"] == 0
    assert "Retrieval-Augmented Generation" in doc.page_content


def test_split_respects_chunk_size_and_gives_unique_stable_ids():
    docs = load_file(SAMPLE_DIR / "nimbus-handbook.md")
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 300 for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "data/sample/nimbus-handbook.md:p0:c0"
    assert [c.metadata["chunk_index"] for c in chunks[:3]] == [0, 1, 2]
    again = [c.metadata["chunk_id"] for c in split_documents(docs, 300, 50)]
    assert again == ids  # same input, same ids: re-ingesting overwrites instead of duplicating


def test_ingest_skips_unchanged_and_replaces_changed_files(tmp_path, store, settings):
    note = tmp_path / "note.md"
    note.write_text("# Note\n\nThe secret word is pineapple.\n" * 3)

    key = source_key(note)
    first = ingest_paths([note], store, settings)
    assert first.added == [key] and first.chunks >= 1
    assert key in load_manifest(settings.manifest_path)

    second = ingest_paths([note], store, settings)
    assert second.skipped == [key] and second.chunks == 0

    note.write_text("Short now.")
    third = ingest_paths([note], store, settings)
    assert third.updated == [key]
    assert stats(store) == {key: 1}  # old chunks were deleted, not left behind


def test_sample_corpus_ingests(sample_store):
    assert set(stats(sample_store)) == {
        "data/sample/nimbus-handbook.md", "data/sample/ollama-notes.md", "data/sample/rag-glossary.md",
    }


def test_files_with_the_same_name_in_different_folders_are_kept_apart(tmp_path, store, settings):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "notes.md").write_text("Alpha facts about apples.")
    (tmp_path / "b" / "notes.md").write_text("Beta facts about bananas.")
    report = ingest_paths([tmp_path / "a" / "notes.md", tmp_path / "b" / "notes.md"], store, settings)
    assert len(report.added) == 2 and report.updated == []
    assert list(stats(store).values()) == [1, 1]


def test_unsupported_files_named_explicitly_are_reported(tmp_path, store, settings):
    docx = tmp_path / "handbook.docx"
    docx.write_bytes(b"not really a docx")
    report = ingest_paths([docx], store, settings)
    assert report.unsupported == [str(docx)] and report.chunks == 0


def test_first_chunk_carries_the_file_headings(sample_store):
    rows = sample_store.get(where={"$and": [{"page": 0}, {"chunk_index": 0}]}, include=["metadatas"])["metadatas"]
    handbook = next(m for m in rows if m["source"] == "nimbus-handbook.md")
    assert handbook["headings"].startswith("Nimbus Coffee Co. — Employee Handbook; Working hours")
