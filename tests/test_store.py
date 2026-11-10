from ragmate.store import HashingEmbeddings, build_retriever, delete_source, reset_store, search_with_scores, stats, topics


def test_hashing_embeddings_match_words_not_meaning():
    emb = HashingEmbeddings()
    vacation = emb.embed_query("vacation days")
    assert abs(sum(v * v for v in vacation) - 1.0) < 1e-9  # unit length
    same = sum(a * b for a, b in zip(vacation, emb.embed_query("how many vacation days")))
    synonym = sum(a * b for a, b in zip(vacation, emb.embed_query("holiday allowance")))
    assert same > 0.5 and synonym < 0.1  # why real RAG uses a learned embedding model


def test_search_finds_the_right_document(sample_store):
    results = search_with_scores(sample_store, "how many vacation days carried over", k=3)
    assert results[0][0].metadata["source"] == "nimbus-handbook.md"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_similarity_and_mmr_retrievers_return_k_docs(sample_store):
    for mmr in (False, True):
        docs = build_retriever(sample_store, k=3, mmr=mmr).invoke("embedding vectors")
        assert len(docs) == 3


def test_delete_and_reset(sample_store):
    assert "data/sample/ollama-notes.md" in stats(sample_store)
    delete_source(sample_store, "data/sample/ollama-notes.md")
    assert "data/sample/ollama-notes.md" not in stats(sample_store)
    reset_store(sample_store)
    assert stats(sample_store) == {}


def test_store_persists_on_disk(sample_store, settings):
    from ragmate.store import open_store

    reopened = open_store(settings, HashingEmbeddings())
    assert stats(reopened) == stats(sample_store)


def test_topics_summarise_each_file_by_its_headings(sample_store):
    lines = topics(sample_store)
    assert len(lines) == 3
    handbook = next(line for line in lines if line.startswith("data/sample/nimbus-handbook.md: "))
    assert "Vacation and leave" in handbook and "Expenses" in handbook
    assert handbook.index("Working hours") < handbook.index("Vacation and leave")  # document order
