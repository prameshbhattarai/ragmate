from fakes import FixedRetriever, Scripted, doc, text
from langchain_core.runnables import RunnableLambda

from ragmate.chain import build_rag_chain, format_docs


def test_format_docs_numbers_chunks_with_sources():
    out = format_docs([doc("alpha", "a.md"), doc("beta", "b.md", 1)])
    assert out.startswith("[1] (a.md)\nalpha")
    assert "[2] (b.md)\nbeta" in out
    assert format_docs([]) == "(no documents found)"


def test_format_docs_cites_pdf_pages_including_the_first():
    first = doc("intro", "report.pdf")  # page 0
    second = doc("results", "report.pdf")
    second.metadata["page"] = 1
    out = format_docs([first, second])
    assert "[1] (report.pdf, page 1)" in out and "[2] (report.pdf, page 2)" in out


def test_chain_puts_retrieved_context_in_the_prompt_and_returns_sources():
    seen = []

    def fake_llm(prompt_value):
        seen.append(prompt_value.to_messages())
        return text("27 days [1]")

    retriever = FixedRetriever(doc("Employees get 27 vacation days.", "handbook.md"))
    result = build_rag_chain(retriever, RunnableLambda(fake_llm)).invoke("How many vacation days?")

    assert result["answer"] == "27 days [1]"
    assert result["question"] == "How many vacation days?"
    assert [d.metadata["source"] for d in result["docs"]] == ["handbook.md"]
    system, human = seen[0]
    assert "27 vacation days" in system.content and "[1] (handbook.md)" in system.content
    assert human.content == "How many vacation days?"
    assert retriever.queries == ["How many vacation days?"]


def test_chain_accepts_any_runnable_model():
    result = build_rag_chain(FixedRetriever(), Scripted(text("x"))).invoke("q")
    assert result["answer"] == "x" and result["docs"] == []


def test_cited_finds_single_and_grouped_citations_in_range():
    from ragmate.chain import cited

    assert cited("27 days [1], carry over 5 [3].", 4) == {1, 3}
    assert cited("both [1, 2]", 2) == {1, 2}
    assert cited("bogus [9] and [0]", 4) == set()
    assert cited("no citations", 2) == set()


def test_describe_and_chunk_number():
    from ragmate.chain import chunk_number, describe

    md = doc("x", "notes.md", 2)  # chunk_id notes.md:p0:c2, no chunk_index (an older store)
    assert describe(md) == "notes.md" and chunk_number(md) == 3
    pdf = doc("y", "report.pdf")
    pdf.metadata.update(page=4, chunk_index=0)
    assert describe(pdf) == "report.pdf, page 5" and chunk_number(pdf) == 1
