"""LangChain concept: LCEL (the LangChain Expression Language). Classic RAG as one pipeline.

    question ─┬─> retriever ──> docs ──┐
              └────────────────────────┴─> format_docs ──> prompt ──> chat model ──> answer text

``a | b`` means "feed a's output into b". Every piece (retriever, prompt, model, parser) is a
Runnable with the same ``invoke`` / ``stream`` / ``batch`` methods, and so is the whole pipeline.

This is "naive" RAG: retrieve once, then answer, whatever came back. It works well when the
question is clear and the documents are good. graph.py adds the decisions this chain can't make:
"is this question about the documents at all?", "are these chunks actually relevant?", "should I
search again with better words?".
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

# LangChain: a ChatPromptTemplate is a list of (role, template) messages. invoke({"context": ...,
# "question": ...}) fills the {placeholders} and returns messages ready for the chat model.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the context below, which comes from the user's documents.\n"
            "- Cite the sources you used as [1], [2] ... matching the numbers in the context.\n"
            "- If the context does not contain the answer, say you could not find it in the documents.\n"
            "  Do not use outside knowledge, and do not guess.\n"
            "- Keep the answer short and plain.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def describe(doc: Document) -> str:
    """Where a chunk came from, for people: ``report.pdf, page 3`` or ``notes.md``.

    Only PDFs have real pages; a Markdown file is "page 0" too, but has no page to cite.
    """
    source = str(doc.metadata.get("source", "?"))
    if source.lower().endswith(".pdf"):
        return f"{source}, page {int(doc.metadata.get('page', 0)) + 1}"
    return source


def chunk_number(doc: Document) -> int | None:
    """1-based position of the chunk within its file (or PDF page), if known."""
    index = doc.metadata.get("chunk_index")
    if index is None:  # stores ingested before chunk_index existed: read it from the id
        match = re.search(r":c(\d+)$", str(doc.metadata.get("chunk_id", "")))
        index = int(match.group(1)) if match else None
    return None if index is None else int(index) + 1


def cited(answer: str, count: int) -> set[int]:
    """The source numbers an answer cites: ``"... [1] and [3]."`` -> {1, 3} (ignoring out-of-range ones).

    The prompt asks the model to cite by number, so this is how to check it actually did: a
    source that was retrieved but never cited is a hint that retrieval brought in noise.
    """
    found = {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in re.split(r"\s*,\s*", group)}
    return {n for n in found if 1 <= n <= count}


def format_docs(docs: list[Document]) -> str:
    """Number the chunks so the model can cite them, and so the UI can show which one is which.

    This string *is* the "augmentation" in Retrieval-Augmented Generation: the retrieved text is
    pasted into the prompt, and the model answers from it.
    """
    if not docs:
        return "(no documents found)"
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[{i}] ({describe(doc)})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def build_rag_chain(retriever, llm):
    """Question in, ``{"question", "docs", "answer"}`` out.

    LangChain: RunnableParallel runs its branches on the same input (here: keep the question, and
    retrieve docs for it). RunnablePassthrough.assign adds a new key computed from the others, so the
    retrieved docs are still in the output for the UI to show as sources.
    """
    answer = (
        RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | RAG_PROMPT
        | llm
        | StrOutputParser()  # AIMessage -> plain string
    )
    return RunnableParallel(question=RunnablePassthrough(), docs=retriever) | RunnablePassthrough.assign(answer=answer)
