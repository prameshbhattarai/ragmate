"""LangChain concepts: structured output and prompts. The small decisions the graph asks the model.

The graph (graph.py) never parses free text. Each question it asks the model has a Pydantic schema,
and the model must fill it in:

1. ``Route``    is the question about the user's documents, or general chat?
2. ``Grade``    is this retrieved chunk relevant to the question?
3. ``Rewrite``  a better search query, when nothing relevant was found
4. prompts      the text the model reads for each decision, as (role, text) messages
5. ``Models``   the bundle of runnables the graph uses: real ones here, scripted fakes in tests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.documents import Document
from pydantic import BaseModel, Field

# --- 1-3. Structured output: the schemas the model fills in ---------------------------------------

# LangChain: with_structured_output() turns a Pydantic model into a JSON schema, tells the model to
# reply in that shape, and parses the reply back into the object. A reply that doesn't fit raises,
# which graph.py catches and replaces with a safe default.


class Route(BaseModel):
    # Field order matters: the model writes JSON top to bottom, so putting `reason` first makes it
    # think before it commits to an answer. A small model decides noticeably better this way.
    reason: str = Field("", description="One short sentence: what the user is asking for.")
    datasource: Literal["docs", "general"] = Field(
        description='"docs" if answering needs the user\'s documents, "general" for greetings or small talk.'
    )


class Grade(BaseModel):
    relevant: Literal["yes", "no"] = Field(description='"yes" if the chunk helps answer the question, else "no".')


class Rewrite(BaseModel):
    query: str = Field(description="A better search query: the key terms, synonyms, no filler words.")


class Structured(Protocol):
    """Anything with ``invoke(messages) -> <pydantic object>``: a structured chat model, or a fake.

    LangChain: every model, prompt and chain is a Runnable with the same ``invoke()`` method, which
    is why a ten-line fake (tests/fakes.py) can stand in for the real model.
    """

    def invoke(self, messages: list) -> Any: ...


# --- 4. Prompts -------------------------------------------------------------------------------------

# Few-shot prompting: a handful of worked examples teaches a small model the decision far better
# than a longer rule. Note the examples use other words than the documents do ("holiday").
ROUTE_SYSTEM = """You decide whether a message needs a search of the user's documents.

The user's documents (file: headings):
{topics}

Reply "docs" for ANY request for information: facts, rules, policies, numbers, names, how-tos or
explanations, even if it uses different words than the headings, and even if you are not sure
the documents cover it. The search finds out; you do not need to.
Reply "general" ONLY for greetings, thanks, goodbyes, or questions about you, the assistant.

Examples:
- "How many days off do I get?" -> docs
- "what's the policy on hotels" -> docs
- "explain embeddings" -> docs
- "What is the capital of France?" -> docs
- "hi!" -> general
- "thanks, that helped" -> general
- "who are you?" -> general"""

GRADE_SYSTEM = """You check whether a document chunk is relevant to a question.
Reply "yes" if the chunk contains information that helps answer the question, even partly.
Reply "no" if it is about something else. Do not answer the question itself."""

REWRITE_SYSTEM = """A search over the user's documents found nothing relevant for this question.
Write one better search query: keep the key terms, add likely synonyms, drop filler words.
Reply with the query only, not an answer."""

GENERAL_SYSTEM = """You are ragmate, a friendly assistant for chatting with the user's documents.
This message is small talk, so answer briefly and naturally. If the user seems to want facts
from their documents, suggest they ask a specific question about them."""


def route_messages(question: str, topics: list[str]) -> list[tuple[str, str]]:
    topic_text = "\n".join(f"- {t}" for t in topics) if topics else "(no documents ingested yet)"
    return [("system", ROUTE_SYSTEM.format(topics=topic_text)), ("human", question)]


def grade_messages(question: str, doc: Document) -> list[tuple[str, str]]:
    return [("system", GRADE_SYSTEM), ("human", f"Question: {question}\n\nChunk:\n{doc.page_content}")]


def rewrite_messages(question: str, last_query: str) -> list[tuple[str, str]]:
    return [("system", REWRITE_SYSTEM), ("human", f"Question: {question}\nPrevious query: {last_query}")]


def general_messages(question: str, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [("system", GENERAL_SYSTEM), *history, ("human", question)]


def answer_messages(question: str, docs: list[Document]) -> list:
    """The same grounded-answer prompt as the plain RAG chain, so both modes answer alike."""
    from .chain import RAG_PROMPT, format_docs

    return RAG_PROMPT.invoke({"context": format_docs(docs), "question": question}).to_messages()


# --- 5. The models the graph uses -----------------------------------------------------------------

@dataclass(frozen=True)
class Models:
    """Everything the graph calls. Each field only needs an ``invoke(messages)`` method."""

    llm: Any  # plain chat model: messages -> AIMessage (answers, condensing follow-ups)
    router: Structured  # messages -> Route
    grader: Structured  # messages -> Grade
    rewriter: Structured  # messages -> Rewrite


def build_models(model: str | None = None) -> Models:
    """The real models: one ChatOllama, wrapped three times for the three schemas."""
    from .config import chat_model

    llm = chat_model(model)
    # method="json_schema" uses Ollama's native JSON mode (its `format` option), which small local
    # models follow far more reliably than tool calling.
    return Models(
        llm=llm,
        router=llm.with_structured_output(Route, method="json_schema"),
        grader=llm.with_structured_output(Grade, method="json_schema"),
        rewriter=llm.with_structured_output(Rewrite, method="json_schema"),
    )
