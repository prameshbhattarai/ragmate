"""Scripted stand-ins for the models, so no test ever calls Ollama.

Testing LLM apps: the graph only calls ``.invoke(messages)`` on its models, so a fake that returns a
fixed list of replies lets a test script a whole run ("route to docs, grade everything irrelevant,
rewrite, grade relevant, answer") and check what the graph does with it: fast, free, repeatable.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from ragmate.agent import Grade, Models, Rewrite, Route


class Scripted:
    """Returns the given replies in order; the last one repeats. An exception in the list is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def __call__(self, messages):  # callable, so LCEL can pipe it: prompt | Scripted(...)
        return self.invoke(messages)


def text(reply: str) -> AIMessage:
    return AIMessage(content=reply)


def route(datasource: str) -> Route:
    return Route(datasource=datasource, reason="test")


def grade(relevant: bool) -> Grade:
    return Grade(relevant="yes" if relevant else "no")


def rewrite(query: str) -> Rewrite:
    return Rewrite(query=query)


def models(llm=None, router=None, grader=None, rewriter=None) -> Models:
    return Models(
        llm=llm or Scripted(text("an answer [1]")),
        router=router or Scripted(route("docs")),
        grader=grader or Scripted(grade(True)),
        rewriter=rewriter or Scripted(rewrite("better query")),
    )


class FixedRetriever:
    """Returns the same documents for every query, and records the queries."""

    def __init__(self, *docs: Document):
        self.docs = list(docs)
        self.queries: list[str] = []

    def invoke(self, query, *args, **kwargs):
        self.queries.append(query)
        return list(self.docs)

    def __call__(self, query):  # callable, so LCEL can use it inside RunnableParallel
        return self.invoke(query)


def doc(textual: str, source: str = "a.md", index: int = 0) -> Document:
    return Document(page_content=textual, metadata={"source": source, "page": 0, "chunk_id": f"{source}:p0:c{index}"})
