r"""LangGraph concepts: state, nodes, conditional edges, loops and checkpointing. Agentic RAG.

chain.py retrieves once and answers whatever comes back. This graph makes decisions along the way
(the pattern is often called "corrective RAG"):

    START -> condense -> route --general--> direct_answer --> END
                           |
                          docs
                           v
                       retrieve -> grade --some relevant--> generate --> END
                           ^          |
                           |      none relevant
                           |          v
                           +---- rewrite  (at most max_retries times, then: fallback --> END)

Read it top to bottom: the state, the nodes (plain functions that return the keys they change),
the routing functions (which edge to take), then how they are wired into a graph.

Why hand-build a StateGraph instead of ``langchain.agents.create_agent``? create_agent gives a model
tools and lets it loop as it likes. Here every step is explicit, so you can see (and test) each
decision: the graph's path is part of the answer the UI shows.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .agent import (
    Models,
    answer_messages,
    general_messages,
    grade_messages,
    rewrite_messages,
    route_messages,
)
from .memory import condense, trim_history

NOT_FOUND = (
    "I couldn't find this in your documents. Try rephrasing, or ingest a document that covers it "
    "(`ragmate ingest <file>`)."
)


# --- 1. State -------------------------------------------------------------------------------------

class RagState(TypedDict, total=False):
    """Everything the nodes share. Each node returns only the keys it changes.

    LangGraph: a key annotated with a reducer (here ``operator.add``) is *appended to* instead of
    replaced. ``history`` uses it, so with a checkpointer it grows turn after turn. The other keys
    are reset at the start of every turn by ``condense``.
    """

    question: str  # what the user typed this turn
    standalone: str  # the question, condensed with the history so it makes sense on its own
    query: str  # what is actually searched: starts as `standalone`, and may be rewritten
    route: str  # "docs" | "general"
    docs: list[Document]  # retrieved chunks; after grading, only the relevant ones
    retrieved: int  # how many chunks the last retrieval returned (before grading)
    retries: int  # rewrites so far this turn
    answer: str
    steps: list[str]  # the path taken this turn, for the UI and tests
    history: Annotated[list[tuple[str, str]], operator.add]


# --- 2. Nodes, built around the models and retriever they need ----------------------------------------

def make_nodes(models: Models, retriever, topics: list[str]):
    """Return the node functions, closed over the models and retriever.

    LangGraph: a node is any function ``state -> partial state``. Nothing else is required, so a node
    can call a model, a database, or plain Python.
    """

    def condense_node(state: RagState) -> dict:
        history = trim_history(state.get("history", []))
        query = condense(models.llm, history, state["question"])  # unchanged when there is no history
        return {"standalone": query, "query": query, "docs": [], "retrieved": 0, "retries": 0, "answer": "", "steps": ["condense"]}

    def route_node(state: RagState) -> dict:
        try:
            route = models.router.invoke(route_messages(state["query"], topics)).datasource
        except Exception:  # a reply that didn't fit the schema: searching is the safe default
            route = "docs"
        return {"route": route, "steps": state["steps"] + [f"route:{route}"]}

    def retrieve_node(state: RagState) -> dict:
        docs = retriever.invoke(state["query"])
        return {"docs": docs, "retrieved": len(docs), "steps": state["steps"] + [f"retrieve:{len(docs)}"]}

    def grade_node(state: RagState) -> dict:
        kept = []
        for doc in state["docs"]:
            try:
                relevant = models.grader.invoke(grade_messages(state["standalone"], doc)).relevant == "yes"
            except Exception:
                relevant = True  # unsure? keep it: the answer prompt can still ignore it
            if relevant:
                kept.append(doc)
        return {"docs": kept, "steps": state["steps"] + [f"grade:{len(kept)}/{len(state['docs'])}"]}

    def rewrite_node(state: RagState) -> dict:
        try:
            query = models.rewriter.invoke(rewrite_messages(state["standalone"], state["query"])).query.strip()
        except Exception:
            query = ""
        return {
            "query": query or state["query"],
            "retries": state.get("retries", 0) + 1,
            "steps": state["steps"] + ["rewrite"],
        }

    def finish(state: RagState, answer: str, step: str) -> dict:
        turn = [("human", state["question"]), ("ai", answer)]
        return {"answer": answer, "history": turn, "steps": state["steps"] + [step]}

    def generate_node(state: RagState) -> dict:
        reply = models.llm.invoke(answer_messages(state["standalone"], state["docs"]))
        return finish(state, reply.content.strip(), "generate")

    def direct_answer_node(state: RagState) -> dict:
        reply = models.llm.invoke(general_messages(state["question"], trim_history(state.get("history", []))))
        return {**finish(state, reply.content.strip(), "direct_answer"), "docs": []}

    def fallback_node(state: RagState) -> dict:
        return {**finish(state, NOT_FOUND, "fallback"), "docs": []}

    return {
        "condense": condense_node,
        "route": route_node,
        "retrieve": retrieve_node,
        "grade": grade_node,
        "rewrite": rewrite_node,
        "generate": generate_node,
        "direct_answer": direct_answer_node,
        "fallback": fallback_node,
    }


# --- 3. Routing: which edge to take -------------------------------------------------------------------

def after_route(state: RagState) -> Literal["retrieve", "direct_answer"]:
    return "direct_answer" if state.get("route") == "general" else "retrieve"


def make_after_grade(max_retries: int):
    def after_grade(state: RagState) -> Literal["generate", "rewrite", "fallback"]:
        if state["docs"]:
            return "generate"
        # LangGraph: a loop is just an edge back to an earlier node. Always bound it: a small model
        # can rewrite forever without ever finding anything.
        return "rewrite" if state.get("retries", 0) < max_retries else "fallback"

    return after_grade


# --- 4. Wiring -------------------------------------------------------------------------------------

def build_graph(models: Models, retriever, topics: list[str] | None = None, max_retries: int = 2, checkpointer=None):
    """Compile the graph. Run it with ``graph.invoke({"question": ...}, config)``.

    LangGraph: with a checkpointer, every run is saved under ``config["configurable"]["thread_id"]``,
    and the next run on that thread starts from the saved state: that is the chat memory. Try:
    ``print(build_graph(...).get_graph().draw_mermaid())`` to see this graph as a diagram.
    """
    nodes = make_nodes(models, retriever, topics or [])
    builder = StateGraph(RagState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "condense")
    builder.add_edge("condense", "route")
    builder.add_conditional_edges("route", after_route, ["retrieve", "direct_answer"])
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", make_after_grade(max_retries), ["generate", "rewrite", "fallback"])
    builder.add_edge("rewrite", "retrieve")
    for terminal in ("generate", "direct_answer", "fallback"):
        builder.add_edge(terminal, END)

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
