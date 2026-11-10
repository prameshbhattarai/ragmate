"""RAG concept: conversation memory, and why a follow-up question can't be searched as it is.

Turn 1: "How many vacation days do I get?"   ← searchable on its own
Turn 2: "And can I carry them over?"         ← "them"? The vector store has no idea.

So before retrieving, a follow-up is *condensed* into a standalone question using the history:
"Can unused vacation days be carried over to next year?". That is what gets embedded and searched.

Where the history lives: graph.py stores it in the graph state, and LangGraph's checkpointer saves
that state per ``thread_id``. Same thread id → the graph remembers; new thread id → fresh chat.
"""

from __future__ import annotations

import uuid

MAX_TURNS = 6  # keep the last few exchanges: older ones cost context and rarely matter

CONDENSE_SYSTEM = """Rewrite the user's latest message as one standalone question that can be
understood without the conversation, replacing words like "it", "them" or "that" with what they
refer to. If it is already standalone, return it unchanged. Reply with the question only."""


def trim_history(history: list[tuple[str, str]], max_turns: int = MAX_TURNS) -> list[tuple[str, str]]:
    """The last ``max_turns`` (human, ai) pairs, as tuples.

    The checkpointer saves state as JSON-like data, so tuples come back as 2-item lists; turning
    them back into tuples keeps them valid LangChain ``(role, text)`` messages.
    """
    return [tuple(m) for m in history[-2 * max_turns :]]


def condense_messages(history: list[tuple[str, str]], question: str) -> list[tuple[str, str]]:
    transcript = "\n".join(f"{role}: {text}" for role, text in trim_history(history))
    return [
        ("system", CONDENSE_SYSTEM),
        ("human", f"Conversation:\n{transcript}\n\nLatest message: {question}"),
    ]


def condense(llm, history: list[tuple[str, str]], question: str) -> str:
    """Return a standalone version of ``question``; the question itself if there is no history."""
    if not history:
        return question
    reply = llm.invoke(condense_messages(history, question))
    text = (getattr(reply, "content", reply) or "").strip()
    return text or question


def new_thread_id() -> str:
    return uuid.uuid4().hex[:12]
