"""LangGraph concept: streaming. Plus the chat session the interactive mode is built on.

``graph.invoke()`` waits for the whole run and returns the final state. ``graph.stream()`` yields
while the graph runs, in one or more *stream modes* at once:

- ``"updates"``   after each node finishes: ``{node name: the keys it returned}``. ragmate shows
                  these as the live path: route:docs, retrieve:4, grade:1/4 ...
- ``"messages"``  every token a chat model produces inside any node, with metadata saying which
                  node it came from. ragmate keeps only the tokens of the answering nodes, so the
                  JSON of the router and grader never reaches the screen.

A :class:`ChatSession` turns that into a small stream of events the front end draws:
``("step", "route:docs")``, ``("token", "Every ")``, ..., and finally ``("done", Turn)``.
The CLI (cli.py) only renders those events, so the session can be tested without a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from .graph import thread_config
from .memory import new_thread_id

# Only these nodes write the answer; tokens from every other node (condense, route, grade,
# rewrite) are the model talking to the graph, not to the user.
ANSWER_NODES = {"generate", "direct_answer"}
MODES = ("agent", "rag")
EXIT = object()  # returned by run_command for /exit


@dataclass
class Turn:
    question: str
    answer: str
    docs: list[Document] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


HELP = """Commands:
  /help              this list
  /new               start a new conversation (forget the history)
  /mode agent|rag    agent = LangGraph corrective RAG (with memory), rag = one LCEL chain (no memory)
  /k N               retrieve N chunks per search
  /mmr on|off        MMR retrieval: relevant *and* diverse chunks
  /sources           the full text of the chunks behind the last answer
  /history           the conversation so far
  /stats             what is in the vector store
  /ingest PATH ...   add files or folders to the store
  /exit              quit (or Ctrl-D)"""


class ChatSession:
    """One conversation: settings, memory (thread id + checkpointer) and the last answer."""

    def __init__(self, runtime, mode: str = "agent", k: int | None = None, mmr: bool = False, models=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.rt = runtime
        self.mode = mode
        self.k = k or runtime.settings.top_k
        self.mmr = mmr
        self.thread_id = new_thread_id()
        # LangGraph: the checkpointer is the memory. Keeping it here (not inside one compiled graph)
        # means the graph can be rebuilt after /k or /ingest without forgetting the conversation.
        self.checkpointer = InMemorySaver()
        self._models = models
        self._graph = None
        self.turns: list[Turn] = []

    # --- the pieces, built on first use -----------------------------------------------------------

    @property
    def models(self):
        if self._models is None:
            from .agent import build_models

            self._models = build_models(self.rt.settings.model)
        return self._models

    def graph(self):
        if self._graph is None:
            self._graph = self.rt.agent_graph(self.k, self.mmr, models=self.models, checkpointer=self.checkpointer)
        return self._graph

    def _rebuild(self) -> None:
        self._graph = None  # settings or documents changed; the next question compiles a fresh graph

    # --- asking -------------------------------------------------------------------------------

    def stream(self, question: str) -> Iterator[tuple[str, object]]:
        """Yield ``("step", str)`` and ``("token", str)`` events, then ``("done", Turn)``."""
        turn = yield from (self._stream_rag(question) if self.mode == "rag" else self._stream_agent(question))
        self.turns.append(turn)
        yield "done", turn

    def _stream_agent(self, question: str):
        config = thread_config(self.thread_id)
        graph = self.graph()
        # LangGraph: several stream modes at once yield (mode, chunk) pairs.
        for mode, chunk in graph.stream({"question": question}, config, stream_mode=["updates", "messages"]):
            if mode == "updates":
                for update in chunk.values():
                    if update and update.get("steps"):
                        yield "step", update["steps"][-1]
            else:
                message, meta = chunk
                if meta.get("langgraph_node") in ANSWER_NODES and message.content:
                    yield "token", message.content
        # The final state is in the checkpointer, under this thread.
        values = graph.get_state(config).values
        return Turn(question, values["answer"], values.get("docs", []), values["steps"])

    def _stream_rag(self, question: str):
        # LangChain: .stream() on an LCEL chain yields partial outputs. For the dict this chain
        # returns, that is {"question": ...}, then {"docs": [...]}, then the answer piece by piece.
        docs, answer, steps = [], [], []
        for chunk in self.rt.rag_chain(self.k, self.mmr, models=self.models).stream(question):
            if "docs" in chunk:
                docs = chunk["docs"]
                steps.append(f"retrieve:{len(docs)}")
                yield "step", steps[-1]
            if chunk.get("answer"):
                answer.append(chunk["answer"])
                yield "token", chunk["answer"]
        steps.append("generate")
        yield "step", "generate"
        return Turn(question, "".join(answer).strip(), docs, steps)

    # --- slash commands ---------------------------------------------------------------------------

    def run_command(self, line: str):
        """Handle a ``/command``. Returns text to show, or :data:`EXIT`."""
        parts = line[1:].split()
        if not parts:  # just "/": show the commands rather than fail
            return HELP
        name, args = parts[0].lower(), parts[1:]
        if name in ("exit", "quit", "q"):
            return EXIT
        if name == "help":
            return HELP
        if name == "new":
            self.thread_id = new_thread_id()
            self.turns = []
            return "New conversation: the history is forgotten."
        if name == "mode":
            if not args or args[0] not in MODES:
                return f"Mode is {self.mode}. Use /mode agent or /mode rag."
            self.mode = args[0]
            note = " (no memory: each question stands alone)" if self.mode == "rag" else ""
            return f"Mode: {self.mode}{note}."
        if name == "k":
            if not args or not args[0].isdigit() or int(args[0]) < 1:
                return f"k is {self.k}. Use /k N with N >= 1."
            self.k = int(args[0])
            self._rebuild()
            return f"Retrieving {self.k} chunks per search."
        if name == "mmr":
            if not args or args[0] not in ("on", "off"):
                return f"MMR is {'on' if self.mmr else 'off'}. Use /mmr on or /mmr off."
            self.mmr = args[0] == "on"
            self._rebuild()
            return f"MMR {args[0]}."
        if name == "sources":
            return self.sources_text()
        if name == "history":
            if not self.turns:
                return "No questions yet."
            return "\n\n".join(f"you> {t.question}\nragmate> {t.answer}" for t in self.turns)
        if name == "stats":
            from .store import stats

            counts = stats(self.rt.store)
            return "\n".join(f"{n:>4} chunks  {s}" for s, n in counts.items()) or "The store is empty."
        if name == "ingest":
            if not args:
                return "Use /ingest PATH [PATH ...]"
            return self.ingest([Path(a).expanduser() for a in args])
        return f"Unknown command /{name}. Type /help."

    def sources_text(self) -> str:
        from .chain import format_docs

        if not self.turns or not self.turns[-1].docs:
            return "The last answer used no document chunks."
        return format_docs(self.turns[-1].docs)

    def ingest(self, paths: list[Path]) -> str:
        from .ingest import ingest_paths

        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            return f"Not found: {', '.join(missing)}"
        report = ingest_paths(paths, self.rt.store, self.rt.settings)
        self._rebuild()  # the router's topic list must include the new files
        message = (f"{report.chunks} chunks embedded: {len(report.added)} added, "
                   f"{len(report.updated)} updated, {len(report.skipped)} unchanged.")
        if report.unsupported:
            from .ingest import SUPPORTED_SUFFIXES

            message += f"\nSkipped (unsupported type): {', '.join(report.unsupported)}. Supported: {', '.join(SUPPORTED_SUFFIXES)}."
        return message
