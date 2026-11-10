"""The terminal front end. It never decides anything: it calls the chain or graph and draws the result.

    ragmate                             interactive chat (same as `ragmate chat`): /help for commands
    ragmate ingest data/sample          load, split, embed and store files (unchanged files are skipped)
    ragmate stats                       what is in the vector store (no model call)
    ragmate inspect "question"          retrieval only: the chunks and scores a question finds
    ragmate ask "question"              one answer (--mode agent | rag)
    ragmate chat                        multi-turn chat with memory, streaming and /commands
    ragmate ui                          the Streamlit web UI

Every command that answers or searches calls Ollama. ``stats`` and ``--help`` never do.
"""

from __future__ import annotations

import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Chat with your documents on a local model, and see every step of RAG while you do.",
)
console = Console()

class Mode(str, Enum):
    agent = "agent"
    rag = "rag"


ModeOpt = typer.Option(Mode.agent, "--mode", "-m", case_sensitive=False, help='"agent" (LangGraph, corrective RAG) or "rag" (plain LCEL chain).')
KOpt = typer.Option(None, "--k", "-k", help="How many chunks to retrieve (default RAGMATE_TOP_K).")
MmrOpt = typer.Option(False, "--mmr", help="Use MMR retrieval: relevant and diverse chunks.")


def _runtime():
    from .runtime import Runtime

    return Runtime.open()


def _version(value: bool):
    if value:
        console.print(f"ragmate {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version."),
):
    """Chat with your documents on a local model. With no command, starts the interactive chat."""
    if ctx.invoked_subcommand is None:
        _chat_loop("agent", None, False)


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or folders (.md, .txt, .pdf)."),
    reset: bool = typer.Option(False, "--reset", help="Delete everything in the store first."),
):
    """Load, split, embed and store documents."""
    from .ingest import ingest_paths, save_manifest
    from .store import reset_store

    rt = _runtime()
    if reset:
        reset_store(rt.store)
        save_manifest(rt.settings.manifest_path, {})
    with console.status("Embedding chunks with the local model..."):
        report = ingest_paths(paths, rt.store, rt.settings)
    for label, names, style in (
        ("added", report.added, "green"),
        ("updated", report.updated, "yellow"),
        ("unchanged", report.skipped, "dim"),
        ("skipped", report.unsupported, "bold red"),
    ):
        for name in names:
            console.print(f"[{style}]{label:>9}[/] {name}")
    console.print(f"[bold]{report.chunks}[/] chunks embedded. Store: {rt.settings.chroma_dir}")
    if report.unsupported:
        from .ingest import SUPPORTED_SUFFIXES

        console.print(f"[red]Unsupported file type(s) skipped; ragmate reads {', '.join(SUPPORTED_SUFFIXES)}.[/]")


@app.command()
def stats():
    """Show what is in the vector store. No model call."""
    from .store import stats as store_stats

    rt = _runtime()
    counts = store_stats(rt.store)
    if not counts:
        console.print("The store is empty. Try: [bold]ragmate ingest data/sample[/]")
        return
    table = Table("source", "chunks", title=f"{rt.settings.chroma_dir}")
    for source, n in counts.items():
        table.add_row(source, str(n))
    console.print(table)


@app.command()
def inspect(question: str, k: Optional[int] = KOpt):
    """Retrieval only: the chunks a question finds, with their relevance scores. No chat model."""
    from .store import search_with_scores

    rt = _runtime()
    results = search_with_scores(rt.store, question, k or rt.settings.top_k)
    if not results:
        console.print("Nothing found: is the store empty? (ragmate stats)")
        return
    for i, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content.strip().replace("\n", " ")
        console.print(Panel(preview[:500], title=f"[{i}] {doc.metadata['chunk_id']}", subtitle=f"score {score:.3f}"))


def _print_sources(answer: str, docs) -> None:
    """The key for the answer's [n] citations; the sources it actually cited are marked."""
    from rich.text import Text

    from .chain import chunk_number, cited, describe

    if not docs:
        return
    used = cited(answer, len(docs))
    console.print(Text("Sources", style="bold"))
    for i, doc in enumerate(docs, start=1):
        number = chunk_number(doc)
        line = Text(f"  [{i}] ", style="cyan" if i in used else "dim")
        line.append(describe(doc), style="default" if i in used else "dim")
        if number is not None:
            line.append(f" · chunk {number}", style="dim")
        line.append("  cited" if i in used else "  not cited", style="green" if i in used else "dim")
        console.print(line)  # Text objects are never markup-parsed or auto-highlighted


def _render(session, question: str) -> None:
    """Draw one answer while it streams: each graph step as it finishes, then the answer's tokens."""
    streamed = False
    turn = None
    try:
        with console.status("[dim]thinking...[/]") as status:
            for kind, payload in session.stream(question):
                if kind == "step":
                    if not streamed:  # once the answer is streaming, steps would land inside it;
                        console.print(f"  · {payload}", style="dim", markup=False, highlight=False)
                elif kind == "token":
                    if not streamed:
                        status.stop()  # the spinner and streamed text can't share the line
                        console.print("[bold green]ragmate>[/] ", end="")
                        streamed = True
                    console.print(payload, end="", markup=False, highlight=False)
                else:
                    turn = payload
    except KeyboardInterrupt:
        console.print("\n[yellow](stopped)[/]")
        return
    if streamed:
        console.print()
    elif turn is not None:  # e.g. the fallback node: a fixed answer, no model tokens to stream
        console.print("[bold green]ragmate>[/] " + turn.answer)
    if turn is not None:
        # highlight=False: Rich's auto-highlighter would colour fragments like "e:d" in "route:docs"
        console.print(f"path: {' → '.join(turn.steps)}", style="dim", markup=False, highlight=False)
        _print_sources(turn.answer, turn.docs)


def _session(mode, k: Optional[int], mmr: bool):
    from .chat import ChatSession

    return ChatSession(_runtime(), mode=Mode(mode).value, k=k, mmr=mmr)


@app.command()
def ask(question: str, mode: Mode = ModeOpt, k: Optional[int] = KOpt, mmr: bool = MmrOpt):
    """Ask one question."""
    _render(_session(mode, k, mmr), question)


def _enable_line_editing(history_file: Path) -> None:
    """Arrow keys and up-arrow history at the prompt, kept between sessions. Optional nicety."""
    try:
        import readline
    except ImportError:  # Windows without pyreadline
        return
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        if history_file.exists():
            readline.read_history_file(history_file)
        import atexit

        atexit.register(readline.write_history_file, history_file)
    except OSError:
        pass


def _prompt() -> str:
    """The ``you>`` prompt, passed to input() itself so readline knows its width.

    Rich's console.input prints the prompt separately, so readline thinks the line starts at column
    0 and redraws over the prompt on up-arrow or long lines. The ANSI colour codes are wrapped in
    \\001 ... \\002, which tells readline they take no space on screen.
    """
    if not console.is_terminal:
        return "you> "
    return "\001\033[1;36m\002you>\001\033[0m\002 "


def _chat_loop(mode, k: Optional[int], mmr: bool) -> None:
    from .chat import EXIT
    from .store import stats as store_stats

    session = _session(mode, k, mmr)
    settings = session.rt.settings
    _enable_line_editing(settings.home / "history")
    counts = store_stats(session.rt.store)
    docs_line = f"{sum(counts.values())} chunks from {len(counts)} files" if counts else "no documents yet: /ingest data/sample"
    console.print(Panel(
        f"[bold]ragmate[/] {__version__} · chat [cyan]{settings.model}[/] · embeddings [cyan]{settings.embed_model}[/]\n"
        f"Store: {docs_line}\n"
        "[dim]Ask anything about your documents. /help for commands, Ctrl-D to quit.[/]",
        border_style="green",
    ))
    while True:
        try:
            line = input(_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line.startswith("/"):
            reply = session.run_command(line)
            if reply is EXIT:
                break
            console.print(reply, markup=False, highlight=False)
            continue
        _render(session, line)


@app.command()
def chat(mode: Mode = ModeOpt, k: Optional[int] = KOpt, mmr: bool = MmrOpt):
    """Interactive chat with memory, streaming answers and /commands (also: just `ragmate`)."""
    _chat_loop(mode, k, mmr)


@app.command()
def ui():
    """Open the Streamlit web UI."""
    ui_main()


def ui_main() -> None:
    """Entry point for ``ragmate-ui``."""
    ui_file = Path(__file__).with_name("ui.py")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ui_file)]))
