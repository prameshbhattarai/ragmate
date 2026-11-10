import pytest
from conftest import SAMPLE_DIR
from fakes import Scripted, grade, models, text
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ragmate.chat import EXIT, ChatSession
from ragmate.ingest import ingest_paths
from ragmate.runtime import Runtime
from ragmate.store import HashingEmbeddings


@pytest.fixture
def rt(settings):
    runtime = Runtime.open(settings, HashingEmbeddings())
    ingest_paths([SAMPLE_DIR], runtime.store, settings)
    return runtime


def events(session, question):
    return list(session.stream(question))


def test_agent_streams_steps_then_answer_tokens(rt):
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="You get 27 vacation days [1].")]))
    session = ChatSession(rt, models=models(llm=llm, grader=Scripted(grade(True), grade(False))))
    out = events(session, "How many vacation days?")

    steps = [p for k, p in out if k == "step"]
    tokens = [p for k, p in out if k == "token"]
    kind, turn = out[-1]
    assert steps == ["condense", "route:docs", "retrieve:4", "grade:1/4", "generate"]
    assert "".join(tokens) == "You get 27 vacation days [1]." and len(tokens) > 1  # really streamed
    assert kind == "done" and turn.answer == "You get 27 vacation days [1]." and len(turn.docs) == 1


def test_router_and_grader_output_never_reaches_the_user(rt):
    session = ChatSession(rt, models=models(llm=Scripted(text("answer"))))
    out = events(session, "q")
    assert not any(k == "token" and "datasource" in str(p) for k, p in out)


def test_memory_survives_a_rebuild_and_new_forgets(rt):
    session = ChatSession(rt, models=models(llm=Scripted(text("first"), text("standalone"), text("second"))))
    events(session, "How many vacation days?")
    session.run_command("/k 2")  # rebuilds the graph...
    _, turn = events(session, "and carry over?")[-1]
    assert "condense" in turn.steps and session.models.llm.calls[1][0][0] == "system"  # ...history kept: condensed
    assert session.run_command("/new").startswith("New conversation") and session.turns == []


def test_rag_mode_streams_without_the_agent(rt):
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="27 days [1]")]))
    session = ChatSession(rt, mode="rag", models=models(llm=llm))
    out = events(session, "vacation days")
    assert [p for k, p in out if k == "step"] == ["retrieve:4", "generate"]
    assert out[-1][1].answer == "27 days [1]"


@pytest.mark.parametrize(
    "command,expected",
    [
        ("/mode rag", "Mode: rag"),
        ("/mode nope", "Use /mode agent or /mode rag"),
        ("/k 0", "Use /k N"),
        ("/k 6", "Retrieving 6 chunks"),
        ("/mmr on", "MMR on"),
        ("/stats", "nimbus-handbook.md"),
        ("/sources", "no document chunks"),
        ("/history", "No questions yet"),
        ("/ingest /does/not/exist", "Not found"),
        ("/bogus", "Unknown command /bogus"),
        ("/", "Commands:"),
        ("/   ", "Commands:"),
    ],
)
def test_commands(rt, command, expected):
    assert expected in ChatSession(rt, models=models()).run_command(command)


def test_unknown_mode_is_rejected(rt):
    with pytest.raises(ValueError):
        ChatSession(rt, mode="RAG-ish", models=models())


def test_ingest_command_reports_unsupported_files(rt, tmp_path):
    docx = tmp_path / "x.docx"
    docx.write_bytes(b"x")
    assert "Skipped (unsupported type)" in ChatSession(rt, models=models()).run_command(f"/ingest {docx}")


def test_exit_and_ingest(rt, tmp_path):
    session = ChatSession(rt, models=models())
    assert session.run_command("/exit") is EXIT
    note = tmp_path / "note.md"
    note.write_text("# Note\n\nPineapple.")
    assert "1 added" in session.run_command(f"/ingest {note}")
