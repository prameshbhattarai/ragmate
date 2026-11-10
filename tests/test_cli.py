import pytest
from conftest import SAMPLE_DIR
from fakes import Scripted, models, text
from typer.testing import CliRunner

from ragmate import cli, runtime
from ragmate.store import HashingEmbeddings

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_ollama(monkeypatch):
    """Every Runtime uses the model-free embeddings, and the fake models."""
    real_open = runtime.Runtime.open.__func__
    monkeypatch.setattr(runtime.Runtime, "open", classmethod(lambda cls, s=None, e=None: real_open(cls, s, HashingEmbeddings())))
    monkeypatch.setattr("ragmate.agent.build_models", lambda model=None: models(llm=Scripted(text("27 days [1]"))))


def test_help_and_version():
    assert runner.invoke(cli.app, ["--help"]).exit_code == 0
    assert "ragmate 0.1.0" in runner.invoke(cli.app, ["--version"]).output


def test_stats_on_empty_store():
    result = runner.invoke(cli.app, ["stats"])
    assert result.exit_code == 0 and "empty" in result.output


def test_ingest_then_stats_then_inspect():
    result = runner.invoke(cli.app, ["ingest", str(SAMPLE_DIR)])
    assert result.exit_code == 0, result.output
    assert "added" in result.output and "nimbus-handbook.md" in result.output

    again = runner.invoke(cli.app, ["ingest", str(SAMPLE_DIR)])
    assert "unchanged" in again.output and "0 chunks" in again.output

    assert "nimbus-handbook.md" in runner.invoke(cli.app, ["stats"]).output
    inspected = runner.invoke(cli.app, ["inspect", "vacation days carried over", "-k", "2"])
    assert inspected.exit_code == 0 and "nimbus-handbook.md" in inspected.output and "score" in inspected.output


@pytest.mark.parametrize("mode", ["agent", "rag"])
def test_ask_in_both_modes(mode):
    runner.invoke(cli.app, ["ingest", str(SAMPLE_DIR)])
    result = runner.invoke(cli.app, ["ask", "How many vacation days?", "--mode", mode])
    assert result.exit_code == 0, result.output
    assert "27 days" in result.output and "path:" in result.output


def test_mode_is_validated_and_case_insensitive():
    runner.invoke(cli.app, ["ingest", str(SAMPLE_DIR)])
    assert runner.invoke(cli.app, ["ask", "q", "--mode", "bogus"]).exit_code == 2
    assert runner.invoke(cli.app, ["ask", "q", "--mode", "RAG"]).exit_code == 0


def test_steps_never_print_inside_a_streamed_answer(monkeypatch):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    llm = GenericFakeChatModel(messages=iter([AIMessage(content="You get 27 days [1].")]))
    monkeypatch.setattr("ragmate.agent.build_models", lambda model=None: models(llm=llm))
    runner.invoke(cli.app, ["ingest", str(SAMPLE_DIR)])
    result = runner.invoke(cli.app, ["ask", "How many vacation days?"])
    answer_line = next(line for line in result.output.splitlines() if "You get 27 days" in line)
    assert "·" not in answer_line
    assert "path: condense → route:docs → retrieve:4 → grade:4/4 → generate" in result.output
    assert "Sources" in result.output
    assert "[1] nimbus-handbook.md · chunk" in result.output and "  cited" in result.output
    assert "[2]" in result.output and "not cited" in result.output  # retrieved but unused


def test_ingest_warns_about_unsupported_files(tmp_path):
    docx = tmp_path / "handbook.docx"
    docx.write_bytes(b"x")
    result = runner.invoke(cli.app, ["ingest", str(docx)])
    assert "skipped" in result.output and "Unsupported file type" in result.output


def test_bare_ragmate_opens_the_chat_and_runs_commands():
    runner.invoke(cli.app, ["ingest", str(SAMPLE_DIR)])
    result = runner.invoke(cli.app, [], input="/\n/help\nHow many vacation days?\n/mode rag\n/exit\n")
    assert result.exit_code == 0, result.output
    assert "/help for commands" in result.output  # the banner
    assert "/mode agent|rag" in result.output  # /help
    assert "27 days [1]" in result.output and "path: condense" in result.output
    assert "Mode: rag" in result.output
