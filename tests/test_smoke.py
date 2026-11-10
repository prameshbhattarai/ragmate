import importlib

import pytest


def test_importing_never_builds_a_model(monkeypatch):
    import langchain_ollama

    def boom(*args, **kwargs):
        raise AssertionError("a model was built at import time")

    monkeypatch.setattr(langchain_ollama, "ChatOllama", boom)
    monkeypatch.setattr(langchain_ollama, "OllamaEmbeddings", boom)
    for name in ("config", "ingest", "store", "chain", "agent", "memory", "graph", "runtime", "cli"):
        importlib.reload(importlib.import_module(f"ragmate.{name}"))


def test_settings_read_the_environment(monkeypatch):
    from ragmate.config import get_settings

    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:1b")
    monkeypatch.setenv("RAGMATE_TOP_K", "7")
    s = get_settings()
    assert (s.model, s.top_k, s.embed_model) == ("gemma3:1b", 7, "nomic-embed-text")
    assert s.chroma_dir.name == "chroma"


@pytest.mark.parametrize("history,expected", [([], "q"), ([("human", "a"), ("ai", "b")], "standalone q")])
def test_condense_only_calls_the_model_with_history(history, expected):
    from fakes import Scripted, text

    from ragmate.memory import condense

    llm = Scripted(text("standalone q"))
    assert condense(llm, history, "q") == expected
    assert len(llm.calls) == (1 if history else 0)
