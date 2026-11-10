import pytest
from fakes import doc
from pydantic import ValidationError

from ragmate.agent import Grade, Route, answer_messages, grade_messages, route_messages


def test_schemas_reject_values_outside_the_choices():
    assert Route(datasource="docs").datasource == "docs"
    with pytest.raises(ValidationError):
        Route(datasource="internet")
    with pytest.raises(ValidationError):
        Grade(relevant="maybe")


def test_route_prompt_lists_the_ingested_topics():
    system, human = route_messages("What is MMR?", ["rag-glossary.md: Chunking; MMR", "ollama-notes.md"])
    assert "- rag-glossary.md: Chunking; MMR\n- ollama-notes.md" in system[1]
    assert '"hi!" -> general' in system[1]  # the few-shot examples are in the prompt
    assert human == ("human", "What is MMR?")
    assert "no documents ingested yet" in route_messages("hi", [])[0][1]


def test_route_schema_asks_for_the_reason_before_the_decision():
    assert list(Route.model_json_schema()["properties"]) == ["reason", "datasource"]


def test_grade_prompt_contains_question_and_chunk():
    _, human = grade_messages("Q?", doc("chunk text"))
    assert "Q?" in human[1] and "chunk text" in human[1]


def test_answer_prompt_is_the_rag_chain_prompt():
    messages = answer_messages("Q?", [doc("fact", "f.md")])
    assert "[1] (f.md)\nfact" in messages[0].content
    assert messages[-1].content == "Q?"
