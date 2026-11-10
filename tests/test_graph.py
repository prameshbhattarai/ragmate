from fakes import FixedRetriever, Scripted, doc, grade, models, rewrite, route, text

from ragmate.graph import NOT_FOUND, build_graph, thread_config


def run(graph, question, thread="t1"):
    return graph.invoke({"question": question}, thread_config(thread))


def test_general_questions_skip_retrieval():
    retriever = FixedRetriever(doc("x"))
    graph = build_graph(models(router=Scripted(route("general")), llm=Scripted(text("Hello!"))), retriever)
    result = run(graph, "hi there")
    assert result["answer"] == "Hello!"
    assert result["steps"] == ["condense", "route:general", "direct_answer"]
    assert retriever.queries == []


def test_relevant_docs_are_answered_with_sources():
    retriever = FixedRetriever(doc("27 days", index=0), doc("coffee", index=1))
    graph = build_graph(models(grader=Scripted(grade(True), grade(False))), retriever)
    result = run(graph, "vacation days?")
    assert result["steps"] == ["condense", "route:docs", "retrieve:2", "grade:1/2", "generate"]
    assert [d.page_content for d in result["docs"]] == ["27 days"]  # the irrelevant chunk was dropped


def test_nothing_relevant_rewrites_the_query_and_searches_again():
    retriever = FixedRetriever(doc("27 days"))
    graph = build_graph(
        models(grader=Scripted(grade(False), grade(True)), rewriter=Scripted(rewrite("annual leave allowance"))),
        retriever,
    )
    result = run(graph, "how much time off?")
    assert result["steps"] == [
        "condense", "route:docs", "retrieve:1", "grade:0/1", "rewrite", "retrieve:1", "grade:1/1", "generate",
    ]
    assert retriever.queries == ["how much time off?", "annual leave allowance"]


def test_retries_are_bounded_then_it_admits_not_found():
    llm = Scripted(text("should not be called"))
    graph = build_graph(models(llm=llm, grader=Scripted(grade(False))), FixedRetriever(doc("x")), max_retries=2)
    result = run(graph, "unknowable?")
    assert result["answer"] == NOT_FOUND
    assert result["steps"].count("rewrite") == 2
    assert result["steps"][-1] == "fallback"
    assert llm.calls == []  # no answer is generated from nothing


def test_model_errors_fall_back_to_safe_defaults():
    graph = build_graph(
        models(router=Scripted(ValueError("bad json")), grader=Scripted(ValueError("bad json"))),
        FixedRetriever(doc("fact")),
    )
    result = run(graph, "q")
    assert result["steps"] == ["condense", "route:docs", "retrieve:1", "grade:1/1", "generate"]


def test_checkpointer_remembers_history_and_condenses_follow_ups():
    llm = Scripted(text("27 days [1]"), text("Can unused vacation days be carried over?"), text("Up to 5 [1]"))
    retriever = FixedRetriever(doc("facts"))
    graph = build_graph(models(llm=llm), retriever)

    run(graph, "How many vacation days?", thread="chat")
    second = run(graph, "can I carry them over?", thread="chat")

    assert [tuple(m) for m in second["history"]] == [
        ("human", "How many vacation days?"), ("ai", "27 days [1]"),
        ("human", "can I carry them over?"), ("ai", "Up to 5 [1]"),
    ]
    assert retriever.queries[-1] == "Can unused vacation days be carried over?"  # searched condensed
    other = run(graph, "fresh question", thread="other")
    assert tuple(other["history"][0]) == ("human", "fresh question")  # new thread: no memory


def test_graph_diagram_has_every_node():
    diagram = build_graph(models(), FixedRetriever()).get_graph().draw_mermaid()
    for node in ("condense", "route", "retrieve", "grade", "rewrite", "generate", "direct_answer", "fallback"):
        assert node in diagram
