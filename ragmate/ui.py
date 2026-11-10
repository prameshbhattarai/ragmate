"""The Streamlit web UI: upload documents, chat with them, and see what RAG did for each answer.

Run with ``ragmate ui`` (or ``streamlit run ragmate/ui.py``). Like the CLI it only draws: answers
come from the same chain (chain.py) and graph (graph.py).

Streamlit re-runs this whole script on every click, so anything expensive (the vector store, the
compiled graph) is kept with ``st.cache_resource``, and the chat itself lives in ``st.session_state``.
"""

from __future__ import annotations

import streamlit as st

from ragmate.chain import chunk_number, cited, describe, format_docs
from ragmate.graph import thread_config
from ragmate.ingest import ingest_paths
from ragmate.memory import new_thread_id
from ragmate.runtime import Runtime
from ragmate.store import stats

st.set_page_config(page_title="ragmate", page_icon="📚", layout="wide")


@st.cache_resource
def runtime() -> Runtime:
    return Runtime.open()


@st.cache_resource
def agent_graph(k: int, mmr: bool, topics: tuple[str, ...]):
    # topics (paths + headings) is part of the cache key: after an ingest, even of an edited file
    # with the same name, the router must see the new topics
    return runtime().agent_graph(k, mmr)


rt = runtime()
state = st.session_state
state.setdefault("thread_id", new_thread_id())
state.setdefault("messages", [])  # [{"role", "content", "docs", "steps"}]

# --- Sidebar: documents and settings --------------------------------------------------------------

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add files", type=["md", "txt", "pdf"], accept_multiple_files=True)
    if uploads and st.button("Ingest", type="primary"):
        rt.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for upload in uploads:
            path = rt.settings.uploads_dir / upload.name
            path.write_bytes(upload.getbuffer())
            paths.append(path)
        with st.spinner("Embedding chunks..."):
            report = ingest_paths(paths, rt.store, rt.settings)
        st.success(f"{report.chunks} chunks from {len(report.added) + len(report.updated)} files "
                   f"({len(report.skipped)} unchanged).")

    counts = stats(rt.store)
    if counts:
        st.dataframe({"source": list(counts), "chunks": list(counts.values())}, hide_index=True)
    else:
        st.info("The store is empty. Upload files, or run `ragmate ingest data/sample`.")

    st.header("Settings")
    mode = st.radio("Mode", ["agent", "rag"], help="agent: LangGraph corrective RAG. rag: one LCEL chain.")
    k = st.slider("Chunks to retrieve (k)", 1, 10, rt.settings.top_k)
    mmr = st.toggle("MMR (diverse chunks)", value=False)
    if st.button("New chat"):
        state.thread_id = new_thread_id()
        state.messages = []

# --- Chat -------------------------------------------------------------------------------------

st.title("📚 ragmate")
st.caption(f"Chat model `{rt.settings.model}` · embeddings `{rt.settings.embed_model}` · all local via Ollama")


def show_details(message: dict) -> None:
    if message.get("steps"):
        st.caption("path: " + " → ".join(message["steps"]))
    if message.get("docs"):
        used = cited(message["content"], len(message["docs"]))
        lines = []
        for i, doc in enumerate(message["docs"], start=1):
            number = chunk_number(doc)
            where = describe(doc) + (f" · chunk {number}" if number else "")
            lines.append(f"**[{i}]** {where}" if i in used else f":gray[[{i}] {where} · not cited]")
        st.markdown("**Sources**  \n" + "  \n".join(lines))
        with st.expander(f"Retrieved chunks ({len(message['docs'])}) — exactly what the model read"):
            st.text(format_docs(message["docs"]))


for message in state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_details(message)

if question := st.chat_input("Ask about your documents"):
    state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if mode == "rag":
                result = rt.rag_chain(k, mmr).invoke(question)
                steps = ["retrieve", "generate"]
            else:
                graph = agent_graph(k, mmr, tuple(rt.topics()))
                result = graph.invoke({"question": question}, thread_config(state.thread_id))
                steps = result["steps"]
        reply = {"role": "assistant", "content": result["answer"], "docs": result.get("docs", []), "steps": steps}
        st.markdown(reply["content"])
        show_details(reply)
    state.messages.append(reply)
