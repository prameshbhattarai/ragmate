# ragmate

**Chat with your documents on a local model, and learn RAG by reading how it works.** Point it at
Markdown, text or PDF files; ragmate splits them into chunks, embeds them into a local vector
database (Chroma), and answers your questions from them with citations. It runs fully on your
machine: Gemma (`gemma3:4b`) via Ollama writes the answers and `nomic-embed-text` makes the
embeddings, so nothing leaves your laptop.

The code is small on purpose and commented where each RAG, LangChain or LangGraph idea is used, so
it doubles as a worked example: [Learn RAG by reading ragmate](#learn-rag-by-reading-ragmate).

## Quick start

```bash
# 1. Local models (one-off)
brew install ollama
ollama pull gemma3:4b          # the chat model
ollama pull nomic-embed-text   # the embedding model

# 2. Install
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 3. Run
ragmate ingest data/sample                       # index the bundled sample documents
ragmate inspect "how many vacation days?"        # see what retrieval finds, before any LLM
ragmate ask "How many days of holiday do I get?" # an answer, with sources and the path taken
ragmate                                          # interactive chatbot: streaming, memory, /help
ragmate ui                                       # the web UI (Streamlit)
```

`data/sample/` holds three short documents. One is the handbook of **Nimbus Coffee Co., a
fictional company**, so its facts (27 vacation days, the "Brewhouse" HR portal ...) are things no
model can know without retrieval. Ask about them and check that the answer cites the handbook.

Two modes, on the CLI (`--mode`) and in the UI:

| Mode | What runs | Use it to see |
|---|---|---|
| `rag` | one LCEL chain: retrieve → prompt → answer ([chain.py](ragmate/chain.py)) | classic RAG |
| `agent` (default) | a LangGraph graph that routes, grades, rewrites and retries ([graph.py](ragmate/graph.py)) | agentic / corrective RAG |

Every answer prints its path, for example
`condense → route:docs → retrieve:4 → grade:2/4 → generate`.

### Interactive chat

Run `ragmate` with no command (or `ragmate chat`). Each graph step appears as it finishes, the
answer streams in token by token, and follow-up questions use the conversation so far. Up-arrow
recalls earlier questions. Ctrl-C stops an answer; Ctrl-D quits.

| Command | Does |
|---|---|
| `/mode agent` · `/mode rag` | LangGraph agent with memory, or the plain chain without it |
| `/k N` · `/mmr on\|off` | how many chunks to retrieve, and whether to diversify them |
| `/sources` | the full text of the chunks behind the last answer |
| `/history` · `/new` | show the conversation, or forget it |
| `/stats` · `/ingest PATH` | what is in the store, or add files without leaving the chat |
| `/help` · `/exit` | |

Streaming is its own concept, in [`chat.py`](ragmate/chat.py): `graph.stream(..., stream_mode=["updates", "messages"])`
yields node updates and model tokens as they happen, and only the answering nodes' tokens are shown.

## How RAG works, in one picture

```
 INDEXING (once per file)                         ANSWERING (every question)
 ───────────────────────                          ──────────────────────────
 file.pdf ─load─> text ─split─> chunks            question ─embed─> vector
                                  │                                    │
                         embed (nomic-embed-text)          nearest neighbours (cosine)
                                  │                                    │
                                  v                                    v
                        ┌──────────────────┐  top-k chunks   ┌───────────────────┐
                        │ Chroma (on disk) │ ──────────────> │ prompt = rules +  │
                        │ vectors + text   │                 │ chunks + question │
                        └──────────────────┘                 └─────────┬─────────┘
                                                                       v
                                                         gemma3:4b → answer with [1] [2] citations
```

The vector database is a folder inside the project, **`.ragmate/`** (gitignored):

| Path | Contents |
|---|---|
| `.ragmate/chroma/` | Chroma: `chroma.sqlite3` (chunk text + metadata) and the vector index files |
| `.ragmate/manifest.json` | `{file: sha256}`, so unchanged files are not re-embedded |
| `.ragmate/uploads/` | files uploaded through the web UI |

Set `RAGMATE_HOME` to put it elsewhere (relative paths are taken from the project root). Delete
the folder, or run `ragmate ingest --reset ...`, to start over.

More detail and the full graph diagram: [docs/how-it-works.md](docs/how-it-works.md).

## Learn RAG by reading ragmate

Read the files in this order. Each step names the concept, where to look, and a small change to
try. Every file starts with a docstring naming its concept, and `# LangChain:` / `# LangGraph:`
comments mark the lines that matter.

### 1. Two models: [`config.py`](ragmate/config.py)

RAG needs a **chat model** (reads and writes text) and an **embedding model** (text → vector).
Both are built lazily, so importing ragmate never contacts Ollama.

> **Try:** `python -c "from ragmate.config import embeddings; v = embeddings().embed_query('hello'); print(len(v), v[:5])"`:
> a sentence is now 768 numbers.

### 2. Loading and chunking: [`ingest.py`](ragmate/ingest.py)

Files become LangChain `Document`s (text + metadata), then `RecursiveCharacterTextSplitter` cuts
them into overlapping chunks with stable ids. A hash per file means re-ingesting only re-embeds
files that changed.

> **Try:** `RAGMATE_CHUNK_SIZE=250 ragmate ingest --reset data/sample`, then `ragmate stats`: more,
> smaller chunks. Ask the same question before and after.

### 3. Embeddings and the vector store: [`store.py`](ragmate/store.py)

Chroma stores each chunk's vector next to its text, in a folder on disk. Search embeds the question
and returns the closest chunks by cosine similarity. `HashingEmbeddings` is a keyword-only
"embedding" for tests; compare it with the real one to see why RAG uses a learned model.

> **Try:** `ragmate inspect "how much time off do I get?"`. The word "vacation" is not in the
> question, yet the handbook's vacation chunk should score highest. That is semantic search.
> Then `ragmate inspect "..." --k 8` and look at where the scores drop off.

### 4. Classic RAG as a pipeline: [`chain.py`](ragmate/chain.py)

LCEL composes Runnables with `|`. The retrieved chunks are numbered and pasted into the prompt
(the "augmentation"), and the prompt tells the model to answer only from them and to cite them.

> **Try:** remove the "Do not use outside knowledge" line from `RAG_PROMPT`, then ask
> `ragmate ask --mode rag "What is the capital of France?"`. Then put it back.

### 5. Structured output: [`agent.py`](ragmate/agent.py)

The agent asks the model three small questions, each with a Pydantic schema: `Route` (docs or
small talk?), `Grade` (is this chunk relevant?), `Rewrite` (a better search query).
`with_structured_output(..., method="json_schema")` makes `invoke()` return those objects instead
of text.

> **Try:** add `confidence: int` to `Grade` and print it in `grade_node` in graph.py.

### 6. Agentic RAG as a graph: [`graph.py`](ragmate/graph.py)

A LangGraph `StateGraph`: typed state, nodes that are plain functions, conditional edges that
choose the next node, and a bounded loop (rewrite → retrieve again). If nothing relevant turns up
after `RAGMATE_MAX_RETRIES` rewrites, it says "not found" instead of guessing.

> **Try:** `ragmate ask "What is the dress code?"` (it isn't in the documents): watch
> `grade:0/4 → rewrite → ... → fallback`. Then try `RAGMATE_MAX_RETRIES=0`.

### 7. Memory: [`memory.py`](ragmate/memory.py) and the checkpointer

The graph's checkpointer saves state per `thread_id`, so the history carries over between turns.
A follow-up like "can I carry them over?" is *condensed* into a standalone question before it is
searched, because the vector store doesn't know what "them" means.

> **Try:** in `ragmate chat`, ask "How many vacation days do I get?" then "and can I carry them
> over?". Add a `print(query)` in `condense_node` to see the rewritten question.

### 8. Measuring it: [`evals/`](evals/)

`evals/cases.jsonl` lists questions, the file each answer is in, and the route the router should
take. Most questions use *different words* from the documents, on purpose.

```bash
python evals/run_evals.py --check              # no model: is the eval set itself consistent? (CI runs this)
python evals/run_evals.py --model gemma3:4b    # real models: hit@k for retrieval, router accuracy
```

> **Try:** run the scored eval with `RAGMATE_CHUNK_SIZE=300` and with `1200`, and compare hit@k.

## Testing

```bash
pytest -q
```

No test calls Ollama. `tests/fakes.py` scripts the models' replies, so a test can say "the grader
rejects everything, then the rewrite finds it" and check the exact path the graph takes. Retrieval
tests use a real Chroma store with the keyword-only `HashingEmbeddings`.

## Project layout

```
ragmate/
  config.py   settings and the two models (lazy)
  ingest.py   load → split → label, incremental by file hash
  store.py    Chroma, retrievers (similarity / MMR), HashingEmbeddings
  chain.py    classic RAG as one LCEL chain
  agent.py    structured-output schemas and prompts
  graph.py    corrective RAG as a LangGraph StateGraph
  memory.py   history trimming and follow-up condensing
  runtime.py  wiring shared by the CLI and UI
  cli.py      Typer CLI
  ui.py       Streamlit UI
evals/        eval cases and runner
data/sample/  the bundled corpus
tests/        pytest, fakes only
```

## License

MIT
