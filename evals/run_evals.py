"""Score ragmate's retrieval and routing on evals/cases.jsonl, against the bundled sample corpus.

Each case is a question plus what good behaviour looks like:

    expect_route   "docs" or "general": what the router should decide
    expect_source  the file the answer is in; it should be among the top-k retrieved chunks
    reference      a keyword query that must find expect_source, used by --check

Note that the questions mostly use *different words* than the documents ("holiday" vs "vacation").
That is on purpose: it is what an embedding model is for, and a keyword search would fail them.

Two modes:

    python evals/run_evals.py --check
        No model. Checks the eval set itself: every case is well-formed, and every reference query
        retrieves its expected source using the keyword-only HashingEmbeddings. This is what CI runs.

    python evals/run_evals.py --model gemma3:4b [--k 4] [--out results.json]
        Ingests the sample corpus with the real embedding model into a temporary store, then scores
        retrieval (hit@k) for every docs case and the router's decision for every case.
        This calls your local models once per case: run it when the machine can take the load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CASES = Path(__file__).with_name("cases.jsonl")
SAMPLE = ROOT / "data" / "sample"
FIELDS = {"id", "question", "expect_route", "expect_source", "reference"}


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def temp_store(embedding):
    from ragmate.config import get_settings
    from ragmate.ingest import ingest_paths
    from ragmate.store import open_store

    os.environ["RAGMATE_HOME"] = tempfile.mkdtemp(prefix="ragmate-eval-")
    settings = get_settings()
    store = open_store(settings, embedding)
    ingest_paths([SAMPLE], store, settings)
    return store, settings


def hit(store, query: str, source: str, k: int) -> bool:
    return any(d.metadata["source"] == source for d in store.similarity_search(query, k=k))


def check(k: int) -> int:
    from ragmate.store import HashingEmbeddings

    cases = load_cases()
    sources = {p.name for p in SAMPLE.iterdir()}
    store, _ = temp_store(HashingEmbeddings())
    problems = []
    ids = [c.get("id") for c in cases]
    if len(set(ids)) != len(ids):
        problems.append("duplicate case ids")
    for case in cases:
        if set(case) != FIELDS:
            problems.append(f"{case.get('id')}: fields must be exactly {sorted(FIELDS)}")
            continue
        if case["expect_route"] not in ("docs", "general"):
            problems.append(f"{case['id']}: expect_route must be docs or general")
        elif case["expect_route"] == "docs":
            if case["expect_source"] not in sources:
                problems.append(f"{case['id']}: {case['expect_source']} is not in data/sample")
            elif not hit(store, case["reference"], case["expect_source"], k):
                problems.append(f"{case['id']}: reference query does not retrieve {case['expect_source']}")
        elif case["expect_source"] or case["reference"]:
            problems.append(f"{case['id']}: general cases have no source or reference")
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    print(f"{len(cases)} cases: all cases ok")
    return 0


def score(model: str, k: int, out: str | None) -> int:
    from ragmate.agent import build_models, route_messages
    from ragmate.config import embeddings
    from ragmate.store import topics as store_topics

    store, _ = temp_store(embeddings())
    models = build_models(model)
    topics = store_topics(store)
    rows = []
    for case in load_cases():
        try:
            routed = models.router.invoke(route_messages(case["question"], topics)).datasource
        except Exception as exc:  # noqa: BLE001 - a bad reply is a failed case, not a crash
            routed = f"error: {exc.__class__.__name__}"
        retrieved = hit(store, case["question"], case["expect_source"], k) if case["expect_route"] == "docs" else None
        rows.append({"id": case["id"], "route_ok": routed == case["expect_route"], "routed": routed, "hit": retrieved})
        print(f"{case['id']:<16} route {'ok ' if rows[-1]['route_ok'] else 'BAD'} ({routed:<7}) hit@{k} {retrieved}")
    docs_rows = [r for r in rows if r["hit"] is not None]
    print("\n| model | route accuracy | hit@%d |\n|---|---|---|" % k)
    print(f"| {model} | {sum(r['route_ok'] for r in rows)}/{len(rows)} | {sum(r['hit'] for r in docs_rows)}/{len(docs_rows)} |")
    if out:
        Path(out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="validate the eval set; no model")
    parser.add_argument("--model", help="chat model to score, e.g. gemma3:4b")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", help="write per-case results as JSON")
    args = parser.parse_args()
    if args.check:
        return check(args.k)
    if args.model:
        return score(args.model, args.k, args.out)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
