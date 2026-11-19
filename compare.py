"""
Compare the LangChain and LangGraph engines on ANY document and ANY questions.

Run from the project root (where app.py is), with your venv active:

    python compare.py --pdf data/MyDoc.pdf --questions my_questions.json
    python compare.py --pdf a.pdf --pdf b.pdf --questions q.csv --repeats 3 --pause 6

Questions file (JSON list or CSV):
    JSON: [{"question": "...", "kind": "in"|"out", "keywords": ["18"], "gold": "optional text"}, ...]
    CSV : question,kind,keywords,gold        (keywords separated with |)
  kind "in"  -> the document contains the answer; every keyword must appear in the answer
  kind "out" -> the document does NOT contain it; a good engine says "not found"
  gold       -> optional text that should appear in a retrieved chunk (gives Hit@k)

Each question makes 2 model calls per repeat (one per engine). Results are saved to
compare_results.csv (per question) and compare_summary.md (metrics table).
"""

import argparse
import csv
import io
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from pipeline import (  # noqa: E402
    SCORE_THRESHOLD,
    build_chain_engine,
    build_graph_engine,
    build_retriever,
)

NOT_FOUND_PATTERNS = (
    "couldn't find", "could not find", "not found", "no information", "don't know", "do not know",
    "not mentioned", "not provided", "not specified", "not covered", "not available",
    "cannot answer", "can't answer", "unable to find", "outside the scope", "not in the document",
    "does not contain", "doesn't contain", "does not specify", "doesn't specify",
)


class LocalFile(io.BytesIO):
    """Makes a file on disk look like a Streamlit upload."""

    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name


# ---------------------------------------------------------------- questions
def load_questions(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        raw = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        for r in raw:
            r["keywords"] = [x.strip() for x in (r.get("keywords") or "").split("|") if x.strip()]
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for r in raw:
        kind = str(r.get("kind", r.get("type", "in"))).lower()
        if kind not in ("in", "out"):
            raise SystemExit(f"kind must be 'in' or 'out', got {kind!r} for: {r.get('question')}")
        out.append({"question": r["question"].strip(), "kind": kind,
                    "keywords": list(r.get("keywords", r.get("expected", [])) or []),
                    "gold": (r.get("gold") or "").strip()})
    if not out:
        raise SystemExit("No questions found in the questions file.")
    return out


# ---------------------------------------------------------------- scoring
def normalize(text: str) -> str:
    t = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"(?<=\d),(?=\d)", "", t)          # "20,000" == "20000"


def said_not_found(answer: str) -> bool:
    a = normalize(answer)
    return any(p in a for p in NOT_FOUND_PATTERNS)


def is_error(answer: str) -> bool:
    return answer.startswith("ERROR")


def is_correct(answer: str, kind: str, keywords) -> bool:
    if is_error(answer):
        return False
    if kind == "out":
        return said_not_found(answer)
    if said_not_found(answer):
        return False
    a = normalize(answer)
    return all(normalize(k) in a for k in keywords)


def timed_ask(ask, question: str):
    """Ask once; retry one time after a pause if the API rate limit is hit."""
    for attempt in range(2):
        start = time.perf_counter()
        try:
            answer = ask(question)
            return str(answer), time.perf_counter() - start
        except Exception as e:
            msg = str(e)
            if attempt == 0 and ("429" in msg or "RESOURCE_EXHAUSTED" in msg):
                print("   rate limited, waiting 35s and retrying...")
                time.sleep(35)
                continue
            return f"ERROR: {msg[:120]}", time.perf_counter() - start
    return "ERROR", 0.0


def percentile(values, p):
    s = sorted(values)
    return s[int(round(p * (len(s) - 1)))] if s else 0.0


def best_threshold(rows):
    """Score cutoff that best separates in-document from off-topic questions (max F1 for 'in')."""
    pts = [(r["top_score"], r["kind"] == "in") for r in rows]
    if len({k for _, k in pts}) < 2:
        return None, None
    best = (None, -1.0)
    for th in sorted({s for s, _ in pts}):
        tp = sum(1 for s, k in pts if k and s >= th)
        fp = sum(1 for s, k in pts if not k and s >= th)
        fn = sum(1 for s, k in pts if k and s < th)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[1]:
            best = (th, f1)
    return best


def engine_metrics(rows, prefix):
    ins = [r for r in rows if r["kind"] == "in"]
    outs = [r for r in rows if r["kind"] == "out"]
    ok = lambda r: r[f"{prefix}_ok"]
    refused = lambda r: said_not_found(r[f"{prefix}_answer"]) and not is_error(r[f"{prefix}_answer"])
    tp = sum(1 for r in outs if refused(r))
    fp = sum(1 for r in ins if refused(r))
    times = [r[f"{prefix}_time"] for r in rows if not is_error(r[f"{prefix}_answer"])]
    return {
        "in_accuracy": f"{sum(ok(r) for r in ins)}/{len(ins)}",
        "out_refused": f"{tp}/{len(outs)}",
        "false_refusals": f"{fp}/{len(ins)}",
        "refusal_precision": f"{tp / (tp + fp):.0%}" if (tp + fp) else "-",
        "errors": sum(is_error(r[f"{prefix}_answer"]) for r in rows),
        "time_mean": f"{statistics.mean(times):.2f}s" if times else "-",
        "time_p95": f"{percentile(times, 0.95):.2f}s" if times else "-",
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", action="append", required=True, help="PDF to index (repeatable)")
    ap.add_argument("--questions", required=True, help="JSON or CSV file with your questions")
    ap.add_argument("--repeats", type=int, default=1, help="ask each question N times and average the time")
    ap.add_argument("--pause", type=float, default=4, help="seconds between model calls (rate limits)")
    ap.add_argument("--k", type=int, default=4, help="chunks retrieved for the similarity score")
    ap.add_argument("--out", default="compare_results.csv")
    a = ap.parse_args()

    pdfs = [Path(p) for p in a.pdf]
    for p in pdfs:
        if not p.exists():
            raise SystemExit(f"PDF not found: {p}")
    questions = load_questions(a.questions)

    print(f"Indexing {', '.join(p.name for p in pdfs)} ...")
    retriever, n_pages, n_chunks = build_retriever([LocalFile(p) for p in pdfs])
    print(f"Indexed {n_pages} pages into {n_chunks} chunks. {len(questions)} questions.\n")

    chain_ask = build_chain_engine(retriever)
    graph_ask = build_graph_engine(retriever)

    rows = []
    for i, q in enumerate(questions, start=1):
        question, kind = q["question"], q["kind"]
        print(f"[{i}/{len(questions)}] {question}")

        scored = retriever.vectorstore.similarity_search_with_relevance_scores(question, k=a.k)
        top_score = max((s for _, s in scored), default=0.0)
        gate_fires = top_score < SCORE_THRESHOLD
        hit = None
        if q["gold"]:
            hit = any(normalize(q["gold"]) in normalize(d.page_content) for d, _ in scored)

        chain_times, graph_times = [], []
        chain_answer = graph_answer = ""
        for _ in range(max(1, a.repeats)):
            chain_answer, t = timed_ask(chain_ask, question)
            chain_times.append(t)
            time.sleep(a.pause)
            graph_answer, t = timed_ask(graph_ask, question)
            graph_times.append(t)
            time.sleep(a.pause)

        rows.append({
            "question": question,
            "kind": kind,
            "top_score": round(top_score, 3),
            "gate_fires": gate_fires,
            "hit_at_k": hit,
            "chain_ok": is_correct(chain_answer, kind, q["keywords"]),
            "graph_ok": is_correct(graph_answer, kind, q["keywords"]),
            "chain_time": round(statistics.mean(chain_times), 2),
            "graph_time": round(statistics.mean(graph_times), 2),
            "chain_answer": chain_answer.replace("\n", " "),
            "graph_answer": graph_answer.replace("\n", " "),
        })

    # ---------------- results table ----------------
    width = 96
    print("\n" + "=" * width)
    print(f"{'#':<3}{'Type':<5}{'Score':<7}{'Gate':<6}{'LangChain':<12}{'LangGraph':<12}"
          f"{'Chain s':<9}{'Graph s':<9}Question")
    print("-" * width)
    for i, r in enumerate(rows, start=1):
        print(f"{i:<3}{r['kind']:<5}{r['top_score']:<7}{('yes' if r['gate_fires'] else 'no'):<6}"
              f"{('OK' if r['chain_ok'] else 'WRONG'):<12}{('OK' if r['graph_ok'] else 'WRONG'):<12}"
              f"{r['chain_time']:<9}{r['graph_time']:<9}{r['question'][:40]}")
    print("=" * width)

    ins = [r for r in rows if r["kind"] == "in"]
    outs = [r for r in rows if r["kind"] == "out"]
    chain, graph = engine_metrics(rows, "chain"), engine_metrics(rows, "graph")

    summary = [
        ("Questions answerable (in) / off-topic (out)", f"{len(ins)} / {len(outs)}", ""),
        ("In-document questions correct", chain["in_accuracy"], graph["in_accuracy"]),
        ("Off-topic questions refused", chain["out_refused"], graph["out_refused"]),
        ("False refusals (in-document question refused)", chain["false_refusals"], graph["false_refusals"]),
        ("Refusal precision", chain["refusal_precision"], graph["refusal_precision"]),
        ("Errors (API failures)", chain["errors"], graph["errors"]),
        ("Mean time per answer", chain["time_mean"], graph["time_mean"]),
        ("p95 time per answer", chain["time_p95"], graph["time_p95"]),
    ]
    print("\nSUMMARY                                          LangChain    LangGraph")
    for label, c, g in summary:
        print(f"  {label:<46} {str(c):<12} {g}")

    print("\nRETRIEVAL / GATE")
    print(f"  Score threshold used by LangGraph gate : {SCORE_THRESHOLD}")
    print(f"  Times the gate fired                   : {sum(r['gate_fires'] for r in rows)} of {len(rows)}")
    print(f"  Gate correctly blocked off-topic       : {sum(r['gate_fires'] for r in outs)} of {len(outs)}")
    wrong_blocks = sum(r["gate_fires"] for r in ins)
    print(f"  Gate wrongly blocked in-document       : {wrong_blocks} of {len(ins)}")
    margin_line = ""
    if ins and outs:
        margin = min(r["top_score"] for r in ins) - max(r["top_score"] for r in outs)
        margin_line = f"{margin:+.3f}"
        verdict = "a threshold can separate them" if margin > 0 else "scores overlap: no threshold separates them"
        print(f"  Score margin (lowest in - highest out) : {margin:+.3f}  ({verdict})")
        th, f1 = best_threshold(rows)
        if th is not None:
            print(f"  Best threshold on these questions      : {th:.3f} (F1 {f1:.2f}); current {SCORE_THRESHOLD}")
    hits = [r["hit_at_k"] for r in rows if r["hit_at_k"] is not None]
    if hits:
        print(f"  Hit@{a.k} (gold text retrieved)           : {sum(hits)}/{len(hits)}")

    out_file = Path(a.out)
    if not out_file.is_absolute():
        out_file = ROOT / out_file
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = ["| Metric | LangChain | LangGraph |", "|---|---|---|"]
    md += [f"| {l} | {c} | {g} |" for l, c, g in summary]
    md += ["", f"Gate threshold {SCORE_THRESHOLD}; gate fired {sum(r['gate_fires'] for r in rows)}/{len(rows)}; "
               f"wrongly blocked {wrong_blocks}/{len(ins)}; score margin {margin_line or '-'}"]
    (out_file.parent / "compare_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nFull answers saved to {out_file.name}; summary table in compare_summary.md")


if __name__ == "__main__":
    main()