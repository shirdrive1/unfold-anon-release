#!/usr/bin/env python3
"""
eval.py — Batch evaluator for UNFOLD (unfold.unfold).

Reads a benchmark file (JSON list or JSONL), runs unfold() per question
in parallel, computes EM/F1 against gold, writes a per-question .jsonl log,
and prints a summary at the end.

Resume: if --output already exists, qids/indices already present are skipped
(checkpointing per line).

Timeouts: per-question wall-clock cap. A timed-out question is logged with
status=timeout and counts as wrong (em=0, f1=0).

Designed to be format-agnostic across MuSiQue, HotpotQA, 2WikiMQA,
ConcurrentQA, and Drowzee — picks up the question and answer fields by name.
"""
import argparse
import collections
import json
import os
import re
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unfold import unfold


# --------------------------------------------------------------------------
# SQuAD-style EM/F1
# --------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _em(pred: str, gold: str) -> int:
    return int(_normalize(pred) == _normalize(gold))


def _f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = collections.Counter(p_toks) & collections.Counter(g_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p_toks)
    recall = n / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _score(pred: str, gold) -> (int, float):
    """Score against either a single gold string or a list of acceptable golds."""
    if isinstance(gold, list):
        return (max((_em(pred, g) for g in gold), default=0),
                max((_f1(pred, g) for g in gold), default=0.0))
    return _em(pred, gold), _f1(pred, gold)


# --------------------------------------------------------------------------
# Benchmark loader — handles both JSON-list and JSONL inputs, normalizes fields
# --------------------------------------------------------------------------

def _load_records(path: str):
    records = []
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        records = json.load(open(p))
    return records


def _normalize_record(rec: dict, idx: int) -> dict:
    """Extract canonical (qid, question, gold, context) regardless of source."""
    qid = (rec.get("qid") or rec.get("id") or rec.get("_id")
           or rec.get("question_id") or str(idx))
    question = rec.get("question") or rec.get("Question") or rec.get("q")
    gold = (rec.get("answer") if "answer" in rec
            else rec.get("gold_answer") or rec.get("answers") or rec.get("label"))
    context = rec.get("context") or rec.get("paragraphs") or None
    return {
        "qid": str(qid),
        "question": question,
        "gold": gold,
        "context": context,
        "raw": rec,
    }


# --------------------------------------------------------------------------
# Output / resume
# --------------------------------------------------------------------------

def _load_done_qids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["qid"])
            except Exception:
                continue
    return done


# --------------------------------------------------------------------------
# Per-question worker
# --------------------------------------------------------------------------

def _process_one(entry, client, model, timeout, skip_kg, skip_grounding, src):
    qid = entry["qid"]
    q = entry["question"]
    gold = entry["gold"]
    context = entry["context"]
    t0 = time.time()
    # NOTE: explicit shutdown(wait=False) on timeout. A `with`-block would call
    # shutdown(wait=True) by default, which blocks until the submitted task
    # finishes — but a thread stuck in a C-level network call can't be
    # interrupted by Python, so the `with` block would hang forever and the
    # outer eval would stall. We accept the leaked thread (it dies with the
    # process) in exchange for a real per-question timeout.
    inner = ThreadPoolExecutor(max_workers=1)
    try:
        try:
            fut = inner.submit(
                unfold, q, context,
                client=client, model=model,
                skip_kg=skip_kg, skip_grounding=skip_grounding, src=src,
            )
            result = fut.result(timeout=timeout)
        finally:
            inner.shutdown(wait=False)
    except Exception as e:
        return {
            "qid": qid, "question": q, "gold": gold,
            "answer": "", "em": 0, "f1": 0.0,
            "status": "timeout_or_error",
            "error": f"{type(e).__name__}: {str(e)[:160]}",
            "elapsed_s": round(time.time() - t0, 2),
        }
    pred = result.get("answer") or ""
    em, f1 = _score(pred, gold)
    return {
        "qid": qid,
        "question": q,
        "gold": gold,
        "answer": pred,
        "em": em,
        "f1": round(f1, 4),
        "status": result.get("status", "ok"),
        "subgraph_size": len(result.get("subgraph") or []),
        "pruned_subgraph_size": len(result.get("pruned_subgraph") or []),
        "ignored_size": len(result.get("ignored_subgraph") or []),
        "subgraph": result.get("subgraph") or [],
        "pruned_subgraph": result.get("pruned_subgraph") or [],
        "ignored_subgraph": result.get("ignored_subgraph") or [],
        "annotated_context": result.get("annotated_context"),
        "bag": result.get("bag"),
        "t_search": result.get("t_search", 0),
        "t_compose": result.get("t_compose", 0),
        "elapsed_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Benchmark file (.json list or .jsonl)")
    ap.add_argument("--output", required=True,
                    help="Per-question results .jsonl (created/appended)")
    ap.add_argument("--summary", default=None,
                    help="Summary stats JSON (default: <output>.summary.json)")
    ap.add_argument("--model", default="deepseek/deepseek-chat-v3-0324",
                    help="OpenRouter model for parse/ground/extract/compose")
    ap.add_argument("--src", default=None,
                    help="Benchmark name passed to unfold (e.g. drowzee_transitive)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N records")
    ap.add_argument("--workers", type=int, default=20,
                    help="Concurrent in-flight questions (default 20)")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-question wall-clock cap in seconds (default 300)")
    ap.add_argument("--skip-kg", action="store_true",
                    help="Skip KG short-circuit in CONSTRUCT (e.g. MuSiQue)")
    ap.add_argument("--skip-grounding", action="store_true",
                    help="Skip predicate-grounding LLM call")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="Print running EM/F1 every N questions (default 10)")
    args = ap.parse_args()

    summary_path = args.summary or (args.output + ".summary.json")

    # Build OpenRouter client.
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed — `pip install openai`", file=sys.stderr)
        sys.exit(2)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=api_key, timeout=60)

    raw = _load_records(args.input)
    entries = [_normalize_record(r, i) for i, r in enumerate(raw)]
    if args.limit:
        entries = entries[:args.limit]

    done_qids = _load_done_qids(args.output)
    todo = [e for e in entries if e["qid"] not in done_qids]
    print(f"Loaded {len(entries)} records, {len(done_qids)} already done, "
          f"{len(todo)} to run. workers={args.workers} timeout={args.timeout}s "
          f"model={args.model}")

    log_lock = threading.Lock()
    state_lock = threading.Lock()
    counter = {"done": len(done_qids),
               "em_sum": 0, "f1_sum": 0.0, "ok": 0, "fail": 0}
    total = len(entries)

    # Pre-load already-done results into the running totals so progress
    # reflects cumulative accuracy across resume.
    if done_qids:
        with open(args.output) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    counter["em_sum"] += rec.get("em", 0)
                    counter["f1_sum"] += rec.get("f1", 0.0)
                    if rec.get("status") == "ok":
                        counter["ok"] += 1
                    else:
                        counter["fail"] += 1
                except Exception:
                    continue

    log_f = open(args.output, "a")
    t_start = time.time()

    def _on_done(rec):
        with log_lock:
            log_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            log_f.flush()
        with state_lock:
            counter["done"] += 1
            counter["em_sum"] += rec.get("em", 0)
            counter["f1_sum"] += rec.get("f1", 0.0)
            if rec.get("status") == "ok":
                counter["ok"] += 1
            else:
                counter["fail"] += 1
            done = counter["done"]
            mean_em = counter["em_sum"] / max(done, 1)
            mean_f1 = counter["f1_sum"] / max(done, 1)
        if done % args.progress_every == 0 or done == total:
            elapsed = time.time() - t_start
            print(f"  [{done}/{total}] EM={mean_em:.3f} F1={mean_f1:.3f} "
                  f"ok={counter['ok']} fail={counter['fail']} "
                  f"({elapsed:.0f}s, {elapsed / max(done - len(done_qids), 1):.1f}s/q)")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_one, e, client, args.model,
                               args.timeout, args.skip_kg, args.skip_grounding,
                               args.src)
                   for e in todo]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:
                # _process_one swallows exceptions; this shouldn't happen
                rec = {"qid": "?", "em": 0, "f1": 0.0,
                       "status": "worker_error",
                       "error": f"{type(e).__name__}: {str(e)[:160]}"}
            _on_done(rec)

    log_f.close()

    # Final summary
    total_done = counter["done"]
    summary = {
        "input": args.input,
        "model": args.model,
        "src": args.src,
        "total": total,
        "done": total_done,
        "ok": counter["ok"],
        "fail": counter["fail"],
        "em_mean": counter["em_sum"] / max(total_done, 1),
        "f1_mean": counter["f1_sum"] / max(total_done, 1),
        "elapsed_s": round(time.time() - t_start, 2),
        "skip_kg": args.skip_kg,
        "skip_grounding": args.skip_grounding,
        "workers": args.workers,
        "timeout": args.timeout,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 60)
    print(f"DONE: {total_done}/{total} questions")
    print(f"  EM mean: {summary['em_mean']:.4f}")
    print(f"  F1 mean: {summary['f1_mean']:.4f}")
    print(f"  status: ok={summary['ok']} fail={summary['fail']}")
    print(f"  log:     {args.output}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
