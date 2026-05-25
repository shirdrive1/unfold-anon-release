"""UNFOLD-nowalk — lightweight variant of UNFOLD.

UNFOLD-nowalk replaces the forward-walk KG construction (UNFOLD Algorithm,
Lines 2-16) with a single LLM call that emits all the triples linking the
query entities, given the question and the context. The rest of UNFOLD's
pipeline is unchanged:

  1. EXTRACT entities and predicates from the question (same as UNFOLD).
  2. NOWALK   one LLM call asks the LLM to emit triples that link the
              bag entities using only the bag predicates, drawing facts
              from the context.
              On empty output, fall back to zero-shot.
  3. COMPOSE  iterative independent yes/no sub-queries over the emitted
              triples + final answer (same as UNFOLD).

This is the variant referred to in the paper as `\\tool-nowalk`.

Public function:
  unfold_nowalk(question, context, client, model) -> (answer, triples)

CLI:
  python3 unfold_nowalk.py --input <benchmark.jsonl> --output out.jsonl \\
                          --model meta-llama/llama-3.3-70b-instruct \\
                          --src 2wiki --workers 20

Environment:
  OPENROUTER_API_KEY  required.
  UNFOLD_TEMPERATURE  optional, default 0.1 (structured-output stages).
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unfold


# ----------------------------------------------------------------------
# nowalk prompt (verbatim from the paper appendix).
# ----------------------------------------------------------------------
_NOWALK_PROMPT = (
    "You extract factual triples (subject, predicate, object) from a CONTEXT, "
    "given a BAG of entities and predicates extracted from a QUESTION.\n\n"
    "CORE RULE --- fact support: every triple must express a FACT directly stated in "
    "the CONTEXT. Subject and object must appear in the context (literally or via "
    "clear coreference), and the relation must be the one the context describes, in "
    "the same direction. Forbidden: inventing facts, transitively combining two "
    "separate facts into one triple, background knowledge.\n\n"
    "Predicate matching is SEMANTIC, not literal. Match a BAG predicate to the "
    "context by MEANING --- accept paraphrases, idiomatic phrasings, synonymous verbs.\n\n"
    "Emit each triple using the BAG's predicate label exactly as listed (do not "
    "substitute the context's wording). Each predicate is annotated with a generic "
    "type tag hinting at the kind of object the relation should land on; use it as "
    "a soft preference, not a hard veto.\n\n"
    "Coverage. Each emitted triple's subject or object must either be one of the "
    "BAG entities or an intermediate entity needed to chain BAG entities together. "
    "Direction follows the context.\n\n"
    "Chaining. The QUESTION may be multi-hop: a single BAG predicate may require "
    "traversing several intermediate entities in the context before reaching the "
    "answer. Starting from each BAG entity, emit the chain of triples (using BAG "
    "predicates throughout) that follows the context's links until you reach the "
    "answer or another BAG entity. Do not stop at one hop if the question requires "
    "more.\n\n"
    'Output strict JSON: {"triples": [[s, p, o], ...]}. Empty list if nothing qualifies.'
)


def _flatten_context(ctx) -> str:
    """Flatten a HotpotQA/2Wiki/MuSiQue-style context (list of [title, body])
    into a single block of text."""
    if not ctx:
        return ""
    if isinstance(ctx, str):
        return ctx
    parts = []
    for item in ctx:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, body = item[0], item[1]
            if isinstance(body, list):
                body = " ".join(str(s) for s in body)
            parts.append(f"[{title}]\n{body}")
        else:
            parts.append(str(item))
    return "\n\n".join(parts)


def _zs_call(question: str, client, model: str) -> str:
    """Closed-book zero-shot fallback (no context)."""
    try:
        resp = client.chat.completions.create(
            model=model, temperature=unfold._COMPOSE_TEMPERATURE, max_tokens=200,
            messages=[
                {"role": "system",
                 "content": "Answer concisely. Output only the answer."},
                {"role": "user", "content": question},
            ],
        )
        return unfold._extract_final_answer(
            (resp.choices[0].message.content or "").strip())
    except Exception as e:
        return f"[ZS ERROR: {e}]"


def _build_kg(bag, question: str, ctx, client, model: str):
    """Single LLM call: given the bag (entities + predicates extracted from
    the question) and the context, emit the triples linking those entities."""
    ents = [e.label for e in bag.entities]
    preds = [{"label": p.label, "type": p.type} for p in bag.predicates]
    bag_blob = (f"Entities: {json.dumps(ents)}\n"
                f"Predicates: {json.dumps(preds)}")
    user_msg = (f"BAG (from QUESTION):\n{bag_blob}\n\n"
                f"CONTEXT:\n{_flatten_context(ctx)}\n\n"
                f"QUESTION: {question}")
    try:
        resp = client.chat.completions.create(
            model=model, temperature=unfold._LLM_TEMPERATURE, max_tokens=1500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _NOWALK_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        obj = unfold._safe_loads(m.group(0)) or {}
        out = []
        for t in (obj.get("triples", []) or []):
            if isinstance(t, list) and len(t) >= 3:
                s, p, o = str(t[0]), str(t[1]), str(t[2])
                if s.strip() and p.strip() and o.strip():
                    out.append((s, p, o))
        return out
    except Exception:
        return []


def unfold_nowalk(question: str, context, client, model: str):
    """Run UNFOLD-nowalk on a single (question, context) pair.

    Returns (answer, triples, path) where:
      - answer  is the final string,
      - triples is the list of (s, p, o) the LLM emitted (empty on fallback),
      - path    is "nowalk" or "ZS" (the latter when no triples were emitted).
    """
    bag = unfold.parse_question(question, client, model)
    if bag is None:
        return _zs_call(question, client, model), [], "ZS"
    triples = _build_kg(bag, question, context, client, model)
    if not triples:
        return _zs_call(question, client, model), [], "ZS"
    try:
        ans = unfold.compose(question, triples, client, model, verbose=False)
    except Exception as e:
        ans = f"[COMPOSE ERROR: {e}]"
    return ans, triples, "nowalk"


# ----------------------------------------------------------------------
# Batch CLI (mirror eval.py).
# ----------------------------------------------------------------------
_FIELD_QUESTION = ("question", "input", "query")
_FIELD_ANSWER = ("answer", "answers", "gold", "label")
_FIELD_CONTEXT = ("context", "paragraphs", "supporting_facts")


def _pick(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _records(path):
    with open(path) as f:
        first = f.read(2)
    if first.startswith("["):
        return json.load(open(path))
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _process(entry, client, model):
    q = _pick(entry, _FIELD_QUESTION)
    ctx = _pick(entry, _FIELD_CONTEXT)
    qid = entry.get("qid") or entry.get("id")
    gold = _pick(entry, _FIELD_ANSWER)
    t0 = time.time()
    try:
        ans, triples, path = unfold_nowalk(q, ctx, client, model)
        status = "ok"
    except Exception as e:
        ans, triples, path, status = "", [], "error", f"error: {e}"
    return {
        "qid": qid, "question": q, "gold": gold,
        "answer": ans, "n_triples": len(triples), "triples": triples,
        "path": path, "status": status, "elapsed_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="JSON or JSONL benchmark file")
    ap.add_argument("--output", required=True, help="Per-question JSONL output")
    ap.add_argument("--model", default="deepseek/deepseek-chat-v3-0324",
                    help="OpenRouter model name")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N records")
    args = ap.parse_args()

    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=api_key, timeout=120)

    entries = _records(args.input)
    if args.limit:
        entries = entries[: args.limit]
    print(f"[unfold-nowalk] {len(entries)} questions, model={args.model}")

    out_f = open(args.output, "w")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_process, e, client, args.model) for e in entries]
        for fut in as_completed(futs):
            rec = fut.result()
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{len(entries)}]", flush=True)
    out_f.close()
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
