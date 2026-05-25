# Examples

This folder contains a small cross-dataset sample of 125 questions (25 from
each of 2WikiMultiHopQA, HotpotQA, ConcurrentQA, MuSiQue-Ans, MuSiQue-Full)
chosen so that the full UNFOLD pipeline answers them correctly with raw
Exact Match (no judge rescue), on DeepSeek-v3.

## Files

- `multihop_sample.jsonl` — 125 questions in the standard format
  (`qid`, `question`, `context`, `answer`, `dataset`).

## Try UNFOLD on the sample

```bash
export OPENROUTER_API_KEY=<your-key>
python3 ../eval.py \
    --input multihop_sample.jsonl \
    --output multihop_unfold.jsonl \
    --model deepseek/deepseek-chat-v3-0324 \
    --workers 4
```

Expected: EM ≈ 1.0 across these 125 questions (selected as raw-correct positives).

## Try UNFOLD-nowalk on the same sample

```bash
python3 ../unfold_nowalk.py \
    --input multihop_sample.jsonl \
    --output multihop_nowalk.jsonl \
    --model deepseek/deepseek-chat-v3-0324 \
    --workers 4
```

## A note on the knowledge graph

UNFOLD can short-circuit fact retrieval through a local SQLite knowledge
graph at `./local_graph/knowledge_graph.db` (Wikidata-derived in the
paper). For the multi-hop benchmarks above, the context paragraphs alone
are sufficient and UNFOLD works without a KG, but the code still tries to
open the SQLite file on startup.

If you do not have a KG database, set the environment variable to skip KG
access entirely:

```bash
export UNFOLD_NO_KG=1
```

UNFOLD will then rely only on the context provided with each question.
For benchmarks that require KG access (e.g., Drowzee, whose questions
have no context), point `local_graph/knowledge_graph.db` at a Wikidata
dump materialized in the schema described in `unfold.py` (tables
`entities(name)` and `triples(subject, predicate, object)`).
