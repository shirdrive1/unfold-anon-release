# UNFOLD: Guiding LLM Question Answering through Fact Decomposition

This repository accompanies the paper *UNFOLD: Guiding LLM Question Answering
through Fact Decomposition*. UNFOLD is a training-free, model-agnostic
algorithm that wraps an LLM, builds a small query-specific knowledge graph at
inference time from the query's entities and predicates, and guides the LLM
through the assembled facts one at a time.

The repo ships two scripts:

- **`unfold.py`** — the full UNFOLD algorithm (forward walk + iterative
  compose). Run it via `eval.py` to evaluate on a benchmark file.
- **`unfold_nowalk.py`** — the lightweight `UNFOLD-nowalk` variant from the
  paper, which replaces the forward-walk KG construction with a single LLM
  call.

## Requirements

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key (used for all LLM calls).

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=<your-key>
```

## Quickstart

Run UNFOLD on the bundled cross-dataset sample (125 questions, 25 from each
of 2WikiMultiHopQA, HotpotQA, ConcurrentQA, MuSiQue-Ans, MuSiQue-Full):

```bash
python3 eval.py \
    --input examples/multihop_sample.jsonl \
    --output runs/multihop_unfold.jsonl \
    --model deepseek/deepseek-chat-v3-0324 \
    --workers 10
```

Run `UNFOLD-nowalk` on the same sample:

```bash
python3 unfold_nowalk.py \
    --input examples/multihop_sample.jsonl \
    --output runs/multihop_nowalk.jsonl \
    --model deepseek/deepseek-chat-v3-0324 \
    --workers 10
```

Each script writes a per-question JSONL log with the predicted answer,
extracted triples, and timing.

## Reproducing the paper

Run the two scripts above on each benchmark file (one per row of Tables 1–3
in the paper). Datasets must be downloaded separately; see
[`data/README.md`](data/README.md).

## Repository layout

```
unfold.py             # core UNFOLD algorithm (extract → walk → compose)
unfold_nowalk.py      # UNFOLD-nowalk variant (no walk; single-call KG)
eval.py               # batch evaluator (UNFOLD + EM/F1 + checkpointing)

examples/             # tiny sample inputs + expected outputs
docs/                 # algorithm walkthrough, optimizations
data/                 # dataset download instructions
```

## Citation

```bibtex
@inproceedings{unfold2026,
  title  = {{UNFOLD}: Guiding {LLM} Question Answering through Fact Decomposition},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
