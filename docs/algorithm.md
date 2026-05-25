# UNFOLD algorithm — code-level walkthrough

This document maps the algorithm box in the paper (Algorithm 1) to the
implementation in `unfold.py`. Line numbers in the algorithm column refer to
the algorithm pseudo-code in the paper, not source lines.

## Pipeline

```
EXTRACT       Line 1        unfold.parse_question()
              after Line 1  unfold._resolve_coreferences()  (optimization 1)
              after Line 1  unfold._llm_filter_paragraphs() (context pruning, when > 10 paragraphs)
              after Line 1  unfold.ground_bag()             (optimization 2)

CONSTRUCT     Lines 2-16    unfold.search()
                            - multi-anchor predicate-budget BFS
                            - KG short-circuit per (entity, predicate) when grounded
                            - per-paragraph triple extraction (`_guided_extract`) on KG miss

WALK          Lines 17-20   unfold._relevance_trim_single_anchor()
                            + unfold._select_compose_subgraph()
              before Line 19 unfold._link_uncovered_entities() (optimization 4)
              before Line 21 unfold._llm_pick_best_leaf()      (optimization 3)

POSE          Lines 21-24   unfold.compose()
                            - iterative independent sub-queries (yes/no per triple)
                            - final answer with resolved-facts context
```

## Top-level entry

```python
from unfold import unfold
result = unfold(question, context, client, model)
# result is a dict with keys: answer, subgraph, pruned_subgraph,
# ignored_subgraph, bag, t_search, t_compose, status, ...
```

## Hyperparameters

| Symbol | Default | Where in code |
| --- | --- | --- |
| `R_max`  | 2  | `UNFOLD_MAX_TRANS_HOPS` env var (`_trans_cap` in `search`) |
| `K_pred` | 30 | `common_top_n=30` argument default in `ground_bag` |
| `K_sim`  | 20 | `hard_cap_per_entity=20` argument default in `ground_bag` |
| `K_rank` | 1  | top-1 selection in `_llm_pick_best_leaf` |

Other environment variables: `UNFOLD_TEMPERATURE` (default 0.0),
`UNFOLD_COMPOSE_TEMPERATURE` (defaults to `UNFOLD_TEMPERATURE`),
`UNFOLD_NO_KG=1` (disable KG entirely), `UNFOLD_LINK_UNCOVERED=1`.
