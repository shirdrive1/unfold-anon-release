# UNFOLD optimizations

The four optimizations in the paper (and the context-pruning step) run in
the execution chain shown in `docs/algorithm.md`. Each corresponds to one
prompt in the paper appendix.

## 1. Coreference annotation (first optimization)

Before the walk, annotates pronouns, role titles, acronyms, and partial
names in the context with their referent in the bag of entities. This lets
the LLM resolve paraphrased references when extracting triples.

Code: `unfold._resolve_coreferences()`. Prompt: see paper appendix
(`appendix:prompt-coref`).

## 2. Context pruning

When the context contains more than 10 paragraphs, the LLM filters down to
the top-5 most relevant paragraphs for the question.

Code: `unfold._llm_filter_paragraphs()`. Prompt: `appendix:prompt-filter`.

## 3. Predicate grounding (second optimization)

Maps each query predicate to one or more KG predicates so KG lookups can
match directly. The mapping is contextual: it depends on the entity (e.g.
"leader" maps to "head of state" for a country but "CEO" for a company).

Code: `unfold.ground_bag()`. Prompt: `appendix:prompt-ground`.

## 4. Per-paragraph triple extraction (in the walk, Lines 6 & 12)

Extracts factual triples from one paragraph for a specific
(entity, predicate) pair, used when the KG short-circuit misses.

Code: `unfold._guided_extract()`. Prompt: `appendix:prompt-subquery`.

## 5. Leaf ranking (third optimization)

Before posing the iterative sub-queries, when multiple objects share the
same (subject, predicate), the LLM picks the most plausible candidate
(top-1). Avoids posing redundant sub-queries.

Code: `unfold._llm_pick_best_leaf()`. Prompt: `appendix:prompt-leaf`.

## 6. Cover-uncovered (fourth optimization)

Before the bidirectional BFS, if any bag entity is missing from the
constructed subgraph, the LLM is asked to propose triples that link
uncovered entities to the existing subgraph.

Code: `unfold._link_uncovered_entities()`. Prompt: `appendix:prompt-cover`.
Off by default; set `UNFOLD_LINK_UNCOVERED=1` to enable.
