# Datasets

The benchmarks are not included in this repo. Download them from the
authoritative sources below and place the resulting `.jsonl` files anywhere
convenient; pass the path to `eval.py --input <path>` and
`unfold_nowalk.py --input <path>`.

| Dataset | License | Source |
| --- | --- | --- |
| 2WikiMultiHopQA | Apache-2.0 | https://github.com/Alab-NII/2wikimultihop |
| HotpotQA (distractor) | CC BY-SA 4.0 | https://hotpotqa.github.io/ |
| ConcurrentQA | Apache-2.0 | https://github.com/facebookresearch/concurrentqa |
| MuSiQue (Answerable / Full) | CC BY 4.0 | https://github.com/StonyBrookNLP/musique |
| Drowzee | MIT | https://github.com/zhiyilll/Drowzee |
| Wikidata (KG used as source) | CC0 | https://www.wikidata.org/ |

For the multi-hop benchmarks we evaluate on the 500-query split used by
Relink (Huang et al., AAAI 2026); see the Relink repo for that exact
subset. For Drowzee we use a stratified 2{,}000-question sample (500 per
reasoning type).

All datasets are used consistently with their intended use as documented by
their authors.
