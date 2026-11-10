# Phase 7 — Prior Case Retrieval

Retrieve exactly 7 relevant prior cases per query. **Best submission: micro-F1@7 0.4525,
rank 18.**

| Run | Backbone | Technique | Score |
|---|---|---|---|
| BM25, full-document query | — | lexical | 0.1164 |
| 02_A_lorma | A | LoRMA+ | 0.1494 |
| 03_B_full | B | full fine-tuning | 0.1217 |
| 04_C_ia3 | C | IA³ | 0.1246 |
| 05_D_prompt | D | prompt tuning | 0.1282 |
| 06_A_distill | A → 6-layer A | distillation | 0.1488 |
| TF-IDF, trigram features | — | lexical | 0.4253 |
| + hubness centring | — | lexical | 0.4501 |
| + CSLS (k=5) | — | lexical | **0.4525** |

## Architecture

A shared-tower bi-encoder: one encoder produces vectors for both queries and candidates by
masked mean pooling, a trained 256-d projection and L2 normalisation, so relevance is a
cosine. One tower rather than two, because 817 training queries cannot support twice the
trainable parameters.

Multi-positive InfoNCE at temperature 0.05:

```python
sim = (qv @ cv.t()) / cfg.TEMP
num = sim.masked_fill(~pos, torch.finfo(sim.dtype).min)
loss = -(torch.logsumexp(num, -1) - torch.logsumexp(sim, -1)).mean()
```

Written as a difference of two log-sum-exp terms it is numerically stable, handles a query
with several positives without reweighting, and makes it impossible for a candidate that is
genuinely relevant to be counted against the query as an in-batch negative.

## No threshold to tune

Because exactly 7 are always returned, micro-precision@7 is `hits / (7·n)` and
micro-recall@7 is `hits / total_gold` — both monotone in the hit count. Maximising
micro-F1@7 *is* maximising hits in the top 7.

## What actually worked

Every neural variant lost to a lexical one by nearly three to one. The decisive correction:
**BM25 assumes a short query**, and the query here is an entire judgment, so term-presence
weighting was being spread across thousands of boilerplate terms. TF-IDF cosine similarity
is the right model for document-to-document comparison; with phrase features it reached
0.4253.

**Hubness centring** then added 0.025. A few documents sit near the collection centroid and
score highly against every query; subtracting each candidate's mean similarity across all
queries removes that bias and raised recall@100 from 0.831 to 0.863. **CSLS** — the same
correction computed over a candidate's *k* nearest queries rather than all of them — added
a further 0.002.

One tuning note worth recording: the CSLS neighbourhood size scales with pool size.
Development picked k=5 of 1023 candidates (0.49%); training picked k=25 of 4320 (0.58%).
Two independent splits agreeing on a ratio, not a constant.

## Failed approaches

- **Score propagation over the candidate graph**, on the reasoning that cases cited together
  are related, hurt monotonically: 0.4636 → 0.4591 at γ=0.2 → 0.4503 at γ=0.6. Topical
  similarity and precedent relevance are different relations.
- **Finer hyperparameter sweeps on the 118-query dev set** overfit: one variant scored
  higher on dev and lower on the board (0.4495 vs 0.4525).

## Output format

A JSON **dict** `{query_id: [7 unique candidate ids]}` — not the list of objects used in
Phases 4 to 6. The writer asserts exactly seven unique identifiers, all drawn from the test
pool, for every test query.

## Files

- `CS781_Phase7_PCR.ipynb`, `build_p7.py`, `test_p7.py` (17 CPU runs)
