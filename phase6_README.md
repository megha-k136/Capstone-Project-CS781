# Phase 6 — Abstractive Summarization

The first generation task. **Best submission: BERTScore F1 0.8492** (tuned extractive);
best neural 0.8235 (B, LoRMA).

| Run | Backbone | Technique | BERTScore | val ROUGE-1 |
|---|---|---|---|---|
| 00_lead | — | extractive lead | 0.8226 | — |
| 02_B_lorma | B | LoRMA+ | 0.8235 | 0.3258 |
| 03_A_lora | A | LoRA | 0.8398 | 0.3585 |
| 04_C_houlsby | C | Houlsby adapters | 0.8278 | — |
| 05_D_prompt | D | prompt tuning | 0.8238 | — |
| 06_B_distill | B → 3-dec B | distillation | 0.8218 | — |
| 20_extractive | — | cue + centroid | **0.8492** | 0.4531 |

## Two input shapes

**B** summarises as an encoder–decoder: document in the encoder, summary decoded with
cross-attention. **A, C, D** have no cross-attention, so document and summary are
concatenated and the loss is masked to `-100` everywhere except the summary tokens —
without that mask the model spends its capacity predicting the judgment.

## Models stay causal

Phases 2–5 and 7 un-mask attention for classification. This phase does not: a decoder that
sees its own future trains on information it will not have at inference. `set_bidirectional`
appears nowhere in this notebook, and the test harness asserts all four architectures remain
causal by checking that changing the last token does not move an earlier position.

## Decoding

Greedy, with trigram repetition blocking and a minimum length before `<eos>` is permitted.
No architecture implements a key–value cache, so every step re-runs the prefix: B generates
at 0.77 s/doc because its encoder runs once and only the short summary is re-decoded, while
A runs at 4.43 s/doc re-running the whole document.

## The negative result

Model B reached loss 4.05 and produced fluent but ungrounded text — one sample reads *"The
assessee was a partner in the firm of a firm. The firm was a firm of the firm."* Model A
reached 3.23 and named real entities from the source. The hypothesis is that Phase-1
pretraining fed the **same sequence to both encoder and decoder**, so B's cross-attention
learned to attend to a copy of what the decoder already held and never learned to retrieve
from a distinct source.

## Files

- `CS781_Phase6_Summ.ipynb`, `build_p6.py`, `test_p6.py` (15 CPU runs)
