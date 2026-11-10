# Phase 1 — Legal LM Pretraining

_TODO: add your Phase-1 notebooks here._

Four architectures were pretrained under the 50M–300M parameter cap. All share `d_model`
768, 12 heads, 2048 max sequence length, a 32k byte-level BPE vocabulary, RMSNorm, SwiGLU
and rotary position embeddings.

| id | File | Family | Layers | Params | Steps |
|----|------|--------|--------|--------|-------|
| A | `A_decoder.pt` | decoder-only, fused QKV | 12 | 109.53 M | 6600 |
| B | `B_encdec.pt` | encoder–decoder | 6 + 6 | 123.69 M | 6600 |
| C | `C_hybrid.pt` | GLR + attention every 3rd layer | 12 | 118.99 M | 1800 |
| D | `D_gqa.pt` | GQA sliding window (4 KV heads, window 256) | 12 | 100.09 M | 6600 |

Architecture definitions: [`../shared/`](../shared/).

_TODO: training objective, data mix across the four sources, token budget, decoding
configuration used for the Phase-1 submission, and the leaderboard score._
