# Phase 3 — Rhetorical Role Prediction

Sentence-level classification over rhetorical roles. **Best submission: macro-F1 0.4945**
(ensemble); best single model 0.4780 (A, LoRMA).

## Representation

Sentences are packed whole into 1024-token windows and classified jointly, so each sentence
is represented by the mean of its own token hidden states after the backbone has processed
its neighbours:

```python
valid = (spos >= 0).unsqueeze(-1)
idx = spos.clamp(min=0).unsqueeze(-1).expand(-1, -1, H)
P = h.new_zeros(B, n_slots, H).scatter_add_(1, idx, h * valid)
C = h.new_zeros(B, n_slots, 1).scatter_add_(
    1, spos.clamp(min=0).unsqueeze(-1), valid.to(h.dtype))
return self.head(self.drop(P / C.clamp(min=1e-6)))
```

The scatter-add formulation allows a variable number of sentences per window without
padding to a fixed count.

## The Model B fix

In Phase 2, B collapsed to predicting the outside label everywhere — exactly 0.0000 across
all twelve entity types. The cause was that Phase-1 training reached B's encoder only
through cross-attention, so the encoder alone produced representations that had never
received direct supervision. Here `trunk_embeds` runs the **full** stack — encoder, encoder
norm, then decoder with cross-attention — and the collapse did not recur.

## Class weighting

`PrecedentOverruled` has three training instances. Uncapped inverse-frequency weighting gave
it ~37× the weight of `Fact`; capping and renormalising brought the ratio to 21:1. Note that
renormalisation rescales everything uniformly, so the cap value is not the final weight —
what matters is the ratio.

Development macro-F1 is capped near 0.846 because two roles have zero development support,
which is why leaderboard scores ran above local ones throughout this phase.

## Files

- `CS781_Phase3_RR.ipynb`, `build_p3.py`, `test_p3.py` (13 CPU runs)
