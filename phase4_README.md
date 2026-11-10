# Phase 4 — Court Judgment Prediction

Binary outcome prediction over long judgments. **Best submission: macro-F1 0.7264**
(Model A, full fine-tuning).

| Run | Backbone | Technique | Score |
|---|---|---|---|
| 01_A_lorma | A | LoRMA+ | 0.4989 |
| 02_C_adapter | C | adapters | 0.5290 |
| 03_D_adalora | D | AdaLoRA | 0.5612 |
| 04_A_full | A | **full fine-tuning** | **0.7264** |
| 05_A_distill | A → 6-layer A | distillation | 0.3323 |
| 06_B_bitfit | B | BitFit | 0.5221 |

## The result worth reporting

With 32,305 training documents, full fine-tuning beat every parameter-efficient method by
roughly 0.17 macro-F1. This is the exact reverse of Phase 2, where 140 training documents
made low-rank regularisation the better bet. The prediction made before running it was
0.57–0.62; the direction was right and the magnitude badly wrong.

## Long-document encoding

Two-level pooling. Token mean within a chunk, then learned attention across chunks, with
chunks flattened across the batch so the backbone runs once:

```python
h = self.backbone.trunk(input_ids)                  # (M, T, H)
m = tok_mask.unsqueeze(-1).to(h.dtype)
cvec = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)    # (M, H)
counts = torch.bincount(owner, minlength=n_docs)
Cmax = int(counts.max().item())
slot = torch.cat([torch.arange(int(c), device=owner.device) for c in counts]) \
       if Cmax > 1 else torch.zeros_like(owner)
P = cvec.new_zeros(n_docs, Cmax, cvec.size(-1)); P[owner, slot] = cvec
```

Three representations were compared — last window, first window, and hierarchical chunking
from the end.

## Output format

Phase 4 changed the schema: a JSON **list** of `{"id": str, "pred": int}`, not the dict used
in Phases 2 and 3. The writer asserts it; a format mistake scores zero with no error.

## Files

- `CS781_Phase4_CJPE.ipynb`, `build_p4.py`, `test_p4.py` (17 CPU runs)
