# Phase 2 — Legal NER

Token classification over 12 entity types with exact character-offset reconstruction.
**Best submission: macro-F1 0.4579** (three-architecture ensemble).

| Run | Backbone | Technique | Score |
|---|---|---|---|
| single backbone, 512-token windows | A | LoRA | 0.1240 |
| 1024-token windows + 21× oversampling | C | LoRA r=32 | 0.3973 |
| refit on train+dev | C | LoRA r=32 | 0.4168 |
| three-architecture ensemble | A, C, D | LoRA | **0.4579** |

## The offset problem

Strict span matching means an offset wrong by one character scores zero. The tokenizer's
`offset_mapping` is never trusted. Text is atomised first:

```python
ATOM_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
def atomise(text):
    return [(m.group(0), m.start(), m.end()) for m in ATOM_RE.finditer(text)]
```

Offsets come straight off the source string, and predicted spans are rebuilt by copying the
first atom's start and the last atom's end. No arithmetic is ever done on an offset.
Round-tripping every gold span recovers 99.27% exactly on real data, 100% on synthetic.

## What moved the score

Six of the twelve types denote party roles and live in the preamble. At 512 tokens the model
saw them a handful of times across the whole training set and predicted them at near-zero
rate. Widening to 1024 tokens and oversampling rare-label-bearing windows 21× was the single
largest gain. Ensembling three architectures by averaging per-token log-probabilities added
5.4 points, and a swept bias on the outside-label logit added more at no training cost.

## Files

- `CS781_Phase2_LNER_FINAL.ipynb` — the notebook
- `build_final.py` — generates it
- `test_final.py` — 13 CPU runs; asserts offset round-trip and zip schema
