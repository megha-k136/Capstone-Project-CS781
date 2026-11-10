# Phase 5 — Legal Statute Identification

Multi-label classification over 100 statute identifiers. **Best submission: macro-F1 0.2818**
(probability-level ensemble); best single model 0.2755 (B, full fine-tuning).

| Run | Backbone | Technique | Macro-F1 |
|---|---|---|---|
| 01_A_LoRMA | A | LoRMA+ | 0.2604 |
| 02_B_FullFT | B | full fine-tuning | 0.2755 |
| 03_C_AdaLoRA | C | AdaLoRA | — |
| 04_D_Adapter | D | adapters | — |
| 05_KD_A6_student | A → 6-layer A | distillation | — |
| 06_ensemble | all | probability average | **0.2818** |

Dataset: ~42,750 training documents, ~10k dev, ~13k test, 100 distinct statute classes.

## Head and loss

Independent sigmoids over all 100 statutes — binary relevance, no label-correlation
modelling. `BCEWithLogitsLoss` with per-class positive weighting set to the
negative-to-positive ratio and clipped at `POS_WEIGHT_CAP = 10`.

Because macro-F1 averages over all 100 classes equally, a statute the model never predicts
contributes exactly zero, and the long tail dominates the metric.

## Long documents

Three 512-token chunks sampled **evenly across the document** with `np.linspace`, so the
beginning, middle and end are all represented, then pooled by learned attention across
chunks — the same two-level scheme as Phase 4.

## Thresholding

Two stages. A global threshold is swept over `linspace(0.10, 0.80, 15)` to maximise macro-F1,
then each statute with at least three dev positives gets its own threshold swept over
`linspace(0.05, 0.90, 18)`. Statutes with fewer than three positives keep the global value,
because a per-label sweep on one or two examples fits noise. If a document ends up with no
label above threshold, the single highest-probability statute is emitted, so no prediction
is ever empty.

## Two AMP bugs fixed here

Both produced `c10::Half` overflow before being corrected:

- The chunk-attention mask uses `-1e4` rather than `torch.finfo(dtype).min`, which overflows
  in fp16.
- The BCE loss is computed in fp32 **outside** the autocast block, with `logits.float()` and
  `y.float()`, rather than inside it.

## What limited the score

The V1 runs were badly under-trained and this is measurable rather than suspected.
`MAX_STEPS_PER_EXPERIMENT = 2000` at an effective batch of 8 processes 16,000 documents,
under 40% of a single epoch over the 42,750-document training set — and `MAX_TRAIN_DOCS` was
itself capped at 6,000. Every number in the table above comes from a model that saw a
fraction of the data.

A V2 configuration was prepared — full training set, 6000 steps, `POS_WEIGHT_CAP = 30` to
push the rare statutes, chunks taken from the document end, and a threshold floor dropped to
0.01 to catch weak signals on rare classes. It reached roughly step 2250 of 6000 on Model B
before the deadline forced an interrupt, and the attempt to rescue the partially trained
model failed: interrupting inside a nested CUDA call destroys the context for the whole
process, so the model survived in RAM but could no longer reach the GPU to run inference.
The V1 ensemble was submitted instead.

**The lesson worth carrying:** a mid-run rescue only works if the interrupt lands between
CUDA calls. Phases 6 and 7 avoided this by writing each submission zip immediately after
inference, so an interrupted run loses at most the run in progress rather than the whole
session.

## Corroboration of the Phase-4 finding

Full fine-tuning on Model B (0.2755) beat LoRMA on Model A (0.2604) at 42,750 training
documents, the same direction as Phase 4 at 32,305. Note the confound: the two runs differ
in architecture as well as technique, so this is suggestive rather than controlled.

## Output format

A JSON **list** of `{"id": str, "pred": [int, ...]}` with unique integer statute identifiers
and at least one per document.

## Files

- `CS781_Phase5_LSI.ipynb` — the notebook as run
