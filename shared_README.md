# Shared architecture definitions

`arch_components.py` — RMSNorm, SwiGLU, RoPE, CausalSelfAttention, GeneralizedAttention,
EncoderBlock, DecoderBlock, GatedLinearRecurrence, GQASlidingWindowAttention, GenericBlock.

`arch_models.py` — GPTModel (A), EncoderDecoderModel (B), GenericLM (C and D), the mixer
factories, `load_phase1()` and `trunk_blocks()`.

These files are **reproduced verbatim** inside every phase notebook rather than imported,
so each notebook is self-contained on Kaggle and every checkpoint loads with `strict=True`.
The builder scripts read them from here at generation time, so this directory is the single
source of truth: edit here, then re-run the relevant `build_*.py`.

## Bidirectional patching

Phases 2–5 and 7 un-mask causal attention, because classification and retrieval read a whole
span at once. The procedure is architecture-specific:

- **A, D** — all 12 mixers switched.
- **B** — the encoder is natively bidirectional; its 6 decoder self-attention blocks switch.
- **C** — only its 4 attention layers can be patched. The 8 gated-recurrence layers encode
  direction in the recurrence operator itself, not in a mask, so C enters every
  discriminative phase with two-thirds of its depth still strictly causal. This is a
  property of the architecture, not a defect of the patch.

**Phase 6 does none of this.** A generator that sees its own future trains on information it
will not have at inference.
