import json

cells = []

def _lines(src):
    return (src.strip("\n") + "\n").splitlines(keepends=True)

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(src)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(src)})

ARCH_COMPONENTS = open("arch_components.py").read().rstrip()
ARCH_MODELS     = open("arch_models.py").read().rstrip()

# =============================================================================
md(r"""
# CS-781 Capstone — Phase 4: Court Judgment Prediction

Binary outcome prediction over **long** legal judgments, on all four Phase-1 models.

## The six submissions

| # | Phase-1 model | Technique | Why it is distinct |
|---|---------------|-----------|--------------------|
| 1 | **A** decoder-only | **LoRMA+ (compulsory)** | multiplicative low-rank reparameterisation |
| 2 | **C** GLR hybrid | Adapters | trained bottleneck modules inserted into the residual stream |
| 3 | **D** GQA windowed | AdaLoRA | SVD parameterisation with adaptive rank budget |
| 4 | **B** encoder-decoder | Full fine-tuning | every weight updated |
| 5 | **A** | BitFit | biases and norm scales only |
| 6 | **A → half-depth A** | **Distillation (compulsory)** | teacher-student compression |

Phase 4's permitted list (§10.2.1) differs again from Phase 3's: **LoRA, IA³ and prompt
tuning are not on it**, while full fine-tuning, BitFit and adapters return. Every technique
above is drawn from the Phase 4 list.

## Two things that are new this phase

**The output format changed.** Phases 2 and 3 wanted a dict keyed by document. Phase 4 wants
a JSON **list of objects**, each `{"id": ..., "pred": 0 or 1}` with `pred` an integer. The
writer asserts all of it — a format mistake here scores zero with no error message.

**Document length is the real problem.** Judgments run to thousands of tokens. Section 7
compares three representations — the last window, the first window, and hierarchical chunking
with learned attention pooling — on a short budget, and you carry the winner into the six
runs. On the corpus this task is drawn from, the *end* of a judgment is well known to predict
the outcome far better than the beginning, so expect `last` to be strong.

## Run order
Cells 1-9, then the smoke test; upload its zip immediately. Then section 7's three-way
comparison (about ten minutes), set `cfg.REPR` to the winner, and run the six.
""")

# =============================================================================
md(r"""
## 1. Setup and config
""")

code(r"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, json, math, time, random, zipfile, gc, glob
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
print("torch", torch.__version__, "| device:", DEVICE)
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
set_seed()

# ---- pull Phase-4 data (needs the HF_TOKEN Kaggle secret) ---------------------------
if not glob.glob("/kaggle/**/train.json", recursive=True):
    from huggingface_hub import snapshot_download
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        _tok = None
    snapshot_download(repo_id="Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                      allow_patterns="Phase_4_CJPE/*", local_dir="/kaggle/working/p4",
                      token=_tok)


class CFG:
    TOK_GLOB   = "/kaggle/input/**/bpe32k.json"
    CKPT_GLOBS = {"A": "/kaggle/input/**/A_decoder.pt",
                  "B": "/kaggle/input/**/B_encdec.pt",
                  "C": "/kaggle/input/**/C_hybrid.pt",
                  "D": "/kaggle/input/**/D_gqa.pt"}
    P4_GLOB    = "/kaggle/**/Phase_4_CJPE/**/train.json"

    DIM, N_HEADS, MAX_SEQ = 768, 12, 2048

    # ---- long-document representation ---------------------------------------
    REPR          = "last"   # "last" | "first" | "hier"   (section 7 picks this)
    CHUNK_LEN     = 1024     # tokens per chunk
    MAX_CHUNKS    = 4        # hier only: chunks kept per document
    HIER_FROM_END = True     # hier only: keep the LAST MAX_CHUNKS chunks
    CHUNK_POOL    = "attn"   # hier only: "attn" | "mean"
    BIDIRECTIONAL = True

    # ---- training -----------------------------------------------------------
    EPOCHS        = 4
    BATCH_SIZE    = 2
    GRAD_ACCUM    = 8
    LR_BACKBONE   = 2e-5     # full fine-tuning
    LR_HEAD       = 1e-3
    LR_PEFT       = 1e-3     # lorma / adalora / bitfit / adapters
    WARMUP_FRAC   = 0.06
    WEIGHT_DECAY  = 0.01
    MAX_GRAD_NORM = 1.0
    CLASS_WEIGHTS = True     # inverse-frequency, capped
    CLASS_WEIGHT_CAP = 4.0
    AMP           = True

    # ---- adaptation ---------------------------------------------------------
    LORMA_R, LORMA_ALPHA = 32, 64
    ADALORA_R, ADALORA_ALPHA = 32, 64
    ADALORA_TARGET_R = 16
    ADALORA_ORTH = 0.1
    ADAPTER_DIM = 64         # bottleneck width
    PARTIAL_TOP_K = 3

    KD_T, KD_ALPHA = 2.0, 0.5

    OUT_DIR = "/kaggle/working/submissions"

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"{what} not found ({pattern}) — attach the Kaggle dataset"
    return Path(hits[0])

TOK_PATH = find_one(cfg.TOK_GLOB, "tokenizer bpe32k.json")
P4_TRAIN = find_one(cfg.P4_GLOB, "Phase-4 train.json")
CKPTS    = {k: find_one(v, f"Phase-1 model {k}") for k, v in cfg.CKPT_GLOBS.items()}
print("tokenizer :", TOK_PATH)
print("phase-4   :", P4_TRAIN.parent)
for k, v in CKPTS.items():
    print(f"model {k}   : {v}  ({v.stat().st_size/1e6:.0f} MB)")
""")

# =============================================================================
md(r"""
## 2. Data

Examples are `{id, text, label}`. The loader is tolerant of a list of objects or a dict keyed
by id, and of the usual spellings for the label field.

The length histogram it prints is the thing to read: it tells you how much of each judgment
the chunking actually reaches, and therefore how much the three representations in section 7
can possibly differ.
""")

code(r"""
def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_cjpe(path, with_label=True):
    raw = _read_json(path)
    items = raw if isinstance(raw, list) else [
        (v if isinstance(v, dict) else {"text": v}) | {"id": k} for k, v in raw.items()]
    out = {}
    for d in items:
        did = str(d.get("id", d.get("case_id", d.get("doc_id", len(out)))))
        txt = d.get("text", d.get("document", d.get("case_text", "")))
        if isinstance(txt, list):
            txt = " ".join(txt)
        lab = d.get("label", d.get("pred", d.get("outcome", d.get("judgment"))))
        rec = {"text": txt}
        if with_label and lab is not None:
            rec["label"] = int(lab)
        out[did] = rec
    return out

def _p(name):
    hits = sorted(glob.glob(str(P4_TRAIN.parent.parent / "**" / name), recursive=True))
    assert hits, f"{name} not found"
    return hits[0]

train_docs = load_cjpe(_p("train.json"))
dev_docs   = load_cjpe(_p("dev.json"))
test_docs  = load_cjpe(_p("test.json"), with_label=False)

assert all("label" in d for d in train_docs.values()), "train is missing labels"
assert all("label" in d for d in dev_docs.values()), "dev is missing labels"
LABELSET = sorted({d["label"] for d in train_docs.values()})
assert LABELSET == [0, 1], f"expected binary 0/1, found {LABELSET}"

ytr = Counter(d["label"] for d in train_docs.values())
ydv = Counter(d["label"] for d in dev_docs.values())
print(f"docs  train {len(train_docs)} | dev {len(dev_docs)} | test {len(test_docs)}")
print(f"train labels {dict(ytr)}  ({100*ytr[1]/max(len(train_docs),1):.1f}% positive)")
print(f"dev   labels {dict(ydv)}  ({100*ydv[1]/max(len(dev_docs),1):.1f}% positive)")

wc = np.array([len(d["text"].split()) for d in train_docs.values()])
print(f"\ndocument length (words): median {int(np.median(wc))} | "
      f"mean {int(wc.mean())} | p90 {int(np.percentile(wc, 90))} | max {int(wc.max())}")
for q in (50, 75, 90, 95, 99):
    print(f"  p{q:<3} {int(np.percentile(wc, q)):>8}")
""")

# =============================================================================
md(r"""
## 3. Tokenising and the three representations

A document becomes a **list of chunks**, and the three strategies differ only in which chunks
survive:

- **`last`** — one chunk, the final `CHUNK_LEN` tokens. The judgment's conclusion.
- **`first`** — one chunk, the opening `CHUNK_LEN` tokens. Facts and procedural history.
- **`hier`** — up to `MAX_CHUNKS` chunks taken from the end, each encoded separately and then
  pooled together by a learned attention over chunks.

Chunks are flattened across the batch into one tensor so the backbone runs once over all of
them, then gathered back per document. That keeps the hierarchical path from costing a
separate forward pass per chunk.
""")

code(r"""
from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(TOK_PATH))
VOCAB_SIZE = tok.get_vocab_size()
PAD_ID = int(tok.token_to_id("<|pad|>") or 0)
print(f"vocab {VOCAB_SIZE:,} | pad {PAD_ID}")

_ENC_CACHE = {}

def doc_chunks(did, text, repr_mode=None, store=None):
    '''-> list of token-id lists, per the active representation.'''
    mode = repr_mode or cfg.REPR
    key = (did, cfg.CHUNK_LEN)
    ids = (store or _ENC_CACHE).get(key)
    if ids is None:
        ids = tok.encode(text).ids or [PAD_ID]
        (store if store is not None else _ENC_CACHE)[key] = ids
    L = cfg.CHUNK_LEN
    if mode == "last":
        return [ids[-L:]]
    if mode == "first":
        return [ids[:L]]
    if mode == "hier":
        ch = [ids[i:i + L] for i in range(0, max(len(ids), 1), L)] or [[PAD_ID]]
        return ch[-cfg.MAX_CHUNKS:] if cfg.HIER_FROM_END else ch[:cfg.MAX_CHUNKS]
    raise ValueError(mode)


class CJPEDocs(Dataset):
    def __init__(self, docs, with_label=True, repr_mode=None):
        self.ids = list(docs)
        self.docs, self.with_label, self.mode = docs, with_label, repr_mode
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        did = self.ids[i]
        d = self.docs[did]
        return {"did": did, "chunks": doc_chunks(did, d["text"], self.mode),
                "y": int(d["label"]) if (self.with_label and "label" in d) else -100}


def collate(batch):
    flat, owner = [], []
    for bi, b in enumerate(batch):
        for c in b["chunks"]:
            flat.append(c); owner.append(bi)
    T = max(len(c) for c in flat)
    ids  = torch.full((len(flat), T), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(flat), T), dtype=torch.bool)
    for i, c in enumerate(flat):
        ids[i, :len(c)] = torch.tensor(c, dtype=torch.long)
        mask[i, :len(c)] = True
    return {"input_ids": ids, "tok_mask": mask,
            "owner": torch.tensor(owner, dtype=torch.long),
            "n_docs": len(batch),
            "labels": torch.tensor([b["y"] for b in batch], dtype=torch.long),
            "dids": [b["did"] for b in batch]}


def make_sets(repr_mode=None):
    return (CJPEDocs(train_docs, repr_mode=repr_mode),
            CJPEDocs(dev_docs,   repr_mode=repr_mode),
            CJPEDocs(test_docs,  with_label=False, repr_mode=repr_mode))

ds_train, ds_dev, ds_test = make_sets()
_nc = [len(ds_train[i]["chunks"]) for i in range(min(len(ds_train), 200))]
print(f"repr={cfg.REPR} | chunks/doc over a 200-doc sample: "
      f"mean {np.mean(_nc):.2f} max {max(_nc)}")
""")

# =============================================================================
md(r"""
## 4. Phase-1 architecture components

Reproduced from your Phase-1 notebooks so every checkpoint loads with `strict=True`. The only
additions are the `causal` / `bidirectional` flags, which default to the trained behaviour.

**Why un-mask.** Classification reads the whole chunk at once rather than generating, so there
is no reason to hide the right-hand context. Model B's encoder was already trained
bidirectionally; A and D are patched; C can only be patched in its four attention layers,
since its eight gated-recurrence layers have direction built into the operator.
""")

code(ARCH_COMPONENTS)

# =============================================================================
md(r"""
## 5. The four models
""")

code(ARCH_MODELS)

# =============================================================================
md(r"""
## 6. The classifier

Two levels of pooling. Within a chunk, hidden states are mean-pooled over the real (non-pad)
tokens. Across chunks, documents with more than one chunk are pooled by a small learned
attention: each chunk gets a score, the scores are softmaxed within the document, and the
chunk vectors are combined by those weights. With `last` or `first` there is one chunk and
the cross-chunk step is a no-op, so all three representations share one code path.
""")

code(r"""
def set_bidirectional(model, on=True):
    n = 0
    for mod in model.modules():
        if isinstance(mod, CausalSelfAttention):
            mod.causal = not on; n += 1
        elif isinstance(mod, GQASlidingWindowAttention):
            mod.bidirectional = on; n += 1
        elif isinstance(mod, GeneralizedAttention) and not mod.cross:
            if mod.causal:
                mod.causal = not on; n += 1
    return n


class CJPEModel(nn.Module):
    def __init__(self, backbone, arch, hidden, dropout=0.1):
        super().__init__()
        self.backbone, self.arch, self.hidden = backbone, arch, hidden
        self.chunk_attn = nn.Sequential(nn.Linear(hidden, hidden // 4), nn.Tanh(),
                                        nn.Linear(hidden // 4, 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2))

    def forward(self, input_ids, tok_mask, owner, n_docs):
        h = self.backbone.trunk(input_ids)                       # (M, T, H)
        m = tok_mask.unsqueeze(-1).to(h.dtype)
        cvec = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)         # (M, H) per chunk

        # group chunks by document, pad to (B, Cmax, H)
        counts = torch.bincount(owner, minlength=n_docs)
        Cmax = int(counts.max().item())
        slot = torch.cat([torch.arange(int(c), device=owner.device) for c in counts]) \
               if Cmax > 1 else torch.zeros_like(owner)
        P = cvec.new_zeros(n_docs, Cmax, cvec.size(-1))
        P[owner, slot] = cvec
        valid = torch.zeros(n_docs, Cmax, dtype=torch.bool, device=cvec.device)
        valid[owner, slot] = True

        if Cmax == 1:
            doc = P[:, 0]
        elif cfg.CHUNK_POOL == "mean":
            doc = (P * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        else:
            s = self.chunk_attn(P).squeeze(-1)                   # (B, Cmax)
            s = s.masked_fill(~valid, torch.finfo(s.dtype).min)
            doc = (P * s.softmax(-1).unsqueeze(-1)).sum(1)
        return self.head(self.drop(doc))                         # (B, 2)


def trunk_blocks(m, arch):
    return m.decoder_blocks if arch == "B" else m.blocks


def build_cjpe(arch, bidirectional=None, keep_layers=None, pretrained=True, verbose=True):
    bb = load_phase1(arch, pretrained=pretrained)
    if keep_layers is not None:
        blocks = trunk_blocks(bb, arch)
        keep = np.linspace(0, len(blocks) - 1, keep_layers).round().astype(int).tolist()
        new = nn.ModuleList([blocks[i] for i in keep])
        if arch == "B": bb.decoder_blocks = new
        else:           bb.blocks = new
        if verbose: print(f"  student trunk {len(blocks)} -> {keep_layers} blocks {keep}")
    bb.head = nn.Identity()
    bi = cfg.BIDIRECTIONAL if bidirectional is None else bidirectional
    n = set_bidirectional(bb, bi)
    m = CJPEModel(bb, arch, cfg.DIM).to(DEVICE)
    if verbose:
        tot = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
        print(f"  [{arch}] {ARCH_INFO[arch]} | {tot/1e6:.1f}M params | "
              f"bidirectional={bi} ({n} mixers switched)")
    return m
""")

# =============================================================================
md(r"""
## 7. Adaptation techniques

All six come from §10.2.1. LoRA, IA³ and prompt tuning are **not** on Phase 4's list and are
deliberately absent.

### LoRMA — multiplicative low-rank *(compulsory)*
LoRMA (Exploration-Lab, ACL Findings 2025) replaces LoRA's additive update with a
multiplicative one, transforming the weight rather than offsetting it. A bare product would
cap the adapted weight at rank `r` and has no zero initialisation, so **LoRMA+** adds an
identity: `W = (I + (alpha/r)·BA)·W`, which is exactly `W` at init and stays near full rank.
The implementation never materialises `BA` — it computes `y = Wx`, then `y + (yAᵀ)Bᵀ`.

### AdaLoRA — adaptive rank budget
The update is written in singular-value form `P·diag(λ)·Q` with an orthogonality penalty.
Each singular value is scored by sensitivity, and partway through training the weakest are
masked to hit a budget, so layers that need capacity keep it and layers that don't give it up.

### Adapters — bottleneck modules
Small trained modules inserted into the residual stream after the mixer and after the
feed-forward block: down-project to `ADAPTER_DIM`, nonlinearity, up-project, add back. The
up-projection is zero-initialised so each adapter starts as the identity. Unlike the low-rank
methods this adds *depth* rather than reparameterising existing weights — a genuinely
different mechanism, and the reason it belongs on the list beside them.

### Full fine-tuning
Every weight trainable. The reference point the brief asks the efficient methods to be
measured against.

### BitFit
Biases and norm scales only — a few hundred thousand parameters. It shifts and rescales
activations rather than changing what the weights compute.

### Knowledge distillation *(compulsory, cell 16)*
Teacher-to-student on softened logits, covered at its own cell.
""")

code(r"""
class LoRMALinear(nn.Module):
    # W = (I + (alpha/r) B A) W0
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        d = base.out_features
        self.A = nn.Parameter(torch.zeros(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))      # zero -> starts exactly at W0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
    def forward(self, x):
        y = self.base(x)
        return y + ((y @ self.A.t()) @ self.B.t()) * self.scale


class AdaLoRALinear(nn.Module):
    # dW = P diag(lam) Q, orthogonality penalty + importance-based rank masking
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.Q = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.P = nn.Parameter(torch.randn(base.out_features, r) * 0.02)
        self.lam = nn.Parameter(torch.zeros(r))
        self.register_buffer("mask", torch.ones(r))
        self.register_buffer("imp", torch.zeros(r))
        self.scale, self.r = alpha / r, r
    def forward(self, x):
        z = (x @ self.Q.t()) * (self.lam * self.mask)
        return self.base(x) + (z @ self.P.t()) * self.scale
    def orth_penalty(self):
        I = torch.eye(self.r, device=self.P.device, dtype=self.P.dtype)
        return ((self.P.t() @ self.P - I).pow(2).mean()
                + (self.Q @ self.Q.t() - I).pow(2).mean())
    @torch.no_grad()
    def update_importance(self, beta=0.85):
        if self.lam.grad is not None:
            self.imp.mul_(beta).add_((1 - beta) * (self.lam * self.lam.grad).abs())


def adalora_housekeeping(model, step, total):
    mods = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    if not mods:
        return
    for m in mods:
        m.update_importance()
    if step == max(1, total // 2):
        allimp = torch.cat([m.imp for m in mods])
        budget = int(cfg.ADALORA_TARGET_R * len(mods))
        if budget < allimp.numel():
            thr = torch.topk(allimp, budget).values.min()
            kept = sum(int(m.mask.copy_((m.imp >= thr).float()).sum().item()) for m in mods)
            print(f"    [adalora] pruned to {kept}/{allimp.numel()} singular values")


class AdapterWrap(nn.Module):
    '''Bottleneck adapter on the output of a frozen Linear, zero-init so it starts as
    the identity: y + Up(GELU(Down(y))).'''
    def __init__(self, base, bottleneck):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        d = base.out_features
        self.down = nn.Linear(d, bottleneck)
        self.up   = nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, std=1e-2); nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight);              nn.init.zeros_(self.up.bias)
    def forward(self, x):
        y = self.base(x)
        return y + self.up(F.gelu(self.down(y)))


MIXERS = (CausalSelfAttention, GeneralizedAttention,
          GQASlidingWindowAttention, GatedLinearRecurrence)

def mixer_linears(model):
    '''Every nn.Linear that is a direct child of a token mixer. Selected by module type,
    never by name -- a keyword match would drag the feed-forward weights in too.'''
    for mod in model.modules():
        if isinstance(mod, MIXERS):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    yield mod, cname, child


def apply_adaptation(model, method):
    bb, arch = model.backbone, model.arch
    for p in bb.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    for p in model.chunk_attn.parameters(): p.requires_grad = True
    blocks = trunk_blocks(bb, arch)
    final_norm = bb.norm_e if arch == "B" else bb.norm_f

    if method == "head_only":
        pass

    elif method == "full":
        for p in bb.tok_emb.parameters(): p.requires_grad = True
        for p in blocks.parameters():     p.requires_grad = True
        for p in final_norm.parameters(): p.requires_grad = True
        if arch == "B":
            for p in bb.decoder_blocks.parameters(): p.requires_grad = True
            for p in bb.norm_f.parameters():         p.requires_grad = True

    elif method == "partial":
        for blk in list(blocks)[-cfg.PARTIAL_TOP_K:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in final_norm.parameters(): p.requires_grad = True

    elif method == "bitfit":
        for mod in (blocks, final_norm):
            for n_, p in mod.named_parameters():
                if n_.endswith("bias") or (n_.endswith("weight") and p.dim() == 1):
                    p.requires_grad = True

    elif method == "lorma":
        for parent, cn, lin in list(mixer_linears(bb)):
            setattr(parent, cn, LoRMALinear(lin, cfg.LORMA_R, cfg.LORMA_ALPHA).to(DEVICE))

    elif method == "adalora":
        for parent, cn, lin in list(mixer_linears(bb)):
            setattr(parent, cn,
                    AdaLoRALinear(lin, cfg.ADALORA_R, cfg.ADALORA_ALPHA).to(DEVICE))

    elif method == "adapter":
        # one adapter after the mixer's output projection, one after the feed-forward
        for parent, cn, lin in list(mixer_linears(bb)):
            if cn == "out_proj":
                setattr(parent, cn, AdapterWrap(lin, cfg.ADAPTER_DIM).to(DEVICE))
        for blk in blocks:
            mlp = getattr(blk, "mlp", None)
            if mlp is not None and isinstance(mlp.w3, nn.Linear):
                mlp.w3 = AdapterWrap(mlp.w3, cfg.ADAPTER_DIM).to(DEVICE)

    else:
        raise ValueError(method)

    tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"  [{method}] trainable {tr/1e6:.3f}M / {tot/1e6:.1f}M = {100*tr/tot:.3f}%")
    return model
""")

# =============================================================================
md(r"""
## 8. Training, scoring and the submission writer

Two things matter more here than in the previous phases.

**Class weighting.** Macro-F1 over two classes means the minority class carries half the
metric however rare it is, so the loss is weighted by inverse frequency with a cap.

**The decision threshold.** This is the single cheapest lever in the whole notebook. The model
outputs a probability; turning it into a 0 or a 1 needs a cut-point, and 0.5 is almost never
the one that maximises macro-F1 under class imbalance. `sweep_threshold` finds the best cut on
dev and it is applied at submission time. It costs nothing — no retraining — and in the
previous two phases the equivalent trick was worth a couple of points each time.
""")

code(r"""
def class_weights():
    w = torch.ones(2)
    if cfg.CLASS_WEIGHTS:
        c = Counter(d["label"] for d in train_docs.values())
        tot = sum(c.values())
        for i in (0, 1):
            w[i] = tot / max(2 * c.get(i, 1), 1)
        w = (w / w.mean()).clamp(max=cfg.CLASS_WEIGHT_CAP)
        w = w / w.mean()
    return w.to(DEVICE)


def make_optimizer(model, method):
    head_ids = {id(p) for p in list(model.head.parameters())
                + list(model.chunk_attn.parameters())}
    body_lr = cfg.LR_BACKBONE if method in ("full", "partial") else cfg.LR_PEFT
    g = [{"params": [p for p in model.parameters()
                     if p.requires_grad and id(p) in head_ids],
          "lr": cfg.LR_HEAD, "weight_decay": cfg.WEIGHT_DECAY},
         {"params": [p for p in model.parameters()
                     if p.requires_grad and id(p) not in head_ids],
          "lr": body_lr, "weight_decay": cfg.WEIGHT_DECAY}]
    return torch.optim.AdamW([x for x in g if x["params"]], betas=(0.9, 0.95), eps=1e-8)


def make_sched(opt, total):
    warm = max(1, int(cfg.WARMUP_FRAC * total))
    def fn(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train(model, ds, method, epochs=None, teacher=None, tag=""):
    epochs = epochs or cfg.EPOCHS
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate,
                    num_workers=2, pin_memory=True)
    steps = max(1, (len(dl) // cfg.GRAD_ACCUM) * epochs)
    opt = make_optimizer(model, method)
    sch = make_sched(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and USE_AMP)
    lossf = nn.CrossEntropyLoss(weight=class_weights(), ignore_index=-100)
    adal = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    if teacher is not None:
        teacher.eval()

    model.train(); t0 = time.time(); gstep = 0
    for ep in range(epochs):
        run = nb = 0
        for i, b in enumerate(dl):
            ids = b["input_ids"].to(DEVICE, non_blocking=True)
            tm  = b["tok_mask"].to(DEVICE, non_blocking=True)
            ow  = b["owner"].to(DEVICE, non_blocking=True)
            y   = b["labels"].to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
                logits = model(ids, tm, ow, b["n_docs"])
                loss = lossf(logits, y)
                if adal:
                    loss = loss + cfg.ADALORA_ORTH * sum(m.orth_penalty() for m in adal)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(ids, tm, ow, b["n_docs"])
                    T = cfg.KD_T
                    kd = F.kl_div(F.log_softmax(logits.float() / T, -1),
                                  F.softmax(tl.float() / T, -1),
                                  reduction="batchmean") * (T * T)
                    loss = (1 - cfg.KD_ALPHA) * loss + cfg.KD_ALPHA * kd
            scaler.scale(loss / cfg.GRAD_ACCUM).backward()
            run += loss.item(); nb += 1
            if (i + 1) % cfg.GRAD_ACCUM == 0 or i + 1 == len(dl):
                scaler.unscale_(opt)
                if adal:
                    adalora_housekeeping(model, gstep, steps)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.MAX_GRAD_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sch.step(); gstep += 1
            if nb % 200 == 0:
                print(f"    ep{ep+1} {nb}/{len(dl)} loss {run/nb:.4f} "
                      f"lr {sch.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        print(f"  [{tag}] epoch {ep+1}/{epochs} loss {run/max(nb,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def predict_proba(model, ds):
    '''-> {doc_id: P(label = 1)}'''
    model.eval()
    dl = DataLoader(ds, batch_size=max(2, cfg.BATCH_SIZE * 2), shuffle=False,
                    collate_fn=collate, num_workers=2)
    out = {}
    for b in dl:
        ids = b["input_ids"].to(DEVICE); tm = b["tok_mask"].to(DEVICE)
        ow  = b["owner"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
            p = F.softmax(model(ids, tm, ow, b["n_docs"]).float(), -1)[:, 1].cpu().numpy()
        for did, pi in zip(b["dids"], p):
            out[did] = float(pi)
    return out


def macro_f1_binary(pred, docs):
    tp = Counter(); fp = Counter(); fn = Counter()
    for did, d in docs.items():
        g, p = int(d["label"]), int(pred[did])
        if p == g: tp[g] += 1
        else:      fp[p] += 1; fn[g] += 1
    rows, Fs, Ps, Rs = [], [], [], []
    for c in (0, 1):
        P = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        R = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        Fm = 2 * P * R / (P + R) if P + R else 0.0
        rows.append((c, tp[c], fp[c], fn[c], P, R, Fm))
        Ps.append(P); Rs.append(R); Fs.append(Fm)
    acc = sum(tp.values()) / max(len(docs), 1)
    return {"macro_f1": float(np.mean(Fs)), "macro_p": float(np.mean(Ps)),
            "macro_r": float(np.mean(Rs)), "acc": acc, "per_class": rows}


def sweep_threshold(proba, docs, lo=0.05, hi=0.95, n=91, tol=0.005):
    # Best macro-F1 cut-point on dev. Takes the MIDDLE of the widest near-optimal
    # plateau rather than the bare argmax: with a small dev set the single best
    # threshold is often a knife-edge that does not survive the shift to test,
    # whereas the centre of a flat region does.
    ts = np.linspace(lo, hi, n)
    f1s = np.array([macro_f1_binary({d: int(p >= t) for d, p in proba.items()},
                                    docs)["macro_f1"] for t in ts])
    top = f1s.max()
    near = np.flatnonzero(f1s >= top - tol)
    runs, start = [], near[0]
    for a, b in zip(near, near[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    runs.append((start, near[-1]))
    i, j = max(runs, key=lambda r: r[1] - r[0])
    return float(ts[(i + j) // 2]), float(top)


def report(res):
    print(f"  {'class':<8}{'TP':>7}{'FP':>7}{'FN':>7}{'P':>8}{'R':>8}{'F1':>8}")
    for c, a, b_, d_, P, R, Fm in res["per_class"]:
        print(f"  {c:<8}{a:>7}{b_:>7}{d_:>7}{P:>8.3f}{R:>8.3f}{Fm:>8.3f}")
    print(f"  {'MACRO':<8}{'':>21}{res['macro_p']:>8.3f}{res['macro_r']:>8.3f}"
          f"{res['macro_f1']:>8.3f}   acc {res['acc']:.3f}")


def write_submission(proba, name, thr=0.5):
    '''Phase-4 format: a JSON LIST of {"id": str, "pred": int}.'''
    payload = [{"id": did, "pred": int(proba[did] >= thr)} for did in test_docs]
    assert isinstance(payload, list) and len(payload) == len(test_docs)
    assert {r["id"] for r in payload} == set(test_docs), "test id mismatch"
    for r in payload:
        assert isinstance(r["pred"], int) and r["pred"] in (0, 1), r
        assert set(r) == {"id", "pred"}, r

    jp = os.path.join(cfg.OUT_DIR, f"prediction_{name}.json")
    zp = os.path.join(cfg.OUT_DIR, f"submission_{name}.zip")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        back = json.loads(z.read("prediction.json"))
    assert isinstance(back, list) and len(back) == len(test_docs)
    mix = Counter(r["pred"] for r in payload)
    print(f"  -> {zp}  | {len(payload)} cases | thr {thr:.2f} | "
          f"predicted {dict(mix)} ({100*mix[1]/len(payload):.1f}% positive)")
    return zp


RESULTS, SCORES = [], {}

def run_experiment(name, arch, method, epochs=None, bidirectional=None, keep_layers=None,
                   teacher=None, write=True, seed=42, keep_scores=True, ds=None):
    print(f"\n{'='*72}\n{name}   model {arch} | {method} | repr={cfg.REPR}\n{'='*72}",
          flush=True)
    set_seed(seed); t0 = time.time()
    model = build_cjpe(arch, bidirectional=bidirectional, keep_layers=keep_layers)
    model = apply_adaptation(model, method)
    train(model, ds or ds_train, method, epochs=epochs, teacher=teacher, tag=name)

    pv = predict_proba(model, ds_dev)
    r05 = macro_f1_binary({d: int(p >= 0.5) for d, p in pv.items()}, dev_docs)
    report(r05)
    thr, f1t = sweep_threshold(pv, dev_docs)
    print(f"  threshold sweep: 0.50 -> {r05['macro_f1']:.4f} | "
          f"{thr:.2f} -> {f1t:.4f}")

    pt, zp = None, None
    if write:
        pt = predict_proba(model, ds_test)
        zp = write_submission(pt, name, thr=thr)
    if keep_scores:
        SCORES[name] = {"dev": pv, "test": pt, "thr": thr}

    RESULTS.append({"name": name, "model": arch, "method": method, "repr": cfg.REPR,
                    "dev_f1_at_0.5": round(r05["macro_f1"], 4),
                    "dev_f1_tuned": round(f1t, 4), "thr": round(thr, 2),
                    "acc": round(r05["acc"], 4),
                    "trainable_M": round(sum(p.numel() for p in model.parameters()
                                             if p.requires_grad) / 1e6, 3),
                    "minutes": round((time.time() - t0) / 60, 1), "zip": zp})
    print("\n " + json.dumps(RESULTS[-1]))
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, {"macro_f1": f1t, "thr": thr, "at05": r05["macro_f1"]}
""")

# =============================================================================
md(r"""
## 9. SMOKE TEST — run this first

Two to three minutes with a frozen backbone. If it prints a zip path, the pipeline is sound:
checkpoint loading, chunking, two-level pooling, training, threshold sweep and the new
list-format submission. **Upload it immediately** — it banks one submission and starts the
evaluation clock.
""")

code(r"""
m, _ = run_experiment("00_smoke", "A", "head_only", epochs=1, keep_scores=False)
del m; gc.collect(); torch.cuda.empty_cache()
""")

# =============================================================================
md(r"""
## 10. Which part of the judgment matters

Three short runs, same model and technique, one variable changed. Set `cfg.REPR` to the
winner before running the six.

Expect `last` to win: a judgment's final pages carry the court's reasoning toward its
conclusion, while the opening carries facts and procedural history that are far less
decisive. If `hier` wins instead, it is because pooling several chunks recovers evidence
spread through the document — worth knowing, but it costs roughly four times the compute per
document, so only take it if the gap is real.
""")

code(r"""
REPR_SCORES = {}
for mode in ("last", "first", "hier"):
    cfg.REPR = mode
    ds_train, ds_dev, ds_test = make_sets(mode)
    _, r = run_experiment(f"r_{mode}", "A", "lorma", epochs=2, write=False,
                          keep_scores=False)
    REPR_SCORES[mode] = r["macro_f1"]
    gc.collect(); torch.cuda.empty_cache()

print("\nrepresentation comparison (dev macro-F1, tuned threshold):")
for k, v in sorted(REPR_SCORES.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<6} {v:.4f}")

cfg.REPR = max(REPR_SCORES, key=REPR_SCORES.get)
ds_train, ds_dev, ds_test = make_sets(cfg.REPR)
print(f"\n-> using repr = {cfg.REPR} for the six submissions")
""")

# =============================================================================
md(r"""
## 11. The six submissions

One cell each. Upload every zip as it appears. If time gets tight, runs 1 (LoRMA) and 6
(distillation) are the two the brief makes compulsory.
""")

code(r"""
# --- 1. Model A, LoRMA+  (COMPULSORY technique) --------------------------------------
m_teacher, r_lorma = run_experiment("01_A_lorma", "A", "lorma", epochs=4)
""")

code(r"""
# --- 2. Model C, adapters ------------------------------------------------------------
m, _ = run_experiment("02_C_adapter", "C", "adapter", epochs=4)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 3. Model D, AdaLoRA -------------------------------------------------------------
m, _ = run_experiment("03_D_adalora", "D", "adalora", epochs=4)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 4. Model B, full fine-tuning (full encoder-decoder trunk) -----------------------
m, _ = run_experiment("04_B_full", "B", "full", epochs=3)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 5. Model A, BitFit --------------------------------------------------------------
m, _ = run_experiment("05_A_bitfit", "A", "bitfit", epochs=6)   # tiny param count
del m; gc.collect(); torch.cuda.empty_cache()
""")

md(r"""
### 6. Knowledge distillation *(compulsory)*

The teacher is the trained Phase-4 model from run 1, as §10.2.2 asks — a stronger model on
this task, not a raw Phase-1 checkpoint. The student is the same architecture with six of its
twelve blocks, taken at evenly spaced depths so it inherits a spread of the teacher's feature
hierarchy rather than only its early layers.

The loss mixes hard cross-entropy with a temperature-softened KL against the teacher's class
distribution. With only two classes the softened distribution carries less extra information
than it did in Phase 3 — there is no third class for the mass to land on — so what the student
mainly inherits here is the teacher's *confidence calibration*: which cases are near the
boundary and which are clear-cut. That is still a real signal the hard label discards.

Report the student's macro-F1 next to its parameter count; §10.5 asks about the trade-off,
not the score alone.
""")

code(r"""
# --- 6. Model A -> half-depth student, knowledge distillation (COMPULSORY) ----------
for p in m_teacher.parameters():
    p.requires_grad = False

m_student, r_kd = run_experiment("06_A_distill", "A", "lorma", epochs=4,
                                 keep_layers=6, teacher=m_teacher)

tp = sum(p.numel() for p in {id(p): p for p in m_teacher.parameters()}.values())
sp = sum(p.numel() for p in {id(p): p for p in m_student.parameters()}.values())
print(f"\nteacher {tp/1e6:.1f}M -> student {sp/1e6:.1f}M ({100*sp/tp:.0f}% of size)")
print(f"dev macro-F1 {r_lorma['macro_f1']:.4f} -> {r_kd['macro_f1']:.4f}")
del m_student; gc.collect(); torch.cuda.empty_cache()
""")

# =============================================================================
md(r"""
## 12. Results and the ensemble

The ensemble averages predicted probabilities across models and re-sweeps the threshold on the
averaged scores. In Phase 2 this was worth +5.4 macro-F1 and in Phase 3 about +2 over the best
single model, because the four architectures make uncorrelated errors. Here the threshold
re-sweep on top should add a little more.
""")

code(r"""
import pandas as pd
df = pd.DataFrame(RESULTS).sort_values("dev_f1_tuned", ascending=False)
df.to_csv("/kaggle/working/phase4_results.csv", index=False)
display(df[["name", "model", "method", "repr", "dev_f1_at_0.5", "dev_f1_tuned",
            "thr", "acc", "trainable_M", "minutes"]])
print("architectures:", sorted(df["model"].unique()))
print("techniques   :", sorted(df["method"].unique()))

# ---- probability-level ensemble, top-k by dev, threshold re-swept -------------------
ranked = [r["name"] for r in sorted(RESULTS, key=lambda r: -r["dev_f1_tuned"])
          if r["name"] in SCORES and SCORES[r["name"]]["test"] is not None]
best_single = max(r["dev_f1_tuned"] for r in RESULTS)
best = (None, best_single, 0.5)

for k in range(2, len(ranked) + 1):
    use = ranked[:k]
    ed = {d: float(np.mean([SCORES[n]["dev"][d] for n in use])) for d in dev_docs}
    t, f1 = sweep_threshold(ed, dev_docs)
    print(f"  top-{k:<2} thr {t:.2f} -> {f1:.4f}   {use}")
    if f1 > best[1]:
        best = (use, f1, t)

print(f"\nbest single {best_single:.4f} -> best ensemble {best[1]:.4f}")
if best[0]:
    ed = {d: float(np.mean([SCORES[n]["dev"][d] for n in best[0]])) for d in dev_docs}
    report(macro_f1_binary({d: int(p >= best[2]) for d, p in ed.items()}, dev_docs))
    et = {d: float(np.mean([SCORES[n]["test"][d] for n in best[0]])) for d in test_docs}
    write_submission(et, "07_ensemble", thr=best[2])
else:
    print("  no ensemble beat the best single model — submit that one")
""")

md(r"""
## 13. Spares, if you have time left

Each is a legitimate extra submission and each answers one of §10.5's questions.

```python
# partial / layer-wise fine-tuning -- a seventh technique from the §10.2.1 list
m, _ = run_experiment("08_A_partial", "A", "partial", epochs=4)

# mean pooling across chunks instead of learned attention (only meaningful if repr == "hier")
cfg.CHUNK_POOL = "mean"; m, _ = run_experiment("09_hier_meanpool", "A", "lorma", epochs=4)
cfg.CHUNK_POOL = "attn"

# the causal ablation: does un-masking attention actually help classification?
m, _ = run_experiment("10_A_causal", "A", "lorma", epochs=4, bidirectional=False)

# refit on train+dev once the hyperparameters are frozen -- this gained ~2 points in Phase 2
ds_both = CJPEDocs({**train_docs, **dev_docs}, repr_mode=cfg.REPR)
m, _ = run_experiment("11_A_lorma_refit", "A", "lorma", epochs=4, ds=ds_both)
```

Note that the refit's dev score is meaningless — it trains on its own validation set — so keep
the threshold from run 1 rather than the one the refit reports.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

with open("CS781_Phase4_CJPE.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("cells:", len(cells))
