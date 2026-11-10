import json

cells = []

def _lines(src):
    return (src.strip("\n") + "\n").splitlines(keepends=True)

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(src)})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(src)})

# =============================================================================
md(r"""
# CS-781 Capstone — Phase 2: Legal Named Entity Recognition

Fine-tunes **all four of your Phase-1 legal language models** with a token-classification
head and emits strict character-offset spans.

## The six submissions

Each run pairs a different Phase-1 architecture with a different adaptation method, so the
set satisfies both rules at once: four distinct Phase-1 models, six methodologically
distinct experiments, LoRMA and knowledge distillation included.

| # | Phase-1 model | Architecture | Method | Why it is distinct |
|---|---------------|--------------|--------|--------------------|
| 1 | **A** `A_decoder.pt` | decoder-only, 12L, 109.5M | Full fine-tuning | every weight updated |
| 2 | **D** `D_gqa.pt` | GQA + sliding window, 100.1M | **LoRMA+ (compulsory)** | multiplicative low-rank reparameterisation |
| 3 | **B** `B_encdec.pt` | encoder-decoder, 123.7M | Partial / layer-wise | only the top encoder blocks train |
| 4 | **C** `C_hybrid.pt` | GLR + attention hybrid, 119.0M | LoRA | additive low-rank reparameterisation |
| 5 | **A** | decoder-only | BitFit | sparse bias-and-norm only |
| 6 | **A → half-depth A** | decoder-only | **Distillation (compulsory)** | teacher-student compression |

Spares if a run fails or you have time for more: `lorma_pi` on any model, `bitfit` on B,
or any run repeated with `BIDIRECTIONAL=False` — that ablation is a legitimate experiment
in its own right and is worth reporting whatever else happens.

## What you must edit
**Only cell 3 (CONFIG)**, and only if the globs fail to find your Kaggle datasets. The
architecture code is reproduced verbatim from your Phase-1 notebooks, so nothing else
needs touching.

## Run order tonight
1. Run cells 1-21 straight through. Cell 21 is a two-minute smoke test that writes a valid
   `submission.zip`. **Upload it to Codabench immediately** — it banks one valid submission
   and starts the 15-30 minute evaluation clock while the real models train.
2. Then run the six experiment cells one at a time, submitting each zip the moment it appears.

Minimum is 6 valid submissions, cap is 10 per day, and evaluation is slow. Submit as you go;
do not batch them to the end.
""")

# =============================================================================
md(r"""
## 1. Environment
""")

code(r"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, json, math, time, re, random, zipfile, gc, glob
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
""")

# =============================================================================
md(r"""
## 2. CONFIG

The paths are discovered by glob, matching how your Phase-1 notebooks found their
checkpoints. If a glob comes back empty the cell says which Kaggle dataset is not attached.
""")

code(r"""
class CFG:
    # ---- discovery -----------------------------------------------------------
    TOK_GLOB   = "/kaggle/input/**/bpe32k.json"
    CKPT_GLOBS = {"A": "/kaggle/input/**/A_decoder.pt",
                  "B": "/kaggle/input/**/B_encdec.pt",
                  "C": "/kaggle/input/**/C_hybrid.pt",
                  "D": "/kaggle/input/**/D_gqa.pt"}
    P2_GLOB    = "/kaggle/input/**/fold1-text.json"     # Phase-2 Train/ directory

    # ---- shared Phase-1 dimensions (identical across A-D) ---------------------
    DIM, N_HEADS, MAX_SEQ = 768, 12, 2048

    # ---- sequence handling ---------------------------------------------------
    MAX_LEN       = 512     # subword tokens per window
    STRIDE        = 128     # overlap between consecutive windows
    BIDIRECTIONAL = True    # un-mask causal attention for the NER head

    # ---- training ------------------------------------------------------------
    EPOCHS        = 3
    BATCH_SIZE    = 4
    GRAD_ACCUM    = 4
    LR_BACKBONE   = 2e-5    # full / partial fine-tuning
    LR_HEAD       = 1e-3
    LR_PEFT       = 5e-4    # lora / lorma / bitfit
    WARMUP_FRAC   = 0.06
    WEIGHT_DECAY  = 0.01
    MAX_GRAD_NORM = 1.0
    O_CLASS_WEIGHT = 0.3    # down-weight the dominant O tag
    LABEL_SMOOTH  = 0.0
    AMP           = True

    # ---- adaptation ----------------------------------------------------------
    LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
    LORMA_R, LORMA_ALPHA = 16, 32
    PARTIAL_TOP_K = 3       # top blocks to unfreeze (B's encoder has only 6)

    # ---- distillation --------------------------------------------------------
    KD_T, KD_ALPHA = 2.0, 0.5

    # ---- decoding ------------------------------------------------------------
    MIN_SPAN_CHARS = 1
    DROP_O_BIAS    = 0.0    # subtract from the O logit at decode; >0 raises recall

    # ---- split / output ------------------------------------------------------
    VAL_FRAC = 0.15         # held-out documents from the pooled fold1+fold2
    SPLIT_SEED = 42
    OUT_DIR = "/kaggle/working/submissions"

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"{what} not found ({pattern}) — attach the Kaggle dataset"
    return Path(hits[0])

TOK_PATH = find_one(cfg.TOK_GLOB, "tokenizer bpe32k.json")
P2_DIR   = find_one(cfg.P2_GLOB, "Phase-2 fold1-text.json").parent
CKPTS    = {k: find_one(v, f"Phase-1 model {k}") for k, v in cfg.CKPT_GLOBS.items()}

print("tokenizer :", TOK_PATH)
print("phase-2   :", P2_DIR)
for k, v in CKPTS.items():
    print(f"model {k}   : {v}  ({v.stat().st_size/1e6:.0f} MB)")
""")

# =============================================================================
md(r"""
## 3. Data
""")

code(r"""
def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _norm_text(raw):
    return {str(k): (v if isinstance(v, str) else (v.get("text") or v.get("data") or ""))
            for k, v in raw.items()}

def _norm_gold(raw):
    out = {}
    for did, ents in raw.items():
        lst = []
        for e in ents:
            if isinstance(e, dict):
                s = int(e.get("start", e.get("start_char", e.get("begin"))))
                t = int(e.get("end", e.get("end_char")))
                lab = e.get("label", e.get("type", e.get("entity")))
            else:
                s, t, lab = int(e[0]), int(e[1]), str(e[2])
            lst.append((s, t, lab))
        out[str(did)] = lst
    return out

TEST_DIR = P2_DIR.parent / "Test"
f3 = sorted(glob.glob(str(P2_DIR.parent / "**" / "fold3-text.json"), recursive=True))
assert f3, "fold3-text.json (test) not found"

fold_text = {1: _norm_text(_read_json(P2_DIR / "fold1-text.json")),
             2: _norm_text(_read_json(P2_DIR / "fold2-text.json"))}
fold_gold = {1: _norm_gold(_read_json(P2_DIR / "fold1-gold.json")),
             2: _norm_gold(_read_json(P2_DIR / "fold2-gold.json"))}
test_text = _norm_text(_read_json(f3[0]))

LABELS_FALLBACK = ["APP","RESP","A.COUNSEL","R.COUNSEL","JUDGE","WIT",
                   "AUTH","COURT","STAT","PREC","DATE","CASENO"]
try:
    ENTITY_LABELS = [l.strip() for l in open(P2_DIR / "labels.txt") if l.strip()]
except Exception:
    ENTITY_LABELS = LABELS_FALLBACK
ENTITY_LABELS = sorted(set(ENTITY_LABELS))

TAGS = ["O"] + [f"{p}-{l}" for l in ENTITY_LABELS for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}
NUM_TAGS = len(TAGS)

# pool both folds, split by DOCUMENT (never by window — windows of one document
# share text, so a window-level split leaks)
all_text = {**fold_text[1], **fold_text[2]}
all_gold = {**fold_gold[1], **fold_gold[2]}
doc_ids = sorted(all_text)
rng = np.random.RandomState(cfg.SPLIT_SEED); rng.shuffle(doc_ids)
n_val = max(1, int(len(doc_ids) * cfg.VAL_FRAC))
val_ids, train_ids = doc_ids[:n_val], doc_ids[n_val:]

train_text = {d: all_text[d] for d in train_ids}; train_gold = {d: all_gold[d] for d in train_ids}
val_text   = {d: all_text[d] for d in val_ids};   val_gold   = {d: all_gold[d] for d in val_ids}

print(f"docs: fold1 {len(fold_text[1])} + fold2 {len(fold_text[2])} "
      f"-> train {len(train_text)} / val {len(val_text)} | test {len(test_text)}")
print(f"{len(ENTITY_LABELS)} labels -> {NUM_TAGS} BIO tags")
ent_counts = Counter(l for ents in all_gold.values() for _, _, l in ents)
print("entities:", sum(ent_counts.values()), "|", ent_counts.most_common())
""")

# =============================================================================
md(r"""
## 4. Character-offset-safe atomisation

**This is the cell that decides whether you score at all.** Strict evaluation means a span
off by one character scores zero, so nothing here trusts a tokeniser's offset mapping —
your BPE has none.

The document is split into **atoms**: maximal runs of letters, maximal runs of digits, and
single punctuation characters, each carrying its exact `(start, end)` in the original
string. Labels live on atoms, and a predicted span is rebuilt by copying atom offsets
verbatim, so the round trip is exact by construction.

The diagnostic reports what fraction of gold entities land on atom boundaries. Expect above
99%. If it is lower, loosen `ATOM_RE` before training anything.
""")

code(r"""
ATOM_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")

def atomise(text):
    return [(m.group(0), m.start(), m.end()) for m in ATOM_RE.finditer(text)]

def spans_to_bio(atoms, ents):
    tags = ["O"] * len(atoms)
    if not atoms:
        return tags, 0
    starts = np.fromiter((a[1] for a in atoms), dtype=np.int64, count=len(atoms))
    ends   = np.fromiter((a[2] for a in atoms), dtype=np.int64, count=len(atoms))
    aligned = 0
    for (s, e, lab) in ents:
        if lab not in ENTITY_LABELS:
            continue
        idx = np.where((starts < e) & (ends > s))[0]
        if len(idx) == 0:
            continue
        if starts[idx[0]] == s and ends[idx[-1]] == e:
            aligned += 1
        for k, i in enumerate(idx):
            tags[i] = ("B-" if k == 0 else "I-") + lab
    return tags, aligned

def bio_to_spans(atoms, tags, text=None):
    out, i, n = [], 0, len(tags)
    while i < n:
        t = tags[i]
        if t == "O":
            i += 1; continue
        lab = t[2:]
        j = i + 1
        while j < n and tags[j] == "I-" + lab:
            j += 1
        s, e = atoms[i][1], atoms[j - 1][2]
        if text is not None:
            while s < e and text[s].isspace(): s += 1
            while e > s and text[e - 1].isspace(): e -= 1
        if e - s >= cfg.MIN_SPAN_CHARS:
            out.append((s, e, lab))
        i = j
    return out

tot = ok = rt_fail = 0
for did, txt in all_text.items():
    at = atomise(txt)
    ents = all_gold.get(did, [])
    tags, a = spans_to_bio(at, ents)
    tot += len([1 for _, _, l in ents if l in ENTITY_LABELS]); ok += a
    gold = set((s, e, l) for s, e, l in ents if l in ENTITY_LABELS)
    rt_fail += len(gold - set(bio_to_spans(at, tags, txt)))

print(f"gold entities on atom boundaries : {ok}/{tot} = {100*ok/max(tot,1):.2f}%")
print(f"lost in the BIO round trip       : {rt_fail}"
      f"   <- this is your ceiling on recall")
""")

# =============================================================================
md(r"""
## 5. Tokeniser and windowing

Each atom is encoded on its own and labelled on its **first subword**; the rest get `-100`
and are ignored by the loss. That keeps the atom-to-subword mapping exact without needing
offsets from the tokeniser.

Judgments run far past any usable context length, so a document becomes overlapping windows
of `MAX_LEN` subwords. At inference the per-atom log-probabilities from overlapping windows
are averaged, which stitches the document back together and acts as a small free ensemble
over context positions.
""")

code(r"""
from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(TOK_PATH))
VOCAB_SIZE = tok.get_vocab_size()
PAD_ID = int(tok.token_to_id("<|pad|>") or 0)
print(f"tokenizer vocab {VOCAB_SIZE:,} | pad {PAD_ID}")

_enc_cache = {}
def enc_atom(s):
    v = _enc_cache.get(s)
    if v is None:
        v = tok.encode(s).ids or [PAD_ID]
        if len(_enc_cache) < 400_000:
            _enc_cache[s] = v
    return v

def encode_doc(atoms):
    ids, aidx, first = [], [], []
    for i, (s, _, _) in enumerate(atoms):
        piece = s if i == 0 else (" " + s if s[0].isalnum() else s)
        for k, t in enumerate(enc_atom(piece)):
            ids.append(t); aidx.append(i); first.append(k == 0)
    return ids, aidx, first

def build_windows(text, ents=None):
    atoms = atomise(text)
    ids, aidx, first = encode_doc(atoms)
    tags = spans_to_bio(atoms, ents)[0] if ents is not None else None
    n, step = len(ids), max(1, cfg.MAX_LEN - cfg.STRIDE)
    wins = []
    if n == 0:
        return atoms, wins
    for st in range(0, n, step):
        en = min(st + cfg.MAX_LEN, n)
        wa, wf = aidx[st:en], first[st:en]
        lab = ([TAG2ID[tags[a]] if f else -100 for a, f in zip(wa, wf)]
               if tags is not None else [-100] * (en - st))
        wins.append({"ids": ids[st:en], "aidx": wa, "first": wf, "labels": lab})
        if en == n:
            break
    return atoms, wins

class NERWindows(Dataset):
    def __init__(self, texts, golds=None):
        self.items, self.doc_atoms = [], {}
        for did, txt in texts.items():
            atoms, wins = build_windows(txt, golds.get(did, []) if golds is not None else None)
            self.doc_atoms[did] = atoms
            for w in wins:
                w["did"] = did
                self.items.append(w)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    L = max(len(b["ids"]) for b in batch)
    ids  = torch.full((len(batch), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    lab  = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n]  = torch.tensor(b["ids"], dtype=torch.long)
        mask[i, :n] = 1
        lab[i, :n]  = torch.tensor(b["labels"], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask, "labels": lab,
            "meta": [{"did": b["did"], "aidx": b["aidx"], "first": b["first"]} for b in batch]}

ds_train = NERWindows(train_text, train_gold)
ds_val   = NERWindows(val_text, val_gold)
ds_test  = NERWindows(test_text, None)
print(f"windows -> train {len(ds_train)} | val {len(ds_val)} | test {len(ds_test)}")
""")

# =============================================================================
md(r"""
## 6. Phase-1 architecture components

Reproduced from your Phase-1 notebooks so the checkpoints load with `strict=True`. Two
surgical changes, both additive — parameter names and shapes are untouched, so loading is
unaffected:

- `CausalSelfAttention` gains a `causal` flag (default `True`, exactly the trained
  behaviour) instead of hardcoding `is_causal=True`.
- `GQASlidingWindowAttention` gains a `bidirectional` flag and a symmetric window mask.

**Why un-mask at all.** Your Phase-1 models are causal: token *i* cannot see token *i+1*.
For language modelling that is the point; for NER it is the wrong inductive bias, because
the token *after* a name routinely decides its label — "... v. State of Bihar" is a
`RESP` only once you have seen what follows "v.". Fine-tuning with attention un-masked
typically buys several macro-F1 points. It is still your Phase-1 model with your Phase-1
weights; only the mask changes, and the change is declared in the report as part of the
adaptation.
""")

code(r"""
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, mult=4, multiple_of=64):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim, self.max_seq = head_dim, max_seq
        inv_freq = 1.0 / (10000 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
    def forward(self, x, pos0=0):
        dtype = x.dtype
        B, H, T, D = x.shape
        half = D // 2
        x = x.float()
        x1, x2 = x[..., :half], x[..., half:]
        cos = self.cos[pos0:pos0 + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[pos0:pos0 + T].unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq, rope=True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope     = RoPE(self.head_dim, max_seq) if rope else None
        self.causal   = True                     # <- flag; trained behaviour is True
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class GeneralizedAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq, causal=True, cross=False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.causal, self.cross = causal, cross
        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = None if cross else RoPE(self.head_dim, max_seq)
    def forward(self, x, context=None):
        B, T, C = x.shape
        src = context if self.cross else x
        S = src.shape[1]
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=(self.causal and not self.cross))
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, max_seq):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = GeneralizedAttention(dim, n_heads, max_seq, causal=False, cross=False)
        self.norm2 = RMSNorm(dim)
        self.mlp   = SwiGLU(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, max_seq):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = GeneralizedAttention(dim, n_heads, max_seq, causal=True, cross=False)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = GeneralizedAttention(dim, n_heads, max_seq, causal=False, cross=True)
        self.norm3 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)
    def forward(self, x, encoder_output):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context=encoder_output)
        return x + self.mlp(self.norm3(x))


class GatedLinearRecurrence(nn.Module):
    def __init__(self, dim, conv_kernel=3, chunk_size=16):
        super().__init__()
        self.dim, self.chunk_size, self.conv_kernel = dim, chunk_size, conv_kernel
        self.in_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.dwconv = nn.Conv1d(dim, dim, conv_kernel,
                                padding=conv_kernel - 1, groups=dim, bias=False)
        self.decay_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj   = nn.Linear(dim, dim, bias=False)
    def _scan(self, log_a, u):
        B, T, D = u.shape
        L = self.chunk_size
        pad = (-T) % L
        if pad:
            log_a = F.pad(log_a, (0, 0, 0, pad))
            u     = F.pad(u,     (0, 0, 0, pad))
        n = (T + pad) // L
        log_a = log_a.view(B, n, L, D)
        u     = u.view(B, n, L, D)
        A = log_a.cumsum(2)
        d = A.unsqueeze(3) - A.unsqueeze(2)
        tri = torch.tril(torch.ones(L, L, device=u.device, dtype=torch.bool))
        d = d.masked_fill(~tri[None, None, :, :, None], float("-inf"))
        M = torch.exp(d)
        local = torch.einsum("bnijd,bnjd->bnid", M, u)
        h, outs = u.new_zeros(B, D), []
        decay = torch.exp(A)
        for c in range(n):
            outs.append(local[:, c] + decay[:, c] * h[:, None])
            h = outs[-1][:, -1]
        return torch.cat(outs, 1)[:, :T]
    def forward(self, x):
        B, T, D = x.shape
        c = self.dwconv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        v, i, _, o = self.in_proj(c).chunk(4, dim=-1)
        v, i, o = torch.tanh(v), torch.sigmoid(i), torch.sigmoid(o)
        log_a = -0.01 - 0.99 * torch.sigmoid(self.decay_proj(c).float())
        h = self._scan(log_a, i.float() * v.float())
        return self.out_proj((h * o.float()).to(x.dtype))


class GQASlidingWindowAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=4, max_seq=2048,
                 window_size=256, sliding_window=True):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.window_size, self.sliding_window = window_size, sliding_window
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim, max_seq)
        pos = torch.arange(max_seq)
        d = pos[:, None] - pos[None, :]
        causal = d >= 0
        self.register_buffer("causal_mask", causal, persistent=False)
        self.register_buffer("window_mask", causal & (d < window_size), persistent=False)
        # symmetric counterparts, used only when bidirectional is turned on
        self.register_buffer("bi_full_mask", torch.ones_like(causal), persistent=False)
        self.register_buffer("bi_window_mask", d.abs() < window_size, persistent=False)
        self.bidirectional = False               # <- flag; trained behaviour is False
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        if self.bidirectional:
            mask = self.bi_window_mask if self.sliding_window else self.bi_full_mask
        else:
            mask = self.window_mask if self.sliding_window else self.causal_mask
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:T, :T], is_causal=False,
            enable_gqa=(self.n_kv_heads != self.n_heads))
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class GenericBlock(nn.Module):
    def __init__(self, dim, mixer):
        super().__init__()
        self.norm1, self.mixer = RMSNorm(dim), mixer
        self.norm2, self.mlp   = RMSNorm(dim), SwiGLU(dim)
    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))

print("components defined")
""")

# =============================================================================
md(r"""
## 7. The four Phase-1 models
""")

code(r"""
DIM, N_HEADS, MAX_SEQ = cfg.DIM, cfg.N_HEADS, cfg.MAX_SEQ


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, max_seq):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = CausalSelfAttention(dim, n_heads, max_seq)
        self.norm2 = RMSNorm(dim)
        self.mlp   = SwiGLU(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPTModel(nn.Module):
    # Model A -- decoder-only.
    def __init__(self, vocab_size, dim=DIM, n_layers=12, n_heads=N_HEADS, max_seq=MAX_SEQ):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(dim, n_heads, max_seq) for _ in range(n_layers)])
        self.norm_f  = RMSNorm(dim)
        self.head    = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def trunk(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.norm_f(x)


class EncoderDecoderModel(nn.Module):
    # Model B -- encoder-decoder. Only the encoder is used as the NER trunk.
    def __init__(self, vocab_size, dim=DIM, n_encoder_layers=6, n_decoder_layers=6,
                 n_heads=N_HEADS, max_seq=MAX_SEQ):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(dim, n_heads, max_seq) for _ in range(n_encoder_layers)])
        self.norm_e = RMSNorm(dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(dim, n_heads, max_seq) for _ in range(n_decoder_layers)])
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def trunk(self, idx):
        e = self.tok_emb(idx)
        for b in self.encoder_blocks:
            e = b(e)
        return self.norm_e(e)


class GenericLM(nn.Module):
    # Models C and D -- same skeleton, different token mixer per layer.
    def __init__(self, mixer_factory, vocab_size, dim=DIM, n_layers=12, max_seq=MAX_SEQ):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks  = nn.ModuleList(
            [GenericBlock(dim, mixer_factory(i)) for i in range(n_layers)])
        self.norm_f  = RMSNorm(dim)
        self.head    = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def trunk(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.norm_f(x)


C_LAYERS, C_ATTN_EVERY, C_CHUNK = 12, 3, 16
D_LAYERS, D_FULL_EVERY, D_WINDOW, D_KV_HEADS = 12, 3, 256, 4

def c_mixer_factory(i):
    if (i + 1) % C_ATTN_EVERY == 0:
        return CausalSelfAttention(DIM, N_HEADS, max_seq=MAX_SEQ)
    return GatedLinearRecurrence(DIM, conv_kernel=3, chunk_size=C_CHUNK)

def d_mixer_factory(i):
    return GQASlidingWindowAttention(
        DIM, N_HEADS, n_kv_heads=D_KV_HEADS, max_seq=MAX_SEQ,
        window_size=D_WINDOW, sliding_window=((i + 1) % D_FULL_EVERY != 0))


def _build(arch):
    if arch == "A": return GPTModel(VOCAB_SIZE, n_layers=12)
    if arch == "B": return EncoderDecoderModel(VOCAB_SIZE)
    if arch == "C": return GenericLM(c_mixer_factory, VOCAB_SIZE, n_layers=C_LAYERS)
    if arch == "D": return GenericLM(d_mixer_factory, VOCAB_SIZE, n_layers=D_LAYERS)
    raise ValueError(arch)

def trunk_blocks(model, arch):
    # The ModuleList the NER trunk actually runs through.
    return model.encoder_blocks if arch == "B" else model.blocks

def count_unique(m):
    return sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())

def load_phase1(arch, pretrained=True):
    m = _build(arch)
    if pretrained:
        sd = torch.load(CKPTS[arch], map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        missing, unexpected = m.load_state_dict(
            {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()},
            strict=True)
        assert not missing and not unexpected
    return m

ARCH_INFO = {"A": "decoder-only, 12L", "B": "encoder-decoder (encoder trunk, 6L)",
             "C": "GLR + attention hybrid, 12L", "D": "GQA sliding-window, 12L"}
print("model builders ready:", ", ".join(f"{k} ({v})" for k, v in ARCH_INFO.items()))
""")

# =============================================================================
md(r"""
## 8. NER model, and what bidirectional means per architecture

Each architecture reaches un-masked attention differently, and one of them only partly:

- **A** — every layer is `CausalSelfAttention`; setting `causal=False` makes the whole trunk
  bidirectional.
- **B** — the encoder was *already* trained bidirectionally (`causal=False` at Phase 1), so
  nothing is patched and nothing is out of distribution. This is the one trunk whose
  pretraining matches the task's needs exactly, which is worth saying in the report.
- **C** — the four attention layers un-mask, but the eight gated-linear-recurrence layers
  stay causal. A recurrence has a direction built into it: `h_t` is a function of `h_{t-1}`,
  so there is no mask to flip. Running it backwards would be a different operator from the
  one the weights were trained for. So C ends up **partly bidirectional**, and that is a
  genuine finding to report rather than a defect to hide.
- **D** — the window and causal masks are replaced by their symmetric counterparts, so
  windowed layers see `window_size` tokens either side and full layers see everything.
""")

code(r"""
def set_bidirectional(model, on=True):
    # Returns the number of mixer modules switched; B reports 0 by design.
    n = 0
    for mod in model.modules():
        if isinstance(mod, CausalSelfAttention):
            mod.causal = not on; n += 1
        elif isinstance(mod, GQASlidingWindowAttention):
            mod.bidirectional = on; n += 1
    return n


class LegalNER(nn.Module):
    def __init__(self, backbone, arch, hidden, num_tags, dropout=0.1):
        super().__init__()
        self.backbone, self.arch, self.hidden = backbone, arch, hidden
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_tags))
    def forward(self, input_ids, attention_mask=None):
        h = self.backbone.trunk(input_ids)
        return self.head(self.drop(h))


def build_ner(arch, bidirectional=None, keep_layers=None, pretrained=True, verbose=True):
    bb = load_phase1(arch, pretrained=pretrained)

    if keep_layers is not None:                 # student for distillation
        blocks = trunk_blocks(bb, arch)
        keep = np.linspace(0, len(blocks) - 1, keep_layers).round().astype(int).tolist()
        new = nn.ModuleList([blocks[i] for i in keep])
        if arch == "B": bb.encoder_blocks = new
        else:           bb.blocks = new
        if verbose: print(f"  student trunk {len(blocks)} -> {keep_layers} blocks {keep}")

    # the NER trunk never runs B's decoder or either LM head; dropping them keeps
    # memory down and the reported parameter counts honest
    if arch == "B":
        bb.decoder_blocks = nn.ModuleList()
        bb.norm_f = nn.Identity()
    bb.head = nn.Identity()

    bi = cfg.BIDIRECTIONAL if bidirectional is None else bidirectional
    n = set_bidirectional(bb, bi)

    m = LegalNER(bb, arch, cfg.DIM, NUM_TAGS).to(DEVICE)
    if verbose:
        used = sum(p.numel() for p in
                   {id(p): p for p in [bb.tok_emb.weight]
                    + list(trunk_blocks(bb, arch).parameters())}.values())
        print(f"  [{arch}] {ARCH_INFO[arch]} | {len(trunk_blocks(bb, arch))} trunk blocks"
              f" | {used/1e6:.1f}M trunk params | bidirectional={bi}"
              f" ({n} mixers switched)")
    return m
""")

# =============================================================================
md(r"""
## 9. Adaptation methods

### LoRA — additive
The pretrained weight is frozen and a trainable low-rank correction sits beside it:
`W_eff = W0 + (alpha/r) B A`. `B` starts at zero, so training begins at exactly the
pretrained function, and the trainable count falls from `d*k` to `r*(d+k)`.

### LoRMA — multiplicative *(compulsory)*
LoRMA (Exploration-Lab, ACL Findings 2025 — the lab running this course) replaces the
additive update with a multiplicative one: `W = (alpha/r)(B A) W0`, a *transformation of*
the weight rather than an offset to it. Two problems follow, and the paper fixes both with
**rank inflation**:

- `rank((BA)W0) <= r`, so a bare product caps the adapted weight at rank `r` no matter what
  the task needs;
- there is no natural zero initialisation, because the neutral element of a product is the
  identity, not zero.

**LoRMA+** adds an identity: `W = (I + (alpha/r) B A) W0`. At `BA = 0` this is exactly `W0`,
and the inflated factor stays near full rank throughout training. **LoRMA-pi** instead
permutes the rows of `BA` cyclically before use; it is included as a spare.

The implementation never materialises `BA` as a `d x d` matrix. It computes `y = W0 x`
first, then `y + (y A^T) B^T`, which is the same result in two thin matrix products.

### BitFit
Biases and norm scales only. A genuinely different mechanism — it shifts and rescales
activations rather than rotating weight subspaces — with a tiny parameter count and a fast
run, which matters when the clock is against you.

### Partial / layer-wise
Freeze the lower blocks, train the top `k`. Lower layers hold lexical and syntactic
structure that transfers as-is; upper layers are where task-specific entity semantics get
written. Applied to **B**, whose encoder is only six blocks deep, so `k=3` is half the trunk.
""")

code(r"""
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t()) @ self.B.t() * self.scale


class LoRMALinear(nn.Module):
    # W = inflate((alpha/r) B A) @ W0.  inflate: identity-add (+) or row-permute (pi).
    def __init__(self, base: nn.Linear, r, alpha, variant="plus"):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d_out = base.out_features
        self.A = nn.Parameter(torch.zeros(r, d_out))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        if variant == "plus":
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B stays zero -> starts at W0
        else:
            nn.init.normal_(self.A, std=0.02); nn.init.normal_(self.B, std=0.02)
        self.scale, self.variant = alpha / r, variant
        self.register_buffer("perm", torch.arange(d_out))
    def forward(self, x):
        y = self.base(x)                                   # y = W0 x
        m = (y @ self.A.t()) @ self.B.t() * self.scale     # = (BA) y, never materialised
        if self.variant == "plus":
            return y + m                                   # (I + BA) W0 x
        return y[..., self.perm] + m                       # I_pi(BA) W0 x


MIXER_TYPES = (CausalSelfAttention, GeneralizedAttention,
               GQASlidingWindowAttention, GatedLinearRecurrence)

def mixer_linears(model):
    # Every nn.Linear that is a direct child of a token mixer.
    # Selecting by module type, not by name: names like 'blocks' contain stray
    # letters that a keyword match would catch, dragging in the MLPs.
    for mod in model.modules():
        if isinstance(mod, MIXER_TYPES):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    yield mod, cname, child


def apply_adaptation(model, method):
    bb = model.backbone
    arch = model.arch
    for p in bb.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    blocks = trunk_blocks(bb, arch)

    if method == "full":
        for p in bb.tok_emb.parameters(): p.requires_grad = True
        for p in blocks.parameters():     p.requires_grad = True
        for p in (bb.norm_e if arch == "B" else bb.norm_f).parameters():
            p.requires_grad = True

    elif method == "head_only":
        pass

    elif method == "partial":
        for blk in list(blocks)[-cfg.PARTIAL_TOP_K:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in (bb.norm_e if arch == "B" else bb.norm_f).parameters():
            p.requires_grad = True

    elif method == "bitfit":
        for mod in [blocks, (bb.norm_e if arch == "B" else bb.norm_f)]:
            for n_, p in mod.named_parameters():
                if n_.endswith("bias") or (n_.endswith("weight") and p.dim() == 1):
                    p.requires_grad = True

    elif method == "lora":
        for parent, cname, lin in list(mixer_linears(bb)):
            setattr(parent, cname,
                    LoRALinear(lin, cfg.LORA_R, cfg.LORA_ALPHA, cfg.LORA_DROPOUT).to(DEVICE))

    elif method in ("lorma_plus", "lorma_pi"):
        v = "plus" if method == "lorma_plus" else "pi"
        for parent, cname, lin in list(mixer_linears(bb)):
            mod = LoRMALinear(lin, cfg.LORMA_R, cfg.LORMA_ALPHA, v).to(DEVICE)
            if v == "pi":
                d = lin.out_features
                mod.perm.copy_(torch.roll(torch.arange(d), shifts=max(1, d // 7)))
            setattr(parent, cname, mod)

    else:
        raise ValueError(method)

    tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"  [{method}] trainable {tr/1e6:.2f}M / {tot/1e6:.1f}M = {100*tr/tot:.2f}%")
    return model
""")

# =============================================================================
md(r"""
## 10. Training

Class weighting matters more than usual here. `O` is the overwhelming majority tag, and the
metric is **macro**-F1, so every label counts equally however rare it is: one label stuck at
zero costs roughly eight points on its own, regardless of how well the common ones do.
`O_CLASS_WEIGHT` down-weights the majority tag so the rare entity types get gradient.
""")

code(r"""
def make_optimizer(model, method):
    head_ids = {id(p) for p in model.head.parameters()}
    peft = method in ("lora", "lorma_plus", "lorma_pi", "bitfit")
    body_lr = cfg.LR_PEFT if peft else cfg.LR_BACKBONE
    groups = [
        {"params": [p for p in model.head.parameters() if p.requires_grad],
         "lr": cfg.LR_HEAD, "weight_decay": cfg.WEIGHT_DECAY},
        {"params": [p for p in model.parameters()
                    if p.requires_grad and id(p) not in head_ids],
         "lr": body_lr, "weight_decay": cfg.WEIGHT_DECAY}]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def make_sched(opt, total):
    warm = max(1, int(cfg.WARMUP_FRAC * total))
    def fn(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def class_weights():
    w = torch.ones(NUM_TAGS)
    w[TAG2ID["O"]] = cfg.O_CLASS_WEIGHT
    return w.to(DEVICE)


def train(model, method, ds, epochs=None, teacher=None, tag=""):
    epochs = epochs or cfg.EPOCHS
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                    collate_fn=collate, num_workers=2, pin_memory=True)
    steps = max(1, (len(dl) // cfg.GRAD_ACCUM) * epochs)
    opt = make_optimizer(model, method)
    sch = make_sched(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and USE_AMP)
    lossf = nn.CrossEntropyLoss(weight=class_weights(), ignore_index=-100,
                                label_smoothing=cfg.LABEL_SMOOTH)
    if teacher is not None:
        teacher.eval()

    model.train(); t0 = time.time()
    for ep in range(epochs):
        run = nb = 0
        for i, batch in enumerate(dl):
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            lab = batch["labels"].to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
                logits = model(ids)
                loss = lossf(logits.reshape(-1, NUM_TAGS), lab.reshape(-1))
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(ids)
                    sel = (lab != -100).reshape(-1)
                    T = cfg.KD_T
                    kd = F.kl_div(
                        F.log_softmax(logits.reshape(-1, NUM_TAGS)[sel].float() / T, -1),
                        F.softmax(tl.reshape(-1, NUM_TAGS)[sel].float() / T, -1),
                        reduction="batchmean") * (T * T)
                    loss = (1 - cfg.KD_ALPHA) * loss + cfg.KD_ALPHA * kd
            scaler.scale(loss / cfg.GRAD_ACCUM).backward()
            run += loss.item(); nb += 1
            if (i + 1) % cfg.GRAD_ACCUM == 0 or i + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.MAX_GRAD_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sch.step()
            if nb % 100 == 0:
                print(f"    ep{ep+1} {nb}/{len(dl)} loss {run/nb:.4f} "
                      f"lr {sch.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        print(f"  [{tag or method}] epoch {ep+1}/{epochs} loss {run/max(nb,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return model
""")

# =============================================================================
md(r"""
## 11. Inference, span decoding, strict scoring
""")

code(r"""
@torch.no_grad()
def predict_spans(model, ds, texts):
    model.eval()
    dl = DataLoader(ds, batch_size=max(2, cfg.BATCH_SIZE * 2), shuffle=False,
                    collate_fn=collate, num_workers=2)
    acc = {d: np.zeros((len(a), NUM_TAGS), np.float32) for d, a in ds.doc_atoms.items()}
    cnt = {d: np.zeros((len(a), 1), np.float32) for d, a in ds.doc_atoms.items()}
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
            lp = F.log_softmax(model(ids).float(), -1).cpu().numpy()
        for b, meta in enumerate(batch["meta"]):
            A, C = acc[meta["did"]], cnt[meta["did"]]
            for t, (ai, fw) in enumerate(zip(meta["aidx"], meta["first"])):
                if fw:
                    A[ai] += lp[b, t]; C[ai, 0] += 1
    out = {}
    for did, A in acc.items():
        S = A / np.maximum(cnt[did], 1.0)
        if cfg.DROP_O_BIAS:
            S[:, TAG2ID["O"]] -= cfg.DROP_O_BIAS
        tags = [ID2TAG[i] for i in S.argmax(-1)]
        for i, t in enumerate(tags):             # repair illegal I- starts
            if t.startswith("I-") and (i == 0 or tags[i-1] not in ("B-"+t[2:], "I-"+t[2:])):
                tags[i] = "B-" + t[2:]
        out[did] = bio_to_spans(ds.doc_atoms[did], tags, texts[did])
    return out


def strict_macro_f1(pred, gold):
    tp, fp, fn = Counter(), Counter(), Counter()
    for did in gold:
        g = set((s, e, l) for s, e, l in gold[did] if l in ENTITY_LABELS)
        p = set(pred.get(did, []))
        for _, _, l in (p & g): tp[l] += 1
        for _, _, l in (p - g): fp[l] += 1
        for _, _, l in (g - p): fn[l] += 1
    rows, Ps, Rs, Fs = [], [], [], []
    for l in ENTITY_LABELS:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        Fm = 2 * P * R / (P + R) if P + R else 0.0
        rows.append((l, tp[l], fp[l], fn[l], P, R, Fm))
        Ps.append(P); Rs.append(R); Fs.append(Fm)
    return {"macro_f1": float(np.mean(Fs)), "macro_p": float(np.mean(Ps)),
            "macro_r": float(np.mean(Rs)), "per_label": rows}


def report(res):
    print(f"  {'label':<12}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for l, a, b, c, P, R, Fm in res["per_label"]:
        print(f"  {l:<12}{a:>6}{b:>6}{c:>6}{P:>8.3f}{R:>8.3f}{Fm:>8.3f}")
    print(f"  {'MACRO':<12}{'':>18}{res['macro_p']:>8.3f}"
          f"{res['macro_r']:>8.3f}{res['macro_f1']:>8.3f}")
    zero = [r[0] for r in res["per_label"] if r[-1] == 0.0]
    if zero:
        print(f"  !! zero-F1 labels ({len(zero)}): {zero}  "
              f"-> each costs ~{100/len(ENTITY_LABELS):.0f}% of macro-F1")
""")

# =============================================================================
md(r"""
## 12. Submission writer

Three things silently zero a submission: a missing test ID, an unknown extra ID, and a
nested folder inside the zip. All three are asserted before the file is written.
""")

code(r"""
def write_submission(pred, name):
    payload = {}
    for did in test_text:                       # every official ID, in order, no extras
        payload[did] = [[int(s), int(e), l] for s, e, l in pred.get(did, [])]
    assert set(payload) == set(test_text)
    for did, rows in payload.items():
        for s, e, l in rows:
            assert 0 <= s < e <= len(test_text[did]) and l in ENTITY_LABELS, (did, s, e, l)

    jp = os.path.join(cfg.OUT_DIR, f"prediction_{name}.json")
    zp = os.path.join(cfg.OUT_DIR, f"submission_{name}.zip")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")   # root of the archive, no folder
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()

    n = sum(len(v) for v in payload.values())
    empty = sum(1 for v in payload.values() if not v)
    print(f"  -> {zp}")
    print(f"     {len(payload)} docs | {n} entities | {empty} with none | "
          f"{os.path.getsize(zp)/1e3:.0f} KB")
    print(f"     mix: {Counter(r[2] for v in payload.values() for r in v).most_common()}")
    return zp
""")

# =============================================================================
md(r"""
## 13. One experiment, end to end
""")

code(r"""
RESULTS = []

def run_experiment(name, arch, method, epochs=None, bidirectional=None,
                   keep_layers=None, teacher=None, write=True, seed=42):
    print(f"\n{'='*72}\n{name}   model {arch} | {method}\n{'='*72}", flush=True)
    set_seed(seed); t0 = time.time()

    model = build_ner(arch, bidirectional=bidirectional, keep_layers=keep_layers)
    model = apply_adaptation(model, method)
    train(model, method, ds_train, epochs=epochs, teacher=teacher, tag=name)

    res = strict_macro_f1(predict_spans(model, ds_val, val_text), val_gold)
    report(res)

    zp = None
    if write:
        zp = write_submission(predict_spans(model, ds_test, test_text), name)

    RESULTS.append({"name": name, "model": arch, "method": method,
                    "val_macro_f1": round(res["macro_f1"], 4),
                    "val_p": round(res["macro_p"], 4), "val_r": round(res["macro_r"], 4),
                    "params_M": round(sum(p.numel() for p in model.parameters())/1e6, 1),
                    "trainable_M": round(sum(p.numel() for p in model.parameters()
                                             if p.requires_grad)/1e6, 2),
                    "minutes": round((time.time() - t0) / 60, 1), "zip": zp})
    print("\n " + json.dumps(RESULTS[-1]))
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, res
""")

# =============================================================================
md(r"""
## 14. SMOKE TEST — run this first

Two to three minutes on Model A with a frozen trunk. If it prints a zip path, the entire
pipeline is sound: checkpoint loading, offsets, windowing, training, decoding, scoring and
packaging. **Upload that zip to Codabench straight away.** The score will be poor, but it is
a valid submission against the minimum of six and it starts the evaluation clock while the
real runs train.
""")

code(r"""
m, _ = run_experiment("00_smoke", "A", "head_only", epochs=1)
del m; gc.collect(); torch.cuda.empty_cache()
""")

# =============================================================================
md(r"""
## 15. The six submissions

Run these one cell at a time and upload each zip as it appears. If the clock gets tight,
runs 2 (LoRMA+) and 6 (distillation) are the two that are compulsory — do those first and
fill in the rest afterwards.
""")

code(r"""
# --- 1. Model A, full fine-tuning -----------------------------------------------------
m_full, r_full = run_experiment("01_A_full", "A", "full", epochs=3)
""")

code(r"""
# --- 2. Model D, LoRMA+  (COMPULSORY) -------------------------------------------------
m, _ = run_experiment("02_D_lorma_plus", "D", "lorma_plus", epochs=4)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 3. Model B, partial / layer-wise (top-3 of 6 encoder blocks) ----------------------
m, _ = run_experiment("03_B_partial", "B", "partial", epochs=3)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 4. Model C, LoRA -----------------------------------------------------------------
m, _ = run_experiment("04_C_lora", "C", "lora", epochs=4)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 5. Model A, BitFit ---------------------------------------------------------------
m, _ = run_experiment("05_A_bitfit", "A", "bitfit", epochs=4)
del m; gc.collect(); torch.cuda.empty_cache()
""")

md(r"""
### 6. Knowledge distillation *(compulsory)*

The teacher is the fully fine-tuned Model A from run 1 — a Phase-2 model, as the brief
asks, not the raw Phase-1 checkpoint. The student is the same architecture with six of the
twelve blocks, taken at evenly spaced depths so it inherits a spread of the teacher's
feature hierarchy rather than just its early layers.

The loss mixes the hard BIO cross-entropy with a temperature-softened KL against the
teacher's tag distribution. That softened distribution is what carries the useful signal:
for a token the teacher calls `B-COURT`, the mass it still places on `B-AUTH` tells the
student which confusions are reasonable — information the one-hot gold label discards
entirely. Scaling the KL term by `T^2` keeps its gradient magnitude comparable to the
cross-entropy term as `T` changes.

Report the student's macro-F1 **next to its parameter count**, not alone. The phase asks
whether entity-level performance survives compression, which is a question about the
trade-off, not about the score.
""")

code(r"""
# --- 6. Model A -> half-depth student, knowledge distillation (COMPULSORY) -------------
for p in m_full.parameters():
    p.requires_grad = False

m_student, r_kd = run_experiment("06_A_distill", "A", "full", epochs=4,
                                 keep_layers=6, teacher=m_full)

tp = sum(p.numel() for p in m_full.parameters())
sp = sum(p.numel() for p in m_student.parameters())
print(f"\nteacher {tp/1e6:.1f}M -> student {sp/1e6:.1f}M ({100*sp/tp:.0f}% of size)")
print(f"val macro-F1 {r_full['macro_f1']:.4f} -> {r_kd['macro_f1']:.4f} "
      f"({100*(r_kd['macro_f1']-r_full['macro_f1'])/max(r_full['macro_f1'],1e-9):+.1f}%)")
""")

# =============================================================================
md(r"""
## 16. Results table
""")

code(r"""
import pandas as pd
df = pd.DataFrame(RESULTS).sort_values("val_macro_f1", ascending=False)
df.to_csv("/kaggle/working/phase2_results.csv", index=False)
display(df[["name", "model", "method", "val_macro_f1", "val_p", "val_r",
            "params_M", "trainable_M", "minutes"]])
print("\nbest:", df.iloc[0]["name"], f"{df.iloc[0]['val_macro_f1']:.4f}")
print("\narchitectures covered:", sorted(df["model"].unique()))
print("methods covered:", sorted(df["method"].unique()))
""")

# =============================================================================
md(r"""
## 17. Spares and ablations

Each of these is a legitimate additional submission — above the minimum earns extra points,
and the no-bidirectional run in particular answers a question the report should address
anyway.
""")

code(r"""
# LoRMA-pi: permutation inflation instead of identity inflation
# m, _ = run_experiment("07_A_lorma_pi", "A", "lorma_pi", epochs=4)

# Ablation: does un-masking attention actually help? Same model, same method as run 1.
# m, _ = run_experiment("08_A_full_causal", "A", "full", epochs=3, bidirectional=False)

# BitFit on the already-bidirectional encoder
# m, _ = run_experiment("09_B_bitfit", "B", "bitfit", epochs=4)
""")

md(r"""
## 18. If something underperforms

- **Macro-F1 near zero while the loss falls** — offsets. Re-read the alignment diagnostic in
  section 4 and print a few decoded spans beside `text[s:e]` to see what shifted.
- **Good precision, poor recall** — raise `DROP_O_BIAS` to 0.5-1.0, or lower
  `O_CLASS_WEIGHT` to 0.1.
- **One label pinned at zero** — it is starved, and with twelve labels it is costing about
  eight points of macro-F1 by itself. Train an extra epoch or oversample windows containing
  it; that is usually worth more than any amount of tuning on the common labels.
- **Out of memory** — drop `MAX_LEN` to 384 and raise `GRAD_ACCUM` to keep the effective
  batch constant. Model C is the hungriest: its recurrence scan materialises a
  `chunk x chunk` decay matrix per chunk, so lower `BATCH_SIZE` before anything else.
- **Model C much worse than the others** — expected to some degree. Its Phase-1 checkpoint
  trained for 1800 steps against 6600 for A, B and D, so it has seen roughly a quarter of
  the tokens. Say so in the report; it is a confound in the comparison, not a property of
  the architecture.

## For the write-up

The comparison the brief actually asks for is not just which number is highest. Worth
addressing directly:

- **Does bidirectionality explain more than architecture?** B's encoder was pretrained
  bidirectionally and needs no patch; A and D are patched; C can only be patched halfway.
  If B outperforms, that is a statement about pretraining objective, not about depth or
  parameter count. Run 8 above is the controlled test.
- **Do the parameter-efficient methods reach full fine-tuning?** Compare runs 2, 4 and 5
  against run 1 on both macro-F1 and `trainable_M`.
- **Which entity types stay hard, and why?** The per-label table prints TP, FP and FN
  separately — boundary errors show up as simultaneous FP and FN on the same label, which is
  a different failure from simply missing the entity.
- **Does distillation preserve entity-level performance?** Run 6 against run 1, reported as
  score against size.
- **Confounds you should name rather than let a viva panel find**: C's shorter Phase-1
  training budget, the single validation split, and the fact that each architecture is
  paired with one method rather than crossed with all of them.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

with open("CS781_Phase2_LNER_FINAL.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("cells:", len(cells))
