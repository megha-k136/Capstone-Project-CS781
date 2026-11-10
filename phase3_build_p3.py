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
# CS-781 Capstone — Phase 3: Rhetorical Role Prediction

Sentence-level classification on **all four Phase-1 models**, with the six adaptation
techniques the phase permits.

## The six submissions

| # | Phase-1 model | Technique | Why it is distinct |
|---|---------------|-----------|--------------------|
| 1 | **A** decoder-only | **LoRMA+ (compulsory)** | multiplicative low-rank reparameterisation |
| 2 | **C** GLR hybrid | LoRA | additive low-rank reparameterisation |
| 3 | **D** GQA windowed | IA³ | learned activation rescaling, no low-rank term |
| 4 | **B** encoder-decoder | AdaLoRA | SVD parameterisation with adaptive rank budget |
| 5 | **A** | Prompt tuning | soft prompt at the input, backbone untouched |
| 6 | **A → half-depth A** | **Distillation (compulsory)** | teacher-student compression |

Phase 3 permits a **narrower** technique list than Phase 2 did — full fine-tuning, partial
fine-tuning and BitFit are **not** on it. Every method above is drawn from §9.2.1.

## Two things carried over from Phase 2, and one thing fixed

**Carried over.** Document-context windows with overlap, averaged predictions across
overlapping windows, and score-level ensembling — which was worth +5.4 macro-F1 last phase.

**Fixed.** In Phase 2, Model B used only its *encoder* as the trunk and collapsed to a
single class. The encoder never received direct loss during Phase-1 training; gradient
reached it only through cross-attention. Here B runs its **full encoder-decoder stack**, so
the decoder's strong weights do the work. That alone should keep B alive this time.

## What to run
Run cells 1-9 straight through; cell 9 is a two-minute smoke test that writes a valid zip —
**upload it immediately**. Then the six experiment cells one at a time, submitting each zip
as it appears. The minimum is six submissions and four distinct techniques; this gives you
six techniques across four architectures.
""")

# =============================================================================
md(r"""
## 1. Setup, config and data
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

# ---- pull Phase-3 data (needs the HF_TOKEN Kaggle secret, as in Phase 2) ------
if not glob.glob("/kaggle/**/CL_train.json", recursive=True):
    from huggingface_hub import snapshot_download
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        _tok = None
    snapshot_download(repo_id="Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                      allow_patterns="Phase_3_RR/*", local_dir="/kaggle/working/p3",
                      token=_tok)


class CFG:
    # ---- discovery -----------------------------------------------------------
    TOK_GLOB   = "/kaggle/input/**/bpe32k.json"
    CKPT_GLOBS = {"A": "/kaggle/input/**/A_decoder.pt",
                  "B": "/kaggle/input/**/B_encdec.pt",
                  "C": "/kaggle/input/**/C_hybrid.pt",
                  "D": "/kaggle/input/**/D_gqa.pt"}
    P3_GLOB    = "/kaggle/**/CL_train.json"

    # ---- shared Phase-1 dimensions (identical across A-D) --------------------
    DIM, N_HEADS, MAX_SEQ = 768, 12, 2048

    # ---- windowing -----------------------------------------------------------
    MAX_LEN        = 1024   # subword tokens per window
    OVERLAP_SENTS  = 4      # sentences of overlap between consecutive windows
    MAX_SENT_TOK   = 128    # truncate any single monster sentence
    BIDIRECTIONAL  = True   # un-mask causal attention; see section 4

    # ---- training ------------------------------------------------------------
    EPOCHS         = 6
    BATCH_SIZE     = 2
    GRAD_ACCUM     = 8
    LR_HEAD        = 1e-3
    LR_PEFT        = 1e-3
    WARMUP_FRAC    = 0.06
    WEIGHT_DECAY   = 0.01
    MAX_GRAD_NORM  = 1.0
    CLASS_WEIGHT_P = 0.5    # weight = (1/freq)^p ; 0 = none, 1 = full inverse
    LABEL_SMOOTH   = 0.0
    AMP            = True

    # ---- adaptation ----------------------------------------------------------
    LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.05
    LORMA_R, LORMA_ALPHA = 32, 64
    ADALORA_R, ADALORA_ALPHA = 32, 64
    ADALORA_TARGET_R = 16       # rank budget after pruning
    ADALORA_ORTH = 0.1          # orthogonality penalty weight
    PROMPT_LEN = 32

    # ---- distillation --------------------------------------------------------
    KD_T, KD_ALPHA = 2.0, 0.5

    OUT_DIR = "/kaggle/working/submissions"

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"{what} not found ({pattern}) — attach the Kaggle dataset"
    return Path(hits[0])

TOK_PATH = find_one(cfg.TOK_GLOB, "tokenizer bpe32k.json")
P3_DIR   = find_one(cfg.P3_GLOB, "Phase-3 CL_train.json").parent
CKPTS    = {k: find_one(v, f"Phase-1 model {k}") for k, v in cfg.CKPT_GLOBS.items()}
print("tokenizer :", TOK_PATH)
print("phase-3   :", P3_DIR)
for k, v in CKPTS.items():
    print(f"model {k}   : {v}  ({v.stat().st_size/1e6:.0f} MB)")
""")

# =============================================================================
md(r"""
## 2. Data

Documents are `{id, text: [sentences], labels: [one per sentence]}`. The loader is tolerant
of the shapes the organisers have used: a list of documents, or a dict keyed by id.

The official dev splits are the validation set — no need to carve one out of train.
""")

code(r"""
LABELS_FALLBACK = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute",
                   "Dissent", "PrecedentReliedUpon", "PrecedentNotReliedUpon",
                   "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
                   "RulingByPresentCourt", "None"]

def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_rr(path, domain):
    '''-> {doc_id: {"text": [...], "labels": [...] or None, "domain": domain}}'''
    raw = _read_json(path)
    items = raw if isinstance(raw, list) else [
        (v if isinstance(v, dict) else {"text": v}) | {"id": k} for k, v in raw.items()]
    out = {}
    for d in items:
        did = str(d.get("id", d.get("doc_id", d.get("document_id", len(out)))))
        txt = d.get("text", d.get("sentences", d.get("sents")))
        if isinstance(txt, str):
            txt = [txt]
        lab = d.get("labels", d.get("label", d.get("roles")))
        assert txt is not None, f"no sentence list in {did}"
        if lab is not None:
            assert len(lab) == len(txt), f"{did}: {len(lab)} labels vs {len(txt)} sentences"
        out[did] = {"text": list(txt), "labels": (list(lab) if lab is not None else None),
                    "domain": domain}
    return out

def _p(name):
    hits = sorted(glob.glob(str(P3_DIR.parent / "**" / name), recursive=True))
    assert hits, f"{name} not found under {P3_DIR.parent}"
    return hits[0]

train_docs = {**load_rr(_p("CL_train.json"), "CL"), **load_rr(_p("IT_train.json"), "IT")}
dev_docs   = {**load_rr(_p("CL_dev.json"),   "CL"), **load_rr(_p("IT_dev.json"),   "IT")}
test_docs  = {**load_rr(_p("CL_test.json"),  "CL"), **load_rr(_p("IT_test.json"),  "IT")}

try:
    lv = _read_json(_p("label_vocab.json"))
    LABELS = list(lv) if isinstance(lv, list) else sorted(lv, key=lambda k: lv[k]) \
             if all(isinstance(v, int) for v in lv.values()) else list(lv)
except Exception:
    LABELS = LABELS_FALLBACK
seen = {l for d in train_docs.values() for l in d["labels"]}
for l in sorted(seen):
    if l not in LABELS:
        LABELS.append(l)                       # never drop a label the data actually uses
LAB2ID = {l: i for i, l in enumerate(LABELS)}
ID2LAB = {i: l for l, i in LAB2ID.items()}
NUM_LABELS = len(LABELS)

n_tr = sum(len(d["text"]) for d in train_docs.values())
n_dv = sum(len(d["text"]) for d in dev_docs.values())
n_te = sum(len(d["text"]) for d in test_docs.values())
print(f"docs  train {len(train_docs)} | dev {len(dev_docs)} | test {len(test_docs)}")
print(f"sents train {n_tr} | dev {n_dv} | test {n_te}")
print(f"{NUM_LABELS} labels: {LABELS}")

CNT = Counter(l for d in train_docs.values() for l in d["labels"])
for l in LABELS:
    c = CNT.get(l, 0)
    print(f"  {l:<24}{c:>7}  {100*c/max(n_tr,1):>6.2f}%" + ("   <- UNSEEN" if c == 0 else ""))
dom = Counter(d["domain"] for d in train_docs.values())
print("domains:", dict(dom))
""")

# =============================================================================
md(r"""
## 3. Tokenising, windowing, dataset

Each sentence is encoded on its own, then sentences are **packed greedily** into windows of
at most `MAX_LEN` tokens, with `OVERLAP_SENTS` sentences of overlap between consecutive
windows. Packing by sentence rather than by token guarantees no sentence is ever split, so
every sentence gets at least one window in which it is whole.

A sentence's representation is the **mean of its own token hidden states**, taken after the
backbone has seen the surrounding sentences. That is what makes this context-aware rather
than sentence-only: the pooling is local, the encoding is not. Cell 13 runs the sentence-only
ablation the brief asks for, by setting `OVERLAP_SENTS` aside and packing one sentence per
window.

Where a sentence appears in more than one window, its log-probabilities are averaged — the
same trick that acted as a free ensemble in Phase 2.
""")

code(r"""
from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(TOK_PATH))
VOCAB_SIZE = tok.get_vocab_size()
PAD_ID = int(tok.token_to_id("<|pad|>") or 0)
print(f"vocab {VOCAB_SIZE:,} | pad {PAD_ID}")

def encode_doc(sents):
    out = []
    for s in sents:
        ids = tok.encode(" " + s.strip()).ids[:cfg.MAX_SENT_TOK] or [PAD_ID]
        out.append(ids)
    return out

def build_windows(sent_ids, one_per_window=False):
    '''-> list of (first_sentence_index, [sentence indices in this window])'''
    n = len(sent_ids)
    if one_per_window:
        return [[i] for i in range(n)]
    wins, i = [], 0
    while i < n:
        j, tot = i, 0
        while j < n and (tot + len(sent_ids[j]) <= cfg.MAX_LEN or j == i):
            tot += len(sent_ids[j]); j += 1
        wins.append(list(range(i, j)))
        if j >= n:
            break
        i = max(j - cfg.OVERLAP_SENTS, i + 1)
    return wins


class RRWindows(Dataset):
    def __init__(self, docs, with_labels=True, one_per_window=False):
        self.items, self.n_sent, self.docs = [], {}, docs
        for did, d in docs.items():
            sent_ids = encode_doc(d["text"])
            self.n_sent[did] = len(d["text"])
            labs = d["labels"] if (with_labels and d["labels"] is not None) else None
            for slots in build_windows(sent_ids, one_per_window):
                ids, spos = [], []
                for k, si in enumerate(slots):
                    ids.extend(sent_ids[si]); spos.extend([k] * len(sent_ids[si]))
                y = ([LAB2ID.get(labs[si], -100) for si in slots] if labs is not None
                     else [-100] * len(slots))
                self.items.append({"did": did, "ids": ids, "spos": spos,
                                   "slots": slots, "y": y})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    B = len(batch)
    T = max(len(b["ids"]) for b in batch)
    S = max(len(b["slots"]) for b in batch)
    ids  = torch.full((B, T), PAD_ID, dtype=torch.long)
    spos = torch.full((B, T), -1, dtype=torch.long)
    y    = torch.full((B, S), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n]  = torch.tensor(b["ids"], dtype=torch.long)
        spos[i, :n] = torch.tensor(b["spos"], dtype=torch.long)
        y[i, :len(b["y"])] = torch.tensor(b["y"], dtype=torch.long)
    return {"input_ids": ids, "spos": spos, "labels": y, "n_slots": S,
            "meta": [{"did": b["did"], "slots": b["slots"]} for b in batch]}

ds_train = RRWindows(train_docs)
ds_dev   = RRWindows(dev_docs)
ds_test  = RRWindows(test_docs, with_labels=False)
print(f"windows -> train {len(ds_train)} | dev {len(ds_dev)} | test {len(ds_test)}")
print("mean sentences/window:",
      round(np.mean([len(b['slots']) for b in ds_train.items]), 1))
""")

# =============================================================================
md(r"""
## 4. Phase-1 architectures

Reproduced from your Phase-1 notebooks so every checkpoint loads with `strict=True`. The only
additions are the `causal` / `bidirectional` flags, which default to the trained behaviour and
change no parameter name or shape.

**Why un-mask attention.** A causal model cannot see the sentence that follows. For rhetorical
role that is crippling: a sentence is `RulingByPresentCourt` partly because nothing argues
after it, and `Issue` partly because arguments follow. Un-masking lets each sentence see the
whole window in both directions. Model B's encoder was already trained bidirectionally; the
other three are patched. Model C can only be patched in its four attention layers — its eight
gated-recurrence layers have direction built into the operator and no mask to flip — so C
stays **partly** causal, which is a finding to report, not a defect to hide.
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
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        fr = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv)
        self.register_buffer("cos", fr.cos(), persistent=False)
        self.register_buffer("sin", fr.sin(), persistent=False)
    def forward(self, x, pos0=0):
        dtype = x.dtype
        B, H, T, D = x.shape
        half = D // 2
        x = x.float()
        x1, x2 = x[..., :half], x[..., half:]
        cos = self.cos[pos0:pos0 + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[pos0:pos0 + T].unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq, rope=True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope     = RoPE(self.head_dim, max_seq) if rope else None
        self.causal   = True
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, C))


class GeneralizedAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq, causal=True, cross=False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.causal, self.cross = causal, cross
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
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
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=(self.causal and not self.cross))
        return self.out_proj(o.transpose(1, 2).reshape(B, T, C))


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
    def forward(self, x, enc):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context=enc)
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
            log_a = F.pad(log_a, (0, 0, 0, pad)); u = F.pad(u, (0, 0, 0, pad))
        n = (T + pad) // L
        log_a = log_a.view(B, n, L, D); u = u.view(B, n, L, D)
        A = log_a.cumsum(2)
        d = A.unsqueeze(3) - A.unsqueeze(2)
        tri = torch.tril(torch.ones(L, L, device=u.device, dtype=torch.bool))
        d = d.masked_fill(~tri[None, None, :, :, None], float("-inf"))
        local = torch.einsum("bnijd,bnjd->bnid", torch.exp(d), u)
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
        self.register_buffer("bi_full_mask", torch.ones_like(causal), persistent=False)
        self.register_buffer("bi_window_mask", d.abs() < window_size, persistent=False)
        self.bidirectional = False
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        if self.bidirectional:
            m = self.bi_window_mask if self.sliding_window else self.bi_full_mask
        else:
            m = self.window_mask if self.sliding_window else self.causal_mask
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=m[:T, :T], is_causal=False,
            enable_gqa=(self.n_kv_heads != self.n_heads))
        return self.out_proj(o.transpose(1, 2).reshape(B, T, C))


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
## 5. The four models

Each exposes `trunk(ids)` and `trunk_embeds(x)` returning hidden states `(B, T, H)`.
`trunk_embeds` is what prompt tuning needs — it skips the token embedding so a soft prompt
can be prepended.

**B uses its full encoder-decoder stack**, not the encoder alone. The encoder reads the
window, the decoder reads the same window while cross-attending to the encoder output, and
the decoder's final norm gives the representation. This is the fix for Phase 2's collapse.
""")

code(r"""
DIM, N_HEADS, MAX_SEQ = cfg.DIM, cfg.N_HEADS, cfg.MAX_SEQ


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, max_seq):
        super().__init__()
        self.norm1 = RMSNorm(dim); self.attn = CausalSelfAttention(dim, n_heads, max_seq)
        self.norm2 = RMSNorm(dim); self.mlp  = SwiGLU(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPTModel(nn.Module):
    def __init__(self, vocab_size, dim=DIM, n_layers=12, n_heads=N_HEADS, max_seq=MAX_SEQ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(dim, n_heads, max_seq) for _ in range(n_layers)])
        self.norm_f  = RMSNorm(dim)
        self.head    = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def trunk_embeds(self, x):
        for b in self.blocks:
            x = b(x)
        return self.norm_f(x)
    def trunk(self, idx):
        return self.trunk_embeds(self.tok_emb(idx))


class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, dim=DIM, n_encoder_layers=6, n_decoder_layers=6,
                 n_heads=N_HEADS, max_seq=MAX_SEQ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(dim, n_heads, max_seq) for _ in range(n_encoder_layers)])
        self.norm_e = RMSNorm(dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(dim, n_heads, max_seq) for _ in range(n_decoder_layers)])
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def trunk_embeds(self, x):
        e = x
        for b in self.encoder_blocks:
            e = b(e)
        e = self.norm_e(e)
        d = x
        for b in self.decoder_blocks:
            d = b(d, e)
        return self.norm_f(d)
    def trunk(self, idx):
        return self.trunk_embeds(self.tok_emb(idx))


class GenericLM(nn.Module):
    def __init__(self, mixer_factory, vocab_size, dim=DIM, n_layers=12, max_seq=MAX_SEQ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks  = nn.ModuleList(
            [GenericBlock(dim, mixer_factory(i)) for i in range(n_layers)])
        self.norm_f  = RMSNorm(dim)
        self.head    = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def trunk_embeds(self, x):
        for b in self.blocks:
            x = b(x)
        return self.norm_f(x)
    def trunk(self, idx):
        return self.trunk_embeds(self.tok_emb(idx))


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

def trunk_blocks(m, arch):
    return m.decoder_blocks if arch == "B" else m.blocks

def load_phase1(arch, pretrained=True):
    m = _build(arch)
    if pretrained:
        sd = torch.load(CKPTS[arch], map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        miss, unex = m.load_state_dict(
            {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()},
            strict=True)
        assert not miss and not unex
    return m

ARCH_INFO = {"A": "decoder-only, 12L", "B": "encoder-decoder, 6+6L (full stack)",
             "C": "GLR + attention hybrid, 12L", "D": "GQA sliding-window, 12L"}
print("builders ready:", ", ".join(f"{k} ({v})" for k, v in ARCH_INFO.items()))
""")

# =============================================================================
md(r"""
## 6. The sentence-classification model

Hidden states are mean-pooled over each sentence's own tokens, then classified. The pooling is
a scatter-add over the sentence index, so it is one vectorised operation regardless of how many
sentences a window holds.
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
            if mod.causal:                       # decoder self-attention only
                mod.causal = not on; n += 1
    return n


class LegalRR(nn.Module):
    def __init__(self, backbone, arch, hidden, n_labels, dropout=0.1):
        super().__init__()
        self.backbone, self.arch, self.hidden = backbone, arch, hidden
        self.prompt = None                       # populated by prompt tuning
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_labels))

    def encode(self, input_ids):
        x = self.backbone.tok_emb(input_ids)
        if self.prompt is not None:
            p = self.prompt.unsqueeze(0).expand(x.size(0), -1, -1).to(x.dtype)
            h = self.backbone.trunk_embeds(torch.cat([p, x], 1))
            return h[:, self.prompt.size(0):]    # drop the prompt positions
        return self.backbone.trunk_embeds(x)

    def forward(self, input_ids, spos, n_slots):
        h = self.encode(input_ids)               # (B, T, H)
        B, T, H = h.shape
        valid = (spos >= 0).unsqueeze(-1)
        idx = spos.clamp(min=0).unsqueeze(-1).expand(-1, -1, H)
        P = h.new_zeros(B, n_slots, H).scatter_add_(1, idx, h * valid)
        C = h.new_zeros(B, n_slots, 1).scatter_add_(
            1, spos.clamp(min=0).unsqueeze(-1), valid.to(h.dtype))
        return self.head(self.drop(P / C.clamp(min=1e-6)))   # (B, S, n_labels)


def build_rr(arch, bidirectional=None, keep_layers=None, pretrained=True, verbose=True):
    bb = load_phase1(arch, pretrained=pretrained)
    if keep_layers is not None:                  # student for distillation
        blocks = trunk_blocks(bb, arch)
        keep = np.linspace(0, len(blocks) - 1, keep_layers).round().astype(int).tolist()
        new = nn.ModuleList([blocks[i] for i in keep])
        if arch == "B": bb.decoder_blocks = new
        else:           bb.blocks = new
        if verbose: print(f"  student trunk {len(blocks)} -> {keep_layers} blocks {keep}")
    bb.head = nn.Identity()                      # the LM head is never used
    bi = cfg.BIDIRECTIONAL if bidirectional is None else bidirectional
    n = set_bidirectional(bb, bi)
    m = LegalRR(bb, arch, cfg.DIM, NUM_LABELS).to(DEVICE)
    if verbose:
        tot = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
        print(f"  [{arch}] {ARCH_INFO[arch]} | {tot/1e6:.1f}M params | "
              f"bidirectional={bi} ({n} mixers switched)")
    return m
""")

# =============================================================================
md(r"""
## 7. Adaptation techniques

All six are drawn from the list §9.2.1 permits. Full fine-tuning, partial fine-tuning and
BitFit are **not** on that list and are deliberately absent.

### LoRA — additive low-rank
A frozen weight gets a trainable correction beside it, `W + (alpha/r)·BA`, with `B` zero so
training starts exactly at the pretrained function. Trainable count falls from `d·k` to
`r·(d+k)`.

### LoRMA — multiplicative low-rank *(compulsory)*
LoRMA (Exploration-Lab, ACL Findings 2025 — the lab running this course) replaces the additive
update with a multiplicative one, `W = (alpha/r)·(BA)·W`, transforming the weight rather than
offsetting it. Two problems follow and rank inflation fixes both: a bare product caps the
adapted weight at rank `r`, and there is no zero initialisation because the neutral element of
a product is the identity. **LoRMA+** adds an identity, `W = (I + (alpha/r)·BA)·W`, which is
exactly `W` at initialisation and stays near full rank throughout. The implementation never
materialises `BA`: it computes `y = Wx` then `y + (yAᵀ)Bᵀ`, two thin products.

### IA³ — learned rescaling
No low-rank term at all. A learned vector multiplies the keys, the values and the feed-forward
gate, element-wise, initialised to ones. It is the smallest of the methods by parameter count
and a genuinely different mechanism: it reweights existing features instead of rotating weight
subspaces, so it cannot represent a direction the pretrained model does not already carry.

### AdaLoRA — adaptive rank budget
The update is written in singular-value form, `P·diag(λ)·Q`, with an orthogonality penalty
keeping `P` and `Q` near-orthonormal so the triple behaves like a real SVD. Every singular
value is scored by sensitivity (`|λ·∂L/∂λ|`, smoothed), and partway through training the
lowest-scoring ones are masked to reach a target budget. The point is that ranks are not spread
evenly: a layer that needs more capacity keeps more directions, one that needs less gives them up.

### Prompt tuning
The backbone is frozen entirely and a short sequence of trainable embeddings is prepended to
the input. The only parameters are the prompt and the classification head, so it is the most
parameter-efficient method here by a wide margin — and the most likely to underperform, which
is itself worth reporting.

### Knowledge distillation *(compulsory, cell 14)*
Teacher-to-student transfer on softened logits, covered at its own cell.
""")

code(r"""
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale, self.drop = alpha / r, nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t()) @ self.B.t() * self.scale


class LoRMALinear(nn.Module):
    # W = (I + (alpha/r) B A) W0   -- identity (additive) rank inflation
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
    # dW = P diag(lam) Q, with orthogonality penalty and importance-based rank masking
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        self.Q = nn.Parameter(torch.randn(r, d_in) * 0.02)
        self.P = nn.Parameter(torch.randn(d_out, r) * 0.02)
        self.lam = nn.Parameter(torch.zeros(r))       # zero -> starts at W0
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
            s = (self.lam * self.lam.grad).abs()
            self.imp.mul_(beta).add_((1 - beta) * s)


def adalora_housekeeping(model, step, total):
    '''EMA the sensitivities every step; prune once, at the half-way point.'''
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
            kept = 0
            for m in mods:
                m.mask.copy_((m.imp >= thr).float())
                kept += int(m.mask.sum().item())
            print(f"    [adalora] pruned to {kept}/{allimp.numel()} singular values "
                  f"(budget {budget})")


class IA3Scale(nn.Module):
    '''Learned per-feature rescaling of a frozen Linear's output.
    `slice_frac` scales only the trailing fraction of the output (used for a fused QKV,
    where only the K and V thirds should be rescaled).'''
    def __init__(self, base, slice_frac=1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        d = base.out_features
        self.n = int(d * slice_frac)
        self.scale = nn.Parameter(torch.ones(self.n))
        self.full = (self.n == d)
    def forward(self, x):
        y = self.base(x)
        if self.full:
            return y * self.scale
        # cat, never an in-place slice write: the slice is a view of y, and assigning
        # into it bumps y's version counter and breaks the backward pass
        return torch.cat([y[..., :-self.n], y[..., -self.n:] * self.scale], dim=-1)


MIXERS = (CausalSelfAttention, GeneralizedAttention,
          GQASlidingWindowAttention, GatedLinearRecurrence)

def mixer_linears(model):
    '''Every nn.Linear that is a direct child of a token mixer. Selected by module type,
    never by name -- a keyword match would drag in the feed-forward weights too.'''
    for mod in model.modules():
        if isinstance(mod, MIXERS):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    yield mod, cname, child


def apply_adaptation(model, method):
    bb = model.backbone
    for p in bb.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True

    if method == "head_only":
        pass

    elif method == "lora":
        for parent, cn, lin in list(mixer_linears(bb)):
            setattr(parent, cn, LoRALinear(lin, cfg.LORA_R, cfg.LORA_ALPHA,
                                           cfg.LORA_DROPOUT).to(DEVICE))

    elif method == "lorma":
        for parent, cn, lin in list(mixer_linears(bb)):
            setattr(parent, cn, LoRMALinear(lin, cfg.LORMA_R, cfg.LORMA_ALPHA).to(DEVICE))

    elif method == "adalora":
        for parent, cn, lin in list(mixer_linears(bb)):
            setattr(parent, cn, AdaLoRALinear(lin, cfg.ADALORA_R,
                                              cfg.ADALORA_ALPHA).to(DEVICE))

    elif method == "ia3":
        for parent, cn, lin in list(mixer_linears(bb)):
            if cn in ("k_proj", "v_proj"):
                setattr(parent, cn, IA3Scale(lin).to(DEVICE))
            elif cn == "qkv":                       # fused: rescale K and V only
                setattr(parent, cn, IA3Scale(lin, slice_frac=2/3).to(DEVICE))
            elif cn == "in_proj":                   # GLR: rescale the gated projection
                setattr(parent, cn, IA3Scale(lin).to(DEVICE))
        for blk in trunk_blocks(bb, model.arch):    # plus the feed-forward gate
            mlp = getattr(blk, "mlp", None)
            if mlp is not None and isinstance(mlp.w1, nn.Linear):
                mlp.w1 = IA3Scale(mlp.w1).to(DEVICE)

    elif method == "prompt":
        model.prompt = nn.Parameter(
            bb.tok_emb.weight[torch.randint(0, VOCAB_SIZE, (cfg.PROMPT_LEN,))]
            .detach().clone().to(DEVICE))

    else:
        raise ValueError(method)

    tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"  [{method}] trainable {tr/1e6:.3f}M / {tot/1e6:.1f}M = {100*tr/tot:.3f}%")
    return model
""")

# =============================================================================
md(r"""
## 8. Training, evaluation and the submission writer

Macro-F1 weights every label equally however rare it is, so a single label stuck at zero costs
about eight points on its own. `CLASS_WEIGHT_P` raises each class's loss weight by the inverse
of its frequency to the power `p` — `0` is unweighted, `1` is full inverse frequency, and `0.5`
is the usual compromise. This was the single biggest lever in Phase 2 and it is likely to be
again, since the rhetorical roles are severely imbalanced.
""")

code(r"""
def class_weights():
    w = torch.ones(NUM_LABELS)
    if cfg.CLASS_WEIGHT_P > 0:
        tot = sum(CNT.values())
        for i, l in enumerate(LABELS):
            f = max(CNT.get(l, 0), 1) / tot
            w[i] = (1.0 / f) ** cfg.CLASS_WEIGHT_P
        w = w / w.mean()
    return w.to(DEVICE)


def make_optimizer(model):
    head_ids = {id(p) for p in model.head.parameters()}
    g = [{"params": [p for p in model.head.parameters() if p.requires_grad],
          "lr": cfg.LR_HEAD, "weight_decay": cfg.WEIGHT_DECAY},
         {"params": [p for p in model.parameters()
                     if p.requires_grad and id(p) not in head_ids],
          "lr": cfg.LR_PEFT, "weight_decay": cfg.WEIGHT_DECAY}]
    return torch.optim.AdamW([x for x in g if x["params"]], betas=(0.9, 0.95), eps=1e-8)


def make_sched(opt, total):
    warm = max(1, int(cfg.WARMUP_FRAC * total))
    def fn(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train(model, ds, epochs=None, teacher=None, tag="", sampler=None):
    epochs = epochs or cfg.EPOCHS
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=(sampler is None),
                    sampler=sampler, collate_fn=collate, num_workers=2, pin_memory=True)
    steps = max(1, (len(dl) // cfg.GRAD_ACCUM) * epochs)
    opt, sch = make_optimizer(model), None
    sch = make_sched(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and USE_AMP)
    lossf = nn.CrossEntropyLoss(weight=class_weights(), ignore_index=-100,
                                label_smoothing=cfg.LABEL_SMOOTH)
    adal = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    if teacher is not None:
        teacher.eval()

    model.train(); t0 = time.time(); gstep = 0
    for ep in range(epochs):
        run = nb = 0
        for i, b in enumerate(dl):
            ids = b["input_ids"].to(DEVICE, non_blocking=True)
            sp  = b["spos"].to(DEVICE, non_blocking=True)
            y   = b["labels"].to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
                logits = model(ids, sp, b["n_slots"])
                loss = lossf(logits.reshape(-1, NUM_LABELS), y.reshape(-1))
                if adal:
                    loss = loss + cfg.ADALORA_ORTH * sum(m.orth_penalty() for m in adal)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(ids, sp, b["n_slots"])
                    sel = (y != -100).reshape(-1)
                    T = cfg.KD_T
                    kd = F.kl_div(
                        F.log_softmax(logits.reshape(-1, NUM_LABELS)[sel].float() / T, -1),
                        F.softmax(tl.reshape(-1, NUM_LABELS)[sel].float() / T, -1),
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
def predict_logprobs(model, ds):
    '''-> {doc_id: (n_sentences, NUM_LABELS) mean log-probs across overlapping windows}'''
    model.eval()
    dl = DataLoader(ds, batch_size=max(2, cfg.BATCH_SIZE * 2), shuffle=False,
                    collate_fn=collate, num_workers=2)
    acc = {d: np.zeros((n, NUM_LABELS), np.float32) for d, n in ds.n_sent.items()}
    cnt = {d: np.zeros((n, 1), np.float32) for d, n in ds.n_sent.items()}
    for b in dl:
        ids = b["input_ids"].to(DEVICE); sp = b["spos"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
            lp = F.log_softmax(model(ids, sp, b["n_slots"]).float(), -1).cpu().numpy()
        for bi, meta in enumerate(b["meta"]):
            A, C = acc[meta["did"]], cnt[meta["did"]]
            for k, si in enumerate(meta["slots"]):
                A[si] += lp[bi, k]; C[si, 0] += 1
    return {d: acc[d] / np.maximum(cnt[d], 1.0) for d in acc}


def decode(scores):
    return {d: [ID2LAB[i] for i in S.argmax(-1)] for d, S in scores.items()}


def macro_f1(pred, docs, domain=None):
    tp, fp, fn = Counter(), Counter(), Counter()
    for did, d in docs.items():
        if domain is not None and d["domain"] != domain:
            continue
        for p, g in zip(pred[did], d["labels"]):
            if p == g: tp[g] += 1
            else:      fp[p] += 1; fn[g] += 1
    rows, Ps, Rs, Fs = [], [], [], []
    for l in LABELS:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        Fm = 2 * P * R / (P + R) if P + R else 0.0
        rows.append((l, tp[l], fp[l], fn[l], P, R, Fm))
        Ps.append(P); Rs.append(R); Fs.append(Fm)
    return {"macro_f1": float(np.mean(Fs)), "macro_p": float(np.mean(Ps)),
            "macro_r": float(np.mean(Rs)), "per_label": rows}


def report(res, pred=None):
    print(f"  {'label':<24}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for l, a, b_, c, P, R, Fm in res["per_label"]:
        print(f"  {l:<24}{a:>6}{b_:>6}{c:>6}{P:>8.3f}{R:>8.3f}{Fm:>8.3f}")
    print(f"  {'MACRO':<24}{'':>18}{res['macro_p']:>8.3f}"
          f"{res['macro_r']:>8.3f}{res['macro_f1']:>8.3f}")
    z = [r[0] for r in res["per_label"] if r[-1] == 0.0]
    if z:
        print(f"  !! zero-F1 ({len(z)}): {z}  -> each costs ~{100/NUM_LABELS:.0f}% of macro")
    if pred is not None:
        for dm in ("CL", "IT"):
            r = macro_f1(pred, dev_docs, domain=dm)
            print(f"  domain {dm}: macro-F1 {r['macro_f1']:.4f}")


def write_submission(pred, name):
    payload = {}
    for did, d in test_docs.items():
        labs = pred.get(did)
        assert labs is not None, f"missing predictions for {did}"
        assert len(labs) == len(d["text"]), \
            f"{did}: {len(labs)} predictions vs {len(d['text'])} sentences"
        assert all(l in LAB2ID for l in labs), f"{did}: unknown label"
        payload[did] = list(labs)
    assert set(payload) == set(test_docs)

    jp = os.path.join(cfg.OUT_DIR, f"prediction_{name}.json")
    zp = os.path.join(cfg.OUT_DIR, f"submission_{name}.zip")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")     # root of the archive, no folder
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
    n = sum(len(v) for v in payload.values())
    print(f"  -> {zp}  | {len(payload)} docs | {n} sentences | "
          f"{os.path.getsize(zp)/1e3:.0f} KB")
    print(f"     mix: {Counter(l for v in payload.values() for l in v).most_common()}")
    return zp


RESULTS, SCORES = [], {}

def run_experiment(name, arch, method, epochs=None, bidirectional=None,
                   keep_layers=None, teacher=None, write=True, seed=42, keep_scores=True):
    print(f"\n{'='*72}\n{name}   model {arch} | {method}\n{'='*72}", flush=True)
    set_seed(seed); t0 = time.time()
    model = build_rr(arch, bidirectional=bidirectional, keep_layers=keep_layers)
    model = apply_adaptation(model, method)
    train(model, ds_train, epochs=epochs, teacher=teacher, tag=name)

    sv = predict_logprobs(model, ds_dev)
    res = macro_f1(decode(sv), dev_docs)
    report(res, decode(sv))

    st = predict_logprobs(model, ds_test)
    zp = write_submission(decode(st), name) if write else None
    if keep_scores:
        SCORES[name] = {"dev": sv, "test": st}

    RESULTS.append({"name": name, "model": arch, "method": method,
                    "dev_macro_f1": round(res["macro_f1"], 4),
                    "dev_p": round(res["macro_p"], 4), "dev_r": round(res["macro_r"], 4),
                    "CL": round(macro_f1(decode(sv), dev_docs, "CL")["macro_f1"], 4),
                    "IT": round(macro_f1(decode(sv), dev_docs, "IT")["macro_f1"], 4),
                    "trainable_M": round(sum(p.numel() for p in model.parameters()
                                             if p.requires_grad) / 1e6, 3),
                    "minutes": round((time.time() - t0) / 60, 1), "zip": zp})
    print("\n " + json.dumps(RESULTS[-1]))
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, res
""")

# =============================================================================
md(r"""
## 9. SMOKE TEST — run this first

Two to three minutes with a frozen backbone. If it prints a zip path, the whole pipeline is
sound: checkpoint loading, windowing, sentence pooling, training, scoring and packaging.
**Upload that zip to Codabench immediately** — the score will be poor, but it banks one of
the six and starts the 15-30 minute evaluation clock while the real models train.
""")

code(r"""
m, _ = run_experiment("00_smoke", "A", "head_only", epochs=1, keep_scores=False)
del m; gc.collect(); torch.cuda.empty_cache()
""")

# =============================================================================
md(r"""
## 10. The six submissions

One cell each, deliberately. Upload every zip the moment it appears rather than batching them
to the end — Codabench evaluation is slow and the per-day cap is generous. If time runs short,
runs 1 (LoRMA) and 6 (distillation) are the two the brief makes compulsory.
""")

code(r"""
# --- 1. Model A, LoRMA+  (COMPULSORY technique) --------------------------------------
m_teacher, r_lorma = run_experiment("01_A_lorma", "A", "lorma", epochs=6)
""")

code(r"""
# --- 2. Model C, LoRA ----------------------------------------------------------------
m, _ = run_experiment("02_C_lora", "C", "lora", epochs=6)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 3. Model D, IA3 -----------------------------------------------------------------
m, _ = run_experiment("03_D_ia3", "D", "ia3", epochs=8)   # tiny param count, wants longer
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 4. Model B, AdaLoRA (full encoder-decoder trunk) --------------------------------
m, _ = run_experiment("04_B_adalora", "B", "adalora", epochs=6)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 5. Model A, prompt tuning -------------------------------------------------------
m, _ = run_experiment("05_A_prompt", "A", "prompt", epochs=8)
del m; gc.collect(); torch.cuda.empty_cache()
""")

md(r"""
### 6. Knowledge distillation *(compulsory)*

The teacher is the trained Phase-3 model from run 1 — a fine-tuned model, as §9.2.2 asks, not
a raw Phase-1 checkpoint. The student is the same architecture with six of its twelve blocks,
taken at evenly spaced depths so it inherits a spread of the teacher's feature hierarchy
rather than only its early layers.

The loss mixes hard cross-entropy with a temperature-softened KL against the teacher's label
distribution. That softened distribution carries the signal the one-hot label throws away: for
a sentence the teacher calls `RatioOfTheDecision`, the mass it still places on
`RulingByPresentCourt` tells the student which confusions are reasonable — and those two are
exactly the pair that is hardest to separate. Scaling the KL by `T²` keeps its gradient
comparable to the cross-entropy term as `T` changes.

Report the student's macro-F1 **next to its parameter count**. The question §9.5 asks is
whether prediction performance survives compression, which is about the trade-off, not the score.
""")

code(r"""
# --- 6. Model A -> half-depth student, knowledge distillation (COMPULSORY) -----------
for p in m_teacher.parameters():
    p.requires_grad = False

m_student, r_kd = run_experiment("06_A_distill", "A", "lora", epochs=6,
                                 keep_layers=6, teacher=m_teacher)

tp = sum(p.numel() for p in {id(p): p for p in m_teacher.parameters()}.values())
sp = sum(p.numel() for p in {id(p): p for p in m_student.parameters()}.values())
print(f"\nteacher {tp/1e6:.1f}M -> student {sp/1e6:.1f}M ({100*sp/tp:.0f}% of size)")
print(f"dev macro-F1 {r_lorma['macro_f1']:.4f} -> {r_kd['macro_f1']:.4f} "
      f"({100*(r_kd['macro_f1']-r_lorma['macro_f1'])/max(r_lorma['macro_f1'],1e-9):+.1f}%)")
del m_student; gc.collect(); torch.cuda.empty_cache()
""")

# =============================================================================
md(r"""
## 11. Results, ensemble and the context ablation

The ensemble averages per-sentence log-probabilities across models. In Phase 2 this was worth
+5.4 macro-F1 over the best single model, because the architectures make uncorrelated errors —
there is no reason that should be weaker here.
""")

code(r"""
import pandas as pd
df = pd.DataFrame(RESULTS).sort_values("dev_macro_f1", ascending=False)
df.to_csv("/kaggle/working/phase3_results.csv", index=False)
display(df[["name", "model", "method", "dev_macro_f1", "CL", "IT",
            "trainable_M", "minutes"]])
print("architectures:", sorted(df["model"].unique()))
print("techniques   :", sorted(df["method"].unique()))

# ---- score-level ensemble ------------------------------------------------------------
# A weak member drags the average down, so try the top-k by dev score for several k and
# keep whichever wins. All of this is free: the scores are already cached.
ranked = [r["name"] for r in sorted(RESULTS, key=lambda r: -r["dev_macro_f1"])
          if r["name"] in SCORES]
best_single = max(r["dev_macro_f1"] for r in RESULTS)

best = (None, best_single)
for k in range(2, len(ranked) + 1):
    use = ranked[:k]
    e = {d: np.mean([SCORES[n]["dev"][d] for n in use], 0) for d in ds_dev.n_sent}
    f1 = macro_f1(decode(e), dev_docs)["macro_f1"]
    print(f"  top-{k:<2} {f1:.4f}   {use}")
    if f1 > best[1]:
        best = (use, f1)

print(f"\nbest single {best_single:.4f} -> best ensemble {best[1]:.4f}")
if best[0]:
    ens_dev = {d: np.mean([SCORES[n]["dev"][d] for n in best[0]], 0) for d in ds_dev.n_sent}
    report(macro_f1(decode(ens_dev), dev_docs), decode(ens_dev))
    ens_test = {d: np.mean([SCORES[n]["test"][d] for n in best[0]], 0)
                for d in ds_test.n_sent}
    write_submission(decode(ens_test), "07_ensemble")
else:
    print("  no ensemble beat the best single model — submit that one instead")
""")

code(r"""
# ---- the context ablation the brief asks for (§9.1): sentence-only vs document context
ds_train_solo = RRWindows(train_docs, one_per_window=True)
ds_dev_solo   = RRWindows(dev_docs,   one_per_window=True)
ds_test_solo  = RRWindows(test_docs,  with_labels=False, one_per_window=True)

_tr, _dv, _te = ds_train, ds_dev, ds_test
ds_train, ds_dev, ds_test = ds_train_solo, ds_dev_solo, ds_test_solo
m, r_solo = run_experiment("08_A_lorma_nocontext", "A", "lorma", epochs=6, keep_scores=False)
ds_train, ds_dev, ds_test = _tr, _dv, _te

print(f"\ncontext ablation: document-context {r_lorma['macro_f1']:.4f} "
      f"vs sentence-only {r_solo['macro_f1']:.4f}")
del m; gc.collect(); torch.cuda.empty_cache()
""")

md(r"""
## 12. For the write-up

§9.5 asks eight specific questions. The runs above answer seven of them directly:

- **Which technique works best** — the results table, ranked.
- **Whether parameter-efficient methods approach full adaptation** — note that Phase 3 bars
  full fine-tuning, so the honest comparison is *across* the efficient methods, by macro-F1
  against `trainable_M`. Prompt tuning is the floor of that spectrum and IA³ the next step up.
- **Whether context helps** — cell 18, document-context against sentence-only, same model and
  method, one variable changed.
- **Which roles are hard** — the per-label table. Watch `RatioOfTheDecision` against
  `RulingByPresentCourt`, and the three precedent labels against each other; simultaneous high
  FP and FN on a pair is confusion between them, not failure to find them.
- **CL against IT** — the per-domain lines under every report.
- **Whether distillation preserves performance** — run 6, score against parameter count.
- **Whether Phase-1 representations transfer** — the smoke test is the control: a frozen
  backbone with only a head trained. Whatever it scores is what the pretrained features give
  you before any adaptation at all.

Declare these three confounds before a panel finds them: Model C's Phase-1 checkpoint trained
for 1800 steps against 6600 for the others, so it has seen roughly a quarter of the tokens;
each architecture is paired with one technique rather than crossed with all six; and Model C
is only partly bidirectional, since its recurrence layers cannot be un-masked.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

with open("CS781_Phase3_RR.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("cells:", len(cells))
