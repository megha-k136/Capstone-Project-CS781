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
# CS-781 Capstone — Phase 6: Abstractive Legal Summarization

The first **generation** task in the capstone. Everything before this produced labels or
rankings; this produces text, which changes the training objective, the inference loop and
what can go wrong.

## The five experiments

| # | Phase-1 model | Technique | |
|---|---|---|---|
| 1 | **B** encoder-decoder | **LoRMA+** | compulsory |
| 2 | **A** decoder-only | LoRA | |
| 3 | **C** GLR hybrid | Houlsby adapters | |
| 4 | **D** GQA windowed | Prompt tuning | |
| 5 | **B → half-depth B** | **Distillation** | compulsory |

**The permitted list changed again.** §12.2.1 drops full fine-tuning, BitFit, partial
fine-tuning and IA³ — every one of which you used in an earlier phase — and puts **LoRA**
back on, alongside Houlsby and Pfeiffer adapters separately. All five techniques above are
drawn from that list; nothing from Phase 4, 5 or 7 carries over unchecked.

## Four things that are new

**No dev split is provided.** §12.3.2 says to carve one out of train yourself. Cell 3 holds
out a fixed slice by hashed id, so the split is identical on every re-run.

**Two different model shapes.** B is a real encoder-decoder, so it summarises the way it was
built to: encode the judgment, decode the summary with cross-attention. A, C and D are
decoder-only, so the document and summary are concatenated into one sequence and the loss is
masked to the summary tokens. One class handles both.

**Models stay causal.** Phases 2 through 7 un-masked attention because classification reads a
whole span at once. Generation cannot — a decoder that sees the future trains on an answer it
won't have at inference. There is no bidirectional patch in this notebook, by design.

**Generation is the expensive part.** These architectures have no key-value cache, so every
decoding step re-runs the forward pass over the whole prefix. B is far cheaper here than the
others, because its encoder runs once per document and only the short summary is re-decoded.
Budget accordingly; the smoke test reports a measured rate.

## Run order
Cells 1-4, then **submit the lead baseline immediately** — extractive leads are a famously
strong baseline on BERTScore for legal summarization, and it costs nothing. Then the smoke
test, then the five runs.
""")

# =============================================================================
md(r"""
## 1. Setup and config
""")

code(r"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, json, math, time, random, zipfile, gc, glob, re
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

# ---- pull Phase-6 data (needs the HF_TOKEN Kaggle secret) ---------------------------
if not glob.glob("/kaggle/**/Phase_6_Summ/**/test.json", recursive=True):
    from huggingface_hub import snapshot_download
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        _tok = None
    snapshot_download(repo_id="Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                      allow_patterns="Phase_6_Summ/*", local_dir="/kaggle/working/p6",
                      token=_tok)


class CFG:
    TOK_GLOB   = "/kaggle/input/**/bpe32k.json"
    CKPT_GLOBS = {"A": "/kaggle/input/**/A_decoder.pt",
                  "B": "/kaggle/input/**/B_encdec.pt",
                  "C": "/kaggle/input/**/C_hybrid.pt",
                  "D": "/kaggle/input/**/D_gqa.pt"}
    P6_GLOB    = "/kaggle/**/Phase_6_Summ/**/test.json"

    DIM, N_HEADS, MAX_SEQ = 768, 12, 2048

    # ---- lengths ------------------------------------------------------------
    SRC_LEN    = 1024    # document tokens fed to encoder-decoder B
    SRC_LEN_DEC = 768    # document tokens in the decoder-only concat (A/C/D)
    TGT_LEN    = 256     # reference summary tokens kept for training
    MAX_NEW    = 200     # tokens generated at inference
    MIN_NEW    = 40
    NO_REPEAT  = 3       # block repeated n-grams of this size
    VAL_FRAC   = 0.08    # held-out slice of train (no dev split is provided)
    VAL_CAP    = 150     # cap on val size -- generation is slow

    # ---- training -----------------------------------------------------------
    EPOCHS      = 3
    MAX_STEPS   = 1500
    BATCH_SIZE  = 2
    GRAD_ACCUM  = 8
    LR_PEFT     = 1e-3
    LR_HEAD     = 1e-3
    WARMUP_FRAC = 0.06
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    LABEL_SMOOTH = 0.1
    AMP         = True
    GEN_BS      = 8

    # ---- adaptation ---------------------------------------------------------
    LORMA_R, LORMA_ALPHA   = 32, 64
    LORA_R,  LORA_ALPHA    = 32, 64
    ADALORA_R, ADALORA_ALPHA, ADALORA_TARGET_R, ADALORA_ORTH = 32, 64, 16, 0.1
    ADAPTER_DIM = 64
    N_PROMPT    = 48
    KD_T, KD_ALPHA = 2.0, 0.5

    C_CHUNK_OVERRIDE = 8     # C's scan blocking -- lower = less memory, slower
    OUT_DIR = "/kaggle/working/submissions"

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"{what} not found ({pattern}) -- attach the Kaggle dataset"
    return Path(hits[0])

TOK_PATH = find_one(cfg.TOK_GLOB, "tokenizer bpe32k.json")
P6_TEST  = find_one(cfg.P6_GLOB, "Phase-6 test.json")
P6_ROOT  = P6_TEST.parent.parent
CKPTS    = {k: find_one(v, f"Phase-1 model {k}") for k, v in cfg.CKPT_GLOBS.items()}
print("tokenizer :", TOK_PATH)
print("phase-6   :", P6_ROOT)
for k, v in CKPTS.items():
    print(f"model {k}   : {v}  ({v.stat().st_size/1e6:.0f} MB)")
""")

# =============================================================================
md(r"""
## 2. Data and the held-out split

Examples are `{id, document, summary}`. Phase 6 ships no dev split, so one is carved from
train here. The split is taken by a **hash of the document id**, not by `random.shuffle`, so
it is byte-identical across kernel restarts and across the separate runs below — otherwise
run 3 would be scored on documents run 1 trained on, and the comparison between techniques
would be meaningless.

The compression ratio this prints is the number to read. It tells you how long the generated
summaries should be, and generation length is one of the strongest levers on both ROUGE and
BERTScore — too short loses recall, too long dilutes precision.
""")

code(r"""
def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _join(t):
    if isinstance(t, list):
        return " ".join(map(str, t))
    return str(t or "")

def _norm_items(raw):
    if isinstance(raw, list):
        return raw
    return [((v if isinstance(v, dict) else {"document": v}) | {"id": k})
            for k, v in raw.items()]

def load_summ(path, with_summary=True):
    out = {}
    for d in _norm_items(_read_json(path)):
        did = str(d.get("id", d.get("doc_id", d.get("case_id", len(out)))))
        doc = _join(d.get("document", d.get("text", d.get("case_text", ""))))
        rec = {"document": doc}
        s = d.get("summary", d.get("abstract", d.get("target")))
        if with_summary and s is not None:
            rec["summary"] = _join(s)
        out[did] = rec
    return out

def _p(name):
    hits = sorted(glob.glob(str(P6_ROOT / "**" / name), recursive=True))
    assert hits, f"{name} not found under {P6_ROOT}"
    return hits[0]

all_train = load_summ(_p("train.json"))
test_docs = load_summ(_p("test.json"), with_summary=False)
all_train = {k: v for k, v in all_train.items()
             if v.get("summary", "").strip() and v["document"].strip()}

# deterministic held-out slice, by hashed id
import hashlib
def _bucket(did):
    return int(hashlib.md5(str(did).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

val_ids = sorted([d for d in all_train if _bucket(d) < cfg.VAL_FRAC])[: cfg.VAL_CAP]
val_set = set(val_ids)
train_docs = {d: v for d, v in all_train.items() if d not in val_set}
val_docs   = {d: all_train[d] for d in val_ids}

print(f"train {len(train_docs)} | held-out val {len(val_docs)} | test {len(test_docs)}")

dw = np.array([len(v["document"].split()) for v in train_docs.values()])
sw = np.array([len(v["summary"].split())  for v in train_docs.values()])
print(f"\ndocument words : median {int(np.median(dw))} p90 {int(np.percentile(dw,90))} "
      f"max {int(dw.max())}")
print(f"summary  words : median {int(np.median(sw))} p90 {int(np.percentile(sw,90))} "
      f"max {int(sw.max())}")
print(f"compression    : median {np.median(sw)/max(np.median(dw),1)*100:.1f}% "
      f"of the document")
print(f"\n-> MAX_NEW={cfg.MAX_NEW} tokens is about {int(cfg.MAX_NEW*0.75)} words; "
      f"the median reference is {int(np.median(sw))} words")
""")

# =============================================================================
md(r"""
## 3. Scoring and the submission writer

ROUGE is implemented here in a few lines rather than pulled from a package, so the notebook
has no dependency that might be missing on a fresh kernel. ROUGE-1 is unigram overlap F1 and
ROUGE-L is built on the longest common subsequence.

The board metric is **BERTScore F1**, which this notebook does not compute: it needs a
pretrained embedding model downloaded at runtime, which is both unavailable with the network
locked down and the sort of thing to keep well away from a competition that forbids external
pretrained weights. ROUGE is used for model selection instead. The two disagree at the margin
— BERTScore rewards paraphrase that ROUGE misses — so treat a ROUGE gap under about a point
as noise rather than a result.
""")

code(r"""
_W = re.compile(r"[a-z0-9]+")
def _toks(s):
    return _W.findall(s.lower())

def _f1(overlap, np_, nr):
    if not overlap or not np_ or not nr:
        return 0.0
    p, r = overlap / np_, overlap / nr
    return 2 * p * r / (p + r)

def rouge1(pred, ref):
    pc, rc = Counter(_toks(pred)), Counter(_toks(ref))
    return _f1(sum((pc & rc).values()), sum(pc.values()), sum(rc.values()))

def rougeL(pred, ref, cap=600):
    a, b = _toks(pred)[:cap], _toks(ref)[:cap]
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return _f1(prev[-1], len(a), len(b))

def score_summaries(preds, refs):
    r1 = [rouge1(preds[d], refs[d]["summary"]) for d in refs]
    rl = [rougeL(preds[d], refs[d]["summary"]) for d in refs]
    wl = [len(_toks(preds[d])) for d in refs]
    return {"rouge1": float(np.mean(r1)), "rougeL": float(np.mean(rl)),
            "mean_words": float(np.mean(wl)),
            "empty": int(sum(1 for w in wl if w == 0))}

def report(s, label=""):
    print(f"  {label}ROUGE-1 {s['rouge1']:.4f} | ROUGE-L {s['rougeL']:.4f} | "
          f"{s['mean_words']:.0f} words/summary" +
          (f" | {s['empty']} EMPTY" if s["empty"] else ""))
    return s


RESULTS, PREDS = [], {}

def write_submission(preds, name):
    '''Phase-6 format: a JSON LIST of {"id": str, "summary": str} (§12.3.5 option A).'''
    payload = []
    for did in test_docs:
        txt = (preds.get(did) or "").strip()
        if not txt:                      # never submit an empty string
            txt = " ".join(test_docs[did]["document"].split()[:60])
        payload.append({"id": str(did), "summary": txt})
    assert isinstance(payload, list) and len(payload) == len(test_docs)
    assert {r["id"] for r in payload} == {str(d) for d in test_docs}, "test id mismatch"
    for r in payload:
        assert set(r) == {"id", "summary"}, r
        assert isinstance(r["summary"], str) and r["summary"].strip(), r["id"]

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
    mw = np.mean([len(_toks(r["summary"])) for r in payload])
    print(f"  -> {zp}  | {len(payload)} summaries | {mw:.0f} words each")
    return zp
""")

# =============================================================================
md(r"""
## 4. The lead baseline — submit this first

Take the first *k* sentences of the judgment and call them the summary. It sounds like
cheating and it is famously hard to beat: the opening of a legal document states the matter
at issue in the document's own vocabulary, and both ROUGE and BERTScore reward exactly that
overlap.

It costs no GPU and no training, so it banks a real score in under a minute. It is **not** one
of your four required experiments — those all start from a Phase-1 model — but it is the
reference every neural run has to clear to be worth reporting, and §12.5 asks you to say
whether the Phase-1 representations help at all. This is how you answer that honestly.
""")

code(r"""
_SENT = re.compile(r"(?<=[.;:!?])\s+")

def sentences(text):
    s = [x.strip() for x in _SENT.split(text) if x.strip()]
    return s or [text.strip() or "."]

def lead(text, k, max_words=None):
    out = " ".join(sentences(text)[:k])
    if max_words:
        out = " ".join(out.split()[:max_words])
    return out

best_lead = (None, 0.0)
for k in (3, 5, 8, 12, 18, 25):
    s = score_summaries({d: lead(v["document"], k) for d, v in val_docs.items()}, val_docs)
    report(s, label=f"lead-{k:<3} ")
    if s["rouge1"] > best_lead[1]:
        best_lead = (k, s["rouge1"])

# a word cap on top of the sentence count, since sentence length varies wildly
for w in (100, 150, 200, 300):
    s = score_summaries({d: lead(v["document"], 40, w) for d, v in val_docs.items()},
                        val_docs)
    report(s, label=f"lead-{w}w ")
    if s["rouge1"] > best_lead[1]:
        best_lead = (("w", w), s["rouge1"])

sel = best_lead[0]
mk = (lambda t: lead(t, 40, sel[1])) if isinstance(sel, tuple) else (lambda t: lead(t, sel))
print(f"\n-> best lead variant {sel} at ROUGE-1 {best_lead[1]:.4f}")
write_submission({d: mk(v["document"]) for d, v in test_docs.items()}, "00_lead")
RESULTS.append({"name": "00_lead", "model": "-", "method": "extractive lead",
                "rouge1": round(best_lead[1], 4), "minutes": 0.0})
print("\nUPLOAD THIS NOW, then carry on.")
""")

# =============================================================================
md(r"""
## 5. Tokenising and the two input shapes

**B (encoder-decoder)** gets the document in the encoder and the summary in the decoder,
shifted right: input `<bos> y1 … yn`, target `y1 … yn <eos>`. Cross-attention does the rest.
This is what the architecture was pretrained to do.

**A, C and D (decoder-only)** have no cross-attention, so document and summary are
concatenated into one sequence, `document <bos> summary <eos>`, and the loss is masked to
**-100 everywhere except the summary tokens**. Without that mask the model spends most of its
capacity learning to predict the judgment, which is not the task.

Both are built by one dataset class; `seq2seq` picks the shape.
""")

code(r"""
from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(TOK_PATH))
VOCAB_SIZE = tok.get_vocab_size()
def _tid(name, default):
    v = tok.token_to_id(name)
    return default if v is None else int(v)
PAD_ID = _tid("<|pad|>", 0)
BOS_ID = _tid("<|bos|>", 2)
EOS_ID = _tid("<|eos|>", 3)
print(f"vocab {VOCAB_SIZE:,} | pad {PAD_ID} bos {BOS_ID} eos {EOS_ID}")

CHAR_CAP = 12000
def enc_all(texts, maxlen, block=256):
    out = []
    for i in range(0, len(texts), block):
        for e in tok.encode_batch([t[:CHAR_CAP] for t in texts[i:i + block]]):
            out.append((e.ids or [EOS_ID])[:maxlen])
    return out

t0 = time.time()
_all = {**train_docs, **val_docs}
DOC = dict(zip(_all, enc_all([v["document"] for v in _all.values()], cfg.SRC_LEN)))
SUM = dict(zip(_all, enc_all([v["summary"]  for v in _all.values()], cfg.TGT_LEN)))
DOC_TE = dict(zip(test_docs,
                  enc_all([v["document"] for v in test_docs.values()], cfg.SRC_LEN)))
print(f"tokenised in {time.time()-t0:.0f}s")


class SummDS(Dataset):
    def __init__(self, ids, seq2seq):
        self.ids, self.seq2seq = list(ids), seq2seq
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        did = self.ids[i]
        doc, summ = DOC[did], SUM[did]
        if self.seq2seq:
            return {"src": doc, "inp": [BOS_ID] + summ, "lab": summ + [EOS_ID]}
        d = doc[: cfg.SRC_LEN_DEC]
        seq = d + [BOS_ID] + summ + [EOS_ID]
        return {"src": None, "inp": seq[:-1],
                "lab": [-100] * len(d) + (seq[len(d) + 1:])}


def _pad(seqs, fill):
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), fill, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out

def collate(batch):
    b = {"inp": _pad([x["inp"] for x in batch], PAD_ID),
         "lab": _pad([x["lab"] for x in batch], -100)}
    if batch[0]["src"] is not None:
        b["src"] = _pad([x["src"] for x in batch], PAD_ID)
    return b


TR_IDS, VA_IDS = list(train_docs), list(val_docs)
for s2s in (True, False):
    _b = collate([SummDS(TR_IDS, s2s)[i] for i in range(2)])
    sup = int((_b["lab"] != -100).sum())
    print(f"seq2seq={s2s}: inp {tuple(_b['inp'].shape)}"
          + (f" src {tuple(_b['src'].shape)}" if "src" in _b else "")
          + f" | {sup} supervised target tokens")
""")

# =============================================================================
md(r"""
## 6. Phase-1 architecture components

Reproduced from your Phase-1 notebooks so every checkpoint loads with `strict=True`.

Note what is **absent**: there is no call to `set_bidirectional` anywhere in this notebook.
A generator must not see its own future tokens, so A, C and D stay causal and B's decoder
self-attention stays causal. B's encoder was pretrained bidirectional and stays that way,
which is correct — it reads the source, it does not generate it.
""")

code(ARCH_COMPONENTS)

# =============================================================================
md(r"""
## 7. The four models
""")

code(ARCH_MODELS)

# =============================================================================
md(r"""
## 8. The summariser, and decoding

One wrapper over both shapes. The language-model head is the Phase-1 tied head
(`head.weight is tok_emb.weight`) — unlike the classification phases, it is kept, because the
output vocabulary *is* the task.

**Decoding.** Greedy, with three guards that matter far more for a small from-scratch model
than the choice of search:

- **No-repeat n-gram blocking.** Left alone, a 110M model trained on a few thousand summaries
  falls into loops within thirty tokens. Banning any trigram that has already appeared is the
  cheapest effective fix.
- **A minimum length.** Early `<eos>` is the other common failure, and a two-word summary
  scores near zero. `<eos>` is suppressed until `MIN_NEW` tokens have been produced.
- **No key-value cache.** These architectures never implemented one, so each step re-runs the
  prefix. For B that prefix is just the summary so far and the encoder output is computed
  once; for A, C and D it is the whole document plus the summary, every step. That is why B
  generates several times faster than the others and why `SRC_LEN_DEC` is shorter than
  `SRC_LEN`.
""")

code(r"""
class SummModel(nn.Module):
    def __init__(self, backbone, arch, n_prompt=0):
        super().__init__()
        self.backbone, self.arch = backbone, arch
        self.seq2seq = (arch == "B")
        self.n_prompt = n_prompt
        if n_prompt:
            self.prompt = nn.Parameter(torch.randn(n_prompt, cfg.DIM) * 0.02)

    def _prepend(self, x):
        if not self.n_prompt:
            return x
        p = self.prompt.unsqueeze(0).expand(x.size(0), -1, -1).to(x.dtype)
        return torch.cat([p, x], 1)

    def encode(self, src):
        bb = self.backbone
        e = self._prepend(bb.tok_emb(src))
        for b in bb.encoder_blocks:
            e = b(e)
        return bb.norm_e(e)

    def decode(self, inp, enc):
        bb = self.backbone
        d = bb.tok_emb(inp)
        for b in bb.decoder_blocks:
            d = b(d, enc)
        return bb.head(bb.norm_f(d))

    def forward(self, inp, src=None):
        if self.seq2seq:
            return self.decode(inp, self.encode(src))
        bb = self.backbone
        x = self._prepend(bb.tok_emb(inp))
        for b in bb.blocks:
            x = b(x)
        out = bb.head(bb.norm_f(x))
        return out[:, self.n_prompt:] if self.n_prompt else out


def _block_repeats(logits, seqs, n):
    '''Ban any token that would complete an n-gram already present in the sequence.'''
    if n <= 0:
        return logits
    for i, s in enumerate(seqs):
        if len(s) < n:
            continue
        ctx = tuple(s[-(n - 1):]) if n > 1 else ()
        for j in range(len(s) - n + 1):
            if tuple(s[j:j + n - 1]) == ctx:
                logits[i, s[j + n - 1]] = -float("inf")
    return logits


@torch.no_grad()
def generate(model, doc_ids, max_new=None, min_new=None, no_repeat=None):
    model.eval()
    max_new = max_new or cfg.MAX_NEW
    min_new = min_new if min_new is not None else cfg.MIN_NEW
    no_repeat = cfg.NO_REPEAT if no_repeat is None else no_repeat
    B = len(doc_ids)
    amp = dict(device_type="cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP)

    if model.seq2seq:
        src = _pad(doc_ids, PAD_ID).to(DEVICE)
        with torch.autocast(**amp):
            enc = model.encode(src)
        cur = [[BOS_ID] for _ in range(B)]
    else:
        cur = [list(d[: cfg.SRC_LEN_DEC]) + [BOS_ID] for d in doc_ids]
        plen = [len(c) for c in cur]

    gen = [[] for _ in range(B)]
    alive = list(range(B))
    for step in range(max_new):
        if not alive:
            break
        ids = _pad([cur[i] for i in alive], PAD_ID).to(DEVICE)
        with torch.autocast(**amp):
            if model.seq2seq:
                lg = model.decode(ids, enc[alive])
            else:
                lg = model(ids)
        last = torch.stack([lg[r, len(cur[i]) - 1] for r, i in enumerate(alive)]).float()
        last = _block_repeats(last, [gen[i] for i in alive], no_repeat)
        if step < min_new:
            last[:, EOS_ID] = -float("inf")
        last[:, PAD_ID] = -float("inf")
        nxt = last.argmax(-1).tolist()
        still = []
        for r, i in enumerate(alive):
            t = nxt[r]
            if t == EOS_ID:
                continue
            gen[i].append(t); cur[i].append(t)
            still.append(i)
        alive = still
    return [tok.decode(g).strip() for g in gen]


def generate_all(model, ids, store, tag="", bs=None):
    bs = bs or cfg.GEN_BS
    order = sorted(range(len(ids)), key=lambda i: len(store[ids[i]]))
    out, t0 = {}, time.time()
    for s in range(0, len(order), bs):
        sel = [ids[i] for i in order[s:s + bs]]
        for did, txt in zip(sel, generate(model, [store[d] for d in sel])):
            out[did] = txt
        if s and s % (bs * 10) == 0:
            done = s + bs
            print(f"    {tag} {done}/{len(ids)} ({time.time()-t0:.0f}s, "
                  f"{(time.time()-t0)/done:.2f}s/doc)", flush=True)
    print(f"    {tag} generated {len(out)} in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/max(len(ids),1):.2f}s/doc)", flush=True)
    return out


def build_summ(arch, keep_layers=None, n_prompt=0, pretrained=True, verbose=True):
    global C_CHUNK
    if arch == "C" and cfg.C_CHUNK_OVERRIDE:
        C_CHUNK = cfg.C_CHUNK_OVERRIDE
    bb = load_phase1(arch, pretrained=pretrained)
    if keep_layers is not None:
        blocks = trunk_blocks(bb, arch)
        keep = np.linspace(0, len(blocks) - 1, keep_layers).round().astype(int).tolist()
        new = nn.ModuleList([blocks[i] for i in keep])
        if arch == "B": bb.decoder_blocks = new
        else:           bb.blocks = new
        if verbose: print(f"  student trunk {len(blocks)} -> {keep_layers} blocks {keep}")
    m = SummModel(bb, arch, n_prompt=n_prompt).to(DEVICE)
    if verbose:
        tot = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
        print(f"  [{arch}] {ARCH_INFO[arch]} | {tot/1e6:.1f}M params | "
              f"{'seq2seq' if m.seq2seq else 'decoder-only concat'}"
              + (f" | {n_prompt} soft tokens" if n_prompt else ""))
    return m
""")

# =============================================================================
md(r"""
## 9. Adaptation techniques

All five are on §12.2.1's list. **Full fine-tuning, partial fine-tuning, BitFit and IA³ are
not on it** — they were permitted in earlier phases and are deliberately absent here.

### LoRMA — multiplicative low-rank *(compulsory)*
`W = (I + (alpha/r)·BA)·W`, computed as `y = Wx` then `y + (yAᵀ)Bᵀ` so `BA` is never
materialised. Zero-initialised `B` means it starts exactly at the pretrained weight.

### LoRA
The additive counterpart, `W + (alpha/r)·BA`, back on the permitted list for the first time
since Phase 3. Running it beside LoRMA on the same task is the cleanest comparison in the
notebook: identical rank, identical placement, additive versus multiplicative.

### Houlsby adapters
Bottleneck modules after **both** the attention output and the feed-forward block — two per
layer. Zero-initialised up-projection, so each starts as the identity.

### Pfeiffer adapters
The same module after the feed-forward block **only** — one per layer, roughly half the
parameters. §12.2.1 lists the two separately, so the Houlsby/Pfeiffer pair is a second free
comparison if you have time for it.

### Prompt tuning
Trainable embedding vectors prepended to the input. The backbone never moves.

### Knowledge distillation *(compulsory, section 12)*
Covered at its own cell.
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
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
    def forward(self, x):
        y = self.base(x)
        return y + ((y @ self.A.t()) @ self.B.t()) * self.scale


class LoRALinear(nn.Module):
    # W + (alpha/r) B A   -- the additive form
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + ((x @ self.A.t()) @ self.B.t()) * self.scale


class AdapterWrap(nn.Module):
    # bottleneck adapter, zero-init up-projection so it starts as the identity
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
    for mod in model.modules():
        if isinstance(mod, MIXERS):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    yield mod, cname, child


def apply_adaptation(model, method):
    bb = model.backbone
    for p in bb.parameters(): p.requires_grad = False
    if model.n_prompt:
        model.prompt.requires_grad = True

    if method == "prompt":
        pass

    elif method in ("lorma", "lora"):
        cls = LoRMALinear if method == "lorma" else LoRALinear
        r = cfg.LORMA_R if method == "lorma" else cfg.LORA_R
        a = cfg.LORMA_ALPHA if method == "lorma" else cfg.LORA_ALPHA
        for parent, cn, lin in list(mixer_linears(bb)):
            setattr(parent, cn, cls(lin, r, a).to(DEVICE))

    elif method in ("houlsby", "pfeiffer"):
        if method == "houlsby":
            for parent, cn, lin in list(mixer_linears(bb)):
                if cn == "out_proj":
                    setattr(parent, cn, AdapterWrap(lin, cfg.ADAPTER_DIM).to(DEVICE))
        for mlp in [m for m in bb.modules()
                    if isinstance(m, SwiGLU) and isinstance(m.w3, nn.Linear)]:
            mlp.w3 = AdapterWrap(mlp.w3, cfg.ADAPTER_DIM).to(DEVICE)

    else:
        raise ValueError(method)

    tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"  [{method}] trainable {tr/1e6:.3f}M / {tot/1e6:.1f}M = {100*tr/tot:.3f}%")
    assert tr > 0, f"{method} left nothing trainable"
    return model
""")

# =============================================================================
md(r"""
## 10. Training and the experiment driver

Cross-entropy over the target tokens, with `ignore_index=-100` so the decoder-only models are
scored only on the summary, and label smoothing at 0.1 — worth having here because a small
model trained on few examples otherwise becomes overconfident and its greedy decoding turns
repetitive.

Each run trains, generates on the held-out split, scores with ROUGE, then generates on test
and writes the zip. The generation pass is the expensive half, so the smoke test's reported
seconds-per-document is what you should budget from, not the training time.
""")

code(r"""
def make_optimizer(model):
    ps = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(ps, lr=cfg.LR_PEFT, betas=(0.9, 0.95), eps=1e-8,
                             weight_decay=cfg.WEIGHT_DECAY)

def make_sched(opt, total):
    warm = max(1, int(cfg.WARMUP_FRAC * total))
    def fn(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train(model, epochs=None, teacher=None, tag=""):
    epochs = epochs or cfg.EPOCHS
    ds = SummDS(TR_IDS, model.seq2seq)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate,
                    num_workers=2, pin_memory=True)
    planned = min(cfg.MAX_STEPS, max(1, (len(dl) // cfg.GRAD_ACCUM) * epochs))
    opt, sch = make_optimizer(model), None
    sch = make_sched(opt, planned)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and USE_AMP)
    if teacher is not None:
        teacher.eval()

    model.train(); t0 = time.time(); gstep = 0; stop = False
    for ep in range(epochs):
        run = nb = 0
        for i, b in enumerate(dl):
            inp = b["inp"].to(DEVICE, non_blocking=True)
            lab = b["lab"].to(DEVICE, non_blocking=True)
            src = b["src"].to(DEVICE, non_blocking=True) if "src" in b else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
                lg = model(inp, src=src)
                n = min(lg.size(1), lab.size(1))
                loss = F.cross_entropy(lg[:, :n].reshape(-1, lg.size(-1)).float(),
                                       lab[:, :n].reshape(-1), ignore_index=-100,
                                       label_smoothing=cfg.LABEL_SMOOTH)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(inp, src=src)
                    T = cfg.KD_T
                    m = (lab[:, :n] != -100).reshape(-1)
                    sl = lg[:, :n].reshape(-1, lg.size(-1))[m].float()
                    tt = tl[:, :n].reshape(-1, tl.size(-1))[m].float()
                    kd = F.kl_div(F.log_softmax(sl / T, -1), F.softmax(tt / T, -1),
                                  reduction="batchmean") * (T * T)
                    loss = (1 - cfg.KD_ALPHA) * loss + cfg.KD_ALPHA * kd
            scaler.scale(loss / cfg.GRAD_ACCUM).backward()
            run += loss.item(); nb += 1
            if (i + 1) % cfg.GRAD_ACCUM == 0 or i + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.MAX_GRAD_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sch.step(); gstep += 1
                if gstep >= cfg.MAX_STEPS:
                    stop = True; break
            if nb % 200 == 0:
                print(f"    ep{ep+1} {nb}/{len(dl)} loss {run/nb:.4f} "
                      f"lr {sch.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        print(f"  [{tag}] epoch {ep+1}/{epochs} loss {run/max(nb,1):.4f} "
              f"step {gstep}/{planned} ({time.time()-t0:.0f}s)", flush=True)
        if stop:
            print(f"  [{tag}] hit MAX_STEPS={cfg.MAX_STEPS}"); break
    return model


def run_experiment(name, arch, method, epochs=None, keep_layers=None, teacher=None,
                   write=True, seed=42, n_val=None):
    print(f"\n{'='*72}\n{name}   model {arch} | {method}\n{'='*72}", flush=True)
    set_seed(seed); t0 = time.time()
    n_prompt = cfg.N_PROMPT if method == "prompt" else 0
    model = build_summ(arch, keep_layers=keep_layers, n_prompt=n_prompt)
    model = apply_adaptation(model, method)
    train(model, epochs=epochs, teacher=teacher, tag=name)

    vids = VA_IDS[: (n_val or len(VA_IDS))]
    pv = generate_all(model, vids, DOC, tag="val")
    PREDS[name + "__val"] = pv
    s = report(score_summaries(pv, {d: val_docs[d] for d in vids}), label="val  ")
    print("\n  --- sample ---")
    for d in vids[:2]:
        print(f"  GEN: {pv[d][:300]}")
        print(f"  REF: {val_docs[d]['summary'][:300]}\n")

    zp = None
    if write:
        pt = generate_all(model, list(test_docs), DOC_TE, tag="test")
        PREDS[name] = pt
        zp = write_submission(pt, name)

    RESULTS.append({"name": name, "model": arch, "method": method,
                    "rouge1": round(s["rouge1"], 4), "rougeL": round(s["rougeL"], 4),
                    "words": round(s["mean_words"], 1),
                    "trainable_M": round(sum(p.numel() for p in model.parameters()
                                             if p.requires_grad) / 1e6, 3),
                    "minutes": round((time.time() - t0) / 60, 1), "zip": zp})
    print(" " + json.dumps(RESULTS[-1]))
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, s
""")

# =============================================================================
md(r"""
## 11. SMOKE TEST — run this first

A frozen backbone, 30 steps, 12 validation documents, no test pass. It proves the pipeline
end to end and, more usefully, **measures seconds per generated document**. Multiply that by
the test-set size before launching any full run — on these cacheless architectures the
generation pass, not the training, is what will blow your budget.
""")

code(r"""
_ms = cfg.MAX_STEPS
cfg.MAX_STEPS = 30
m, _ = run_experiment("01_smoke", "B", "prompt", epochs=1, write=False, n_val=12)
cfg.MAX_STEPS = _ms
del m; gc.collect(); torch.cuda.empty_cache()
print(f"\nMAX_STEPS back to {cfg.MAX_STEPS}")
print(f"-> at the rate above, one test pass over {len(test_docs)} docs is the number to plan "
      f"from; B is the fastest of the four, A/C/D will be slower.")
""")

# =============================================================================
md(r"""
## 12. The five experiments

Upload each zip as it appears. Runs 1 and 5 are the compulsory pair.
""")

code(r"""
# --- 1. Model B, LoRMA+  (COMPULSORY technique, and the natural summariser) ----------
m_teacher, r1 = run_experiment("02_B_lorma", "B", "lorma", epochs=3)
""")

code(r"""
# --- 2. Model A, LoRA ----------------------------------------------------------------
m, _ = run_experiment("03_A_lora", "A", "lora", epochs=3)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 3. Model C, Houlsby adapters ----------------------------------------------------
_bs = cfg.BATCH_SIZE; cfg.BATCH_SIZE = 1; cfg.GRAD_ACCUM = 16   # C's scan is memory-hungry
m, _ = run_experiment("04_C_houlsby", "C", "houlsby", epochs=3)
cfg.BATCH_SIZE, cfg.GRAD_ACCUM = _bs, 8
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 4. Model D, prompt tuning -------------------------------------------------------
m, _ = run_experiment("05_D_prompt", "D", "prompt", epochs=4)   # tiny param count
del m; gc.collect(); torch.cuda.empty_cache()
""")

md(r"""
### 5. Knowledge distillation *(compulsory)*

The teacher is the **trained** summariser from run 1, not a raw Phase-1 checkpoint, as
§12.2.2 requires. The student is the same encoder-decoder with three of its six decoder
blocks, taken at evenly spaced depths.

The signal is token-level: the student's next-token distribution is pulled toward the
teacher's temperature-softened one, restricted to the supervised summary positions. For
generation this carries much more than it did in the binary phases — at every position the
teacher expresses which *other* words would also have been acceptable, and that distribution
over near-synonyms is exactly what the hard reference token discards. It is also why
distillation tends to hurt generation less than it hurts classification.
""")

code(r"""
# --- 5. Model B -> half-decoder student, knowledge distillation (COMPULSORY) --------
for p in m_teacher.parameters():
    p.requires_grad = False

m_student, r5 = run_experiment("06_B_distill", "B", "lorma", epochs=3,
                               keep_layers=3, teacher=m_teacher)

tp = sum(p.numel() for p in {id(p): p for p in m_teacher.parameters()}.values())
sp = sum(p.numel() for p in {id(p): p for p in m_student.parameters()}.values())
print(f"\nteacher {tp/1e6:.1f}M -> student {sp/1e6:.1f}M ({100*sp/tp:.0f}% of size)")
print(f"val ROUGE-1 {r1['rouge1']:.4f} -> {r5['rouge1']:.4f}")
del m_student; gc.collect(); torch.cuda.empty_cache()
""")

# =============================================================================
md(r"""
## 13. Results, and choosing what to submit

Two things worth doing before you stop.

**Length tuning is free.** Truncating every generated summary to a fixed word count, swept on
the held-out split, costs no GPU and regularly moves ROUGE by a point or two — the model's
natural stopping point is rarely the one that maximises the metric. The same sweep applies to
the lead baseline.

**Check the lead baseline honestly.** If an extractive lead beats all five neural runs, say so
in §12.5 rather than burying it. On a from-scratch 110M model trained on a few thousand
summaries that is a realistic outcome, and "the Phase-1 representations did not beat a lead
baseline on this task" is a finding, supported by five controlled comparisons.
""")

code(r"""
import pandas as pd
df = pd.DataFrame(RESULTS).sort_values("rouge1", ascending=False)
df.to_csv("/kaggle/working/phase6_results.csv", index=False)
display(df[[c for c in ["name","model","method","rouge1","rougeL","words",
                        "trainable_M","minutes"] if c in df]])
_n = [r for r in RESULTS if r["model"] != "-"]
print("architectures:", sorted({r["model"] for r in _n}))
print("techniques   :", sorted({r["method"] for r in _n}))

# ---- length sweep, tuned on the held-out split, applied to test ---------------------
best = max(_n, key=lambda r: r["rouge1"])["name"]
vp = PREDS.get(best + "__val")
if vp:
    vrefs = {d: val_docs[d] for d in vp}
    bw = (None, score_summaries(vp, vrefs)["rouge1"])
    print(f"\nlength sweep on {best} (val, uncapped {bw[1]:.4f})")
    for w in (60, 80, 100, 120, 160, 200, 260, 400):
        s = score_summaries({d: " ".join(t.split()[:w]) for d, t in vp.items()}, vrefs)
        print(f"  cap {w:<4} ROUGE-1 {s['rouge1']:.4f} | {s['mean_words']:.0f} words")
        if s["rouge1"] > bw[1]:
            bw = (w, s["rouge1"])
    if bw[0]:
        print(f"\n-> cap {bw[0]} words improves val {bw[1]:.4f}; writing a capped test zip")
        write_submission({d: " ".join(PREDS[best][d].split()[:bw[0]])
                          for d in test_docs}, f"07_{best}_cap{bw[0]}")
    else:
        print("\n-> no cap beat the uncapped summaries; keep the original zip")
else:
    print("\n(no stored val predictions for the best run -- skipping the length sweep)")
""")

md(r"""
## 14. Spares, if time allows

```python
# Pfeiffer adapters -- half the parameters of Houlsby, same placement minus the attention one
m, _ = run_experiment("08_B_pfeiffer", "B", "pfeiffer", epochs=3)

# LoRA vs LoRMA on the SAME architecture -- the cleanest ablation in the notebook
m, _ = run_experiment("09_B_lora", "B", "lora", epochs=3)

# longer source: does the model use more of the judgment, or drown in it?
cfg.SRC_LEN = 1536
DOC = dict(zip(_all, enc_all([v["document"] for v in _all.values()], cfg.SRC_LEN)))
DOC_TE = dict(zip(test_docs,
                  enc_all([v["document"] for v in test_docs.values()], cfg.SRC_LEN)))
m, _ = run_experiment("10_B_long", "B", "lorma", epochs=3)

# generation length: the cheapest lever on both ROUGE and BERTScore
cfg.MAX_NEW, cfg.MIN_NEW = 320, 80
m, _ = run_experiment("11_B_longgen", "B", "lorma", epochs=3)
```
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

with open("CS781_Phase6_Summ.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("cells:", len(cells))
