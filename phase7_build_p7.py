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
# CS-781 Capstone — Phase 7: Prior Case Retrieval

A **bi-encoder**: the Phase-1 model encodes a query case and every candidate case into one
vector space, and relevance is cosine similarity. Trained with multi-positive InfoNCE over
in-batch and BM25-mined hard negatives.

## The five experiments

| # | Phase-1 model | Technique | |
|---|---|---|---|
| 1 | **A** decoder-only | **LoRMA+** | compulsory |
| 2 | **B** encoder-decoder | Full fine-tuning | |
| 3 | **C** GLR hybrid | IA³ | |
| 4 | **D** GQA windowed | Prompt tuning | |
| 5 | **A → half-depth A** | **Distillation** | compulsory |

Phase 7's permitted list (§13.2.1) is different again: **LoRA, AdaLoRA and adapters are not on
it**, while IA³ returns and QLoRA/prefix/prompt tuning are available. All five above are on it.

## Three things that are new this phase

**The output format is a dict again.** Phases 4 and 5 wanted a JSON list of objects. Phase 7
wants `{"query_id": ["cand1", ..., "cand7"]}` — exactly 7 unique candidate IDs per query,
ordered best-first, drawn only from `test_candidates.json`. The writer asserts all of it.

**There is no threshold to tune.** Because you always return exactly 7, micro-precision@7 is
`hits / (7 · n_queries)` and micro-recall@7 is `hits / total_gold` — both are monotone in the
single quantity `hits`. So maximising micro-F1@7 *is* maximising the number of gold cases that
land in the top 7. Nothing to calibrate; just rank well.

**BM25 matters here.** Legal cases that cite each other share distinctive terms, statute
numbers and party names, so a lexical scorer is a genuinely strong baseline on this task — and
it is computed purely from the provided competition data, no pretrained anything. It is used
for three separate jobs: a zero-training baseline to bank in the first two minutes, hard
negatives for training, and a final score-fusion submission that should beat either component
alone.

## Run order, given the deadline
Cells 1-5 (about 3 minutes), then **submit the BM25 baseline immediately** — it costs nothing
and banks a real score. Then the smoke test, then runs 1-5 in order, uploading each zip as it
appears. Finish with the fusion cell. If time runs out, runs 1 and 5 are the two the brief
makes compulsory.
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

# ---- pull Phase-7 data (needs the HF_TOKEN Kaggle secret) ---------------------------
if not glob.glob("/kaggle/**/Phase_7_PCR/**/test.json", recursive=True):
    from huggingface_hub import snapshot_download
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        _tok = None
    snapshot_download(repo_id="Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                      allow_patterns="Phase_7_PCR/*", local_dir="/kaggle/working/p7",
                      token=_tok)


class CFG:
    TOK_GLOB   = "/kaggle/input/**/bpe32k.json"
    CKPT_GLOBS = {"A": "/kaggle/input/**/A_decoder.pt",
                  "B": "/kaggle/input/**/B_encdec.pt",
                  "C": "/kaggle/input/**/C_hybrid.pt",
                  "D": "/kaggle/input/**/D_gqa.pt"}
    P7_GLOB    = "/kaggle/**/Phase_7_PCR/**/test.json"

    DIM, N_HEADS, MAX_SEQ = 768, 12, 2048

    # ---- document representation --------------------------------------------
    QUERY_LEN  = 512         # tokens kept from a query case
    CAND_LEN   = 512         # tokens kept from a candidate case
    DOC_PART   = "head"      # "head" | "headtail"
    CHAR_CAP   = 6000        # chars fed to the tokenizer (speed; ~6 chars/token)
    PROJ_DIM   = 256         # retrieval vector width
    BIDIRECTIONAL = True

    # ---- training -----------------------------------------------------------
    EPOCHS      = 2
    MAX_STEPS   = 400        # hard cap per run -- the deadline guard
    BATCH_Q     = 4          # queries per step
    HARD_NEG    = 1          # BM25-mined hard negatives per query
    GRAD_ACCUM  = 4
    TEMP        = 0.05       # InfoNCE temperature
    LR_BACKBONE = 2e-5       # full / partial
    LR_HEAD     = 1e-3
    LR_PEFT     = 1e-3
    WARMUP_FRAC = 0.06
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    AMP         = True
    EVAL_BS     = 24

    # ---- adaptation ---------------------------------------------------------
    LORMA_R, LORMA_ALPHA = 32, 64
    N_PROMPT      = 32       # soft tokens for prompt tuning
    PARTIAL_TOP_K = 3
    KD_T, KD_ALPHA = 2.0, 0.5

    # ---- retrieval ----------------------------------------------------------
    TOPK        = 7          # the brief fixes this
    BM25_POOL   = 50         # depth searched for hard negatives
    OUT_DIR = "/kaggle/working/submissions"

cfg = CFG()
os.makedirs(cfg.OUT_DIR, exist_ok=True)

def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"{what} not found ({pattern}) -- attach the Kaggle dataset"
    return Path(hits[0])

TOK_PATH = find_one(cfg.TOK_GLOB, "tokenizer bpe32k.json")
P7_TEST  = find_one(cfg.P7_GLOB, "Phase-7 test.json")
P7_ROOT  = P7_TEST.parent.parent
CKPTS    = {k: find_one(v, f"Phase-1 model {k}") for k, v in cfg.CKPT_GLOBS.items()}
print("tokenizer :", TOK_PATH)
print("phase-7   :", P7_ROOT)
for k, v in CKPTS.items():
    print(f"model {k}   : {v}  ({v.stat().st_size/1e6:.0f} MB)")
""")

# =============================================================================
md(r"""
## 2. Data

Each split has its own candidate pool and the two must be used together. Query `text` arrives
as a list of sentences, so it is joined back into a string. The loaders are tolerant of both a
list of objects and a dict keyed by id, and of the usual spellings for the text and relevance
fields.

Read the two numbers this prints carefully: the **pool size** sets the inference cost of every
run, and the **mean gold-set size** tells you the ceiling. If the average query has fewer than
7 relevant cases, returning 7 forces unavoidable false positives and micro-precision is capped
below 1 no matter how good the ranking is.
""")

code(r"""
def _read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _text_of(d):
    t = d.get("text", d.get("document", d.get("case_text", d.get("sentences", ""))))
    if isinstance(t, list):
        t = " ".join(map(str, t))
    return str(t)

def _norm_items(raw):
    if isinstance(raw, list):
        return raw
    return [((v if isinstance(v, dict) else {"text": v}) | {"id": k})
            for k, v in raw.items()]

def load_queries(path, with_rel=True):
    out = {}
    for d in _norm_items(_read_json(path)):
        qid = str(d.get("id", d.get("query_id", d.get("case_id", len(out)))))
        rec = {"text": _text_of(d)}
        rel = d.get("relevant_candidates",
                    d.get("relevant", d.get("candidates", d.get("labels"))))
        if with_rel and rel is not None:
            rec["rel"] = [str(x) for x in rel]
        out[qid] = rec
    return out

def load_pool(path):
    out = {}
    for d in _norm_items(_read_json(path)):
        cid = str(d.get("id", d.get("candidate_id", d.get("case_id", len(out)))))
        out[cid] = _text_of(d)
    return out

def _p(name):
    hits = sorted(glob.glob(str(P7_ROOT / "**" / name), recursive=True))
    assert hits, f"{name} not found under {P7_ROOT}"
    return hits[0]

train_q = load_queries(_p("train.json"))
dev_q   = load_queries(_p("dev.json"))
test_q  = load_queries(_p("test.json"), with_rel=False)
train_c = load_pool(_p("train_candidates.json"))
dev_c   = load_pool(_p("dev_candidates.json"))
test_c  = load_pool(_p("test_candidates.json"))

# queries with no usable positives cannot train
train_q = {q: v for q, v in train_q.items()
           if v.get("rel") and any(c in train_c for c in v["rel"])}
for v in train_q.values():
    v["rel"] = [c for c in v["rel"] if c in train_c]

print(f"queries   train {len(train_q)} | dev {len(dev_q)} | test {len(test_q)}")
print(f"pools     train {len(train_c)} | dev {len(dev_c)} | test {len(test_c)}")

gsz = np.array([len(v["rel"]) for v in train_q.values()])
dsz = np.array([len(v.get("rel", [])) for v in dev_q.values()])
print(f"\ngold set size (train): mean {gsz.mean():.2f} median {int(np.median(gsz))} "
      f"min {gsz.min()} max {gsz.max()}")
print(f"gold set size (dev)  : mean {dsz.mean():.2f} median {int(np.median(dsz))} "
      f"min {dsz.min()} max {dsz.max()}")
_cap_p = min(1.0, dsz.mean() / cfg.TOPK)
print(f"\n-> with {cfg.TOPK} returned per query, micro-precision on dev is capped at "
      f"{_cap_p:.3f}; a perfect ranking would score micro-F1 "
      f"{2*_cap_p*1.0/(_cap_p+1.0):.3f}")

wc = np.array([len(t.split()) for t in list(train_c.values())[:2000]])
print(f"\ncandidate length (words): median {int(np.median(wc))} "
      f"p90 {int(np.percentile(wc,90))} max {int(wc.max())}")
""")

# =============================================================================
md(r"""
## 3. BM25

Implemented directly over the provided pools — term counts, inverse document frequency and
length normalisation, nothing pretrained. It earns its place three times over: as the baseline
in the next cell, as the source of hard negatives in section 5, and as half of the fusion
submission at the end.

Hard negatives are what make contrastive training work. A randomly drawn negative case is
trivially distinguishable, so the model learns nothing from it; a case that BM25 ranks highly
but that is *not* relevant is exactly the confusion the model needs to resolve.
""")

code(r"""
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp


class BM25:
    def __init__(self, ids, texts, k1=1.2, b=0.75, max_features=200_000):
        self.ids = list(ids)
        self.pos = {c: i for i, c in enumerate(self.ids)}
        self.vec = CountVectorizer(lowercase=True, token_pattern=r"[a-z0-9]{2,}",
                                   max_features=max_features, dtype=np.float32)
        X = self.vec.fit_transform(texts)
        N = X.shape[0]
        dl = np.asarray(X.sum(1)).ravel()
        avgdl = max(dl.mean(), 1.0)
        df = np.asarray((X > 0).sum(0)).ravel()
        idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        Xc = X.tocoo()
        denom = Xc.data + k1 * (1 - b + b * dl[Xc.row] / avgdl)
        data = (Xc.data * (k1 + 1) / denom * idf[Xc.col]).astype(np.float32)
        self.W = sp.csr_matrix((data, (Xc.row, Xc.col)), shape=X.shape).T.tocsr()

    def scores(self, texts, block=256):
        out = np.zeros((len(texts), len(self.ids)), dtype=np.float32)
        for i in range(0, len(texts), block):
            Q = self.vec.transform(texts[i:i + block])
            Q.data[:] = 1.0                      # query term presence
            out[i:i + block] = (Q @ self.W).toarray()
        return out


t0 = time.time()
bm_train = BM25(train_c.keys(), list(train_c.values()))
bm_dev   = BM25(dev_c.keys(),   list(dev_c.values()))
bm_test  = BM25(test_c.keys(),  list(test_c.values()))
print(f"BM25 indexes built in {time.time()-t0:.0f}s")

TRAIN_IDS, DEV_IDS, TEST_IDS = bm_train.ids, bm_dev.ids, bm_test.ids
BM_DEV  = bm_dev.scores([dev_q[q]["text"]  for q in dev_q])
BM_TEST = bm_test.scores([test_q[q]["text"] for q in test_q])
print("bm25 score matrices:", BM_DEV.shape, BM_TEST.shape)


def micro_f1_at_k(ranked, queries, k=None):
    '''ranked: {qid: [cand_id, ...]}. Returns micro P/R/F1 at k and the raw hit count.'''
    k = k or cfg.TOPK
    tp = fp = fn = 0
    for q, v in queries.items():
        gold = set(v.get("rel", []))
        pred = list(dict.fromkeys(ranked[q]))[:k]
        hit = len(gold & set(pred))
        tp += hit; fp += len(pred) - hit; fn += len(gold) - hit
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"micro_f1": 2 * P * R / (P + R) if P + R else 0.0,
            "p": P, "r": R, "hits": tp, "gold": tp + fn}


def rank_from_scores(S, qids, cand_ids, queries=None, k=None):
    '''Top-k per row, with the query's own id removed if it sits in the pool.'''
    k = k or cfg.TOPK
    kk = min(len(cand_ids), k + 2)
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    out = {}
    for r, q in enumerate(qids):
        cols = idx[r][np.argsort(-S[r, idx[r]])]
        picks = [cand_ids[c] for c in cols if cand_ids[c] != q][:k]
        if len(picks) < k:                       # tiny pool fallback
            picks += [c for c in cand_ids if c != q and c not in picks][:k - len(picks)]
        out[q] = picks
    return out


r_bm = micro_f1_at_k(rank_from_scores(BM_DEV, list(dev_q), DEV_IDS), dev_q)
print(f"\nBM25 on dev: micro-F1@7 {r_bm['micro_f1']:.4f} "
      f"(P {r_bm['p']:.3f} R {r_bm['r']:.3f}, {r_bm['hits']}/{r_bm['gold']} gold found)")
""")

# =============================================================================
md(r"""
## 4. The submission writer, and the BM25 baseline

**Submit the zip this cell writes straight away.** It takes seconds, needs no GPU, and banks a
real leaderboard score before any training has happened. It is a baseline for the §13.5
analysis, not one of the four required experiments — those all start from a Phase-1 model.
""")

code(r"""
RESULTS, SCORES = [], {}

def write_submission(ranked, name):
    '''Phase-7 format: a JSON DICT {query_id: [exactly 7 unique candidate ids]}.'''
    payload = {}
    for q in test_q:                             # every test query, in file order
        picks = list(dict.fromkeys(str(c) for c in ranked[q]))[:cfg.TOPK]
        payload[str(q)] = picks
    assert set(payload) == {str(q) for q in test_q}, "test query id mismatch"
    for q, v in payload.items():
        assert isinstance(v, list) and len(v) == cfg.TOPK, (q, len(v))
        assert len(set(v)) == cfg.TOPK, f"{q}: duplicate candidate ids"
        bad = [c for c in v if c not in test_c]
        assert not bad, f"{q}: ids outside the test pool: {bad[:3]}"
        assert all(isinstance(c, str) for c in v), q

    jp = os.path.join(cfg.OUT_DIR, f"prediction_{name}.json")
    zp = os.path.join(cfg.OUT_DIR, f"submission_{name}.zip")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
        back = json.loads(z.read("prediction.json"))
    assert isinstance(back, dict) and len(back) == len(test_q)
    assert all(len(v) == cfg.TOPK for v in back.values())
    print(f"  -> {zp}  | {len(payload)} queries x {cfg.TOPK} candidates")
    return zp


write_submission(rank_from_scores(BM_TEST, list(test_q), TEST_IDS), "00_bm25_baseline")
RESULTS.append({"name": "00_bm25_baseline", "model": "-", "method": "bm25 (lexical)",
                "dev_micro_f1": round(r_bm["micro_f1"], 4), "minutes": 0.0})
print("\nUPLOAD THIS NOW, then carry on.")
""")

# =============================================================================
md(r"""
## 5. Tokenising, hard negatives and the batch

Every query and every candidate is tokenised **once** up front and cached as token ids. This
matters: the pools run to thousands of long documents and re-tokenising them each epoch would
dominate the run time. Texts are first clipped by character count, since nothing past roughly
`CHAR_CAP` characters can survive a `CAND_LEN`-token truncation anyway.

Inside a batch the candidates are de-duplicated and the positive mask marks **every** gold case
present, not only the one that was sampled. That is what keeps in-batch negatives honest: if
another query's positive happens to be relevant to this query too, it lands in the numerator
rather than being punished as a negative.
""")

code(r"""
from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(TOK_PATH))
VOCAB_SIZE = tok.get_vocab_size()
PAD_ID = int(tok.token_to_id("<|pad|>") or 0)
print(f"vocab {VOCAB_SIZE:,} | pad {PAD_ID}")


def encode_all(texts, maxlen, block=256):
    out = []
    for i in range(0, len(texts), block):
        chunk = [t[:cfg.CHAR_CAP] for t in texts[i:i + block]]
        for e in tok.encode_batch(chunk):
            ids = e.ids or [PAD_ID]
            if cfg.DOC_PART == "headtail" and len(ids) > maxlen:
                h = maxlen // 2
                ids = ids[:h] + ids[-(maxlen - h):]
            out.append(ids[:maxlen] or [PAD_ID])
    return out


t0 = time.time()
Q_TR = dict(zip(train_q, encode_all([v["text"] for v in train_q.values()], cfg.QUERY_LEN)))
Q_DV = dict(zip(dev_q,   encode_all([v["text"] for v in dev_q.values()],   cfg.QUERY_LEN)))
Q_TE = dict(zip(test_q,  encode_all([v["text"] for v in test_q.values()],  cfg.QUERY_LEN)))
C_TR = dict(zip(train_c, encode_all(list(train_c.values()), cfg.CAND_LEN)))
C_DV = encode_all([dev_c[c]  for c in DEV_IDS],  cfg.CAND_LEN)
C_TE = encode_all([test_c[c] for c in TEST_IDS], cfg.CAND_LEN)
print(f"tokenised everything in {time.time()-t0:.0f}s")


# ---- BM25-mined hard negatives for the training queries -----------------------------
t0 = time.time()
TR_QIDS = list(train_q)
_S = bm_train.scores([train_q[q]["text"] for q in TR_QIDS])
HARD = {}
for r, q in enumerate(TR_QIDS):
    gold = set(train_q[q]["rel"]) | {q}
    top = np.argpartition(-_S[r], min(cfg.BM25_POOL, len(TRAIN_IDS) - 1))[:cfg.BM25_POOL]
    top = top[np.argsort(-_S[r, top])]
    HARD[q] = [TRAIN_IDS[c] for c in top if TRAIN_IDS[c] not in gold]
del _S; gc.collect()
_hn = np.array([len(v) for v in HARD.values()])
print(f"hard negatives mined in {time.time()-t0:.0f}s | mean pool {_hn.mean():.1f}")


def pad_batch(seqs):
    T = max(len(s) for s in seqs)
    ids  = torch.full((len(seqs), T), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = True
    return ids, mask


class QuerySet(Dataset):
    def __init__(self, qids): self.qids = list(qids)
    def __len__(self): return len(self.qids)
    def __getitem__(self, i): return self.qids[i]


def collate_train(qids):
    cand, seen = [], set()
    for q in qids:
        picks = [random.choice(train_q[q]["rel"])]
        pool = HARD.get(q) or [c for c in TRAIN_IDS if c != q]
        picks += random.sample(pool, min(cfg.HARD_NEG, len(pool)))
        for c in picks:
            if c not in seen:
                seen.add(c); cand.append(c)
    qi, qm = pad_batch([Q_TR[q] for q in qids])
    ci, cm = pad_batch([C_TR[c] for c in cand])
    posm = torch.zeros(len(qids), len(cand), dtype=torch.bool)
    for i, q in enumerate(qids):
        gold = set(train_q[q]["rel"])
        for j, c in enumerate(cand):
            if c in gold:
                posm[i, j] = True
    return {"qi": qi, "qm": qm, "ci": ci, "cm": cm, "pos": posm}


_b = collate_train(TR_QIDS[:cfg.BATCH_Q])
print(f"\nbatch check: {_b['qi'].shape[0]} queries x {_b['ci'].shape[0]} candidates "
      f"@ {_b['qi'].shape[1]}/{_b['ci'].shape[1]} tokens | "
      f"positives per query {_b['pos'].sum(1).tolist()}")
""")

# =============================================================================
md(r"""
## 6. Phase-1 architecture components

Reproduced from your Phase-1 notebooks so every checkpoint loads with `strict=True`. The only
additions are the `causal` / `bidirectional` flags, which default to the trained behaviour.

Retrieval reads a whole document at once rather than generating, so there is no reason to hide
right-hand context. B's encoder is already bidirectional; A and D are patched; C can only be
patched in its four attention layers, since its eight gated-recurrence layers have direction
built into the operator.
""")

code(ARCH_COMPONENTS)

# =============================================================================
md(r"""
## 7. The four models
""")

code(ARCH_MODELS)

# =============================================================================
md(r"""
## 8. The bi-encoder

One encoder, shared between queries and candidates: hidden states are mean-pooled over the
real (non-pad) tokens, passed through a trained projection to `PROJ_DIM`, and L2-normalised, so
a dot product is a cosine. Sharing the tower rather than training two is the right call with a
few hundred training queries — two towers would double the trainable parameters for no extra
signal, and a case is a case whichever side of the pair it sits on.

Soft prompt tokens, when prompt tuning is active, are prepended to the embedding sequence and
are *visible to attention but excluded from the pool*, so the document vector stays a summary
of the document rather than of the prompt.
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


class PCREncoder(nn.Module):
    def __init__(self, backbone, arch, hidden, proj_dim, n_prompt=0, dropout=0.1):
        super().__init__()
        self.backbone, self.arch, self.n_prompt = backbone, arch, n_prompt
        if n_prompt:
            self.prompt = nn.Parameter(torch.randn(n_prompt, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, proj_dim))

    def embed(self, ids, mask):
        x = self.backbone.tok_emb(ids)
        if self.n_prompt:
            p = self.prompt.unsqueeze(0).expand(ids.size(0), -1, -1).to(x.dtype)
            x = torch.cat([p, x], 1)
            pad = torch.zeros(ids.size(0), self.n_prompt, dtype=torch.bool,
                              device=mask.device)
            mask = torch.cat([pad, mask], 1)      # prompt seen, not pooled
        h = self.backbone.trunk_embeds(x)
        m = mask.unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return F.normalize(self.proj(self.drop(v)).float(), dim=-1)


def build_pcr(arch, bidirectional=None, keep_layers=None, n_prompt=0,
              pretrained=True, verbose=True):
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
    m = PCREncoder(bb, arch, cfg.DIM, cfg.PROJ_DIM, n_prompt=n_prompt).to(DEVICE)
    if verbose:
        tot = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
        print(f"  [{arch}] {ARCH_INFO[arch]} | {tot/1e6:.1f}M params | "
              f"bidirectional={bi} ({n} mixers switched)"
              + (f" | {n_prompt} soft tokens" if n_prompt else ""))
    return m
""")

# =============================================================================
md(r"""
## 9. Adaptation techniques

All from §13.2.1. **LoRA, AdaLoRA and adapters are not on Phase 7's list** and are
deliberately absent.

### LoRMA — multiplicative low-rank *(compulsory)*
Replaces LoRA's additive update with a multiplicative one, transforming the weight rather than
offsetting it. A bare product would cap the adapted weight at rank `r` and has no zero
initialisation, so **LoRMA+** adds an identity: `W = (I + (alpha/r)·BA)·W`, exactly `W` at init
and still near full rank. It never materialises `BA` — it computes `y = Wx`, then `y + (yAᵀ)Bᵀ`.

### IA³
Learns one scalar per output unit and rescales the frozen projection's output elementwise:
roughly a thousand parameters per layer, the lightest method on the list. On the fused-QKV
models only the key and value halves are scaled, matching the method as published; the
concatenation is rebuilt with `torch.cat` rather than an in-place slice assignment, which would
bump the tensor's version counter and break the backward pass.

### Prompt tuning
A block of trainable embedding vectors prepended to the sequence. The backbone is untouched —
the only thing learned is a context that steers it, plus the projection.

### Full fine-tuning
Every weight trainable. In Phase 4, at 32,000 training documents, this beat every
parameter-efficient method by a wide margin. PCR gives you a few hundred queries instead, which
is the regime where that advantage reverses, so do not assume it repeats.

### Partial / layer-wise
The top `PARTIAL_TOP_K` blocks only — available as a spare.

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
        self.B = nn.Parameter(torch.zeros(d, r))      # zero -> starts exactly at W0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r
    def forward(self, x):
        y = self.base(x)
        return y + ((y @ self.A.t()) @ self.B.t()) * self.scale


class IA3Linear(nn.Module):
    # learned elementwise rescaling of the LAST n_tail output units
    def __init__(self, base, n_tail=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        d = base.out_features
        self.n = int(n_tail or d)
        self.scale = nn.Parameter(torch.ones(self.n))
    def forward(self, x):
        y = self.base(x)
        if self.n >= y.size(-1):
            return y * self.scale
        # cat, not an in-place slice write -- the latter breaks autograd on fused qkv
        return torch.cat([y[..., :-self.n], y[..., -self.n:] * self.scale], dim=-1)


MIXERS = (CausalSelfAttention, GeneralizedAttention,
          GQASlidingWindowAttention, GatedLinearRecurrence)

def mixer_linears(model):
    # every nn.Linear that is a direct child of a token mixer, selected by module type
    for mod in model.modules():
        if isinstance(mod, MIXERS):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    yield mod, cname, child


def apply_adaptation(model, method):
    bb, arch = model.backbone, model.arch
    for p in bb.parameters(): p.requires_grad = False
    for p in model.proj.parameters(): p.requires_grad = True
    if model.n_prompt:
        model.prompt.requires_grad = True
    blocks = trunk_blocks(bb, arch)

    if method in ("head_only", "prompt"):
        pass

    elif method == "full":
        for p in bb.parameters(): p.requires_grad = True

    elif method == "partial":
        for blk in list(blocks)[-cfg.PARTIAL_TOP_K:]:
            for p in blk.parameters(): p.requires_grad = True
        for mod in ([bb.norm_e, bb.norm_f] if arch == "B" else [bb.norm_f]):
            for p in mod.parameters(): p.requires_grad = True

    elif method == "lorma":
        for parent, cn, lin in list(mixer_linears(bb)):
            setattr(parent, cn, LoRMALinear(lin, cfg.LORMA_R, cfg.LORMA_ALPHA).to(DEVICE))

    elif method == "ia3":
        for parent, cn, lin in list(mixer_linears(bb)):
            if cn == "qkv":                       # scale K and V only
                setattr(parent, cn,
                        IA3Linear(lin, n_tail=2 * (lin.out_features // 3)).to(DEVICE))
            elif cn in ("k_proj", "v_proj", "in_proj"):
                setattr(parent, cn, IA3Linear(lin).to(DEVICE))
        for mlp in [m for m in bb.modules()
                    if isinstance(m, SwiGLU) and isinstance(m.w1, nn.Linear)]:
            mlp.w1 = IA3Linear(mlp.w1).to(DEVICE)

    else:
        raise ValueError(method)

    tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"  [{method}] trainable {tr/1e6:.3f}M / {tot/1e6:.1f}M = {100*tr/tot:.3f}%")
    return model
""")

# =============================================================================
md(r"""
## 10. Training, encoding and evaluation

**The loss.** Multi-positive InfoNCE. For a query, the numerator sums the similarities of every
gold case present in the batch and the denominator sums over all of them, so

`loss = -log( Σ_pos exp(sim/τ) / Σ_all exp(sim/τ) )`

with `τ = TEMP`. Written as a difference of two `logsumexp` terms it is numerically stable, it
handles a query with several positives without any reweighting, and a candidate that is gold
for this query can never be counted against it.

**Encoding.** The pool is sorted by length before batching and unsorted afterwards. These
architectures carry no key-padding mask, so shorter documents padded up to the longest in the
batch would otherwise attend to padding — length-sorting makes batches nearly uniform, which
cuts both that contamination and the wasted compute.

**Evaluation.** Micro-F1@7 on dev, next to BM25 and to the recall ceiling.
""")

code(r"""
def infonce(qv, cv, pos):
    sim = (qv @ cv.t()) / cfg.TEMP
    neg_inf = torch.finfo(sim.dtype).min
    num = sim.masked_fill(~pos, neg_inf)
    keep = pos.any(1)
    if not keep.any():
        return sim.sum() * 0.0
    return -(torch.logsumexp(num[keep], -1) - torch.logsumexp(sim[keep], -1)).mean()


def make_optimizer(model, method):
    head_ids = {id(p) for p in model.proj.parameters()}
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


def train(model, method, epochs=None, teacher=None, tag=""):
    epochs = epochs or cfg.EPOCHS
    dl = DataLoader(QuerySet(TR_QIDS), batch_size=cfg.BATCH_Q, shuffle=True,
                    collate_fn=collate_train, num_workers=2, drop_last=False)
    planned = min(cfg.MAX_STEPS, max(1, (len(dl) // cfg.GRAD_ACCUM) * epochs))
    opt = make_optimizer(model, method)
    sch = make_sched(opt, planned)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and USE_AMP)
    if teacher is not None:
        teacher.eval()

    model.train(); t0 = time.time(); gstep = 0; done = False
    for ep in range(epochs):
        run = nb = 0
        for i, b in enumerate(dl):
            qi, qm = b["qi"].to(DEVICE), b["qm"].to(DEVICE)
            ci, cm = b["ci"].to(DEVICE), b["cm"].to(DEVICE)
            pos = b["pos"].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
                qv, cv = model.embed(qi, qm), model.embed(ci, cm)
                loss = infonce(qv, cv, pos)
                if teacher is not None:
                    with torch.no_grad():
                        tq, tc = teacher.embed(qi, qm), teacher.embed(ci, cm)
                    T = cfg.KD_T
                    kd = F.kl_div(
                        F.log_softmax((qv @ cv.t()) / (cfg.TEMP * T), -1),
                        F.softmax((tq @ tc.t()) / (cfg.TEMP * T), -1),
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
                    done = True; break
            if nb % 100 == 0:
                print(f"    ep{ep+1} {nb}/{len(dl)} loss {run/nb:.4f} "
                      f"lr {sch.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        print(f"  [{tag}] epoch {ep+1}/{epochs} loss {run/max(nb,1):.4f} "
              f"step {gstep}/{planned} ({time.time()-t0:.0f}s)", flush=True)
        if done:
            print(f"  [{tag}] hit MAX_STEPS={cfg.MAX_STEPS}, stopping early")
            break
    return model


@torch.no_grad()
def encode_ids(model, id_lists, bs=None):
    '''Length-sorted batched encoding -> (N, PROJ_DIM) float32, original order.'''
    model.eval()
    bs = bs or cfg.EVAL_BS
    order = np.argsort([len(s) for s in id_lists])
    V = np.zeros((len(id_lists), cfg.PROJ_DIM), dtype=np.float32)
    for i in range(0, len(order), bs):
        sel = order[i:i + bs]
        ids, mask = pad_batch([id_lists[j] for j in sel])
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.AMP and USE_AMP):
            v = model.embed(ids.to(DEVICE), mask.to(DEVICE))
        V[sel] = v.float().cpu().numpy()
    return V


def score_matrix(model, qids, q_store, cand_id_lists):
    Qv = encode_ids(model, [q_store[q] for q in qids])
    Cv = encode_ids(model, cand_id_lists)
    return (Qv @ Cv.T).astype(np.float32)


def report(S, qids, cand_ids, queries, label=""):
    r = micro_f1_at_k(rank_from_scores(S, qids, cand_ids, queries), queries)
    kk = min(100, len(cand_ids))
    top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    rec100 = np.mean([len(set(queries[q].get("rel", [])) &
                          {cand_ids[c] for c in top[r_]}) /
                      max(len(queries[q].get("rel", [])), 1)
                      for r_, q in enumerate(qids)])
    print(f"  {label}micro-F1@7 {r['micro_f1']:.4f} | P {r['p']:.3f} R {r['r']:.3f} | "
          f"hits {r['hits']}/{r['gold']} | recall@{kk} {rec100:.3f}")
    return r


def run_experiment(name, arch, method, epochs=None, bidirectional=None, keep_layers=None,
                   teacher=None, write=True, seed=42, keep_scores=True):
    print(f"\n{'='*72}\n{name}   model {arch} | {method}\n{'='*72}", flush=True)
    set_seed(seed); t0 = time.time()
    n_prompt = cfg.N_PROMPT if method == "prompt" else 0
    model = build_pcr(arch, bidirectional=bidirectional, keep_layers=keep_layers,
                      n_prompt=n_prompt)
    model = apply_adaptation(model, method)
    train(model, method, epochs=epochs, teacher=teacher, tag=name)

    Sd = score_matrix(model, list(dev_q), Q_DV, C_DV)
    r = report(Sd, list(dev_q), DEV_IDS, dev_q, label="dev  ")
    print(f"  (BM25 on the same dev split: {r_bm['micro_f1']:.4f})")

    St, zp = None, None
    if write:
        St = score_matrix(model, list(test_q), Q_TE, C_TE)
        zp = write_submission(rank_from_scores(St, list(test_q), TEST_IDS), name)
    if keep_scores:
        SCORES[name] = {"dev": Sd.astype(np.float16),
                        "test": None if St is None else St.astype(np.float16)}

    RESULTS.append({"name": name, "model": arch, "method": method,
                    "dev_micro_f1": round(r["micro_f1"], 4),
                    "dev_p": round(r["p"], 4), "dev_r": round(r["r"], 4),
                    "trainable_M": round(sum(p.numel() for p in model.parameters()
                                             if p.requires_grad) / 1e6, 3),
                    "minutes": round((time.time() - t0) / 60, 1), "zip": zp})
    print("\n " + json.dumps(RESULTS[-1]))
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return model, r
""")

# =============================================================================
md(r"""
## 11. SMOKE TEST — run this first

Frozen backbone, projection only, capped at 40 steps. If it prints a zip path the pipeline is
sound: checkpoint loading, pooling, the contrastive loss, pool encoding and the new dict-format
submission. Read the reported minutes — multiply by roughly four to budget a real run, and set
`cfg.MAX_STEPS` from what you have left.
""")

code(r"""
_ms = cfg.MAX_STEPS; cfg.MAX_STEPS = 40
m, _ = run_experiment("01_smoke", "A", "head_only", epochs=1, keep_scores=False)
cfg.MAX_STEPS = _ms
del m; gc.collect(); torch.cuda.empty_cache()
print(f"\nMAX_STEPS back to {cfg.MAX_STEPS}")
""")

# =============================================================================
md(r"""
## 12. The five experiments

One cell each — upload every zip as it appears. Runs 1 and 5 are the compulsory pair.
""")

code(r"""
# --- 1. Model A, LoRMA+  (COMPULSORY technique) --------------------------------------
m_teacher, r1 = run_experiment("02_A_lorma", "A", "lorma", epochs=2)
""")

code(r"""
# --- 2. Model B, full fine-tuning (full encoder-decoder trunk) -----------------------
m, _ = run_experiment("03_B_full", "B", "full", epochs=2)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 3. Model C, IA3 -----------------------------------------------------------------
m, _ = run_experiment("04_C_ia3", "C", "ia3", epochs=2)
del m; gc.collect(); torch.cuda.empty_cache()
""")

code(r"""
# --- 4. Model D, prompt tuning -------------------------------------------------------
m, _ = run_experiment("05_D_prompt", "D", "prompt", epochs=3)   # tiny param count
del m; gc.collect(); torch.cuda.empty_cache()
""")

md(r"""
### 5. Knowledge distillation *(compulsory)*

The teacher is the **trained** retriever from run 1, as §13.2.2 asks — a stronger model on this
task, not a raw Phase-1 checkpoint. The student is the same architecture with six of its twelve
blocks, taken at evenly spaced depths so it inherits a spread of the teacher's feature hierarchy
rather than only its early layers.

What transfers is the teacher's **ranking distribution**: the student's softmaxed
query-to-candidate similarities over the batch are pulled toward the teacher's. That is a
retrieval-specific signal rather than a generic logit match — it carries the teacher's view of
how much *more* relevant one candidate is than another, which the hard positive/negative labels
throw away entirely.
""")

code(r"""
# --- 5. Model A -> half-depth student, knowledge distillation (COMPULSORY) ----------
for p in m_teacher.parameters():
    p.requires_grad = False

m_student, r5 = run_experiment("06_A_distill", "A", "lorma", epochs=2,
                               keep_layers=6, teacher=m_teacher)

tp = sum(p.numel() for p in {id(p): p for p in m_teacher.parameters()}.values())
sp_ = sum(p.numel() for p in {id(p): p for p in m_student.parameters()}.values())
print(f"\nteacher {tp/1e6:.1f}M -> student {sp_/1e6:.1f}M ({100*sp_/tp:.0f}% of size)")
print(f"dev micro-F1@7 {r1['micro_f1']:.4f} -> {r5['micro_f1']:.4f}")
del m_student; gc.collect(); torch.cuda.empty_cache()
""")

# =============================================================================
md(r"""
## 13. Results, ensemble and lexical fusion

Three things are combined here, each cheap and each tested on dev before it is trusted.

**Neural ensemble.** Score matrices from the top-k runs are z-normalised per query and averaged.
Per-query normalisation is what makes this legitimate: the four architectures have different
similarity scales, so averaging raw cosines would just let whichever model is most confident
dominate. Ensembling across architectures gained 5.4 macro-F1 in Phase 2 and about 2 in Phase 3.

**Lexical fusion.** The blended score is `α · z(neural) + (1 − α) · z(bm25)`, with `α` swept on
dev. Dense and lexical retrieval fail in genuinely different ways — the dense tower misses an
exact statute number or party name it never learned to weight, BM25 misses a paraphrase — so the
fused ranking usually beats both. Expect this to be the best submission in the notebook.

**The honest check.** If no combination beats the best single run on dev, that is reported and
nothing is written. Submit the best single run instead.
""")

code(r"""
import pandas as pd
df = pd.DataFrame(RESULTS).sort_values("dev_micro_f1", ascending=False)
df.to_csv("/kaggle/working/phase7_results.csv", index=False)
display(df[[c for c in ["name", "model", "method", "dev_micro_f1", "dev_p", "dev_r",
                        "trainable_M", "minutes"] if c in df]])
_neu = [r for r in RESULTS if r["model"] != "-"]
print("architectures:", sorted({r["model"] for r in _neu}))
print("techniques   :", sorted({r["method"] for r in _neu}))


def zrow(S):
    S = S.astype(np.float32)
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-6)

ZBM_D, ZBM_T = zrow(BM_DEV), zrow(BM_TEST)
ranked = [r["name"] for r in sorted(_neu, key=lambda r: -r["dev_micro_f1"])
          if r["name"] in SCORES and SCORES[r["name"]]["test"] is not None]
best_single = max([r["dev_micro_f1"] for r in RESULTS])
print(f"\nbest single (incl. BM25): {best_single:.4f}")

best = (None, None, best_single)       # (names, alpha, f1)

for k in range(1, len(ranked) + 1):
    use = ranked[:k]
    ZD = np.mean([zrow(SCORES[n]["dev"]) for n in use], axis=0)
    for a in np.arange(0.0, 1.01, 0.1):
        S = a * ZD + (1 - a) * ZBM_D
        f1 = micro_f1_at_k(rank_from_scores(S, list(dev_q), DEV_IDS, dev_q),
                           dev_q)["micro_f1"]
        if a in (0.0, 0.5, 1.0) or f1 > best[2]:
            print(f"  top-{k} alpha {a:.1f} -> {f1:.4f}"
                  + ("   <-- best so far" if f1 > best[2] else ""))
        if f1 > best[2]:
            best = (use, float(a), f1)

if best[0]:
    use, a, f1 = best
    print(f"\nfusion wins: {best_single:.4f} -> {f1:.4f} with alpha {a:.1f} over {use}")
    ZD = np.mean([zrow(SCORES[n]["dev"]) for n in use], axis=0)
    report(a * ZD + (1 - a) * ZBM_D, list(dev_q), DEV_IDS, dev_q, label="fused dev  ")
    ZT = np.mean([zrow(SCORES[n]["test"]) for n in use], axis=0)
    write_submission(rank_from_scores(a * ZT + (1 - a) * ZBM_T, list(test_q), TEST_IDS),
                     "07_fusion")
else:
    print("\nnothing beat the best single run on dev -- submit that one")
""")

md(r"""
## 14. Spares, if time allows

```python
# partial / layer-wise fine-tuning -- a sixth technique from the §13.2.1 list
m, _ = run_experiment("08_A_partial", "A", "partial", epochs=2)

# does un-masking attention actually help retrieval?
m, _ = run_experiment("09_A_causal", "A", "lorma", epochs=2, bidirectional=False)

# head+tail instead of head: does the end of a case carry retrieval signal?
cfg.DOC_PART = "headtail"
Q_TR = dict(zip(train_q, encode_all([v["text"] for v in train_q.values()], cfg.QUERY_LEN)))
Q_DV = dict(zip(dev_q,   encode_all([v["text"] for v in dev_q.values()],   cfg.QUERY_LEN)))
Q_TE = dict(zip(test_q,  encode_all([v["text"] for v in test_q.values()],  cfg.QUERY_LEN)))
C_TR = dict(zip(train_c, encode_all(list(train_c.values()), cfg.CAND_LEN)))
C_DV = encode_all([dev_c[c]  for c in DEV_IDS],  cfg.CAND_LEN)
C_TE = encode_all([test_c[c] for c in TEST_IDS], cfg.CAND_LEN)
m, _ = run_experiment("10_A_headtail", "A", "lorma", epochs=2)

# more in-batch negatives -- the single biggest lever in contrastive retrieval,
# if the GPU has the memory for it
cfg.BATCH_Q, cfg.HARD_NEG, cfg.GRAD_ACCUM = 8, 2, 2
m, _ = run_experiment("11_A_bignegs", "A", "lorma", epochs=2)
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

with open("CS781_Phase7_PCR.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("cells:", len(cells))
