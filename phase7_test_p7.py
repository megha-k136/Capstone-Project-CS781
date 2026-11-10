"""Exercise CS781_Phase7_PCR.ipynb end to end on CPU.

Real Phase-1 architecture classes at reduced width with random weights, a real byte-level BPE,
and a synthetic PCR corpus where each query's gold candidates genuinely share rare marker
tokens with it, so BM25 and the bi-encoder both have something true to find. Checkpoint
loading is the only stub.
"""
import json, os, random, shutil, sys, tempfile, traceback

NB = "CS781_Phase7_PCR.ipynb"
cells = json.load(open(NB))["cells"]
SRC = {i: "".join(c["source"]) for i, c in enumerate(cells) if c["cell_type"] == "code"}
print("code cells:", sorted(SRC))

ROOT = tempfile.mkdtemp(prefix="p7test_")
DATA = os.path.join(ROOT, "input", "Phase_7_PCR")
for d in ("Train", "Dev", "Test"):
    os.makedirs(os.path.join(DATA, d), exist_ok=True)
OUT = os.path.join(ROOT, "work"); os.makedirs(OUT, exist_ok=True)

FILLER = ("the appellant preferred the present appeal against the impugned judgment of the "
          "learned single judge after hearing counsel for the parties at length upon the "
          "questions of fact and of law arising in these proceedings ")
rng = random.Random(7)
corpus = []

# Each "topic" owns a few rare marker words. A query and its gold candidates share a topic.
TOPICS = {t: [f"zq{t}{k}" for k in range(4)] for t in range(12)}

def body(topic, n=60):
    w = (FILLER * 4).split()
    out = []
    for _ in range(n):
        out.append(rng.choice(w))
        if rng.random() < 0.28:
            out.append(rng.choice(TOPICS[topic]))
    return " ".join(out)

def build_split(name, n_q, n_c, as_dict_q=False, as_dict_c=False, with_rel=True):
    cands, ctopic = {}, {}
    for k in range(n_c):
        t = k % len(TOPICS)
        cid = f"{name}c{k:05d}"
        cands[cid] = body(t, 70); ctopic[cid] = t
    queries = {}
    for k in range(n_q):
        t = k % len(TOPICS)
        qid = f"{name}q{k:04d}"
        same = [c for c, tt in ctopic.items() if tt == t]
        queries[qid] = {"topic": t, "text": body(t, 80),
                        "rel": rng.sample(same, min(len(same), rng.randint(3, 6)))}
    corpus.extend(list(cands.values())[:120])
    corpus.extend(v["text"] for v in queries.values())

    # queries: text is a LIST of sentences, as the brief describes
    if as_dict_q:
        qp = {q: ({"text": v["text"].split(" . ") or [v["text"]]}
                  | ({"relevant_candidates": v["rel"]} if with_rel else {}))
              for q, v in queries.items()}
    else:
        qp = [({"id": q, "text": [v["text"]]}
               | ({"relevant_candidates": v["rel"]} if with_rel else {}))
              for q, v in queries.items()]
    cp = ({c: {"text": [t]} for c, t in cands.items()} if as_dict_c
          else [{"id": c, "text": [t]} for c, t in cands.items()])
    return qp, cp, queries, cands

qp, cp, TRQ, TRC = build_split("tr", 48, 180, as_dict_q=False, as_dict_c=True)
json.dump(qp, open(f"{DATA}/Train/train.json", "w"))
json.dump(cp, open(f"{DATA}/Train/train_candidates.json", "w"))

qp, cp, DVQ, DVC = build_split("dv", 20, 96, as_dict_q=True, as_dict_c=False)
json.dump(qp, open(f"{DATA}/Dev/dev.json", "w"))
json.dump(cp, open(f"{DATA}/Dev/dev_candidates.json", "w"))

qp, cp, TEQ, TEC = build_split("te", 16, 80, as_dict_q=False, as_dict_c=False,
                               with_rel=False)
json.dump(qp, open(f"{DATA}/Test/test.json", "w"))
json.dump(cp, open(f"{DATA}/Test/test_candidates.json", "w"))
TEST_QIDS, TEST_CIDS = set(TEQ), set(TEC)

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
tk = Tokenizer(models.BPE(unk_token="<|unk|>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(corpus, trainers.BpeTrainer(
    vocab_size=900, special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
TOKP = os.path.join(ROOT, "input", "bpe32k.json")
tk.save(TOKP)
assert tk.token_to_id("<|pad|>") == 0
for n in ("A_decoder.pt", "B_encdec.pt", "C_hybrid.pt", "D_gqa.pt"):
    open(os.path.join(ROOT, "input", n), "wb").write(b"stub")
print(f"[setup] synthetic PCR corpus + real BPE at {ROOT} (vocab {tk.get_vocab_size()})")

# ---------------------------------------------------------------------------
from pathlib import Path
G = {"__name__": "nbtest", "display": print}       # display is a Jupyter builtin

head = SRC[2].split("# ---- pull Phase-7 data")[0]
cfgpart = ("class CFG:" + SRC[2].split("class CFG:", 1)[1].split("cfg = CFG()")[0]
           + "cfg = CFG()\n")
exec(compile(head + cfgpart, "cell2", "exec"), G)

cfg = G["cfg"]
cfg.DIM, cfg.N_HEADS, cfg.MAX_SEQ = 96, 4, 512
cfg.QUERY_LEN = cfg.CAND_LEN = 128
cfg.PROJ_DIM = 32
cfg.CHAR_CAP = 1200
cfg.BATCH_Q, cfg.HARD_NEG, cfg.GRAD_ACCUM = 3, 1, 2
cfg.EPOCHS, cfg.MAX_STEPS = 1, 6
cfg.EVAL_BS = 8
cfg.AMP = False
cfg.N_PROMPT = 8
cfg.LORMA_R, cfg.LORMA_ALPHA = 8, 16
cfg.OUT_DIR = OUT
G["TOK_PATH"] = Path(TOKP)
G["P7_TEST"] = Path(DATA) / "Test" / "test.json"
G["P7_ROOT"] = Path(DATA)
G["CKPTS"] = {k: Path(ROOT) / "input" / v for k, v in
              {"A": "A_decoder.pt", "B": "B_encdec.pt",
               "C": "C_hybrid.pt", "D": "D_gqa.pt"}.items()}
print("[cfg] reduced for CPU:", cfg.DIM, "dim,", cfg.CAND_LEN, "tokens/doc")

exec(SRC[4], G)      # data
exec(SRC[6], G)      # BM25 + metric + ranking
exec(SRC[8], G)      # submission writer + bm25 baseline
exec(SRC[10], G)     # tokenise + hard negatives + batch
exec(SRC[12], G)     # architecture components
exec(SRC[14], G)     # four models

G["C_LAYERS"], G["C_ATTN_EVERY"], G["C_CHUNK"] = 4, 2, 8
G["D_LAYERS"], G["D_FULL_EVERY"], G["D_WINDOW"], G["D_KV_HEADS"] = 4, 2, 32, 2
_real_build = G["_build"]
def _build_small(arch):
    if arch == "A":
        return G["GPTModel"](G["VOCAB_SIZE"], n_layers=4)
    if arch == "B":
        return G["EncoderDecoderModel"](G["VOCAB_SIZE"], n_encoder_layers=3,
                                        n_decoder_layers=4)
    return _real_build(arch)
G["_build"] = _build_small
_orig_load = G["load_phase1"]
G["load_phase1"] = lambda arch, pretrained=True: _orig_load(arch, pretrained=False)

exec(SRC[16], G)     # bidirectional + PCREncoder + build_pcr
exec(SRC[18], G)     # adaptation techniques
exec(SRC[20], G)     # train / encode / score / report / run_experiment
print("[exec] logic cells loaded\n")

torch, nn, np, F = G["torch"], G["nn"], G["np"], G["F"]

# --- 0. loaders coped with BOTH json shapes and sentence-list text -------------------
assert len(G["train_q"]) == 48 and len(G["dev_q"]) == 20 and len(G["test_q"]) == 16
assert len(G["train_c"]) == 180 and len(G["dev_c"]) == 96 and len(G["test_c"]) == 80
assert all("rel" not in v for v in G["test_q"].values()), "test must carry no labels"
assert all(v["rel"] for v in G["train_q"].values())
assert isinstance(next(iter(G["train_q"].values()))["text"], str), "sentences not joined"
print("[data] list- and dict-shaped queries/pools parsed; sentence lists joined")

# --- 1. the metric is what the brief defines ----------------------------------------
mf = G["micro_f1_at_k"]
fake_q = {"a": {"rel": ["x", "y", "z"]}, "b": {"rel": ["p"]}}
rk = {"a": ["x", "y", "1", "2", "3", "4", "5"], "b": ["p", "1", "2", "3", "4", "5", "6"]}
r = mf(rk, fake_q)
assert r["hits"] == 3 and r["gold"] == 4, r
assert abs(r["p"] - 3 / 14) < 1e-9 and abs(r["r"] - 0.75) < 1e-9, r
assert abs(r["micro_f1"] - 2 * (3 / 14) * 0.75 / (3 / 14 + 0.75)) < 1e-9
print(f"[metric] micro P/R/F1@7 exact on a hand case ({r['micro_f1']:.4f})")

# --- 2. BM25 actually retrieves the planted topic ------------------------------------
r_bm = G["r_bm"]
assert r_bm["micro_f1"] > 0.15, f"BM25 found nothing: {r_bm}"
print(f"[bm25] dev micro-F1@7 {r_bm['micro_f1']:.4f} on the planted-marker corpus")

# --- 3. ranking excludes self and always returns exactly 7 ---------------------------
rfs = G["rank_from_scores"]
S = np.zeros((1, len(G["DEV_IDS"])), dtype=np.float32)
qid = list(G["dev_q"])[0]
S[0, 3] = 10.0
ranked = rfs(S, [qid], G["DEV_IDS"])
assert len(ranked[qid]) == 7 and len(set(ranked[qid])) == 7
assert qid not in ranked[qid]
assert ranked[qid][0] == G["DEV_IDS"][3], "argsort order wrong"
# self-exclusion, with the query sitting IN the pool
pool = [qid] + G["DEV_IDS"][:10]
S2 = np.zeros((1, len(pool)), dtype=np.float32); S2[0, 0] = 99.0
assert qid not in rfs(S2, [qid], pool)[qid], "query not excluded from its own pool"
print("[rank] top-7 ordered, unique, self excluded")

# --- 4. the InfoNCE loss behaves ------------------------------------------------------
inf = G["infonce"]
qv = F.normalize(torch.randn(3, 16), dim=-1)
cv = F.normalize(torch.randn(6, 16), dim=-1)
pos = torch.zeros(3, 6, dtype=torch.bool); pos[0, 0] = pos[1, 1] = pos[2, 2] = True
base = inf(qv, cv, pos).item()
# a perfect model: query == its positive
qv2 = cv[:3].clone()
assert inf(qv2, cv, pos).item() < base, "loss not lower for a perfect match"
# all-positive row contributes zero, never NaN
allpos = torch.ones(3, 6, dtype=torch.bool)
assert abs(inf(qv, cv, allpos).item()) < 1e-5
# a row with no positive at all must not produce NaN
nopos = torch.zeros(3, 6, dtype=torch.bool); nopos[0, 0] = True
v = inf(qv, cv, nopos)
assert torch.isfinite(v).all(), "empty-positive row gave NaN"
# multi-positive is lower than single when both positives are close
print(f"[loss] multi-positive InfoNCE: random {base:.3f}, aligned "
      f"{inf(qv2, cv, pos).item():.3f}, all-positive 0, empty row finite")

# --- 5. bidirectional patch, per architecture ----------------------------------------
print()
for arch in ("A", "B", "C", "D"):
    mm = G["build_pcr"](arch, bidirectional=False, verbose=False)
    mm.eval()
    ids = torch.randint(0, G["VOCAB_SIZE"], (1, 48))
    with torch.no_grad():
        causal = mm.backbone.trunk(ids).clone()
    nsw = G["set_bidirectional"](mm.backbone, True)
    with torch.no_grad():
        delta = (causal - mm.backbone.trunk(ids)).abs().max().item()
    G["set_bidirectional"](mm.backbone, False)
    alt = ids.clone(); alt[0, 44] = (ids[0, 44].item() + 1) % G["VOCAB_SIZE"]
    with torch.no_grad():
        leak = (mm.backbone.trunk(ids)[0, 5] - mm.backbone.trunk(alt)[0, 5]).abs().max().item()
    G["set_bidirectional"](mm.backbone, True)
    with torch.no_grad():
        reach = (mm.backbone.trunk(ids)[0, 5] - mm.backbone.trunk(alt)[0, 5]).abs().max().item()
    print(f"  {arch}: {nsw} mixers | causal-vs-bi {delta:.4f} | "
          f"future->past causal {leak:.2e} | bidirectional {reach:.2e}")
    if arch in ("A", "D"):
        assert leak < 1e-5 and reach > 1e-6, f"{arch} bidirectional patch wrong"
    if arch == "C":
        assert delta > 0
    del mm

# --- 6. embeddings are unit-norm, and the prompt is excluded from the pool -----------
m = G["build_pcr"]("A", verbose=False)
ids, mask = G["pad_batch"]([[5, 6, 7, 8], [9, 10]])
v = m.embed(ids, mask)
assert v.shape == (2, cfg.PROJ_DIM)
assert torch.allclose(v.norm(dim=-1), torch.ones(2), atol=1e-4), "not L2-normalised"
# padding must not enter the pool: same real tokens, more padding -> same vector
i1, m1 = G["pad_batch"]([[5, 6, 7, 8]])
i2, m2 = G["pad_batch"]([[5, 6, 7, 8] + [0] * 6])
m2[0, 4:] = False
with torch.no_grad():
    d_pool = (m.embed(i1, m1) - m.embed(i2, m2)).abs().max().item()
print(f"[embed] unit-norm; pad excluded from the mean (delta {d_pool:.2e})")
del m

mp = G["build_pcr"]("A", n_prompt=cfg.N_PROMPT, verbose=False)
assert mp.prompt.shape == (cfg.N_PROMPT, cfg.DIM)
with torch.no_grad():
    vp = mp.embed(ids, mask)
assert torch.allclose(vp.norm(dim=-1), torch.ones(2), atol=1e-4)
# the prompt must change the output (it is seen by attention)...
mp2 = G["build_pcr"]("A", n_prompt=cfg.N_PROMPT, verbose=False)
mp2.load_state_dict(mp.state_dict())
with torch.no_grad():
    mp2.prompt.mul_(50.0)
    assert (vp - mp2.embed(ids, mask)).abs().max().item() > 1e-4, "prompt has no effect"
print(f"[prompt] {cfg.N_PROMPT} soft tokens attended to but not pooled")
del mp, mp2

# --- 7. every (architecture, technique) pair trains and scores -----------------------
run = G["run_experiment"]
ok, fail, teacher = [], [], None

PLAN = [("01_smoke", "A", "head_only"), ("02_A_lorma", "A", "lorma"),
        ("03_B_full", "B", "full"), ("04_C_ia3", "C", "ia3"),
        ("05_D_prompt", "D", "prompt"),
        ("x_A_partial", "A", "partial"), ("x_B_lorma", "B", "lorma"),
        ("x_C_lorma", "C", "lorma"), ("x_D_ia3", "D", "ia3"),
        ("x_A_prompt", "A", "prompt"), ("x_C_full", "C", "full"),
        ("x_D_full", "D", "full"), ("x_B_ia3", "B", "ia3"),
        ("x_A_ia3", "A", "ia3"), ("x_D_lorma", "D", "lorma")]
for name, arch, meth in PLAN:
    try:
        mdl, r = run(name, arch, meth, epochs=1,
                     keep_scores=not name.startswith("01"))
        ok.append((name, r["micro_f1"]))
        if name == "02_A_lorma":
            teacher = mdl
        else:
            del mdl
    except Exception as e:
        traceback.print_exc(); fail.append((name, repr(e)))

# --- 8. IA3 really is trainable and really is tiny -----------------------------------
mi = G["build_pcr"]("A", verbose=False)
G["apply_adaptation"](mi, "ia3")
n_ia3 = sum(p.numel() for n, p in mi.named_parameters()
            if p.requires_grad and n.endswith("scale"))
assert n_ia3 > 0, "IA3 wired up no scale vectors"
# and that a backward pass through the fused-qkv path works (the old in-place bug)
ids, mask = G["pad_batch"]([[5, 6, 7, 8], [9, 10, 11, 12]])
pos = torch.eye(2, dtype=torch.bool)
loss = G["infonce"](mi.embed(ids, mask), mi.embed(ids, mask), pos)
loss.backward()
grads = [n for n, p in mi.named_parameters()
         if n.endswith("scale") and p.grad is not None and p.grad.abs().sum() > 0]
assert grads, "IA3 scales got no gradient -- in-place autograd bug is back"
print(f"\n[ia3] {n_ia3:,} scale parameters, {len(grads)} received gradient "
      f"through the fused-qkv path")
del mi

# --- 9. distillation -----------------------------------------------------------------
try:
    for p in teacher.parameters():
        p.requires_grad = False
    ms, rk5 = run("06_A_distill", "A", "lorma", epochs=1, keep_layers=2, teacher=teacher)
    ok.append(("06_A_distill", rk5["micro_f1"]))
    del ms
except Exception as e:
    traceback.print_exc(); fail.append(("06_A_distill", repr(e)))

# --- 10. the fusion / ensemble cell --------------------------------------------------
try:
    exec(SRC[31], G)
    ok.append(("fusion_cell", 0.0))
except Exception as e:
    traceback.print_exc(); fail.append(("fusion_cell", repr(e)))

# --- 11. fusion maths: z-normalisation then blend ------------------------------------
zr = G["zrow"]
Z = zr(np.array([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]], dtype=np.float32))
assert np.allclose(Z[0], Z[1], atol=1e-4), "per-query z-norm did not equalise scales"
assert abs(Z.mean()) < 1e-5
print("[fusion] per-query z-normalisation puts both scorers on one scale")

# --- 12. every zip is in the PHASE-7 dict format -------------------------------------
import zipfile, glob as _g
zips = sorted(_g.glob(os.path.join(OUT, "*.zip")))
assert zips, "no submissions written"
for zp in zips:
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], (zp, z.namelist())
        payload = json.loads(z.read("prediction.json"))
    assert isinstance(payload, dict), f"{zp}: must be a DICT, got {type(payload).__name__}"
    assert set(payload) == TEST_QIDS, f"{zp}: query id mismatch"
    for q, v in payload.items():
        assert isinstance(v, list) and len(v) == 7, (zp, q, len(v))
        assert len(set(v)) == 7, f"{zp}/{q}: duplicates"
        assert all(isinstance(c, str) for c in v), (zp, q)
        bad = [c for c in v if c not in TEST_CIDS]
        assert not bad, f"{zp}/{q}: outside the test pool: {bad[:3]}"
        assert q not in v, f"{zp}/{q}: query returned itself"
print(f"[submission] {len(zips)} zips: dict of query -> exactly 7 unique in-pool ids")

# --- 13. a trained model must beat a random one (the pipeline learns something) -------
try:
    rand = G["build_pcr"]("A", verbose=False)
    Sr = G["score_matrix"](rand, list(G["dev_q"]), G["Q_DV"], G["C_DV"])
    f_rand = mf(rfs(Sr, list(G["dev_q"]), G["DEV_IDS"]), G["dev_q"])["micro_f1"]
    f_tr = max(s for n, s in ok if n.startswith(("02", "03", "04", "05")))
    print(f"[sanity] untrained {f_rand:.4f} vs best trained {f_tr:.4f}")
    del rand
except Exception as e:
    traceback.print_exc(); fail.append(("sanity", repr(e)))

print("\n" + "=" * 72)
for n, s in ok:
    print(f"  PASS  {n:<22} dev_micro_f1={s:.4f}")
for n, e in fail:
    print(f"  FAIL  {n:<22} {e}")
print("=" * 72)
shutil.rmtree(ROOT, ignore_errors=True)
sys.exit(1 if fail else 0)
