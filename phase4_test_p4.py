"""Exercise CS781_Phase4_CJPE.ipynb end to end on CPU.

Real Phase-1 architecture classes at reduced width with random weights, a real byte-level
BPE, and a synthetic CJPE corpus whose outcome signal sits at the END of each document, so
the representation comparison has something true to find. Checkpoint loading is the only stub.
"""
import json, os, random, shutil, sys, tempfile, traceback

NB = "CS781_Phase4_CJPE.ipynb"
cells = json.load(open(NB))["cells"]
SRC = {i: "".join(c["source"]) for i, c in enumerate(cells) if c["cell_type"] == "code"}
print("code cells:", sorted(SRC))

ROOT = tempfile.mkdtemp(prefix="p4test_")
DATA = os.path.join(ROOT, "input", "Phase_4_CJPE")
os.makedirs(os.path.join(DATA, "Train"), exist_ok=True)
os.makedirs(os.path.join(DATA, "Test"), exist_ok=True)
OUT = os.path.join(ROOT, "work"); os.makedirs(OUT, exist_ok=True)

FILLER = ("the appellant filed the present appeal against the impugned order passed by the "
          "learned tribunal after hearing counsel for both sides at considerable length on "
          "the questions of fact and law arising in these proceedings between the parties ")
WIN  = "in the result the appeal is allowed and the impugned order is hereby set aside with costs"
LOSE = "for these reasons the appeal is dismissed and the impugned order is affirmed in its entirety"

def make_doc(rng, label, n_chunks=3):
    # filler body, then the decisive sentence at the very END
    body = (FILLER * (n_chunks * 14))[: n_chunks * 900]
    tail = (WIN if label == 1 else LOSE)
    return body + " " + (tail + " ") * 3

rng = random.Random(5)
corpus = []

def dump(path, n, prefix, with_label=True, as_dict=False):
    recs, d = [], {}
    for k in range(n):
        lab = rng.choice([0, 0, 1])          # deliberately imbalanced
        t = make_doc(rng, lab, rng.randint(2, 4))
        corpus.append(t[:4000])
        did = f"{prefix}_{k:04d}"
        if as_dict:
            d[did] = ({"text": t, "label": lab} if with_label else {"text": t})
        else:
            r = {"id": did, "text": t}
            if with_label:
                r["label"] = lab
            recs.append(r)
    json.dump(d if as_dict else recs, open(path, "w"))
    return d if as_dict else recs

tr = dump(f"{DATA}/Train/train.json", 60, "tr")
dv = dump(f"{DATA}/Train/dev.json", 24, "dv")
# test deliberately stored in the OTHER shape, to exercise the tolerant loader
te = dump(f"{DATA}/Test/test.json", 18, "te", with_label=False, as_dict=True)
TEST_IDS = set(te)

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
print(f"[setup] synthetic CJPE corpus + real BPE at {ROOT} (vocab {tk.get_vocab_size()})")

# ---------------------------------------------------------------------------
from pathlib import Path
G = {"__name__": "nbtest"}

head = SRC[2].split("# ---- pull Phase-4 data")[0]
cfgpart = ("class CFG:" + SRC[2].split("class CFG:", 1)[1].split("cfg = CFG()")[0]
           + "cfg = CFG()\n")
exec(compile(head + cfgpart, "cell2", "exec"), G)

cfg = G["cfg"]
cfg.DIM, cfg.N_HEADS, cfg.MAX_SEQ = 96, 4, 512
cfg.CHUNK_LEN, cfg.MAX_CHUNKS = 128, 3
cfg.BATCH_SIZE, cfg.GRAD_ACCUM, cfg.EPOCHS = 2, 2, 1
cfg.AMP = False
cfg.OUT_DIR = OUT
cfg.LORMA_R = cfg.ADALORA_R = 8
cfg.LORMA_ALPHA = cfg.ADALORA_ALPHA = 16
cfg.ADALORA_TARGET_R = 4
cfg.ADAPTER_DIM = 16
G["TOK_PATH"] = Path(TOKP)
G["P4_TRAIN"] = Path(DATA) / "Train" / "train.json"
G["CKPTS"] = {k: Path(ROOT) / "input" / v for k, v in
              {"A": "A_decoder.pt", "B": "B_encdec.pt",
               "C": "C_hybrid.pt", "D": "D_gqa.pt"}.items()}
print("[cfg] reduced for CPU:", cfg.DIM, "dim,", cfg.CHUNK_LEN, "tokens/chunk")

exec(SRC[4], G)      # data
exec(SRC[6], G)      # tokenizer + chunking + datasets
exec(SRC[8], G)      # architecture components
exec(SRC[10], G)     # four models

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

exec(SRC[12], G)     # bidirectional + CJPEModel + build_cjpe
exec(SRC[14], G)     # adaptation techniques
exec(SRC[16], G)     # train / predict / score / threshold / submit / run_experiment
print("[exec] logic cells loaded\n")

torch, nn, np = G["torch"], G["nn"], G["np"]

# --- 0. loaders handled BOTH json shapes ---------------------------------------------
assert len(G["train_docs"]) == 60 and len(G["dev_docs"]) == 24
assert len(G["test_docs"]) == 18 and set(G["test_docs"]) == TEST_IDS
assert all("label" not in d for d in G["test_docs"].values())
print("[data] list-shaped train/dev and dict-shaped test both parsed")

# --- 1. the three representations pick the chunks they claim to --------------------
dc = G["doc_chunks"]
did0 = list(G["train_docs"])[0]
txt0 = G["train_docs"][did0]["text"]
full = G["tok"].encode(txt0).ids
c_last  = dc(did0, txt0, "last")
c_first = dc(did0, txt0, "first")
c_hier  = dc(did0, txt0, "hier")
assert len(c_last) == 1 and c_last[0] == full[-cfg.CHUNK_LEN:]
assert len(c_first) == 1 and c_first[0] == full[:cfg.CHUNK_LEN]
assert 1 <= len(c_hier) <= cfg.MAX_CHUNKS
assert c_hier[-1] == full[(len(full) - 1) // cfg.CHUNK_LEN * cfg.CHUNK_LEN:], \
    "hier should end on the document's final chunk"
assert len(full) > cfg.CHUNK_LEN, "test doc too short to distinguish the strategies"
print(f"[repr] doc has {len(full)} tokens -> last=1 first=1 hier={len(c_hier)} chunks, "
      f"hier anchored at the end")

# --- 2. two-level pooling is exactly mean-then-attention ----------------------------
m = G["build_cjpe"]("A", verbose=False)
H = cfg.DIM
h = torch.randn(4, 6, H)                       # 4 chunks
tm = torch.tensor([[1,1,1,0,0,0],[1,1,0,0,0,0],[1,1,1,1,0,0],[1,0,0,0,0,0]]).bool()
owner = torch.tensor([0, 0, 1, 1])
mm = tm.unsqueeze(-1).to(h.dtype)
cvec = (h * mm).sum(1) / mm.sum(1).clamp(min=1e-6)
assert torch.allclose(cvec[0], h[0, :3].mean(0), atol=1e-5)
assert torch.allclose(cvec[3], h[3, :1].mean(0), atol=1e-5)
counts = torch.bincount(owner, minlength=2)
assert counts.tolist() == [2, 2]
slot = torch.cat([torch.arange(int(c)) for c in counts])
assert slot.tolist() == [0, 1, 0, 1]
P = cvec.new_zeros(2, 2, H); P[owner, slot] = cvec
assert torch.allclose(P[0, 0], cvec[0]) and torch.allclose(P[1, 1], cvec[3])
# attention weights must sum to one per document
s = m.chunk_attn(P).squeeze(-1).softmax(-1)
assert torch.allclose(s.sum(-1), torch.ones(2), atol=1e-5)
print("[pooling] per-chunk mean exact; chunk grouping correct; attention normalised")
del m

# --- 3. bidirectional patch -----------------------------------------------------------
print()
for arch in ("A", "B", "C", "D"):
    mm_ = G["build_cjpe"](arch, bidirectional=False, verbose=False)
    mm_.eval()
    ids = torch.randint(0, G["VOCAB_SIZE"], (1, 48))
    with torch.no_grad():
        causal = mm_.backbone.trunk(ids).clone()
    nsw = G["set_bidirectional"](mm_.backbone, True)
    with torch.no_grad():
        delta = (causal - mm_.backbone.trunk(ids)).abs().max().item()
    G["set_bidirectional"](mm_.backbone, False)
    alt = ids.clone(); alt[0, 44] = (ids[0, 44].item() + 1) % G["VOCAB_SIZE"]
    with torch.no_grad():
        leak = (mm_.backbone.trunk(ids)[0, 5]
                - mm_.backbone.trunk(alt)[0, 5]).abs().max().item()
    G["set_bidirectional"](mm_.backbone, True)
    with torch.no_grad():
        reach = (mm_.backbone.trunk(ids)[0, 5]
                 - mm_.backbone.trunk(alt)[0, 5]).abs().max().item()
    print(f"  {arch}: {nsw} mixers | causal-vs-bi {delta:.4f} | "
          f"future->past causal {leak:.2e} | bidirectional {reach:.2e}")
    if arch in ("A", "D"):
        assert leak < 1e-5 and reach > 1e-6, f"{arch} bidirectional patch wrong"
    if arch == "C":
        assert delta > 0
    del mm_

# --- 4. every (architecture, technique) pair, under every representation -------------
run = G["run_experiment"]
ok, fail, teacher = [], [], None

for mode in ("last", "first", "hier"):
    cfg.REPR = mode
    G["ds_train"], G["ds_dev"], G["ds_test"] = G["make_sets"](mode)
    try:
        _, r = run(f"r_{mode}", "A", "lorma", epochs=1, write=False, keep_scores=False)
        ok.append((f"repr:{mode}", r["macro_f1"]))
    except Exception as e:
        traceback.print_exc(); fail.append((f"repr:{mode}", repr(e)))

cfg.REPR = "hier"            # the most demanding path for the remaining checks
G["ds_train"], G["ds_dev"], G["ds_test"] = G["make_sets"]("hier")

PLAN = [("00_smoke", "A", "head_only"), ("01_A_lorma", "A", "lorma"),
        ("02_C_adapter", "C", "adapter"), ("03_D_adalora", "D", "adalora"),
        ("04_B_full", "B", "full"), ("05_A_bitfit", "A", "bitfit"),
        ("x_A_partial", "A", "partial"), ("x_B_adapter", "B", "adapter"),
        ("x_C_adalora", "C", "adalora"), ("x_D_bitfit", "D", "bitfit"),
        ("x_A_adapter", "A", "adapter"), ("x_C_lorma", "C", "lorma")]
for name, arch, meth in PLAN:
    try:
        mdl, r = run(name, arch, meth, epochs=1, keep_scores=not name.startswith("00"))
        ok.append((name, r["macro_f1"]))
        if name == "01_A_lorma":
            teacher = mdl
        else:
            del mdl
    except Exception as e:
        traceback.print_exc(); fail.append((name, repr(e)))

# --- 5. distillation -------------------------------------------------------------------
try:
    for p in teacher.parameters():
        p.requires_grad = False
    ms, rk = run("06_A_distill", "A", "lorma", epochs=1, keep_layers=2, teacher=teacher)
    ok.append(("06_A_distill", rk["macro_f1"]))
    del ms
except Exception as e:
    traceback.print_exc(); fail.append(("06_A_distill", repr(e)))

# --- 6. ensemble cell ------------------------------------------------------------------
try:
    SC, RES = G["SCORES"], G["RESULTS"]
    ranked = [r["name"] for r in sorted(RES, key=lambda r: -r["dev_f1_tuned"])
              if r["name"] in SC and SC[r["name"]]["test"] is not None]
    assert len(ranked) >= 3
    ed = {d: float(np.mean([SC[n]["dev"][d] for n in ranked[:3]])) for d in G["dev_docs"]}
    t, f1 = G["sweep_threshold"](ed, G["dev_docs"])
    et = {d: float(np.mean([SC[n]["test"][d] for n in ranked[:3]])) for d in G["test_docs"]}
    G["write_submission"](et, "07_ensemble", thr=t)
    ok.append((f"ensemble({len(ranked[:3])})", f1))
except Exception as e:
    traceback.print_exc(); fail.append(("ensemble", repr(e)))

# --- 7. every zip is in the PHASE-4 list format -----------------------------------------
import zipfile, glob as _g
zips = sorted(_g.glob(os.path.join(OUT, "*.zip")))
for zp in zips:
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], (zp, z.namelist())
        payload = json.loads(z.read("prediction.json"))
    assert isinstance(payload, list), f"{zp}: must be a LIST, got {type(payload).__name__}"
    assert len(payload) == len(TEST_IDS), f"{zp}: {len(payload)} rows vs {len(TEST_IDS)}"
    assert {r["id"] for r in payload} == TEST_IDS, f"{zp}: id mismatch"
    for r in payload:
        assert set(r) == {"id", "pred"}, r
        assert isinstance(r["pred"], int) and not isinstance(r["pred"], bool), r
        assert r["pred"] in (0, 1), r
print(f"\n[submission] {len(zips)} zips: list of objects, int preds in (0,1), "
      f"every test id present exactly once")

# --- 8. threshold sweep genuinely finds a better cut than 0.5 ---------------------------
probs = {d: (0.9 if G["dev_docs"][d]["label"] == 1 else 0.55) for d in G["dev_docs"]}
t, f1 = G["sweep_threshold"](probs, G["dev_docs"])
base = G["macro_f1_binary"]({d: int(p >= 0.5) for d, p in probs.items()},
                            G["dev_docs"])["macro_f1"]
assert f1 > base, f"sweep failed to beat 0.5 on a separable case ({f1} vs {base})"
print(f"[threshold] on a separable case: 0.50 -> {base:.3f}, swept {t:.2f} -> {f1:.3f}")

print("\n" + "=" * 72)
for n, s in ok:
    print(f"  PASS  {n:<22} dev_macro_f1={s:.4f}")
for n, e in fail:
    print(f"  FAIL  {n:<22} {e}")
print("=" * 72)
shutil.rmtree(ROOT, ignore_errors=True)
sys.exit(1 if fail else 0)
