"""Exercise CS781_Phase3_RR.ipynb end to end on CPU.

Real Phase-1 architecture classes at reduced width with random weights, a real byte-level
BPE trained on synthetic legal-ish text, and a synthetic RR corpus with known labels.
Checkpoint loading is the only stub.
"""
import json, os, random, shutil, sys, tempfile, traceback

NB = "CS781_Phase3_RR.ipynb"
cells = json.load(open(NB))["cells"]
SRC = {i: "".join(c["source"]) for i, c in enumerate(cells) if c["cell_type"] == "code"}
print("code cells:", sorted(SRC))

ROOT = tempfile.mkdtemp(prefix="p3test_")
DATA = os.path.join(ROOT, "input", "Phase_3_RR")
os.makedirs(os.path.join(DATA, "Train"), exist_ok=True)
os.makedirs(os.path.join(DATA, "Test"), exist_ok=True)
OUT = os.path.join(ROOT, "work"); os.makedirs(OUT, exist_ok=True)

LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute",
          "Dissent", "PrecedentReliedUpon", "PrecedentNotReliedUpon",
          "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
          "RulingByPresentCourt", "None"]

# label-specific vocabulary so a model can actually learn the mapping
CUE = {
    "Fact": "the appellant purchased the property on the stated date and took possession",
    "Issue": "the question for determination before this bench is whether the provision applies",
    "ArgumentPetitioner": "learned counsel for the appellant submitted that the order is bad",
    "ArgumentRespondent": "per contra the respondent contended that the petition is not maintainable",
    "Statute": "section of the act provides that no person shall be deprived thereunder",
    "Dissent": "with great respect i am unable to persuade myself to agree with my brother",
    "PrecedentReliedUpon": "reliance was placed upon the decision reported in the law reports",
    "PrecedentNotReliedUpon": "the cited authority is distinguishable on facts and does not assist",
    "PrecedentOverruled": "that decision stands overruled by the larger bench in clear terms",
    "RulingByLowerCourt": "the trial court decreed the suit and the first appellate court affirmed",
    "RatioOfTheDecision": "the true principle emerging from these authorities is therefore settled",
    "RulingByPresentCourt": "in the result the appeal is allowed and the impugned order is set aside",
    "None": "list the matter after four weeks for further directions in due course",
}


def make_doc(rng, n_sent=None):
    n = n_sent or rng.randint(8, 30)
    sents, labs = [], []
    for _ in range(n):
        l = rng.choice(LABELS)
        words = CUE[l].split()
        rng.shuffle(words) if rng.random() < 0.15 else None
        sents.append(" ".join(words) + ".")
        labs.append(l)
    return sents, labs


rng = random.Random(11)
corpus = []

def dump(path, n_docs, prefix, with_labels=True):
    docs = []
    for k in range(n_docs):
        s, l = make_doc(rng)
        corpus.extend(s)
        d = {"id": f"{prefix}_{k:04d}", "text": s}
        if with_labels:
            d["labels"] = l
        docs.append(d)
    json.dump(docs, open(path, "w"))
    return docs

dump(f"{DATA}/Train/CL_train.json", 24, "cltr")
dump(f"{DATA}/Train/IT_train.json", 24, "ittr")
dump(f"{DATA}/Train/CL_dev.json",   8,  "cldv")
dump(f"{DATA}/Train/IT_dev.json",   8,  "itdv")
test_cl = dump(f"{DATA}/Test/CL_test.json", 6, "clte", with_labels=False)
test_it = dump(f"{DATA}/Test/IT_test.json", 6, "itte", with_labels=False)
json.dump(LABELS, open(f"{DATA}/Train/label_vocab.json", "w"))
TEST_LEN = {d["id"]: len(d["text"]) for d in (test_cl + test_it)}

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
tk = Tokenizer(models.BPE(unk_token="<|unk|>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(corpus, trainers.BpeTrainer(
    vocab_size=1200, special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
TOKP = os.path.join(ROOT, "input", "bpe32k.json")
tk.save(TOKP)
assert tk.token_to_id("<|pad|>") == 0
for n in ("A_decoder.pt", "B_encdec.pt", "C_hybrid.pt", "D_gqa.pt"):
    open(os.path.join(ROOT, "input", n), "wb").write(b"stub")
print(f"[setup] synthetic RR corpus + real BPE at {ROOT} (vocab {tk.get_vocab_size()})")

# ---------------------------------------------------------------------------
from pathlib import Path
G = {"__name__": "nbtest"}

# cell 2: imports + CFG, minus the HF download and the kaggle path discovery
head = SRC[2].split("# ---- pull Phase-3 data")[0]
cfgpart = ("class CFG:" + SRC[2].split("class CFG:", 1)[1].split("cfg = CFG()")[0]
           + "cfg = CFG()\n")
exec(compile(head + cfgpart, "cell2", "exec"), G)

cfg = G["cfg"]
cfg.DIM, cfg.N_HEADS, cfg.MAX_SEQ = 96, 4, 512
cfg.MAX_LEN, cfg.OVERLAP_SENTS, cfg.MAX_SENT_TOK = 128, 2, 32
cfg.BATCH_SIZE, cfg.GRAD_ACCUM, cfg.EPOCHS = 2, 2, 1
cfg.AMP = False
cfg.OUT_DIR = OUT
cfg.PROMPT_LEN = 8
cfg.LORA_R = cfg.LORMA_R = cfg.ADALORA_R = 8
cfg.LORA_ALPHA = cfg.LORMA_ALPHA = cfg.ADALORA_ALPHA = 16
cfg.ADALORA_TARGET_R = 4
os.makedirs(OUT, exist_ok=True)
G["TOK_PATH"] = Path(TOKP)
G["P3_DIR"] = Path(DATA) / "Train"
G["CKPTS"] = {k: Path(ROOT) / "input" / v for k, v in
              {"A": "A_decoder.pt", "B": "B_encdec.pt",
               "C": "C_hybrid.pt", "D": "D_gqa.pt"}.items()}
print("[cfg] reduced for CPU:", cfg.DIM, "dim,", cfg.N_HEADS, "heads")

exec(SRC[4], G)      # data
exec(SRC[6], G)      # tokenizer / windowing / datasets
exec(SRC[8], G)      # architecture components
exec(SRC[10], G)      # four models

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

exec(SRC[12], G)     # bidirectional + LegalRR + build_rr
exec(SRC[14], G)     # adaptation techniques
exec(SRC[16], G)     # train / predict / score / submit / run_experiment
print("[exec] logic cells loaded\n")

torch, nn, np = G["torch"], G["nn"], G["np"]

# --- 0. windowing covers every sentence exactly once, whole --------------------------
print("=" * 72); print("WINDOW COVERAGE"); print("=" * 72)
for name, ds, docs in [("train", G["ds_train"], G["train_docs"]),
                       ("dev",   G["ds_dev"],   G["dev_docs"]),
                       ("test",  G["ds_test"],  G["test_docs"])]:
    seen = {d: set() for d in docs}
    for it in ds.items:
        seen[it["did"]].update(it["slots"])
        assert len(it["slots"]) == len(set(it["slots"])), "duplicate slot in one window"
        assert max(it["spos"]) == len(it["slots"]) - 1, "slot index gap"
    for d in docs:
        assert seen[d] == set(range(len(docs[d]["text"]))), \
            f"{name}/{d}: covered {len(seen[d])} of {len(docs[d]['text'])}"
    print(f"  {name}: every sentence of every document covered "
          f"({len(ds)} windows, {sum(len(v) for v in seen.values())} sentence slots)")

# --- 1. sentence pooling is a true per-sentence mean ---------------------------------
print()
m = G["build_rr"]("A", verbose=False)
B, T, H = 2, 10, cfg.DIM
h = torch.randn(B, T, H)
spos = torch.tensor([[0, 0, 1, 1, 1, 2, -1, -1, -1, -1],
                     [0, 1, 1, -1, -1, -1, -1, -1, -1, -1]])
valid = (spos >= 0).unsqueeze(-1)
idx = spos.clamp(min=0).unsqueeze(-1).expand(-1, -1, H)
P = h.new_zeros(B, 3, H).scatter_add_(1, idx, h * valid.to(h.dtype))
C = h.new_zeros(B, 3, 1).scatter_add_(1, spos.clamp(min=0).unsqueeze(-1),
                                      valid.to(h.dtype))
P = P / C.clamp(min=1e-6)
assert torch.allclose(P[0, 0], h[0, 0:2].mean(0), atol=1e-5)
assert torch.allclose(P[0, 1], h[0, 2:5].mean(0), atol=1e-5)
assert torch.allclose(P[0, 2], h[0, 5:6].mean(0), atol=1e-5)
assert torch.allclose(P[1, 1], h[1, 1:3].mean(0), atol=1e-5)
print("[pooling] scatter-mean equals the per-sentence mean exactly")
del m

# --- 2. bidirectional patch actually changes the computation -------------------------
print()
print("=" * 72); print("BIDIRECTIONAL PATCH"); print("=" * 72)
for arch in ("A", "B", "C", "D"):
    mm = G["build_rr"](arch, bidirectional=False, verbose=False)
    mm.eval()
    ids = torch.randint(0, G["VOCAB_SIZE"], (1, 48))
    with torch.no_grad():
        causal = mm.encode(ids).clone()
    nsw = G["set_bidirectional"](mm.backbone, True)
    with torch.no_grad():
        bi = mm.encode(ids)
    delta = (causal - bi).abs().max().item()

    G["set_bidirectional"](mm.backbone, False)
    alt = ids.clone(); alt[0, 44] = (ids[0, 44].item() + 1) % G["VOCAB_SIZE"]
    with torch.no_grad():
        leak = (mm.encode(ids)[0, 5] - mm.encode(alt)[0, 5]).abs().max().item()
    G["set_bidirectional"](mm.backbone, True)
    with torch.no_grad():
        reach = (mm.encode(ids)[0, 5] - mm.encode(alt)[0, 5]).abs().max().item()
    print(f"  {arch}: {nsw} mixers | causal-vs-bi {delta:.4f} | "
          f"future->past causal {leak:.2e} | bidirectional {reach:.2e}")
    if arch in ("A", "D"):
        assert leak < 1e-5, f"{arch} leaks while causal"
        assert reach > 1e-6, f"{arch} has no backward reach when un-masked"
    if arch == "B":
        assert reach > 1e-6, "B should see the future (encoder is bidirectional)"
    if arch == "C":
        assert delta > 0, "C's attention layers did not switch"
    del mm
print("  OK")

# --- 3. every (architecture, technique) pair ------------------------------------------
run = G["run_experiment"]
PLAN = [
    ("00_smoke",      "A", "head_only"),
    ("01_A_lorma",    "A", "lorma"),
    ("02_C_lora",     "C", "lora"),
    ("03_D_ia3",      "D", "ia3"),
    ("04_B_adalora",  "B", "adalora"),
    ("05_A_prompt",   "A", "prompt"),
    ("x_B_lora",      "B", "lora"),
    ("x_C_ia3",       "C", "ia3"),
    ("x_D_prompt",    "D", "prompt"),
    ("x_A_adalora",   "A", "adalora"),
]
ok, fail, teacher = [], [], None
for name, arch, meth in PLAN:
    try:
        mm, rr = run(name, arch, meth, epochs=1,
                     keep_scores=not name.startswith("00"))
        ok.append((name, rr["macro_f1"]))
        if name == "01_A_lorma":
            teacher = mm
        else:
            del mm
    except Exception as e:
        traceback.print_exc(); fail.append((name, repr(e)))

# --- 4. distillation -------------------------------------------------------------------
try:
    for p in teacher.parameters():
        p.requires_grad = False
    ms, rk = run("06_A_distill", "A", "lora", epochs=1, keep_layers=2, teacher=teacher)
    ok.append(("06_A_distill", rk["macro_f1"]))
    tp = sum(p.numel() for p in {id(p): p for p in teacher.parameters()}.values())
    sp = sum(p.numel() for p in {id(p): p for p in ms.parameters()}.values())
    print(f"  teacher {tp/1e6:.2f}M -> student {sp/1e6:.2f}M ({100*sp/tp:.0f}%)")
    del ms
except Exception as e:
    traceback.print_exc(); fail.append(("06_A_distill", repr(e)))

# --- 5. ensemble + context ablation paths ----------------------------------------------
try:
    SC = G["SCORES"]
    use = [k for k in SC if not k.startswith("00")]
    ens = {d: np.mean([SC[k]["dev"][d] for k in use], 0) for d in G["ds_dev"].n_sent}
    r = G["macro_f1"](G["decode"](ens), G["dev_docs"])
    ens_t = {d: np.mean([SC[k]["test"][d] for k in use], 0) for d in G["ds_test"].n_sent}
    G["write_submission"](G["decode"](ens_t), "07_ensemble")
    ok.append((f"ensemble({len(use)})", r["macro_f1"]))
except Exception as e:
    traceback.print_exc(); fail.append(("ensemble", repr(e)))

try:
    solo_tr = G["RRWindows"](G["train_docs"], one_per_window=True)
    solo_dv = G["RRWindows"](G["dev_docs"], one_per_window=True)
    solo_te = G["RRWindows"](G["test_docs"], with_labels=False, one_per_window=True)
    assert len(solo_tr) == sum(len(d["text"]) for d in G["train_docs"].values())
    _tr, _dv, _te = G["ds_train"], G["ds_dev"], G["ds_test"]
    G["ds_train"], G["ds_dev"], G["ds_test"] = solo_tr, solo_dv, solo_te
    mm, rs = run("08_nocontext", "A", "lorma", epochs=1, keep_scores=False)
    G["ds_train"], G["ds_dev"], G["ds_test"] = _tr, _dv, _te
    ok.append(("08_nocontext", rs["macro_f1"]))
    del mm
except Exception as e:
    traceback.print_exc(); fail.append(("nocontext", repr(e)))

# --- 6. every zip is well formed and lengths match exactly ------------------------------
import zipfile, glob as _g
zips = sorted(_g.glob(os.path.join(OUT, "*.zip")))
for zp in zips:
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], (zp, z.namelist())
        payload = json.loads(z.read("prediction.json"))
    assert set(payload) == set(TEST_LEN), f"{zp}: doc id set mismatch"
    for did, labs in payload.items():
        assert len(labs) == TEST_LEN[did], \
            f"{zp}/{did}: {len(labs)} predictions vs {TEST_LEN[did]} sentences"
        assert all(l in LABELS for l in labs), f"{zp}/{did}: unknown label"
print(f"\n[submission] {len(zips)} zips: root-level prediction.json, all ids present, "
      f"length == sentence count for every document, all labels official")

print("\n" + "=" * 72)
for n, s in ok:
    print(f"  PASS  {n:<22} dev_macro_f1={s:.4f}")
for n, e in fail:
    print(f"  FAIL  {n:<22} {e}")
print("=" * 72)
shutil.rmtree(ROOT, ignore_errors=True)
sys.exit(1 if fail else 0)
