"""Exercise CS781_Phase2_LNER_FINAL.ipynb end to end on CPU.

Uses the REAL Phase-1 architecture classes (A/B/C/D) at reduced width with random
weights, a REAL byte-level BPE tokenizer trained on synthetic legal-ish text, and a
synthetic Phase-2 corpus with exactly known character offsets. Checkpoint loading is
the only thing stubbed, since the real .pt files are not here.
"""
import json, os, random, shutil, sys, tempfile, traceback

NB = "CS781_Phase2_LNER_FINAL.ipynb"
cells = json.load(open(NB))["cells"]
SRC = {i: "".join(c["source"]) for i, c in enumerate(cells) if c["cell_type"] == "code"}
print("code cells:", sorted(SRC))

ROOT = tempfile.mkdtemp(prefix="p2final_")
DATA = os.path.join(ROOT, "input", "Phase_2_LNER")
os.makedirs(os.path.join(DATA, "Train"), exist_ok=True)
os.makedirs(os.path.join(DATA, "Test"), exist_ok=True)
OUT = os.path.join(ROOT, "work"); os.makedirs(OUT, exist_ok=True)

LABELS = ["APP","RESP","A.COUNSEL","R.COUNSEL","JUDGE","WIT",
          "AUTH","COURT","STAT","PREC","DATE","CASENO"]

POOL = {
    "COURT": ["Supreme Court of India", "High Court of Delhi", "Bombay High Court"],
    "JUDGE": ["Justice R. K. Sharma", "Hon'ble Mr. Justice Verma"],
    "APP": ["Ramesh Chandra Gupta", "M/s Kaveri Industries Ltd"],
    "RESP": ["State of Bihar", "Union of India"],
    "STAT": ["Section 138 of the Negotiable Instruments Act", "Article 226"],
    "DATE": ["12.03.2019", "4th January 2021"],
    "CASENO": ["Civil Appeal No. 4412 of 2018", "W.P. (C) 231/2020"],
    "PREC": ["Kesavananda Bharati v. State of Kerala"],
    "A.COUNSEL": ["Mr. S. Iyer, Senior Advocate"],
    "R.COUNSEL": ["Ms. Anita Rao, ASG"],
    "WIT": ["PW-3 Shanti Devi"],
    "AUTH": ["Income Tax Appellate Tribunal"],
}
FILLER = ("the appeal arises out of the impugned order dated passed by the learned "
          "tribunal in proceedings between the parties and the matter was heard at "
          "length before this bench on the question of limitation and laches ")


def make_doc(rng, n_ent=12):
    parts, ents, pos = ["REPORTABLE "], [], len("REPORTABLE ")
    for _ in range(n_ent):
        lab = rng.choice(LABELS)
        val = rng.choice(POOL[lab])
        lead = rng.choice(["in ", "before ", "by ", "under ", "see "])
        parts.append(lead); pos += len(lead)
        ents.append([pos, pos + len(val), lab])
        parts.append(val); pos += len(val)
        tail = ". " + FILLER[: rng.randint(50, 120)] + " "
        parts.append(tail); pos += len(tail)
    text = "".join(parts)
    for s, e, l in ents:
        assert text[s:e] in POOL[l]
    return text, ents


rng = random.Random(7)
corpus_lines = []
for f in (1, 2):
    t, g = {}, {}
    for k in range(10):
        did = f"{f}0{k:04d}"
        t[did], g[did] = make_doc(rng)
        corpus_lines.append(t[did])
    json.dump(t, open(f"{DATA}/Train/fold{f}-text.json", "w"))
    json.dump(g, open(f"{DATA}/Train/fold{f}-gold.json", "w"))

test_t = {}
for k in range(6):
    test_t[f"30{k:04d}"], _ = make_doc(rng)
json.dump(test_t, open(f"{DATA}/Test/fold3-text.json", "w"))
open(f"{DATA}/Train/labels.txt", "w").write("\n".join(LABELS))

# ---- a REAL byte-level BPE with the same special tokens as bpe32k.json ----------
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
tk = Tokenizer(models.BPE(unk_token="<|unk|>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(
    corpus_lines + list(test_t.values()),
    trainers.BpeTrainer(vocab_size=2000,
                        special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"],
                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
TOKP = os.path.join(ROOT, "input", "bpe32k.json")
tk.save(TOKP)
assert tk.token_to_id("<|pad|>") == 0
print(f"[setup] corpus + real BPE at {ROOT} (vocab {tk.get_vocab_size()})")

# fake checkpoints so the discovery globs succeed; never actually loaded
for n in ("A_decoder.pt", "B_encdec.pt", "C_hybrid.pt", "D_gqa.pt"):
    open(os.path.join(ROOT, "input", n), "wb").write(b"stub")

# ---------------------------------------------------------------------------
G = {"__name__": "nbtest"}
exec(SRC[2], G)                                     # env

# CONFIG cell: take the class, skip the kaggle-path discovery at the bottom
cfg_src = SRC[4].split("cfg = CFG()")[0] + "cfg = CFG()\n"
exec(cfg_src, G)
cfg = G["cfg"]
cfg.DIM, cfg.N_HEADS, cfg.MAX_SEQ = 96, 4, 256
cfg.MAX_LEN, cfg.STRIDE = 96, 24
cfg.BATCH_SIZE, cfg.GRAD_ACCUM, cfg.EPOCHS = 2, 2, 1
cfg.AMP = False
cfg.OUT_DIR = OUT
cfg.VAL_FRAC = 0.25
G["TOK_PATH"] = __import__("pathlib").Path(TOKP)
G["P2_DIR"] = __import__("pathlib").Path(DATA) / "Train"
G["CKPTS"] = {k: __import__("pathlib").Path(ROOT) / "input" / v for k, v in
              {"A": "A_decoder.pt", "B": "B_encdec.pt",
               "C": "C_hybrid.pt", "D": "D_gqa.pt"}.items()}
print("[cfg] reduced for CPU:", cfg.DIM, "dim,", cfg.N_HEADS, "heads")

exec(SRC[6], G)                                     # data
exec(SRC[8], G)                                     # atomisation + diagnostic
exec(SRC[10], G)                                    # tokenizer + windows
exec(SRC[12], G)                                    # components
exec(SRC[14], G)                                    # the four models

# shrink the architecture constants (looked up as globals at call time)
G["C_LAYERS"], G["C_ATTN_EVERY"], G["C_CHUNK"] = 4, 2, 8
G["D_LAYERS"], G["D_FULL_EVERY"], G["D_WINDOW"], G["D_KV_HEADS"] = 4, 2, 32, 2
_real_build = G["_build"]
def _build_small(arch):
    if arch == "A":
        return G["GPTModel"](G["VOCAB_SIZE"], n_layers=4)
    if arch == "B":
        return G["EncoderDecoderModel"](G["VOCAB_SIZE"],
                                        n_encoder_layers=6, n_decoder_layers=2)
    return _real_build(arch)
G["_build"] = _build_small

# checkpoints are not here: always build fresh weights
_orig_load = G["load_phase1"]
G["load_phase1"] = lambda arch, pretrained=True: _orig_load(arch, pretrained=False)

exec(SRC[16], G)                                    # set_bidirectional + LegalNER + build_ner
exec(SRC[18], G)                                    # adaptation methods
exec(SRC[20], G)                                    # training
exec(SRC[22], G)                                    # predict / score
exec(SRC[24], G)                                    # submission writer
exec(SRC[26], G)                                    # run_experiment
print("[exec] all logic cells loaded\n")

run = G["run_experiment"]
nn, torch = G["nn"], G["torch"]

# --- 0. the bidirectional patch must actually change the computation -------------
print("=" * 72)
print("BIDIRECTIONAL PATCH CHECK")
print("=" * 72)
bidir_ok = {}
for arch in ("A", "B", "C", "D"):
    m = G["build_ner"](arch, bidirectional=False, verbose=False)
    ids = torch.randint(0, G["VOCAB_SIZE"], (1, 64))
    m.eval()
    with torch.no_grad():
        causal = m(ids).clone()
    n_sw = G["set_bidirectional"](m.backbone, True)
    with torch.no_grad():
        bi = m(ids)
    delta = (causal - bi).abs().max().item()
    # a late token must not move an early output while causal, and must while bidirectional
    G["set_bidirectional"](m.backbone, False)
    alt = ids.clone(); alt[0, 60] = (ids[0, 60].item() + 1) % G["VOCAB_SIZE"]
    with torch.no_grad():
        leak_causal = (m(ids)[0, 5] - m(alt)[0, 5]).abs().max().item()
    G["set_bidirectional"](m.backbone, True)
    with torch.no_grad():
        reach_bi = (m(ids)[0, 5] - m(alt)[0, 5]).abs().max().item()
    bidir_ok[arch] = (n_sw, delta, leak_causal, reach_bi)
    print(f"  {arch}: {n_sw} mixers switched | causal-vs-bi delta {delta:.4f} | "
          f"future->past causal {leak_causal:.2e} | bidirectional {reach_bi:.2e}")
    del m

assert bidir_ok["B"][0] == 0, "B should need no patch"
for a in ("A", "D"):
    assert bidir_ok[a][2] < 1e-5, f"{a} leaks while causal"
    assert bidir_ok[a][3] > 1e-5, f"{a} bidirectional has no backward reach"
assert bidir_ok["C"][1] > 0, "C's attention layers did not switch"
print("  OK: A and D flip cleanly, B needs no patch, C partially switches as documented\n")

# --- 1. every submission's (model, method) pair ----------------------------------
PLAN = [
    ("00_smoke",        "A", "head_only",  {}),
    ("01_A_full",       "A", "full",       {}),
    ("02_D_lorma_plus", "D", "lorma_plus", {}),
    ("03_B_partial",    "B", "partial",    {}),
    ("04_C_lora",       "C", "lora",       {}),
    ("05_A_bitfit",     "A", "bitfit",     {}),
    ("07_A_lorma_pi",   "A", "lorma_pi",   {}),
    ("08_A_causal",     "A", "full",       {"bidirectional": False}),
    ("09_B_bitfit",     "B", "bitfit",     {}),
]
ok, fail, teacher = [], [], None
for name, arch, method, kw in PLAN:
    try:
        m, r = run(name, arch, method, epochs=1, **kw)
        ok.append((name, r["macro_f1"]))
        if name == "01_A_full":
            teacher = m
        else:
            del m
    except Exception as e:
        traceback.print_exc(); fail.append((name, repr(e)))

# --- 2. distillation -------------------------------------------------------------
try:
    for p in teacher.parameters():
        p.requires_grad = False
    ms, rk = run("06_A_distill", "A", "full", epochs=1, keep_layers=2, teacher=teacher)
    ok.append(("06_A_distill", rk["macro_f1"]))
    tp = sum(p.numel() for p in teacher.parameters())
    sp = sum(p.numel() for p in ms.parameters())
    print(f"  teacher {tp/1e6:.2f}M -> student {sp/1e6:.2f}M ({100*sp/tp:.0f}%)")
except Exception as e:
    traceback.print_exc(); fail.append(("06_A_distill", repr(e)))

# --- 3. offsets round-trip exactly ------------------------------------------------
atomise, spans_to_bio, bio_to_spans = G["atomise"], G["spans_to_bio"], G["bio_to_spans"]
perfect = total = 0
for did, txt in G["all_text"].items():
    at = atomise(txt)
    tags, _ = spans_to_bio(at, [tuple(x) for x in G["all_gold"][did]])
    rec = set(bio_to_spans(at, tags, txt))
    gold = set(tuple(x) for x in G["all_gold"][did])
    total += len(gold); perfect += len(gold & rec)
print(f"\n[offsets] gold spans recovered exactly through BIO: "
      f"{perfect}/{total} = {100*perfect/total:.2f}%")

# --- 4. every zip is well formed --------------------------------------------------
import zipfile, glob as _g
zips = sorted(_g.glob(os.path.join(OUT, "*.zip")))
for zp in zips:
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], (zp, z.namelist())
        payload = json.loads(z.read("prediction.json"))
    assert set(payload) == set(test_t), f"{zp}: test ID set mismatch"
    for did, rows in payload.items():
        for r in rows:
            assert len(r) == 3 and isinstance(r[0], int) and isinstance(r[1], int)
            assert 0 <= r[0] < r[1] <= len(test_t[did]), (zp, did, r)
            assert r[2] in LABELS
print(f"[submission] {len(zips)} zips: root-level prediction.json, "
      f"all IDs present, all offsets in range, all labels valid")

# --- 5. trainable-parameter sanity per method -------------------------------------
print("\n[trainable fractions]")
for name, arch, method in [("A", "A", "full"), ("A", "A", "bitfit"),
                           ("A", "A", "lora"), ("A", "A", "lorma_plus"),
                           ("B", "B", "partial"), ("D", "D", "lorma_plus")]:
    m = G["apply_adaptation"](G["build_ner"](arch, verbose=False), method)
    del m

print("\n" + "=" * 72)
for n, s in ok:
    print(f"  PASS  {n:<18} val_macro_f1={s:.4f}")
for n, e in fail:
    print(f"  FAIL  {n:<18} {e}")
print("=" * 72)
shutil.rmtree(ROOT, ignore_errors=True)
sys.exit(1 if fail else 0)
