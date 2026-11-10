"""Exercise CS781_Phase6_Summ.ipynb end to end on CPU.

Real Phase-1 architecture classes at reduced width with random weights, a real byte-level
BPE, and a synthetic summarization corpus whose summary is a genuine compression of its
document, so ROUGE has something true to measure. Checkpoint loading is the only stub.
"""
import json, os, random, shutil, sys, tempfile, traceback

NB = "CS781_Phase6_Summ.ipynb"
cells = json.load(open(NB))["cells"]
SRC = {i: "".join(c["source"]) for i, c in enumerate(cells) if c["cell_type"] == "code"}
print("code cells:", sorted(SRC))

ROOT = tempfile.mkdtemp(prefix="p6test_")
DATA = os.path.join(ROOT, "input", "Phase_6_Summ")
for d in ("Train", "Test"):
    os.makedirs(os.path.join(DATA, d), exist_ok=True)
OUT = os.path.join(ROOT, "work"); os.makedirs(OUT, exist_ok=True)

rng = random.Random(11)
HOLD = ["appellant", "respondent", "tribunal", "statute", "injunction", "decree",
        "acquittal", "conviction", "arbitration", "jurisdiction"]
FILL = ("the learned counsel appearing for the parties submitted at considerable length "
        "that the impugned order suffers from an error apparent on the face of the record ")
corpus = []

def make_pair(k):
    topic = HOLD[k % len(HOLD)]
    facts = [f"the court held that the {topic} was entitled to relief",
             f"the matter concerning {topic} was remitted for fresh consideration",
             f"costs in the {topic} proceedings were made easy"]
    body = []
    for f in facts:
        body.append(f)
        body.append(FILL * 2)
    doc = " . ".join(body) * 2
    summ = " . ".join(facts)
    return doc, summ

def dump(path, n, prefix, with_summary=True, as_dict=False):
    recs, dd = [], {}
    for k in range(n):
        doc, summ = make_pair(k)
        corpus.append(doc[:3000]); corpus.append(summ)
        did = f"{prefix}_{k:04d}"
        if as_dict:
            dd[did] = ({"document": [doc], "summary": summ} if with_summary
                       else {"document": [doc]})
        else:
            r = {"id": did, "document": doc}
            if with_summary:
                r["summary"] = summ
            recs.append(r)
    json.dump(dd if as_dict else recs, open(path, "w"))
    return dd if as_dict else recs

dump(f"{DATA}/Train/train.json", 90, "tr")
te = dump(f"{DATA}/Test/test.json", 20, "te", with_summary=False, as_dict=True)
TEST_IDS = set(te)

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
tk = Tokenizer(models.BPE(unk_token="<|unk|>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(corpus, trainers.BpeTrainer(
    vocab_size=800, special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
TOKP = os.path.join(ROOT, "input", "bpe32k.json")
tk.save(TOKP)
assert tk.token_to_id("<|pad|>") == 0
for n in ("A_decoder.pt", "B_encdec.pt", "C_hybrid.pt", "D_gqa.pt"):
    open(os.path.join(ROOT, "input", n), "wb").write(b"stub")
print(f"[setup] synthetic summarization corpus + real BPE at {ROOT} "
      f"(vocab {tk.get_vocab_size()})")

# ---------------------------------------------------------------------------
from pathlib import Path
G = {"__name__": "nbtest", "display": print}

head = SRC[2].split("# ---- pull Phase-6 data")[0]
cfgpart = ("class CFG:" + SRC[2].split("class CFG:", 1)[1].split("cfg = CFG()")[0]
           + "cfg = CFG()\n")
exec(compile(head + cfgpart, "cell2", "exec"), G)

cfg = G["cfg"]
cfg.DIM, cfg.N_HEADS, cfg.MAX_SEQ = 96, 4, 512
cfg.SRC_LEN, cfg.SRC_LEN_DEC, cfg.TGT_LEN = 128, 96, 48
cfg.MAX_NEW, cfg.MIN_NEW, cfg.NO_REPEAT = 24, 4, 3
cfg.VAL_FRAC, cfg.VAL_CAP = 0.15, 8
cfg.EPOCHS, cfg.MAX_STEPS = 1, 4
cfg.BATCH_SIZE, cfg.GRAD_ACCUM, cfg.GEN_BS = 2, 2, 4
cfg.AMP = False
cfg.N_PROMPT, cfg.ADAPTER_DIM = 8, 16
cfg.LORMA_R = cfg.LORA_R = 8
cfg.LORMA_ALPHA = cfg.LORA_ALPHA = 16
cfg.C_CHUNK_OVERRIDE = 4
cfg.OUT_DIR = OUT
G["TOK_PATH"] = Path(TOKP)
G["P6_TEST"] = Path(DATA) / "Test" / "test.json"
G["P6_ROOT"] = Path(DATA)
G["CKPTS"] = {k: Path(ROOT) / "input" / v for k, v in
              {"A": "A_decoder.pt", "B": "B_encdec.pt",
               "C": "C_hybrid.pt", "D": "D_gqa.pt"}.items()}
print("[cfg] reduced for CPU:", cfg.DIM, "dim, src", cfg.SRC_LEN, "tgt", cfg.TGT_LEN)

exec(SRC[4], G)      # data + held-out split
exec(SRC[6], G)      # rouge + submission writer
exec(SRC[8], G)      # lead baseline
exec(SRC[10], G)     # tokenise + datasets
exec(SRC[12], G)     # architecture components
exec(SRC[14], G)     # four models

G["C_LAYERS"], G["C_ATTN_EVERY"], G["C_CHUNK"] = 4, 2, 4
G["D_LAYERS"], G["D_FULL_EVERY"], G["D_WINDOW"], G["D_KV_HEADS"] = 4, 2, 32, 2
_real_build = G["_build"]
def _build_small(arch):
    if arch == "A":
        return G["GPTModel"](G["VOCAB_SIZE"], n_layers=4)
    if arch == "B":
        return G["EncoderDecoderModel"](G["VOCAB_SIZE"], n_encoder_layers=2,
                                        n_decoder_layers=4)
    return _real_build(arch)
G["_build"] = _build_small
_orig_load = G["load_phase1"]
G["load_phase1"] = lambda arch, pretrained=True: _orig_load(arch, pretrained=False)

exec(SRC[16], G)     # SummModel + generate + build_summ
exec(SRC[18], G)     # adaptation techniques
exec(SRC[20], G)     # train / run_experiment
print("[exec] logic cells loaded\n")

torch, nn, np, F = G["torch"], G["nn"], G["np"], G["F"]

# --- 0. data, and the held-out split is deterministic --------------------------------
assert len(G["test_docs"]) == 20 and set(G["test_docs"]) == TEST_IDS
assert all("summary" not in v for v in G["test_docs"].values())
assert len(G["val_docs"]) > 0 and len(G["train_docs"]) > 0
assert not (set(G["val_docs"]) & set(G["train_docs"])), "val leaked into train"
assert len(G["val_docs"]) + len(G["train_docs"]) <= 90
b = G["_bucket"]
assert all(b(d) < cfg.VAL_FRAC for d in G["val_docs"]), "split not by hash"
assert b("tr_0001") == b("tr_0001"), "split not deterministic"
assert isinstance(next(iter(G["test_docs"].values()))["document"], str), "list not joined"
print(f"[data] train {len(G['train_docs'])} / val {len(G['val_docs'])} / "
      f"test {len(G['test_docs'])}, hash split disjoint and reproducible")

# --- 1. ROUGE is correct on hand cases -------------------------------------------------
r1, rl = G["rouge1"], G["rougeL"]
assert abs(r1("a b c", "a b c") - 1.0) < 1e-9
assert r1("x y z", "a b c") == 0.0
# 2 of 3 overlap both ways -> P=R=2/3 -> F1=2/3
assert abs(r1("a b q", "a b c") - 2 / 3) < 1e-9, r1("a b q", "a b c")
assert abs(rl("a b c", "a b c") - 1.0) < 1e-9
# LCS of "a c" within "a b c" is 2 -> P=1.0 R=2/3 -> F1=0.8
assert abs(rl("a c", "a b c") - 0.8) < 1e-9, rl("a c", "a b c")
# order matters for L but not for 1
assert r1("c b a", "a b c") == 1.0 and rl("c b a", "a b c") < 1.0
assert r1("", "a b") == 0.0 and rl("", "a b") == 0.0
print("[rouge] unigram F1 and LCS F1 exact on hand cases; order-sensitivity correct")

# --- 2. the lead baseline found something ----------------------------------------------
lead_r1 = [r["rouge1"] for r in G["RESULTS"] if r["name"] == "00_lead"][0]
assert lead_r1 > 0.1, f"lead baseline scored {lead_r1}"
print(f"[lead] extractive baseline ROUGE-1 {lead_r1:.4f}")

# --- 3. the two input shapes, and the loss mask ----------------------------------------
ds_s2s = G["SummDS"](list(G["train_docs"]), True)
ds_dec = G["SummDS"](list(G["train_docs"]), False)
a = ds_s2s[0]
assert a["src"] is not None and a["inp"][0] == G["BOS_ID"] and a["lab"][-1] == G["EOS_ID"]
assert len(a["inp"]) == len(a["lab"]), "seq2seq teacher forcing misaligned"
assert a["inp"][1:] == a["lab"][:-1], "target not shifted right by exactly one"
d = ds_dec[0]
assert d["src"] is None and len(d["inp"]) == len(d["lab"])
nsup = sum(1 for x in d["lab"] if x != -100)
assert nsup > 0 and nsup <= cfg.TGT_LEN + 1, nsup
assert d["lab"][0] == -100, "document positions must be masked out"
# the supervised labels must be exactly the summary followed by EOS
sup = [x for x in d["lab"] if x != -100]
assert sup == G["SUM"][ds_dec.ids[0]] + [G["EOS_ID"]], "decoder-only labels misaligned"
print(f"[shapes] seq2seq shift-by-one exact; decoder-only masks the document "
      f"({nsup} supervised of {len(d['lab'])} positions)")

# --- 4. models stay CAUSAL (no bidirectional patch anywhere) ---------------------------
assert "set_bidirectional" not in "".join(SRC.values()), "a bidirectional patch crept in"
print()
for arch in ("A", "B", "C", "D"):
    m = G["build_summ"](arch, verbose=False)
    m.eval()
    T = 40
    ids = torch.randint(4, G["VOCAB_SIZE"], (1, T))
    alt = ids.clone(); alt[0, T - 1] = (ids[0, T - 1].item() + 1) % G["VOCAB_SIZE"]
    with torch.no_grad():
        if m.seq2seq:
            src = torch.randint(4, G["VOCAB_SIZE"], (1, 32))
            e = m.encode(src)
            leak = (m.decode(ids, e)[0, 3] - m.decode(alt, e)[0, 3]).abs().max().item()
        else:
            leak = (m(ids)[0, 3] - m(alt)[0, 3]).abs().max().item()
    print(f"  {arch}: changing the LAST token moves position 3 by {leak:.2e}")
    assert leak < 1e-5, f"{arch} is NOT causal -- generation would be training on leaks"
    del m

# --- 5. the LM head is the tied Phase-1 head, and emits vocab logits --------------------
m = G["build_summ"]("A", verbose=False)
assert m.backbone.head.weight is m.backbone.tok_emb.weight, "LM head untied"
with torch.no_grad():
    o = m(torch.randint(4, G["VOCAB_SIZE"], (2, 16)))
assert o.shape == (2, 16, G["VOCAB_SIZE"]), o.shape
print(f"[head] tied LM head retained, logits {tuple(o.shape)}")
del m

# --- 6. decoding guards actually fire ---------------------------------------------------
lgt = torch.zeros(1, 50)
seqs = [[7, 8, 9, 7, 8]]
out = G["_block_repeats"](lgt.clone(), seqs, 3)
assert out[0, 9] == -float("inf"), "trigram 7,8->9 should be banned"
assert out[0, 11] == 0.0, "an unseen continuation must stay allowed"
out1 = G["_block_repeats"](lgt.clone(), seqs, 0)
assert torch.isfinite(out1).all(), "no_repeat=0 must disable blocking"
print("[decode] n-gram blocker bans the seen continuation and only that one")

# --- 7. generation produces non-empty, non-degenerate text for every architecture -------
print()
for arch in ("A", "B", "C", "D"):
    m = G["build_summ"](arch, verbose=False)
    ids = [G["DOC"][d] for d in list(G["train_docs"])[:3]]
    txt = G["generate"](m, ids, max_new=16, min_new=6)
    assert len(txt) == 3, len(txt)
    assert all(isinstance(t, str) for t in txt)
    nw = [len(t.split()) for t in txt]
    # min_new must be honoured: EOS suppressed for the first 6 steps
    assert all(n >= 1 for n in nw), f"{arch} produced an empty summary: {txt}"
    print(f"  {arch}: {nw} words | {txt[0][:60]!r}")
    del m

# --- 8. min_new really suppresses early EOS ---------------------------------------------
_eos = G["EOS_ID"]
_real = G["tok"].encode(" the court").ids[0]        # a token that decodes to real text
# EOS is the argmax everywhere; _real is the runner-up, so blocking EOS yields real text
class _AlwaysEOS(torch.nn.Module):
    seq2seq = False
    n_prompt = 0
    def eval(self): return self
    def __call__(self, ids, src=None):
        o = torch.full((ids.size(0), ids.size(1), G["VOCAB_SIZE"]), -1e4)
        o[..., _eos] = 1e4
        o[..., _real] = 5e3
        return o
docs2 = [G["DOC"][d] for d in list(G["train_docs"])[:2]]
g = G["generate"](_AlwaysEOS(), docs2, max_new=20, min_new=7, no_repeat=0)
nw = [len(G["tok"].encode(t).ids) for t in g]
assert all(n >= 7 for n in nw), f"min_new not honoured: {nw} from {g}"
# and with min_new=0 the very same model must stop immediately
g0 = G["generate"](_AlwaysEOS(), docs2, max_new=20, min_new=0, no_repeat=0)
assert all(t.strip() == "" for t in g0), f"EOS not honoured when allowed: {g0}"
print(f"[decode] EOS forced: min_new=7 gives >=7 tokens {nw}, min_new=0 stops at once")

# --- 9. every (architecture, technique) pair trains, generates and scores ----------------
run = G["run_experiment"]
ok, fail, teacher = [], [], None
PLAN = [("01_smoke", "B", "prompt"), ("02_B_lorma", "B", "lorma"),
        ("03_A_lora", "A", "lora"), ("04_C_houlsby", "C", "houlsby"),
        ("05_D_prompt", "D", "prompt"),
        ("x_B_pfeiffer", "B", "pfeiffer"), ("x_A_lorma", "A", "lorma"),
        ("x_B_lora", "B", "lora"), ("x_C_lorma", "C", "lorma"),
        ("x_D_lora", "D", "lora"), ("x_A_houlsby", "A", "houlsby"),
        ("x_D_pfeiffer", "D", "pfeiffer"), ("x_C_prompt", "C", "prompt")]
for name, arch, meth in PLAN:
    try:
        mdl, s = run(name, arch, meth, epochs=1, write=not name.startswith("01"), n_val=4)
        ok.append((name, s["rouge1"]))
        if name == "02_B_lorma":
            teacher = mdl
        else:
            del mdl
    except Exception as e:
        traceback.print_exc(); fail.append((name, repr(e)))

# --- 10. distillation ---------------------------------------------------------------------
try:
    for p in teacher.parameters():
        p.requires_grad = False
    ms, s5 = run("06_B_distill", "B", "lorma", epochs=1, keep_layers=2,
                 teacher=teacher, n_val=4)
    ok.append(("06_B_distill", s5["rouge1"]))
    del ms
except Exception as e:
    traceback.print_exc(); fail.append(("06_B_distill", repr(e)))

# --- 11. the results / length-sweep cell ---------------------------------------------------
try:
    exec(SRC[31], G)
    ok.append(("results_cell", 0.0))
except Exception as e:
    traceback.print_exc(); fail.append(("results_cell", repr(e)))

# --- 12. every zip is in the PHASE-6 list format -------------------------------------------
import zipfile, glob as _g
zips = sorted(_g.glob(os.path.join(OUT, "*.zip")))
assert zips, "no submissions written"
for zp in zips:
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"], (zp, z.namelist())
        payload = json.loads(z.read("prediction.json"))
    assert isinstance(payload, list), f"{zp}: must be a LIST, got {type(payload).__name__}"
    assert len(payload) == len(TEST_IDS), f"{zp}: {len(payload)} vs {len(TEST_IDS)}"
    assert {r["id"] for r in payload} == TEST_IDS, f"{zp}: id mismatch"
    for r in payload:
        assert set(r) == {"id", "summary"}, r
        assert isinstance(r["summary"], str) and r["summary"].strip(), f"{zp}: empty {r['id']}"
print(f"\n[submission] {len(zips)} zips: list of objects, every test id once, "
      f"no empty summaries")

# --- 13. the empty-summary fallback really fires -------------------------------------------
zp = G["write_submission"]({}, "zz_allempty")
with zipfile.ZipFile(zp) as z:
    pay = json.loads(z.read("prediction.json"))
assert all(r["summary"].strip() for r in pay), "fallback did not fill empty summaries"
print("[submission] a model that emits nothing still produces a valid, non-empty zip")

print("\n" + "=" * 72)
for n, s in ok:
    print(f"  PASS  {n:<22} val_rouge1={s:.4f}")
for n, e in fail:
    print(f"  FAIL  {n:<22} {e}")
print("=" * 72)
shutil.rmtree(ROOT, ignore_errors=True)
sys.exit(1 if fail else 0)
