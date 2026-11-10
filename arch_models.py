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
