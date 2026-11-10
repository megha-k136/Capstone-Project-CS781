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
