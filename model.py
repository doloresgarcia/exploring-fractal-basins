"""Looped bidirectional transformer for latent reasoning.

Architecture per the authors of "Fractal basins trap latent reasoning":
    - encoder-only (bidirectional, full attention, no causal mask)
    - ONE shared transformer block, iterated for K loops
    - 8 heads, d_model=256, MLP ratio 1, RoPE, tied embeddings
    - recurrence:  x <- LN(x + block(x + e))
      i.e. the token embeddings e are re-injected at every loop.

The state x (B, L, d) is the "latent" whose convergence basins we study.
Output logits are read off the answer-slot positions via the tied embedding.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope(seq_len: int, head_dim: int, base: float = 10000.0, device=None):
    assert head_dim % 2 == 0
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    ang = torch.outer(t, freqs)                      # (L, half)
    return torch.cos(ang), torch.sin(ang)            # each (L, half)


def apply_rope(x, cos, sin):
    # x: (B, H, L, D)   cos/sin: (L, D/2)   (compile-friendly, no strided writes)
    B, H, L, D = x.shape
    xr = x.reshape(B, H, L, D // 2, 2)
    x1, x2 = xr[..., 0], xr[..., 1]
    cos = cos[None, None]; sin = sin[None, None]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).reshape(B, H, L, D)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]             # (B, H, L, dh)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)  # bidirectional, no mask
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class Block(nn.Module):
    """Standard pre-LN transformer block with MLP ratio 1."""
    def __init__(self, d_model, n_heads, mlp_ratio=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=256, n_heads=8, mlp_ratio=1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block = Block(d_model, n_heads, mlp_ratio)   # the single shared block
        self.ln_out = nn.LayerNorm(d_model)               # the LN in x <- LN(x + r)
        self.unembed_ln = nn.LayerNorm(d_model)           # final norm before tied readout
        self.logit_scale = nn.Parameter(torch.tensor(1.0))  # learnable temperature
        cos, sin = build_rope(seq_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def embed_tokens(self, tokens):
        return self.embed(tokens)                         # e : (B, L, d)

    def init_state(self, e):
        return torch.zeros_like(e)                        # x_0 = 0 (deterministic)

    def step(self, x, e):
        """One loop: x <- LN(x + block(x + e))."""
        r = self.block(x + e, self.rope_cos, self.rope_sin)
        return self.ln_out(x + r)

    def readout(self, x):
        """Tied (cosine) unembedding -> logits over vocab at every position."""
        w = self.embed.weight
        w = w / (w.norm(dim=-1, keepdim=True) + 1e-6)
        return self.logit_scale * F.linear(self.unembed_ln(x), w)

    def forward(self, tokens, n_loops, x0=None, return_all=False):
        e = self.embed_tokens(tokens)
        x = self.init_state(e) if x0 is None else x0
        states = []
        for _ in range(n_loops):
            x = self.step(x, e)
            if return_all:
                states.append(x)
        logits = self.readout(x)
        if return_all:
            return logits, x, states
        return logits, x


if __name__ == "__main__":
    import numpy as np
    from data import LinSysConfig, make_batch
    cfg = LinSysConfig()
    m = LoopedTransformer(cfg.vocab_size, cfg.seq_len)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params/1e6:.3f}M  seq_len={cfg.seq_len}")
    tok, x = make_batch(8, cfg, np.random.default_rng(0))
    tok = torch.tensor(tok)
    logits, state = m(tok, n_loops=6)
    print("logits", logits.shape, "state", state.shape)
    ans = logits[:, cfg.ans_pos].argmax(-1)
    print("answer-slot preds shape", ans.shape)
