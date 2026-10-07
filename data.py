"""Data generation for the F_p linear-system reasoning task.

Reproduces the encoding described by the authors of
"Fractal basins trap latent reasoning" (Lai, Bao, Quinn, Gilpin, 2026):

    one fixed 97-token sequence per instance:
        a_i1 ... a_i8 | b_i ;          (x8 rows -> 88 tokens)
        =                              (1 token)
        ? ? ? ? ? ? ? ?                (8 answer slots)

The task: given A in F_p^{MxN}, b in F_p^M, output x in F_p^N with A x = b (mod p).
Default p=3, M=N=8  (=> sequence length 97, as stated by the authors).

Vocabulary (tied embedding, tiny):
    0,1,2  -> ids 0,1,2        (field elements, for p=3)
    '|'    -> id p
    ';'    -> id p+1
    '='    -> id p+2
    '?'    -> id p+3
"""
from __future__ import annotations
import numpy as np


def build_vocab(p: int = 3):
    toks = {str(d): d for d in range(p)}
    toks["|"] = p
    toks[";"] = p + 1
    toks["="] = p + 2
    toks["?"] = p + 3
    return toks


class LinSysConfig:
    def __init__(self, p: int = 3, M: int = 8, N: int = 8):
        self.p = p
        self.M = M
        self.N = N
        self.vocab = build_vocab(p)
        self.vocab_size = len(self.vocab)
        self.PIPE = p
        self.SEMI = p + 1
        self.EQ = p + 2
        self.QUES = p + 3
        # sequence length = M*(N + 3) + 1 + N
        self.seq_len = M * (N + 3) + 1 + N
        # answer-slot positions (the N '?' tokens at the end)
        self.ans_pos = np.arange(self.seq_len - N, self.seq_len)


def _row_reduce_mod_p(A: np.ndarray, p: int) -> np.ndarray:
    """Return boolean mask: True where the (batched) square matrix is invertible mod p.

    A: (B, N, N) int array with entries in [0, p).  Uses GF(p) Gaussian
    elimination; p must be prime so every nonzero element has an inverse.
    """
    B, N, _ = A.shape
    work = A.astype(np.int64).copy() % p
    inv = np.array([0] + [pow(a, p - 2, p) for a in range(1, p)], dtype=np.int64)  # modular inverse
    ok = np.ones(B, dtype=bool)
    for col in range(N):
        # find a pivot row >= col with nonzero entry in this column
        pivot_found = np.zeros(B, dtype=bool)
        for row in range(col, N):
            nz = (work[:, row, col] != 0) & (~pivot_found)
            if nz.any():
                # swap row <-> col for those batch elements needing a pivot here
                tmp = work[nz, col, :].copy()
                work[nz, col, :] = work[nz, row, :]
                work[nz, row, :] = tmp
                pivot_found |= nz
        ok &= pivot_found
        active = ok & pivot_found
        if not active.any():
            continue
        # normalise pivot row and eliminate the column below
        piv = work[:, col, col]
        scale = inv[np.where(piv == 0, 1, piv)]  # avoid index 0
        work[active, col, :] = (work[active, col, :] * scale[active, None]) % p
        for row in range(col + 1, N):
            factor = work[:, row, col].copy()
            upd = active & (factor != 0)
            if upd.any():
                work[upd, row, :] = (work[upd, row, :] - factor[upd, None] * work[upd, col, :]) % p
    return ok


def sample_invertible_A(batch: int, cfg: LinSysConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample `batch` invertible NxN matrices over F_p by rejection."""
    N, p = cfg.N, cfg.p
    out = np.empty((batch, N, N), dtype=np.int64)
    filled = 0
    while filled < batch:
        need = batch - filled
        cand = rng.integers(0, p, size=(max(need * 2, 64), N, N))
        good = cand[_row_reduce_mod_p(cand, p)]
        take = min(len(good), need)
        out[filled:filled + take] = good[:take]
        filled += take
    return out


def _sample_A_with_direct(batch, cfg, rng, n_direct):
    """Sample invertible A where `n_direct` equations are unit rows (directly
    substitutable variables: equation "x_j = b"), the rest dense random.

    n_direct=0  -> fully dense random invertible (the hard target distribution).
    n_direct=N  -> a scaled permutation (every variable directly readable).
    """
    p, M, N = cfg.p, cfg.M, cfg.N
    n_direct = int(max(0, min(N, n_direct)))
    out = np.empty((batch, M, N), dtype=np.int64)
    filled = 0
    while filled < batch:
        need = batch - filled
        m = max(need * 2, 64)
        A = rng.integers(0, p, size=(m, M, N))
        if n_direct > 0:
            # first n_direct rows become unit rows on distinct random columns
            cols = np.argsort(rng.random((m, N)), axis=1)[:, :n_direct]   # (m, n_direct)
            A[:, :n_direct, :] = 0
            rows = np.arange(n_direct)
            A[np.arange(m)[:, None], rows[None, :], cols] = 1             # coef 1
        good = A[_row_reduce_mod_p(A, p)]
        take = min(len(good), need)
        out[filled:filled + take] = good[:take]
        filled += take
    return out


def make_batch(batch: int, cfg: LinSysConfig, rng: np.random.Generator, n_direct: int = 0):
    """Return (tokens[B,L] int64, targets[B,N] int64 = x).

    A invertible over F_p, x ~ Uniform(F_p^N), b = A x mod p  (unique solution x).
    `n_direct` = number of directly-substitutable rows (curriculum knob).
    """
    p, M, N = cfg.p, cfg.M, cfg.N
    if n_direct > 0:
        A = _sample_A_with_direct(batch, cfg, rng, n_direct)
    else:
        A = sample_invertible_A(batch, cfg, rng)      # (B, M, N) fully dense
    x = rng.integers(0, p, size=(batch, N))            # (B, N)
    b = (A @ x[:, :, None]).squeeze(-1) % p            # (B, M)

    L = cfg.seq_len
    tokens = np.empty((batch, L), dtype=np.int64)
    pos = 0
    for i in range(M):
        tokens[:, pos:pos + N] = A[:, i, :]
        pos += N
        tokens[:, pos] = cfg.PIPE; pos += 1
        tokens[:, pos] = b[:, i];  pos += 1
        tokens[:, pos] = cfg.SEMI; pos += 1
    tokens[:, pos] = cfg.EQ; pos += 1
    tokens[:, pos:pos + N] = cfg.QUES; pos += N
    assert pos == L, (pos, L)
    return tokens, x


if __name__ == "__main__":
    cfg = LinSysConfig()
    print("seq_len", cfg.seq_len, "vocab", cfg.vocab_size, "ans_pos", cfg.ans_pos)
    rng = np.random.default_rng(0)
    tok, x = make_batch(4, cfg, rng)
    print("tokens shape", tok.shape, "targets shape", x.shape)
    print("example tokens[0]:", tok[0])
    print("example target[0]:", x[0])
    # sanity: verify A x = b for first example by re-parsing
    p, M, N = cfg.p, cfg.M, cfg.N
    A0 = np.zeros((M, N), dtype=int); b0 = np.zeros(M, dtype=int)
    pos = 0
    for i in range(M):
        A0[i] = tok[0, pos:pos+N]; pos += N
        assert tok[0, pos] == cfg.PIPE; pos += 1
        b0[i] = tok[0, pos]; pos += 1
        assert tok[0, pos] == cfg.SEMI; pos += 1
    assert np.array_equal((A0 @ x[0]) % p, b0), "A x != b!"
    print("verified A x = b (mod p) OK")
