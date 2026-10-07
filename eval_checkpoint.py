"""Load a trained checkpoint and report its solve rate on fresh instances.

Example:
    python eval_checkpoint.py checkpoints/n4_f3_step79500.pt --loops 24
"""
import argparse
import numpy as np
import torch

from basins import load_model
from data import make_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--loops", type=int, default=24)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n_direct", type=int, default=0, help="directly-substitutable vars (0 = fully dense)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg, step = load_model(args.ckpt, args.device)
    print(f"loaded step={step}  p={cfg.p}  M={cfg.M}  N={cfg.N}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    rng = np.random.default_rng(12345)
    tc = tot = fc = ft = 0
    with torch.no_grad():
        for _ in range(args.batches):
            tok, x = make_batch(args.batch, cfg, rng, n_direct=args.n_direct)
            pred = model(torch.tensor(tok, device=args.device),
                         n_loops=args.loops)[0][:, cfg.ans_pos].argmax(-1)
            x = torch.tensor(x, device=args.device)
            tc += (pred == x).sum().item(); tot += x.numel()
            fc += (pred == x).all(-1).sum().item(); ft += x.shape[0]
    print(f"@{args.loops} loops, n_direct={args.n_direct}:  "
          f"token_acc={tc/tot:.3f}  exact_solve={fc/ft:.3f}  (n={ft})")


if __name__ == "__main__":
    main()
