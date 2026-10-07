"""Basin-of-attraction maps for the looped linear-system solver (Fig. 5).

For a FIXED problem instance, we perturb the initial latent state x_0 over a
random orthonormal 2-plane in latent space (QR of a Gaussian, following the
repo of the paper) and iterate the deterministic map x <- LN(x + block(x+e))
up to `max_steps`. Each grid point (pixel) is coloured by:
    - convergence correctness  (does the settled answer equal the true x?)
    - convergence time         (loops until the decoded answer stops changing)
and labelled by its attractor id (the final decoded answer), from which we
compute the Sprott-Daza basin entropy.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch

from data import LinSysConfig, make_batch
from model import LoopedTransformer


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    cfg = LinSysConfig(a["p"], a["M"], a["N"])
    model = LoopedTransformer(cfg.vocab_size, cfg.seq_len, a["d_model"],
                              a["n_heads"], a["mlp_ratio"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["step"]


def fixed_instance(cfg, seed, n_direct=0):
    tok, x = make_batch(1, cfg, np.random.default_rng(seed), n_direct=n_direct)
    return tok[0], x[0]              # (L,), (N,)


def random_plane(dim, seed):
    g = np.random.default_rng(seed).standard_normal((dim, 2)).astype(np.float32)
    q, _ = np.linalg.qr(g)          # (dim, 2) orthonormal columns
    return q[:, 0], q[:, 1]


@torch.no_grad()
def compute_fields(model, cfg, tokens, true_x, u, v, radius, res, max_steps,
                   device, chunk=4096):
    """Return settle(res,res), correct(res,res) bool, attractor(res,res) int."""
    L, d = cfg.seq_len, model.d_model
    e = model.embed_tokens(torch.tensor(tokens[None], device=device))  # (1,L,d)
    u_t = torch.tensor(u.reshape(L, d), device=device)
    v_t = torch.tensor(v.reshape(L, d), device=device)
    true = torch.tensor(true_x, device=device)
    ans_pos = torch.tensor(cfg.ans_pos, device=device)
    p, N = cfg.p, cfg.N
    powers = (p ** torch.arange(N, device=device)).long()

    lin = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    S, T = np.meshgrid(lin, lin)                      # (res,res)
    coords = np.stack([S.ravel(), T.ravel()], 1)      # (res*res, 2)
    npix = coords.shape[0]

    settle = np.zeros(npix, np.int32)
    correct = np.zeros(npix, bool)
    attractor = np.zeros(npix, np.int64)

    for i in range(0, npix, chunk):
        c = torch.tensor(coords[i:i + chunk], device=device)      # (B,2)
        B = c.shape[0]
        # initial latent perturbation (base state x_0 = 0)
        x = radius * (c[:, 0, None, None] * u_t[None] + c[:, 1, None, None] * v_t[None])  # (B,L,d)
        e_b = e.expand(B, L, d)
        answers = torch.empty(max_steps, B, N, dtype=torch.long, device=device)
        for t in range(max_steps):
            x = model.step(x, e_b)
            logits = model.readout(x)[:, ans_pos]                 # (B,N,V)
            answers[t] = logits.argmax(-1)
        final = answers[-1]                                       # (B,N)
        # settling time: last step where answer changed
        changed = (answers[1:] != answers[:-1]).any(-1)          # (max_steps-1, B)
        idx = torch.arange(1, max_steps, device=device)[:, None].expand(-1, B)
        last_change = torch.where(changed, idx, torch.zeros_like(idx)).max(0).values  # (B,)
        settle[i:i + B] = last_change.cpu().numpy()
        correct[i:i + B] = (final == true).all(-1).cpu().numpy()
        attractor[i:i + B] = (final * powers).sum(-1).cpu().numpy()

    return (settle.reshape(res, res), correct.reshape(res, res),
            attractor.reshape(res, res))


def basin_entropy(attractor, box=5):
    """Sprott-Daza basin entropy Sb over non-overlapping boxes."""
    res = attractor.shape[0]
    ents = []
    for i in range(0, res - box + 1, box):
        for j in range(0, res - box + 1, box):
            blk = attractor[i:i + box, j:j + box].ravel()
            _, counts = np.unique(blk, return_counts=True)
            prob = counts / counts.sum()
            ents.append(-(prob * np.log(prob)).sum())
    return float(np.mean(ents)) if ents else 0.0


def boundary_density(attractor):
    a = attractor
    diff = np.zeros_like(a, bool)
    diff[:-1] |= a[:-1] != a[1:]
    diff[1:] |= a[1:] != a[:-1]
    diff[:, :-1] |= a[:, :-1] != a[:, 1:]
    diff[:, 1:] |= a[:, 1:] != a[:, :-1]
    return float(diff.mean())


def pick_radius(model, cfg, tokens, true_x, u, v, radii, res, max_steps, device):
    """Choose the radius with the richest boundary structure (auto)."""
    best = None
    for R in radii:
        s, c, a = compute_fields(model, cfg, tokens, true_x, u, v, R,
                                 res, max_steps, device)
        frac = float(c.mean()); bd = boundary_density(a); be = basin_entropy(a)
        # score: want mixed outcomes (frac near 0.5) AND structured boundaries
        score = bd * (1.0 - abs(frac - 0.5) * 2) + 0.1 * be
        print(f"  radius {R:6.2f}  solve_frac {frac:.3f}  bdry {bd:.3f}  "
              f"Sb {be:.3f}  score {score:.3f}", flush=True)
        if best is None or score > best[0]:
            best = (score, R, frac, bd, be)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run dir (contains ckpt/)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--scan_res", type=int, default=96)
    ap.add_argument("--max_steps", type=int, default=24)
    ap.add_argument("--instance_seed", type=int, default=7)
    ap.add_argument("--instance_direct", type=int, default=0,
                    help="#directly-substitutable vars in the fixed instance")
    ap.add_argument("--plane_seed", type=int, default=3)
    ap.add_argument("--radius", type=float, default=0.0, help="0 => auto-pick")
    ap.add_argument("--n_ckpts", type=int, default=16, help="evenly spaced ckpts to map")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = args.out or os.path.join(args.run, "basins")
    os.makedirs(out, exist_ok=True)
    device = args.device

    ckpts = sorted(glob.glob(os.path.join(args.run, "ckpt", "step*.pt")))
    assert ckpts, f"no checkpoints in {args.run}/ckpt"
    # evenly spaced subset
    if len(ckpts) > args.n_ckpts:
        sel = np.linspace(0, len(ckpts) - 1, args.n_ckpts).round().astype(int)
        ckpts = [ckpts[i] for i in sorted(set(sel))]

    # fixed instance + plane (use final ckpt's cfg)
    model, cfg, _ = load_model(ckpts[-1], device)
    tokens, true_x = fixed_instance(cfg, args.instance_seed, args.instance_direct)
    dim = cfg.seq_len * model.d_model
    u, v = random_plane(dim, args.plane_seed)

    radius = args.radius
    if radius <= 0:
        print("auto-picking radius on final checkpoint:", flush=True)
        radii = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
        radius = pick_radius(model, cfg, tokens, true_x, u, v, radii,
                             args.scan_res, args.max_steps, device)
        print(f"chosen radius = {radius}", flush=True)

    summary = {"radius": radius, "instance_seed": args.instance_seed,
               "plane_seed": args.plane_seed, "true_x": true_x.tolist(),
               "max_steps": args.max_steps, "res": args.res, "frames": []}

    for cp in ckpts:
        model, cfg, step = load_model(cp, device)
        s, c, a = compute_fields(model, cfg, tokens, true_x, u, v, radius,
                                 args.res, args.max_steps, device)
        be = basin_entropy(a); bd = boundary_density(a); frac = float(c.mean())
        npz = os.path.join(out, f"basin_step{step:07d}.npz")
        np.savez_compressed(npz, settle=s, correct=c, attractor=a,
                            step=step, radius=radius, frac_correct=frac,
                            basin_entropy=be, boundary_density=bd)
        summary["frames"].append({"step": int(step), "frac_correct": frac,
                                  "basin_entropy": be, "boundary_density": bd,
                                  "npz": os.path.basename(npz)})
        print(f"step {step:7d}  solve_frac {frac:.3f}  Sb {be:.3f}  bdry {bd:.3f}",
              flush=True)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("basins done ->", out, flush=True)


if __name__ == "__main__":
    main()
