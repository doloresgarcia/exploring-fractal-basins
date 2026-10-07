"""Test whether the basin boundaries are actually FRACTAL (not just intricate).

Two diagnostics on a trained checkpoint, for a fixed instance + random 2-plane:
  (1) Uncertainty exponent alpha (Grebogi-McDonald final-state sensitivity):
        perturb random points by epsilon, measure fraction f(eps) whose final
        attractor flips.  f(eps) ~ eps^alpha.
        alpha ~ 1  => smooth boundary, box-dimension D = 2 - alpha = 1 (NOT fractal)
        alpha < 1  => fractal boundary, D = 2 - alpha > 1
  (2) Zoom sequence: attractor maps at nested windows; a true fractal keeps
        showing fine structure, a fragile-but-smooth boundary resolves to flat.
"""
import argparse
import numpy as np
import torch

from basins import load_model, fixed_instance, random_plane


@torch.no_grad()
def attractor_ids(model, cfg, tokens, u_t, v_t, coords, radius, max_steps, device, chunk=8192):
    """coords: (B,2) plane coords -> attractor id (int) of the settled answer."""
    L, d = cfg.seq_len, model.d_model
    e = model.embed_tokens(torch.tensor(tokens[None], device=device))
    ans_pos = torch.tensor(cfg.ans_pos, device=device)
    powers = (cfg.p ** torch.arange(cfg.N, device=device)).long()
    out = np.empty(coords.shape[0], np.int64)
    for i in range(0, coords.shape[0], chunk):
        c = torch.tensor(coords[i:i+chunk], device=device, dtype=torch.float32)
        B = c.shape[0]
        x = radius * (c[:, 0, None, None] * u_t[None] + c[:, 1, None, None] * v_t[None])
        e_b = e.expand(B, L, d)
        for _ in range(max_steps):
            x = model.step(x, e_b)
        final = model.readout(x)[:, ans_pos].argmax(-1)      # (B,N)
        out[i:i+B] = (final * powers).sum(-1).cpu().numpy()
    return out


def uncertainty_exponent(model, cfg, tokens, u_t, v_t, radius, max_steps, device,
                         n_points=120000, eps_list=None, seed=1):
    rng = np.random.default_rng(seed)
    if eps_list is None:
        eps_list = np.geomspace(1e-5, 3e-2, 12)
    base = rng.uniform(-1, 1, size=(n_points, 2)).astype(np.float32)
    ang = rng.uniform(0, 2*np.pi, size=n_points).astype(np.float32)
    dirx, diry = np.cos(ang), np.sin(ang)
    a_base = attractor_ids(model, cfg, tokens, u_t, v_t, base, radius, max_steps, device)
    fs = []
    for eps in eps_list:
        pert = base.copy()
        pert[:, 0] += eps * dirx; pert[:, 1] += eps * diry
        a_p = attractor_ids(model, cfg, tokens, u_t, v_t, pert, radius, max_steps, device)
        fs.append(float(np.mean(a_base != a_p)))
    fs = np.array(fs)
    # local slopes (between consecutive eps) -- the true exponent is the eps->0 trend
    lf = np.log(np.clip(fs, 1e-9, None)); le = np.log(eps_list)
    local = (lf[1:] - lf[:-1]) / (le[1:] - le[:-1])
    # eps->0 fit: use the small-eps unsaturated regime with enough statistics
    m = (fs > 3.0 / n_points * 20) & (fs < 0.08)   # >~20 flips, unsaturated
    alpha = float(np.polyfit(le[m], lf[m], 1)[0]) if m.sum() >= 2 else np.nan
    return eps_list, fs, alpha, local


def zoom_maps(model, cfg, tokens, u_t, v_t, radius, max_steps, device,
              center, halfwidths, res=350):
    maps = []
    for w in halfwidths:
        lin_s = np.linspace(center[0]-w, center[0]+w, res, dtype=np.float32)
        lin_t = np.linspace(center[1]-w, center[1]+w, res, dtype=np.float32)
        S, T = np.meshgrid(lin_s, lin_t)
        coords = np.stack([S.ravel(), T.ravel()], 1)
        a = attractor_ids(model, cfg, tokens, u_t, v_t, coords, radius, max_steps, device)
        maps.append(a.reshape(res, res))
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instance_seed", type=int, default=7)
    ap.add_argument("--instance_direct", type=int, default=0)
    ap.add_argument("--plane_seed", type=int, default=3)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--n_points", type=int, default=20000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="fractality")
    ap.add_argument("--zoom_center", type=float, nargs=2, default=[0.0, 0.0])
    args = ap.parse_args()

    model, cfg, step = load_model(args.ckpt, args.device)
    tokens, true_x = fixed_instance(cfg, args.instance_seed, args.instance_direct)
    dim = cfg.seq_len * model.d_model
    u, v = random_plane(dim, args.plane_seed)
    u_t = torch.tensor(u.reshape(cfg.seq_len, model.d_model), device=args.device)
    v_t = torch.tensor(v.reshape(cfg.seq_len, model.d_model), device=args.device)

    print(f"ckpt step {step}, p={cfg.p} N={cfg.N}, radius={args.radius}", flush=True)
    eps, fs, alpha, local = uncertainty_exponent(model, cfg, tokens, u_t, v_t, args.radius,
                                                 args.max_steps, args.device, args.n_points)
    print("\nUNCERTAINTY EXPONENT (final-state sensitivity):")
    print("  eps        f(eps)     local_slope(eps->)")
    for k, (e, f) in enumerate(zip(eps, fs)):
        ls = f"{local[k]:.2f}" if k < len(local) else ""
        print(f"  {e:.2e}  {f:.5f}    {ls}")
    D = 2 - alpha
    print(f"\n  eps->0 fit alpha = {alpha:.3f}  =>  boundary box-dim D = 2 - alpha = {D:.3f}")
    print("  (local slope approaching ~1 as eps->0 => smooth/non-fractal;"
          " staying <1 at small eps => genuinely fractal)")
    print("  verdict:", "FRACTAL (D>1 persists to small eps)" if alpha < 0.9 else
          "NOT a clean fractal (alpha->~1, D->1 at small scales): intricate but smooth boundaries")

    # auto-pick a BOUNDARY center for the zoom (not an interior point)
    cg = 220
    lin = np.linspace(-1, 1, cg, dtype=np.float32)
    S, T = np.meshgrid(lin, lin)
    ac = attractor_ids(model, cfg, tokens, u_t, v_t,
                       np.stack([S.ravel(), T.ravel()], 1), args.radius,
                       args.max_steps, args.device).reshape(cg, cg)
    bnd = np.zeros_like(ac, bool)
    bnd[:-1] |= ac[:-1] != ac[1:]; bnd[:, :-1] |= ac[:, :-1] != ac[:, 1:]
    # prefer a boundary pixel near the centre of the frame
    yy, xx = np.where(bnd)
    if len(xx):
        d2 = (xx - cg/2)**2 + (yy - cg/2)**2
        j = yy[np.argmin(d2)], xx[np.argmin(d2)]
        center = [float(lin[j[1]]), float(lin[j[0]])]
    else:
        center = args.zoom_center
    print(f"\n  zoom centred on boundary point {center}")

    halfwidths = [1.0, 0.25, 0.0625, 0.015625]
    maps = zoom_maps(model, cfg, tokens, u_t, v_t, args.radius, args.max_steps,
                     args.device, center, halfwidths)
    np.savez_compressed(args.out + "_data.npz", eps=eps, fs=fs, alpha=alpha,
                        halfwidths=np.array(halfwidths),
                        **{f"zoom{i}": m for i, m in enumerate(maps)})

    # render
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    VOID = "#07080c"
    fig = plt.figure(figsize=(16, 4.5), facecolor=VOID)
    gs = fig.add_gridspec(1, len(maps) + 1, width_ratios=[1]*len(maps) + [1.1])
    for i, (m, w) in enumerate(zip(maps, halfwidths)):
        ax = fig.add_subplot(gs[0, i]); ax.set_facecolor(VOID)
        ax.imshow(m % 20, cmap="twilight", origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"zoom x{int(1/w)}  (±{w:g})", color="#c9d4e0", fontsize=9)
    axp = fig.add_subplot(gs[0, -1]); axp.set_facecolor(VOID)
    axp.loglog(eps, np.clip(fs, 1e-5, None), "o-", color="#2fd8e8", label="measured f(ε)")
    # anchor reference lines at the smallest eps with decent statistics
    m = (fs > 3.0 / args.n_points * 20) & (fs < 0.08)
    if m.any():
        e0 = eps[m][0]; f0 = fs[m][0]
        axp.loglog(eps, f0 * (eps / e0) ** alpha, "--", color="#e85f8a",
                   label=f"ε→0 fit  α={alpha:.2f}\nD=2−α={2-alpha:.2f}")
        axp.loglog(eps, f0 * (eps / e0) ** 1.0, ":", color="#9aa4b2",
                   label="slope 1 (smooth)")
    axp.set_xlabel("ε (perturbation)", color="#c9d4e0")
    axp.set_ylabel("f(ε) fraction flipped", color="#c9d4e0")
    axp.tick_params(colors="#c9d4e0"); axp.legend(fontsize=8, labelcolor="#c9d4e0",
                                                   facecolor=VOID)
    for s in axp.spines.values(): s.set_color("#30384a")
    axp.set_title("uncertainty exponent", color="#c9d4e0", fontsize=9)
    fig.suptitle(f"Fractality test — 4×4 F3 looped solver (step {step})",
                 color="#eaffff", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=160, facecolor=VOID, bbox_inches="tight")
    print("\nwrote", args.out + ".png")


if __name__ == "__main__":
    main()
