"""Render a Figure-5-style panel from basin npz frames + training metrics.

Top row:   basin maps coloured by convergence correctness (per attractor).
Mid row:   same maps coloured by convergence time (loops to settle).
Bottom:    fraction correct, basin entropy, boundary density vs training step,
           plus the training solve-rate curve.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

VOID = "#07080c"
CMAP_TIME = LinearSegmentedColormap.from_list(
    "t", ["#07080c", "#17215e", "#3b2fd0", "#8a4fe8", "#2fd8e8", "#eaffff"])


def load_frames(basin_dir):
    summ = json.load(open(os.path.join(basin_dir, "summary.json")))
    frames = []
    for fr in summ["frames"]:
        d = np.load(os.path.join(basin_dir, fr["npz"]))
        frames.append({**fr, "settle": d["settle"], "correct": d["correct"],
                       "attractor": d["attractor"]})
    frames.sort(key=lambda f: f["step"])
    return summ, frames


def correctness_rgb(correct, attractor):
    """Green where correct; distinct hues per wrong attractor (vectorized)."""
    import matplotlib.colors as mcolors
    res = correct.shape[0]
    rgb = np.zeros((res, res, 3))
    wrong = ~correct
    if wrong.any():
        uniqw = np.unique(attractor[wrong])
        hue_idx = np.zeros_like(attractor)
        for i, u in enumerate(uniqw):
            hue_idx[attractor == u] = i
        hsv = np.zeros((res, res, 3))
        hsv[..., 0] = (hue_idx % 12) / 12.0
        hsv[..., 1] = 0.85
        hsv[..., 2] = 0.75
        rgbw = mcolors.hsv_to_rgb(hsv)
        rgb[wrong] = rgbw[wrong]
    rgb[correct] = np.array([0.18, 0.95, 0.65])          # correct basin
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basins", required=True, help="dir with basin npz + summary.json")
    ap.add_argument("--metrics", default=None, help="training metrics.jsonl")
    ap.add_argument("--out", default="figure5.png")
    ap.add_argument("--ncols", type=int, default=6)
    args = ap.parse_args()

    summ, frames = load_frames(args.basins)
    # choose evenly spaced columns
    n = min(args.ncols, len(frames))
    sel = np.linspace(0, len(frames) - 1, n).round().astype(int)
    cols = [frames[i] for i in sorted(set(sel))]
    n = len(cols)

    tmax = max(f["settle"].max() for f in cols) or 1
    fig = plt.figure(figsize=(2.3 * n, 8.2), facecolor=VOID)
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.9], hspace=0.18, wspace=0.06)

    for k, f in enumerate(cols):
        ax = fig.add_subplot(gs[0, k]); ax.set_facecolor(VOID)
        ax.imshow(correctness_rgb(f["correct"], f["attractor"]),
                  origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {f['step']//1000}k\n{f['frac_correct']*100:.0f}% correct",
                     color="#c9d4e0", fontsize=8)
        if k == 0:
            ax.set_ylabel("correctness", color="#c9d4e0", fontsize=9)

        ax2 = fig.add_subplot(gs[1, k]); ax2.set_facecolor(VOID)
        ax2.imshow(f["settle"], origin="lower", cmap=CMAP_TIME, vmin=0, vmax=tmax,
                   interpolation="nearest")
        ax2.set_xticks([]); ax2.set_yticks([])
        if k == 0:
            ax2.set_ylabel("loops to settle", color="#c9d4e0", fontsize=9)

    steps = [f["step"] for f in frames]
    fc = [f["frac_correct"] for f in frames]
    be = [f["basin_entropy"] for f in frames]
    bd = [f["boundary_density"] for f in frames]

    axc = fig.add_subplot(gs[2, :]); axc.set_facecolor(VOID)
    axc.plot(steps, fc, "-o", color="#2fd8e8", ms=3, label="fraction correct")
    axc.plot(steps, bd, "-o", color="#e85f8a", ms=3, label="boundary density")
    axc.set_xlabel("training step", color="#c9d4e0")
    axc.tick_params(colors="#c9d4e0")
    for s in axc.spines.values():
        s.set_color("#30384a")
    axc.set_ylim(0, 1.02)
    axb = axc.twinx()
    axb.plot(steps, be, "-s", color="#8a7fe8", ms=3, label="basin entropy $S_b$")
    axb.tick_params(colors="#8a7fe8")
    axb.set_ylabel("basin entropy", color="#8a7fe8")

    if args.metrics and os.path.exists(args.metrics):
        recs = [json.loads(l) for l in open(args.metrics)]
        ms = [r["step"] for r in recs]; sr = [r["solve_rate"] for r in recs]
        axc.plot(ms, sr, "-", color="#9aa4b2", lw=1, alpha=0.7,
                 label="train solve rate")

    lines = axc.get_lines() + axb.get_lines()
    axc.legend(lines, [l.get_label() for l in lines], loc="center left",
               fontsize=7, facecolor=VOID, labelcolor="#c9d4e0", framealpha=0.3)
    axc.set_title(f"Convergence landscape vs training  (radius={summ['radius']:.1f}, "
                  f"instance={summ['instance_seed']}, plane={summ['plane_seed']})",
                  color="#c9d4e0", fontsize=9, loc="left")

    fig.suptitle("Fractal basins of a looped $\\mathbb{F}_3$ linear-system solver "
                 "(reproduction of Fig. 5)", color="#eaffff", fontsize=12, y=0.98)
    fig.savefig(args.out, dpi=200, facecolor=VOID, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
