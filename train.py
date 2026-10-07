"""Train the looped transformer to solve F_p linear systems.

Random loop-count curriculum (depth ~ U[1, max_loops]) with cross-entropy on the
N answer slots. Checkpoints are saved frequently so the training trajectory can be
scanned afterwards for the convergence-landscape bifurcation (Fig. 5).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from data import LinSysConfig, make_batch
from model import LoopedTransformer


def evaluate(model, cfg, device, eval_loops, n_direct=0, n_batches=4, batch=512, seed=12345):
    model.eval()
    rng = np.random.default_rng(seed)
    tok_correct = tot = full_correct = full_tot = 0
    with torch.no_grad():
        for _ in range(n_batches):
            tok, x = make_batch(batch, cfg, rng, n_direct=n_direct)
            tok = torch.tensor(tok, device=device)
            x = torch.tensor(x, device=device)
            logits, _ = model(tok, n_loops=eval_loops)
            pred = logits[:, cfg.ans_pos].argmax(-1)       # (B, N)
            tok_correct += (pred == x).sum().item()
            tot += x.numel()
            full = (pred == x).all(-1)
            full_correct += full.sum().item()
            full_tot += full.numel()
    model.train()
    return tok_correct / tot, full_correct / full_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--mlp_ratio", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--curriculum_steps", type=int, default=60000,
                    help="anneal #directly-substitutable rows N->0 over this many steps")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--min_loops", type=int, default=1)
    ap.add_argument("--max_loops", type=int, default=20)
    ap.add_argument("--eval_loops", type=int, default=32)
    ap.add_argument("--deep_sup", action="store_true",
                    help="supervise answer at every (last K) loop -> fixed-point training")
    ap.add_argument("--deep_sup_loops", type=int, default=8)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = LinSysConfig(args.p, args.M, args.N)

    model = LoopedTransformer(cfg.vocab_size, cfg.seq_len, args.d_model,
                              args.n_heads, args.mlp_ratio).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)
    if args.compile:
        # compile the shared block (fixed shapes, reused every loop) -> big win
        model.block = torch.compile(model.block)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + np.cos(np.pi * prog))

    metrics_path = os.path.join(args.out, "metrics.jsonl")
    mf = open(metrics_path, "a")
    t0 = time.time()
    run_loss = 0.0
    nseen = 0
    for step in range(args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        n_loops = int(rng.integers(args.min_loops, args.max_loops + 1))
        # curriculum: anneal directly-substitutable rows N -> 0, keep easy skills alive
        progress = min(1.0, step / max(1, args.curriculum_steps))
        scheduled = int(round(args.N * (1.0 - progress)))
        if rng.random() < 0.7:
            n_direct = scheduled
        else:
            n_direct = int(rng.integers(scheduled, args.N + 1))
        tok, x = make_batch(args.batch, cfg, rng, n_direct=n_direct)
        tok = torch.tensor(tok, device=device)
        x = torch.tensor(x, device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.deep_sup:
                # supervise the answer at EVERY loop (last deep_sup_loops) ->
                # pushes the latent map toward a solution-holding fixed point.
                _, _, states = model(tok, n_loops=n_loops, return_all=True)
                k = min(args.deep_sup_loops, len(states))
                tgt = x.reshape(-1)
                loss = 0.0
                for st in states[-k:]:
                    al = model.readout(st)[:, cfg.ans_pos]
                    loss = loss + F.cross_entropy(
                        al.reshape(-1, cfg.vocab_size).float(), tgt)
                loss = loss / k
            else:
                logits, _ = model(tok, n_loops=n_loops)
                ans_logits = logits[:, cfg.ans_pos]            # (B, N, V)
                loss = F.cross_entropy(ans_logits.reshape(-1, cfg.vocab_size).float(),
                                       x.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item(); nseen += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            rate = (step + 1) / dt
            print(f"step {step:7d}  loss {run_loss/nseen:.4f}  lr {lr_at(step):.2e}  "
                  f"{rate:.1f} it/s", flush=True)
            run_loss = 0.0; nseen = 0

        if step % args.eval_every == 0:
            tok_acc, solve = evaluate(model, cfg, device, args.eval_loops, n_direct=0)
            tok_e, solve_e = evaluate(model, cfg, device, args.eval_loops, n_direct=4)
            rec = {"step": step, "loss": loss.item(), "lr": lr_at(step),
                   "tok_acc": tok_acc, "solve_rate": solve,
                   "tok_acc_easy4": tok_e, "solve_rate_easy4": solve_e,
                   "n_direct": n_direct, "n_loops": n_loops, "t": time.time() - t0}
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            print(f"  [eval @ {args.eval_loops} loops] hard: tok {tok_acc:.4f} solve {solve:.4f}"
                  f"  | easy4: tok {tok_e:.4f} solve {solve_e:.4f}  (train n_direct={n_direct})",
                  flush=True)

        if step % args.ckpt_every == 0:
            sd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
            torch.save({"step": step, "model": sd, "args": vars(args)},
                       os.path.join(args.out, "ckpt", f"step{step:07d}.pt"))

    mf.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
