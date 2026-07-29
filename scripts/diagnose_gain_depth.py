"""Is `gain` depth-converged on the real trajectory, or is it the depth trap again?

`rho` was caught once already: at Lanczos depths (4, 8, 12) it reported ~9.8 on
the MNIST MLP and 0.045-0.19 at depths (25, 50, 75, 100), because `lambda_g` is
a harmonic mean and Lanczos resolves the smallest eigenvalues last.  `gain` was
added afterwards and has never been checked the same way.

The concern is structural, not incidental.  For a spectral `P = f(H)^-1` the
Gauss quadrature evaluates

    g'Pg     ~  sum_j w_j / f(lam_j)
    g'PHPg   ~  sum_j w_j lam_j / f(lam_j)^2
    g'Hg     ~  sum_j w_j lam_j                     <- degree 1: EXACT at any depth

The first two are dominated by the *smallest* |lam|, where `1/(|lam| + delta)`
is largest and where -- for small `delta` -- the integrand is nearly singular
and therefore badly approximated by any polynomial.  The third is a degree-one
polynomial, which a Gauss rule integrates exactly at any depth >= 1.  So
`R(I)`'s denominator is exact while `R(P)`'s is not, and the *ratio* inherits
whatever error the numerator's near-singular integrand carries.

This script measures it: `rho` and `gain` at a ladder of depths, per checkpoint,
per `delta`, on DSN's own MNIST trajectory.  A statistic that is still moving at
the deepest rung is not converged, and its value is not a measurement of
anything.

    python scripts/diagnose_gain_depth.py --steps 40 --max-depth 150
"""

import argparse
import json
import pathlib
import time

import torch

from _device import add_device_arg, resolve_device
from autoresearch.gate import statistics
from autoresearch.preconditioners import IdentityPreconditioner, dsn_preconditioner
from autoresearch.probes import build_probe_context

LOGDIR = pathlib.Path("autoresearch/logs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=150)
    ap.add_argument("--curv-batch", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    add_device_arg(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")
    torch.set_default_dtype(torch.float64)
    from torchvision import datasets, transforms

    from dsn import DSN
    from dsn.subspace import KrylovBuilder

    depths = tuple(d for d in (5, 10, 25, 50, 75, 100, 125, 150) if d <= args.max_depth)
    deltas = (1e-3, 1e-2, 1e-1)

    torch.manual_seed(0)
    ds = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    n_pool = args.curv_batch + args.micro * args.micro_batch
    x = torch.stack([ds[i][0] for i in range(n_pool)]).to(torch.get_default_dtype()).to(device)
    y = torch.tensor([ds[i][1] for i in range(n_pool)]).to(device)
    x_c, y_c = x[: args.curv_batch], y[: args.curv_batch]
    x_m, y_m = x[args.curv_batch :], y[args.curv_batch :]

    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(784, 96), torch.nn.Tanh(),
        torch.nn.Linear(96, 10),
    ).to(device)
    params = list(model.parameters())
    opt = DSN(model.parameters(), lr=1e-2, weight_decay=0.0,
              builder=KrylovBuilder(k_max=8, m_recycle=4, tau=0.1))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)
    ce = torch.nn.functional.cross_entropy

    checkpoints = sorted({0, args.steps // 2, args.steps - 1})
    out = []
    t0 = time.time()
    it = iter(loader)
    for step in range(args.steps):
        if step in checkpoints:
            ctx = build_probe_context(
                lambda: ce(model(x_c), y_c),
                params,
                [
                    (lambda j=j: ce(
                        model(x_m[j * args.micro_batch : (j + 1) * args.micro_batch]),
                        y_m[j * args.micro_batch : (j + 1) * args.micro_batch],
                    ))
                    for j in range(args.micro)
                ],
                depths=depths,
                curv_batch=args.curv_batch,
                micro_batch=args.micro_batch,
                train_batch=args.batch,
                step=step,
                label="mnist-mlp",
            )
            g_rules, _ = ctx.rules()
            row = {
                "step": step,
                "neg_frac": ctx.neg_frac(),
                "loss": float(ce(model(x_c), y_c).detach()),
                "cells": [],
                "spectrum": {},
            }
            # Where the quadrature nodes actually sit. If the smallest node is
            # far above 0 while `delta` is far below it, the integrand's
            # near-singular region is entirely unsampled and `1/(|lam|+delta)`
            # is being extrapolated, not integrated.
            for d in depths:
                nodes = g_rules[d].nodes
                row["spectrum"][str(d)] = {
                    "min_abs_node": float(nodes.abs().min()),
                    "max_abs_node": float(nodes.abs().max()),
                    "n_negative": int((nodes < 0).sum()),
                }
            for delta in deltas:
                P = dsn_preconditioner(delta)
                for d in depths:
                    rho, gain = statistics(P.forms(ctx, d))
                    row["cells"].append(
                        {"delta": delta, "depth": d, "rho": rho, "gain": gain}
                    )
            ident = IdentityPreconditioner()
            row["identity"] = [
                dict(zip(("rho", "gain"), statistics(ident.forms(ctx, d))))
                for d in depths
            ]
            out.append(row)
            print(f"[ckpt {step:4d}] loss {row['loss']:.4f} neg_frac {row['neg_frac']:.3f} "
                  f"({time.time() - t0:.0f}s)")

        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        xb, yb = xb.to(torch.get_default_dtype()).to(device), yb.to(device)
        opt.step(lambda: ce(model(xb), yb))

    LOGDIR.mkdir(parents=True, exist_ok=True)
    path = LOGDIR / "gain_depth_diagnosis.json"
    path.write_text(json.dumps({"depths": list(depths), "deltas": list(deltas),
                                "rows": out}, indent=1))

    print(f"\n{'step':>5} {'delta':>8} " + "".join(f"{d:>11}" for d in depths))
    for stat in ("rho", "gain"):
        print(f"\n--- {stat} ---")
        for r in out:
            for delta in deltas:
                vals = [c[stat] for c in r["cells"] if c["delta"] == delta]
                lo, hi = min(vals), max(vals)
                spread = (hi - lo) / abs(hi) if hi else float("nan")
                print(f"{r['step']:5d} {delta:8.0e} "
                      + "".join(f"{v:11.4g}" for v in vals)
                      + f"   spread {spread:.2f}")
    print("\nsmallest |node| reached, by depth (step 0):")
    for d, s in out[0]["spectrum"].items():
        print(f"  depth {d:>4}: min|lam| {s['min_abs_node']:.4g}  "
              f"max|lam| {s['max_abs_node']:.4g}  n_neg {s['n_negative']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
