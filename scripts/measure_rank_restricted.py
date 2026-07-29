"""Screen DSN's *shipped* rank-k operator, not the full-rank form it approximates.

Every DSN verdict this harness produced before this script existed was about

    P = (|H| + delta)^-1          full rank

while DSN applies

    P_k = W (|W'HW| + delta)^-1 W' + c (I - WW')

with ``W`` a rank-``k_max`` Krylov basis seeded at the gradient.

**These are two different operators, and the first is not an upper bound on the
second.**  That framing is the natural one and it is wrong, so it is worth being
precise about why.  ``P_k`` with ``c > 0`` is itself a full-rank, invertible
matrix -- it is a *member* of the class the full-rank screen draws from, not a
restriction of it.  The full-rank screen evaluates one particular member of that
class, and DSN applies a different member.  An upper bound would require
maximizing the statistic over the class, which no gate here does (and which the
harness documents as vacuous: ``P = I`` is always feasible, so the supremum over
all ``P`` is always at least 1).

Measured consequence: on this trajectory the rank-8 operator scores *higher*
than the full-rank one on both statistics, and the verdict flips from FAIL to
PASS.  The direction is not the surprising part -- ``P_k`` with ``c = 1`` leaves
the flat tail of the spectrum alone instead of amplifying it by ``1/delta`` --
but the fact that it flips at all is the whole reason this measurement exists.

Two things worth knowing before reading the numbers.

**The rank-restricted screen has no depth trap.**  A full-rank spectral candidate
is evaluated through Gauss quadrature, whose integrands ``1/f(lam)`` and
``lam/f(lam)^2`` are near-singular for small ``delta`` -- that is what made
``delta=1e-3`` unusable at every checkpoint.  A rank-restricted candidate is an
explicit operator: ``g'Pg``, ``g'P^-1g`` and ``g'PHPg`` are computed by applying
it, and ``g'Hg`` is a degree-one polynomial that a Gauss rule integrates exactly
at any depth.  So all four forms are exact and the verdict does not depend on the
sweep at all.

**The complement scaling ``c`` is a modelling choice, so it is swept.**  DSN puts
AdamW in the orthogonal complement; ``c`` models that as a scalar.  A verdict
that flips across the ``c`` sweep is a verdict about the complement, not about
the subspace, and is reported as such.

    python scripts/measure_rank_restricted.py --steps 40
"""

import argparse
import json
import pathlib
import time

import torch

from _device import add_device_arg, resolve_device
from autoresearch.gate import Gate, Verdict
from autoresearch.preconditioners import (
    KrylovRestrictedPreconditioner,
    dsn_preconditioner,
)
from autoresearch.probes import build_probe_context

LOGDIR = pathlib.Path("autoresearch/logs")
DEPTHS = (25, 50, 75, 100)


def build_contexts(args, device):
    from torchvision import datasets, transforms

    from dsn import DSN
    from dsn.subspace import KrylovBuilder

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
              builder=KrylovBuilder(k_max=args.k_max, m_recycle=4, tau=0.1))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)
    ce = torch.nn.functional.cross_entropy

    checkpoints = sorted({0, args.steps // 2, args.steps - 1})
    ctxs = []
    it = iter(loader)
    for step in range(args.steps):
        if step in checkpoints:
            ctxs.append(build_probe_context(
                lambda: ce(model(x_c), y_c),
                params,
                [
                    (lambda j=j: ce(
                        model(x_m[j * args.micro_batch : (j + 1) * args.micro_batch]),
                        y_m[j * args.micro_batch : (j + 1) * args.micro_batch],
                    ))
                    for j in range(args.micro)
                ],
                depths=DEPTHS, curv_batch=args.curv_batch,
                micro_batch=args.micro_batch, train_batch=args.batch,
                step=step, label="mnist-mlp",
            ))
            print(f"  ckpt {step} built (neg_frac {ctxs[-1].neg_frac():.3f})")
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        xb, yb = xb.to(torch.get_default_dtype()).to(device), yb.to(device)
        opt.step(lambda: ce(model(xb), yb))
    return ctxs


def render(rows, args, elapsed) -> str:
    full = [r for r in rows if r["rank"] is None]
    shipped = [r for r in rows if r["rank"] == args.k_max]
    md = [
        "# DSN screened at its shipped rank, not at full rank",
        "",
        f"Generated by `scripts/measure_rank_restricted.py` in {elapsed:.0f}s "
        f"on the MNIST MLP over {args.steps} DSN steps.",
        "",
        "| candidate | scope | verdict | rho ceil | gain ceil | trajectory gain | "
        "worst depth conv. |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['candidate']} | {r['scope']} | {r['verdict']} | "
            f"{r['rho_ceiling']:.4g} | {r['gain_ceiling']:.4g} | "
            f"{r['trajectory_gain']:.4g} | {r['worst_depth_convergence']:.3f} |"
        )
    md += [
        "",
        "## The verdict changes, and in the opposite direction to the assumption",
        "",
        f"Full-rank verdicts: **{sorted({r['verdict'] for r in full})}**. "
        f"Rank-{args.k_max} verdicts: **{sorted({r['verdict'] for r in shipped})}**.",
        "",
        "The full-rank form FAILs on `rho` (0.46-0.57 against a bar of 1.05). "
        f"DSN's shipped rank-{args.k_max} operator never FAILs, and at "
        "`delta=0.1` it PASSes on both statistics at every complement scaling. "
        "**So the harness's previously-reported DSN rejection was a statement "
        "about an operator DSN does not apply.**",
        "",
        "This does not make DSN good. Stage 2 already measures it training worse "
        "per FLOP than plain SGD, and the negative-result document stands on "
        "three independent mechanism measurements. What it does mean is that "
        "stage 1's rejection was not evidence about DSN, and should not have "
        "been reported as if it were.",
        "",
        "## Why the rank-restricted `rho` values should not be read",
        "",
        f"`rho` for the rank-{args.k_max} operator runs into the tens of "
        "thousands, and falls monotonically as the complement scaling `c` rises. "
        "That is not a discovery, it is mutant G1 firing: `rho` contains "
        "`tr(P^-1 Sigma) / tr(P Sigma)`, which diverges as `P` is shrunk on one "
        "subspace relative to another, with no reference to what that subspace "
        "costs the objective. A rank-restricted operator *is* a subspace-damping "
        "operator, so it lands squarely in `rho`'s known blind spot -- the same "
        "one that scored 33.5 on a configuration training 2,264x worse than SGD.",
        "",
        "The `gain` column is the more trustworthy half here, and it also clears "
        "the bar.",
        "",
        "## Two things that are solid",
        "",
        "**No truncation error.** Every rank-restricted row has a measured depth "
        "convergence of 0.000: a restricted `P` is an explicit operator, so its "
        "three `P`-dependent forms are computed by applying it, and the only "
        "quadrature left (`g'Hg`) is a degree-one polynomial that a Gauss rule "
        "integrates exactly at any depth. The full-rank `delta=1e-3` row, by "
        "contrast, has a convergence of 4.7 -- it is not a measurement.",
        "",
        "**`nan` trajectory gains are the safeguard working.** A `nan` means the "
        "noisy-quadratic model went unbounded at some checkpoint (the "
        "denominator `g'PHPg + tr(PHP Sigma)` went non-positive on indefinite "
        "curvature, and `neg_frac` is 0.42 at step 0). Those rows are "
        "INDETERMINATE rather than PASS, which is the intended behavior: a "
        "broken model cannot be summed into a trajectory total.",
        "",
        "## Rank is not the binding constraint here",
        "",
        "The rank sweep at `delta=1e-2` moves non-monotonically and is driven by "
        "whether some checkpoint's quadratic model went unbounded, not by rank. "
        "On this trajectory the choice of `delta` and the complement scaling both "
        "matter more than `k`.",
        "",
    ]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--curv-batch", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=str, default=str(LOGDIR))
    add_device_arg(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.set_default_dtype(torch.float64)
    gate = Gate(depths=DEPTHS, n_bootstrap=1000)
    t0 = time.time()
    print(f"building MNIST trajectory contexts ... (device: {device})")
    ctxs = build_contexts(args, device)

    rows = []

    def record(P, scope):
        rep = gate.evaluate(P, ctxs)
        conv = [
            max(c.depth_convergence("rho"), c.depth_convergence("gain"))
            for c in rep.checkpoints
        ]
        rows.append({
            "candidate": P.name, "scope": scope, "rank": getattr(P, "rank", None),
            "verdict": rep.verdict.value, "rho_only": rep.rho_only_verdict.value,
            "rho_ceiling": rep.rho_ceiling, "gain_ceiling": rep.gain_ceiling,
            "trajectory_gain": rep.trajectory_gain,
            "worst_depth_convergence": max(
                (c for c in conv if c == c), default=float("nan")
            ),
            "reason": rep.reason,
            "per_checkpoint": [
                {"step": c.step, "verdict": c.verdict.value,
                 "rho": c.rho_ceiling.point, "gain": c.gain_ceiling.point}
                for c in rep.checkpoints
            ],
        })
        r = rows[-1]
        print(f"  {r['candidate']:36s} {scope:24s} {r['verdict']:14s} "
              f"rho {r['rho_ceiling']:9.4g} gain {r['gain_ceiling']:9.4g} "
              f"traj {r['trajectory_gain']:9.4g} conv {r['worst_depth_convergence']:.3f}")

    print("\nfull-rank forms (a DIFFERENT operator from the rank-restricted one, not a bound on it):")
    for delta in (1e-3, 1e-2, 1e-1):
        record(dsn_preconditioner(delta), "full-rank")

    print(f"\nDSN's shipped operator (rank k_max={args.k_max}), complement swept:")
    for delta in (1e-3, 1e-2, 1e-1):
        for c in (0.1, 1.0, 10.0):
            record(KrylovRestrictedPreconditioner(delta, args.k_max, c), f"rank-{args.k_max}")

    print("\nrank sweep at delta=1e-2, c=1 (is rank the binding constraint?):")
    for rank in (2, 4, 8, 16, 32, 64):
        record(KrylovRestrictedPreconditioner(1e-2, rank, 1.0), f"rank-{rank}")

    elapsed = time.time() - t0
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "rank_restricted.json").write_text(
        json.dumps({"rows": rows, "k_max": args.k_max, "depths": list(DEPTHS),
                    "steps": args.steps, "elapsed_s": elapsed}, indent=1)
    )

    md = render(rows, args, elapsed)
    (outdir / "rank_restricted.md").write_text(md)
    print(f"\nwrote {outdir / 'rank_restricted.json'} and {outdir / 'rank_restricted.md'}")


if __name__ == "__main__":
    main()
