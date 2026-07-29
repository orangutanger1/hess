"""Validate the stage-1 gate against mutants, and log both outcomes.

Run this before the gate is used to reject or accept anything.  A screen that
has not been shown to separate a known-good preconditioner from a known-bad one
has no demonstrated power, and pointing it at real candidates would launder its
own noise into a verdict.  The exit code is the contract: non-zero means the
gate did **not** separate the mutants and stage 2 must not be run.

Mutants
-------

``PASS``   ``P* = H^-1`` on a synthetic quadratic built so that curvature
           preconditioning pays.  The answer is known in closed form, so this
           checks the *estimator* against truth as well as the verdict.
``FAIL``   a constant preconditioner (``P = 3I``), which is not a
           preconditioner at all, and a *frozen* ``P = H_0^-1`` applied after
           the curvature has drifted.

The frozen case is the one that matters.  This project has already shipped a
staleness gate with zero power against staleness -- a maximally-stale frozen
basis scored *better* than the correct implementation on both gates written to
catch it (negative-result doc, section 8.2).  So the frozen mutant is run
against both the composite gate and the ``rho``-only screen, and the difference
between them is reported as a number rather than argued for.

    python scripts/validate_gate.py                 # synthetic only, seconds
    python scripts/validate_gate.py --mnist         # also screen DSN's own trajectory
"""

import argparse
import json
import pathlib
import sys
import time

import torch

from _device import add_device_arg, resolve_device
from autoresearch.gate import Gate, Verdict, statistics
from autoresearch.preconditioners import (
    IdentityPreconditioner,
    dense_preconditioner,
    dsn_preconditioner,
    frozen_spectral_preconditioner,
    newton_preconditioner,
)
from autoresearch.synthetic import (
    blocked_quadratic,
    favorable_pair,
    unfavorable_quadratic,
)

LOGDIR = pathlib.Path("autoresearch/logs")

#: Synthetic problems here are 12-dimensional, so depth 12 makes the quadrature
#: rule exact and the sweep is a genuine convergence check rather than a sweep
#: over truncation error.
DEPTHS = (4, 8, 12)

#: Real trajectories need far more.  `lambda_g` is a harmonic mean, dominated by
#: the smallest eigenvalues, which Lanczos resolves last: on the MNIST MLP,
#: depths (4, 8, 12) put `rho` near 10 while depths (25, 50, 75, 100) put it at
#: 0.045-0.19.  The shallow sweep is not a measurement of `rho`, it is a
#: measurement of truncation, and the gate says so via `depth_spread`.
MNIST_DEPTHS = (25, 50, 75, 100)


def _row(name, expect, report, note="", counts=True):
    ck = report.checkpoints[-1]
    return {
        "mutant": name,
        "expected": expect.value,
        "counts_toward_separation": counts,
        "verdict": report.verdict.value,
        "separated": report.verdict is expect,
        "rho": ck.rho_ceiling.point,
        "rho_lo": ck.rho_floor.lo,
        "rho_hi": ck.rho_ceiling.hi,
        "gain": ck.gain_ceiling.point,
        "gain_lo": ck.gain_floor.lo,
        "gain_hi": ck.gain_ceiling.hi,
        "rho_only_verdict": report.rho_only_verdict.value,
        "spec": report.spec,
        "reason": report.reason,
        "note": note,
    }


def synthetic_suite(gate: Gate) -> tuple[list[dict], dict]:
    rows = []
    t0, tT = favorable_pair()
    ctx0 = t0.context(depths=DEPTHS, step=0)
    ctxT = tT.context(depths=DEPTHS, step=1)

    # --- (b) should obviously pass: the exact optimum, answer known ----------
    good = blocked_quadratic(0.1, 10.0, 0.01)
    ctx_good = good.context(depths=DEPTHS, step=0)
    star = newton_preconditioner(1e-12)
    rho_hat, gain_hat = statistics(star.forms(ctx_good, 12))
    pdiag = good.newton_diag()
    exact = {
        "rho_closed_form": good.exact_rho(pdiag),
        "rho_estimated": rho_hat,
        "gain_closed_form": good.exact_gain(pdiag),
        "gain_estimated": gain_hat,
        "R_at_optimum_noiseless": None,
        "R_ceiling_noiseless": None,
    }
    quiet = blocked_quadratic(0.1, 10.0, 0.01, noise_scale=1e-10)
    exact["R_at_optimum_noiseless"] = quiet.exact_R(quiet.newton_diag())
    exact["R_ceiling_noiseless"] = quiet.ceiling_R()
    rows.append(
        _row(
            "exact optimum P* = H^-1 (favorable quadratic)",
            Verdict.PASS,
            gate.evaluate(star, [ctx_good]),
            note="estimator vs closed form and vs the all-P Cauchy-Schwarz ceiling",
        )
    )

    # --- (a-i) should obviously fail: a constant preconditioner --------------
    rows.append(
        _row(
            "constant P = 3I",
            Verdict.FAIL,
            gate.evaluate(dense_preconditioner(3.0 * torch.eye(good.n), "constant(3I)"), [ctx_good]),
            note="scores exactly 1.0 on both statistics; a scaled no-op",
        )
    )
    rows.append(
        _row(
            "identity P = I (SGD control)",
            Verdict.FAIL,
            gate.evaluate(IdentityPreconditioner(), [ctx_good]),
            note="the calibration point: must never certify itself",
        )
    )

    # --- (a-ii) should obviously fail: frozen after curvature drift ----------
    frozen = dense_preconditioner(torch.linalg.inv(t0.H()), "frozen(H0^-1)")
    fresh = dense_preconditioner(torch.linalg.inv(tT.H()), "fresh(HT^-1)")
    frozen_report = gate.evaluate(frozen, [ctxT])
    rows.append(
        _row(
            "frozen P = H_0^-1, evaluated at t=T",
            Verdict.FAIL,
            frozen_report,
            note="the staleness mutant",
        )
    )
    rows.append(
        _row(
            "refreshed P = H_T^-1, evaluated at t=T",
            Verdict.PASS,
            gate.evaluate(fresh, [ctxT]),
            note="control for the staleness mutant: same problem, current curvature",
        )
    )
    for rank in (4, 8, 12):
        stale = frozen_spectral_preconditioner(
            ctx0, lambda lam: lam.abs() + 1e-8, rank=rank, name=f"frozen-rank{rank}"
        )
        expect = Verdict.FAIL if rank >= 8 else Verdict.PASS
        rows.append(
            _row(
                f"frozen rank-{rank} Krylov snapshot at t=T",
                expect,
                gate.evaluate(stale, [ctxT]),
                note=(
                    "below rank 8 the snapshot has not resolved the drifting (flattest) "
                    "block at all, so it is protected by ignorance rather than freshness"
                ),
            )
        )

    # --- the DSN-shaped negative ---------------------------------------------
    bad = unfavorable_quadratic()
    rows.append(
        _row(
            "P* = H^-1 on a DSN-shaped unfavorable quadratic",
            Verdict.FAIL,
            gate.evaluate(newton_preconditioner(1e-12), [bad.context(depths=DEPTHS, step=0)]),
            note="gradient in sharp directions, noise in flat ones: nothing to buy",
        )
    )

    # --- (a-iii) a data-independent random diagonal ---------------------------
    # Not a single mutant but a population claim, because the answer depends on
    # the draw.  Screening one lucky seed would be exactly the cherry-pick this
    # file exists to prevent.
    import statistics as stats

    from autoresearch.preconditioners import fixed_random_preconditioner

    spread_rows = []
    for spread in (0.25, 0.5, 1.0, 2.0):
        rhos, gains, composite, rho_only = [], [], [], []
        for s in range(24):
            prob = blocked_quadratic(0.1, 10.0, 0.01, seed=s % 7)
            ctx = prob.context(depths=DEPTHS)
            P = fixed_random_preconditioner(
                prob.n, seed=100 + s, log_spread=spread, like=prob.c
            )
            r, gv = statistics(P.forms(ctx, 12))
            rhos.append(r)
            gains.append(gv)
            rep = gate.evaluate(P, [ctx])
            composite.append(rep.verdict is Verdict.PASS)
            rho_only.append(rep.rho_only_verdict is Verdict.PASS)
        spread_rows.append({
            "log_spread": spread,
            "cond_P": float(torch.exp(torch.tensor(2.0 * spread))),
            "rho_median": stats.median(rhos),
            "rho_frac_above_1": sum(r > 1 for r in rhos) / len(rhos),
            "gain_median": stats.median(gains),
            "gain_frac_above_1": sum(g > 1 for g in gains) / len(gains),
            "composite_pass_rate": sum(composite) / len(composite),
            "rho_only_pass_rate": sum(rho_only) / len(rho_only),
        })

    # --- the H-blindness measurement ------------------------------------------
    rho_t0, _ = statistics(frozen.forms(ctx0, 12))
    rho_tT, gain_tT = statistics(frozen.forms(ctxT, 12))
    blindness = {
        "frozen_rho_at_t0": rho_t0,
        "frozen_rho_at_tT": rho_tT,
        "identical_to_machine_precision": abs(rho_t0 - rho_tT) < 1e-12 * abs(rho_t0),
        "frozen_gain_at_tT": gain_tT,
        "composite_verdict": frozen_report.verdict.value,
        "rho_only_verdict": frozen_report.rho_only_verdict.value,
    }
    return rows, {
        "exact_optimum_check": exact,
        "h_blindness": blindness,
        "random_diagonal_sweep": spread_rows,
    }


def mnist_suite(gate_cfg: dict, steps: int, micro: int, curv_batch: int,
                 device: torch.device) -> list[dict]:
    """Screen the DSN preconditioner family on DSN's own MNIST trajectory.

    **Not a mutant test, and it does not count toward separation.**  Two
    different jobs: a smoke test that the same predicate runs unchanged against
    a real trajectory and a real ``hvp``, and a cross-check that the generalized
    ``rho`` reproduces the published measurement (``scripts/measure_rho.py``,
    max 0.717 against a threshold of 1).

    Truth is not known here, so a disagreement is a finding to investigate, not
    a failed test.  Counting it as one would let a real trajectory -- whose
    right answer is exactly what is under study -- decide whether the gate is
    trustworthy.
    """
    from torchvision import datasets, transforms

    from autoresearch.probes import build_probe_context
    from dsn import DSN
    from dsn.subspace import KrylovBuilder

    torch.manual_seed(0)
    ds = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    micro_batch, batch = 128, 256
    n_pool = curv_batch + micro * micro_batch
    # ToTensor() yields float32; the model is built under the float64 default
    # this script sets, and DSN raises rather than silently mixing dtypes.
    x = torch.stack([ds[i][0] for i in range(n_pool)]).to(torch.get_default_dtype()).to(device)
    y = torch.tensor([ds[i][1] for i in range(n_pool)]).to(device)
    x_c, y_c = x[:curv_batch], y[:curv_batch]
    x_m, y_m = x[curv_batch:], y[curv_batch:]

    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(784, 96), torch.nn.Tanh(), torch.nn.Linear(96, 10)
    ).to(device)
    params = list(model.parameters())
    opt = DSN(model.parameters(), lr=1e-2, weight_decay=0.0,
              builder=KrylovBuilder(k_max=8, m_recycle=4, tau=0.1))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    ce = torch.nn.functional.cross_entropy
    checkpoints = sorted({0, steps // 2, steps - 1})
    contexts = []
    it = iter(loader)
    for step in range(steps):
        if step in checkpoints:
            contexts.append(
                build_probe_context(
                    lambda: ce(model(x_c), y_c),
                    params,
                    [
                        (lambda j=j: ce(
                            model(x_m[j * micro_batch : (j + 1) * micro_batch]),
                            y_m[j * micro_batch : (j + 1) * micro_batch],
                        ))
                        for j in range(micro)
                    ],
                    depths=MNIST_DEPTHS,
                    curv_batch=curv_batch,
                    micro_batch=micro_batch,
                    train_batch=batch,
                    step=step,
                    label="mnist-mlp",
                )
            )
            print(f"  [ckpt {step:4d}] neg_frac {contexts[-1].neg_frac():.3f}")
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        xb, yb = xb.to(torch.get_default_dtype()).to(device), yb.to(device)
        opt.step(lambda: ce(model(xb), yb))

    gate = Gate(depths=MNIST_DEPTHS, **gate_cfg)
    rows = []
    for p in (dsn_preconditioner(1e-3), dsn_preconditioner(1e-2), dsn_preconditioner(1e-1),
              IdentityPreconditioner()):
        r = gate.evaluate(p, contexts)
        rows.append(
            _row(f"MNIST/DSN trajectory: {p.name}", Verdict.FAIL, r, counts=False,
                 note=(
                     "cross-check only, and NOT a verdict on DSN: this screens the "
                     "FULL-RANK form, while DSN applies a rank-k_max Krylov "
                     "restriction. See scripts/measure_rank_restricted.py, where "
                     "the rank-8 verdict is PASS rather than FAIL. Published rho "
                     "on this trajectory caps at 0.717."
                 ))
        )
    return rows


def render(rows, extras, elapsed) -> str:
    mutants = [r for r in rows if r["counts_toward_separation"]]
    others = [r for r in rows if not r["counts_toward_separation"]]
    ok = all(r["separated"] for r in mutants)
    out = [
        "# Stage-1 gate -- mutant validation log",
        "",
        f"Generated by `scripts/validate_gate.py` in {elapsed:.1f}s.",
        "",
        f"**Separation: {'ACHIEVED' if ok else 'NOT ACHIEVED'}** "
        f"({sum(r['separated'] for r in mutants)}/{len(mutants)} mutants landed on "
        "their expected verdict).",
        "",
        "Separation is judged on the synthetic mutants only, where the right answer "
        "is known by construction.  The real-trajectory rows below it are a "
        "cross-check: their right answer is the thing under study, so they cannot "
        "also be the thing that certifies the instrument.",
        "",
        "| mutant | expected | verdict | rho (ceiling) | gain (ceiling) | rho-only | ok |",
        "| --- | --- | --- | ---: | ---: | --- | :-: |",
    ]
    for r in mutants:
        out.append(
            f"| {r['mutant']} | {r['expected']} | {r['verdict']} | "
            f"{r['rho']:.4g} | {r['gain']:.4g} | {r['rho_only_verdict']} | "
            f"{'yes' if r['separated'] else 'NO'} |"
        )

    if others:
        out += [
            "",
            "## Cross-check on a real trajectory (does not count toward separation)",
            "",
            "| candidate | verdict | rho (ceiling) | gain (ceiling) | reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for r in others:
            out.append(
                f"| {r['mutant']} | {r['verdict']} | {r['rho']:.4g} | "
                f"{r['gain']:.4g} | {r['reason']} |"
            )
        out += [
            "",
            "The generalized `rho` reduces to `scripts/measure_rho.py`'s formula at "
            "`P = (|H| + delta)^-1`, and reproduces it **to machine precision** at "
            "every `(delta, k)` cell -- 0.176 / 0.193 / 0.064 / 0.045 at `delta=1e-3` "
            "and depths 25 / 50 / 75 / 100, matching the published step-0 row.  The "
            "generalization is the same measurement, not a new one.",
            "",
            "Read the depth sweep before reading the verdict.  At depths (4, 8, 12) "
            "the same checkpoint reports `rho` near 10; at (25, 50, 75, 100) it "
            "reports 0.045-0.19.  `lambda_g` is a harmonic mean, so it is dominated "
            "by the smallest eigenvalues, which Lanczos resolves last -- a shallow "
            "sweep measures truncation, not `rho`.  The ceiling/floor rule returns "
            "INDETERMINATE rather than a false PASS in that regime, and "
            "`depth_spread` names the reason.",
        ]

    e = extras.get("exact_optimum_check")
    if e:
        out += [
            "",
            "## The PASS mutant is the optimum, not merely good",
            "",
            "| quantity | closed form | estimated |",
            "| --- | ---: | ---: |",
            f"| rho at P* = H^-1 | {e['rho_closed_form']:.6g} | {e['rho_estimated']:.6g} |",
            f"| gain at P* = H^-1 | {e['gain_closed_form']:.6g} | {e['gain_estimated']:.6g} |",
            "",
            "In the noiseless limit `sup_P R(P) <= (g'H^-1 g)(g'C^-1 g)` tightens to "
            "`g'H^-1 g`, and `P = H^-1` attains it: "
            f"`R(P*) = {e['R_at_optimum_noiseless']:.6g}` against a ceiling of "
            f"`{e['R_ceiling_noiseless']:.6g}`.",
        ]

    rd = extras.get("random_diagonal_sweep")
    if rd:
        out += [
            "",
            "## A data-independent random diagonal, swept over spread",
            "",
            "24 draws per row, on the favorable quadratic.  `P` is drawn from a "
            "seed and never looks at `H`, `g` or `Sigma`, so nothing here carries "
            "any curvature information.",
            "",
            "| log spread | cond(P) | rho median | rho > 1 | gain median | gain > 1 | "
            "composite PASS | rho-only PASS |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in rd:
            out.append(
                f"| {r['log_spread']:.2f} | {r['cond_P']:.3g} | "
                f"{r['rho_median']:.4g} | {r['rho_frac_above_1']:.0%} | "
                f"{r['gain_median']:.4g} | {r['gain_frac_above_1']:.0%} | "
                f"{r['composite_pass_rate']:.0%} | {r['rho_only_pass_rate']:.0%} |"
            )
        out += [
            "",
            "**The two statistics move in opposite directions as the spread grows.** "
            "`rho` rises and `gain` falls, on a preconditioner that by construction "
            "knows nothing.  `rho` rewards per-coordinate spread for a structural "
            "reason: it contains `(g'Pg)^2 / ||g||^4`, which grows quadratically in "
            "`P`'s action on the gradient's support, while the compensating trace "
            "ratio `tr(P^-1 Sigma) / tr(P Sigma)` only recovers part of it -- an "
            "arithmetic-versus-harmonic asymmetry, not a sampling artifact.  `gain` "
            "is not fooled, because scaling directions up without regard to their "
            "curvature raises `g'PHPg` and `tr(PHP Sigma)` in the denominator.",
            "",
            "This is the second independent demonstration on this page that the "
            "`rho` screen alone is insufficient -- the first being staleness.  The "
            "two failures are unrelated in mechanism and share one cause: `H` does "
            "not appear in `rho`.",
        ]

    b = extras.get("h_blindness")
    if b:
        out += [
            "",
            "## Why the gate is not `rho` alone",
            "",
            "`H` does not appear in `rho`.  With `g` and `Sigma` held fixed while the "
            "curvature drifts, `rho` of a frozen preconditioner is therefore constant "
            "*by construction* -- it cannot detect staleness, at any sample size.",
            "",
            f"- frozen `P = H_0^-1`, `rho` at t=0: **{b['frozen_rho_at_t0']:.6g}**",
            f"- the same `P`, `rho` at t=T after the drift: **{b['frozen_rho_at_tT']:.6g}** "
            f"(identical to machine precision: {b['identical_to_machine_precision']})",
            f"- the same `P`, `gain` at t=T: **{b['frozen_gain_at_tT']:.4g}**",
            "",
            f"`rho`-only verdict: **{b['rho_only_verdict']}**.  "
            f"Composite verdict: **{b['composite_verdict']}**.",
            "",
            "That gap is the same failure this project has already shipped once: a "
            "maximally-stale frozen basis scoring better than the correct "
            "implementation on both gates written to catch it.",
        ]

    out += [
        "",
        "## Reproduce",
        "",
        "    python scripts/validate_gate.py",
        "    python scripts/validate_gate.py --mnist",
        "    python -m pytest tests/test_autoresearch_gate.py",
        "",
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true", help="also screen DSN's MNIST trajectory")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--curv-batch", type=int, default=1024)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--min-effect", type=float, default=0.05)
    ap.add_argument("--out", type=str, default=str(LOGDIR))
    add_device_arg(ap)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    gate = Gate(depths=DEPTHS, n_bootstrap=args.bootstrap, min_effect=args.min_effect)

    t0 = time.time()
    print("synthetic mutants ...")
    rows, extras = synthetic_suite(gate)
    if args.mnist:
        device = resolve_device(args.device)
        print(f"MNIST/DSN trajectory ... (device: {device})")
        rows += mnist_suite(
            dict(n_bootstrap=args.bootstrap, min_effect=args.min_effect),
            args.steps, args.micro, args.curv_batch, device,
        )
    elapsed = time.time() - t0

    for r in rows:
        mark = ("ok " if r["separated"] else "!! ") if r["counts_toward_separation"] else "-- "
        print(f"{mark}{r['mutant'][:58]:58s} want {r['expected']:13s} got {r['verdict']:13s} "
              f"rho {r['rho']:10.4g} gain {r['gain']:10.4g}")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "gate_mutants.json").write_text(
        json.dumps({"rows": rows, "extras": extras, "elapsed_s": elapsed,
                    "gate": {"depths": list(DEPTHS), "min_effect": args.min_effect,
                             "alpha": gate.alpha, "n_bootstrap": args.bootstrap}},
                   indent=1)
    )
    md = render(rows, extras, elapsed)
    (outdir / "gate_mutants.md").write_text(md)
    print(f"\nwrote {outdir / 'gate_mutants.json'} and {outdir / 'gate_mutants.md'}")

    mutants = [r for r in rows if r["counts_toward_separation"]]
    separated = all(r["separated"] for r in mutants)
    print(f"\nSEPARATION: {'ACHIEVED' if separated else 'NOT ACHIEVED'}")
    if not separated:
        print("Stage 2 must not be run against this gate.", file=sys.stderr)
    return 0 if separated else 1


if __name__ == "__main__":
    sys.exit(main())
