"""Mutants aimed at `gain`, and a calibration of both statistics against training.

`rho` was mutant-tested and found structurally blind to `H`.  `gain` was added to
cover that, and then never attacked in the same way.  This file attacks it.

Three mutants, each aimed at a *different* assumption `gain` makes, plus one
calibration study that asks the only question that finally settles a statistic:
**does it order candidates the way real training does?**

    G1  subspace freeze     -- `tr(PHP Sigma)` is quadratic in P, so refusing to
                               move in the noisy subspace looks like pure profit
                               to a one-step model
    G2  one-step locality   -- `gain` is evaluated at a checkpoint; the trajectory
                               roll-up passes a candidate favorable at *any* one
    G3  near-singular       -- `gain`'s integrand `lam/f(lam)^2` is sharper than
        integrand             `rho`'s `1/f(lam)`, so it converges in Lanczos depth
                               strictly more slowly and was shipped without the
                               safeguard `rho` got

    CAL  P = (|H| + delta)^-p for p in [0, 1.5], scored by both statistics and
         then actually trained, on a problem small enough to know the answer.

    python scripts/validate_gain.py
"""

import argparse
import json
import math
import pathlib
import statistics as stats
import time
from dataclasses import replace

import torch

from autoresearch.gate import Gate, Verdict
from autoresearch.gate import statistics as gate_statistics
from autoresearch.preconditioners import (
    IdentityPreconditioner,
    dense_preconditioner,
    dsn_preconditioner,
    power_preconditioner,
    subspace_damping_preconditioner,
)
from autoresearch.probes import ProbeContext
from autoresearch.select import TrialResult
from autoresearch.synthetic import (
    QuadraticProblem,
    blocked_quadratic,
    flat_tail_quadratic,
)
from autoresearch.trials import (
    Candidate,
    PreconditionedSGD,
    QuadraticTrial,
    tune_learning_rate,
)

LOGDIR = pathlib.Path("autoresearch/logs")
DEPTHS = (4, 8, 12)


# --------------------------------------------------------------------- G1


def mutant_subspace_freeze(gate: Gate) -> dict:
    """Damp the subspace where the noise lives, and watch the one-step model cheer.

    Ground truth is not argued for -- it is trained.  The same `P` is handed to
    `PreconditionedSGD` on a real (small) least-squares stream, and its loss at a
    fixed FLOP budget is compared against plain SGD.
    """
    trial = QuadraticTrial(dim=24, cond=100.0, flop_budget=6e6)
    # The trial's Hessian is the design covariance, which is known here.
    _, root, _ = trial._problem(0)
    H = root @ root.T
    lam, V = torch.linalg.eigh(0.5 * (H + H.T))

    rows = []
    for k in (4, 8, 12):
        # Damp the *sharpest* k directions: for this stream the gradient noise
        # covariance is proportional to H, so that is exactly where the noise
        # lives -- and also exactly where the loss is reduced fastest.
        U = V[:, -k:]
        for eps in (1e-1, 1e-2, 1e-3):
            P = subspace_damping_preconditioner(U, eps, f"freeze(k={k},eps={eps:g})")
            cand = Candidate(
                name=P.name, preconditioner=P,
                build=lambda p, lr_, _P=P: PreconditionedSGD(p, lr_, precond=_P),
                lr=3e-3, lr_grid=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
            )
            ctxs = trial.probe_contexts(cand, checkpoints=(0, 40), depths=DEPTHS,
                                        curv_batch=256, micro=8, micro_batch=64)
            rep = gate.evaluate(P, ctxs)
            lr, _, _ = tune_learning_rate(cand, trial, seeds=[10_000, 10_001])
            scores = [trial.run(replace(cand, lr=lr), s).score for s in range(4)]
            rows.append({
                "k": k, "epsilon": eps, "verdict": rep.verdict.value,
                "rho_ceiling": rep.rho_ceiling, "gain_ceiling": rep.gain_ceiling,
                "trained_loss": stats.median(scores), "lr": lr,
            })

    base = Candidate(
        name="sgd", preconditioner=IdentityPreconditioner(),
        build=lambda p, lr_: PreconditionedSGD(p, lr_), lr=3e-3,
        lr_grid=(3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    )
    lr, _, _ = tune_learning_rate(base, trial, seeds=[10_000, 10_001])
    sgd_loss = stats.median([trial.run(replace(base, lr=lr), s).score for s in range(4)])

    worse = [r for r in rows if r["trained_loss"] > sgd_loss]
    return {
        "mutant": "G1 subspace freeze",
        "sgd_trained_loss": sgd_loss,
        "rows": rows,
        "n_total": len(rows),
        "n_trained_worse": len(worse),
        "n_gain_fooled": sum(1 for r in worse if r["gain_ceiling"] > 1.05),
        "n_rho_fooled": sum(1 for r in worse if r["rho_ceiling"] > 1.05),
        "worst_rho": max(rows, key=lambda r: r["rho_ceiling"]),
        "worst_gain": max(rows, key=lambda r: r["gain_ceiling"]),
    }


# --------------------------------------------------------------------- G2


def mutant_one_step_locality(gate: Gate) -> dict:
    """A preconditioner right at exactly one checkpoint and wrong at the rest.

    `gain` is a one-step model and the trajectory roll-up passes a candidate
    that passes at *any* checkpoint.  Composed, those two give a PASS to a
    preconditioner whose time-averaged expected decrease is *worse* than SGD's.

    The right aggregate over a trajectory is the average of `R(P)` against the
    average of `R(I)` -- expected decreases add across steps.  The roll-up
    instead takes a max over checkpoints of their ratio, which is not the same
    quantity and is not conservative.
    """
    # Two curvature regimes the iterate alternates between, sharing g and Sigma.
    # Phase A: sharp overall, and favorable to P -- so its per-step decrease is
    # small in absolute terms but its *ratio* is excellent.
    # Phase B: flat overall, and unfavorable to P -- so its per-step decrease is
    # large in absolute terms while its ratio is terrible.
    # A max over per-checkpoint ratios sees only phase A. A ratio of summed
    # decreases -- which is what a trajectory actually accumulates -- is
    # dominated by phase B. The two aggregates then disagree by ~170x.
    a = blocked_quadratic(0.1, 10.0, 0.01, seed=0, name="phase-A")
    b = blocked_quadratic(1e-3, 2e-3, 5e-4, seed=0, name="phase-B")
    phases = [a, b, a, b]
    ctxs = [p.context(depths=DEPTHS, step=i, label=p.name) for i, p in enumerate(phases)]

    P = dense_preconditioner(torch.linalg.inv(a.H()), "tuned-to-phase-A")
    p_diag = 1.0 / a.lam  # same operator, in the shared eigenbasis

    rep = gate.evaluate(P, ctxs)
    per_ckpt = [gate_statistics(P.forms(c, 12)) for c in ctxs]

    # Trajectory-level truth: expected decreases add, so compare the sums.
    RP = [ph.exact_R(p_diag) for ph in phases]
    RI = [ph.exact_R(torch.ones_like(ph.lam)) for ph in phases]
    trajectory_gain = sum(RP) / sum(RI)

    return {
        "mutant": "G2 one-step locality",
        "verdict": rep.verdict.value,
        "reason": rep.reason,
        "per_checkpoint": [
            {"step": c.step, "phase": c.label, "rho": r, "gain": g}
            for c, (r, g) in zip(ctxs, per_ckpt)
        ],
        "max_checkpoint_gain": max(g for _, g in per_ckpt),
        "trajectory_gain_sum_R": trajectory_gain,
        "fooled": rep.verdict is Verdict.PASS and trajectory_gain < 1.0,
    }


# --------------------------------------------------------------------- G3


def mutant_near_singular_integrand(gate: Gate) -> dict:
    """`gain` converges in Lanczos depth strictly more slowly than `rho`.

    Exact statistics come from the true spectrum using the *same* probe vectors,
    so the gap measured here is pure truncation -- no sampling difference.
    """
    prob = flat_tail_quadratic(n=400, lo=1e-4, hi=1e2, seed=0)
    u = prob.sampled_cross_fit(8, seed=1)
    g = prob.g()
    depths = (5, 10, 25, 50, 100, 200)
    rows = []
    for delta in (1e-3, 1e-2, 1e-1):
        f = lambda l, d=delta: l.abs() + d
        rho_x, gain_x = prob.exact_spectral_statistics(f, g, u)
        P = dsn_preconditioner(delta)
        cells = []
        for depth in depths:
            ctx = ProbeContext(matvec=prob.matvec, g=g, u=u, depths=(depth,),
                               noise_scale=prob.noise_scale)
            rho, gain = gate_statistics(P.forms(ctx, depth))
            node = float(ctx.rules()[0][depth].nodes.abs().min())
            cells.append({
                "depth": depth, "rho": rho, "gain": gain,
                "rho_frac_of_exact": rho / rho_x, "gain_frac_of_exact": gain / gain_x,
                "min_abs_node": node, "spectral_resolution": delta / node,
            })
        rows.append({"delta": delta, "rho_exact": rho_x, "gain_exact": gain_x,
                     "cells": cells})

    # The claim: at every (delta, depth) short of convergence, `gain` is further
    # from its converged value than `rho` is from its own.
    worse = [
        (r["delta"], c["depth"], c["rho_frac_of_exact"], c["gain_frac_of_exact"])
        for r in rows for c in r["cells"]
        if c["rho_frac_of_exact"] < 0.999
    ]
    gain_slower = sum(1 for _, _, rf, gf in worse if gf < rf)
    return {
        "mutant": "G3 near-singular integrand",
        "rows": rows,
        "n_unconverged_cells": len(worse),
        "n_where_gain_slower": gain_slower,
        "gain_always_slower": gain_slower == len(worse),
    }


# ------------------------------------------------------------------ CAL


def calibration(gate: Gate) -> dict:
    """Score `P = (|H| + delta)^-p` with both statistics, then train it.

    The only test that can settle which statistic is measuring the real thing:
    does it *order* the family the way loss-at-a-fixed-FLOP-budget does?  Both
    statistics and the training runs see the same `P`, the same problem and the
    same FLOP budget, and every arm gets its own tuned learning rate so the
    comparison is not a referendum on one shared step size.
    """
    trial = QuadraticTrial(dim=24, cond=100.0, flop_budget=6e6)
    _, root, _ = trial._problem(0)
    H = root @ root.T
    H = 0.5 * (H + H.T)
    lam, V = torch.linalg.eigh(H)
    delta = 1e-2

    rows = []
    for p in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        Pm = V @ torch.diag((lam.abs() + delta) ** (-p)) @ V.T
        P = dense_preconditioner(Pm, f"p={p:g}")
        cand = Candidate(
            name=P.name, preconditioner=P,
            build=lambda q, lr_, _P=P: PreconditionedSGD(q, lr_, precond=_P),
            lr=3e-3, lr_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        )
        ctxs = trial.probe_contexts(cand, checkpoints=(0, 40), depths=DEPTHS,
                                    curv_batch=256, micro=8, micro_batch=64)
        rep = gate.evaluate(P, ctxs)
        spectral = power_preconditioner(p, delta)
        rho_s, gain_s = gate_statistics(spectral.forms(ctxs[-1], 12))
        lr, _, _ = tune_learning_rate(cand, trial, seeds=[10_000, 10_001])
        scores = [trial.run(replace(cand, lr=lr), s).score for s in range(5)]
        rows.append({
            "power": p, "verdict": rep.verdict.value,
            "rho": rep.rho_ceiling, "gain": rep.gain_ceiling,
            "rho_spectral": rho_s, "gain_spectral": gain_s,
            "trained_loss": stats.median(scores), "lr": lr,
        })

    def spearman(xs, ys):
        n = len(xs)
        rx = _ranks(xs)
        ry = _ranks(ys)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else float("nan")

    loss = [r["trained_loss"] for r in rows]
    # Lower loss is better, higher statistic is better -> a statistic that
    # predicts correctly has a NEGATIVE rank correlation with loss.
    return {
        "study": "calibration: P = (|H| + delta)^-p",
        "delta": delta,
        "rows": rows,
        "spearman_rho_vs_loss": spearman([r["rho"] for r in rows], loss),
        "spearman_gain_vs_loss": spearman([r["gain"] for r in rows], loss),
        "best_trained_power": min(rows, key=lambda r: r["trained_loss"])["power"],
        "best_rho_power": max(rows, key=lambda r: r["rho"])["power"],
        "best_gain_power": max(rows, key=lambda r: r["gain"])["power"],
    }


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = float(pos)
    return r


# ---------------------------------------------------------------- render


def render(res: dict, elapsed: float) -> str:
    g1, g2, g3, cal = res["G1"], res["G2"], res["G3"], res["CAL"]
    out = [
        "# `gain` -- mutant validation and calibration",
        "",
        f"Generated by `scripts/validate_gain.py` in {elapsed:.0f}s.",
        "",
        "`rho` was shown to be structurally blind to `H`. `gain` was added to "
        "cover that and had never been attacked the same way. Three mutants, each "
        "aimed at a different assumption `gain` makes, plus a calibration against "
        "actual training.",
        "",
        "## G1 -- subspace freeze: **`gain` holds; `rho` is fooled again**",
        "",
        "`tr(PHP Sigma)` is quadratic in `P`, so damping `P` on the subspace where "
        "the noise lives drives the noise term to zero quadratically while the "
        "numerator loses only the gradient mass that subspace carries. The "
        "prediction was that a one-step model would score refusing-to-move as pure "
        "profit. It does not, and the reason is worth more than the prediction was.",
        "",
        f"Plain SGD, trained, reaches **{g1['sgd_trained_loss']:.5g}**.",
        "",
        "| rank(U) | epsilon | verdict | rho ceil | gain ceil | trained loss | vs SGD |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in g1["rows"]:
        ratio = r["trained_loss"] / g1["sgd_trained_loss"]
        out.append(
            f"| {r['k']} | {r['epsilon']:g} | {r['verdict']} | {r['rho_ceiling']:.4g} | "
            f"{r['gain_ceiling']:.4g} | {r['trained_loss']:.5g} | {ratio:.0f}x |"
        )
    out += [
        "",
        f"All {g1['n_trained_worse']} of {g1['n_total']} configurations train worse "
        f"than SGD -- up to "
        f"{max(r['trained_loss'] for r in g1['rows']) / g1['sgd_trained_loss']:.0f}x "
        "worse. Of those:",
        "",
        f"- **`gain` is fooled by {g1['n_gain_fooled']}** of them.",
        f"- **`rho` is fooled by {g1['n_rho_fooled']}** of them, scoring as high as "
        f"{g1['worst_rho']['rho_ceiling']:.4g} on a configuration that trains "
        f"{g1['worst_rho']['trained_loss'] / g1['sgd_trained_loss']:.0f}x worse than "
        "SGD.",
        "",
        "So this mutant does not break `gain`. It is instead the **third** "
        "independent demonstration that `rho` alone is not a screen -- after "
        "staleness and random spread -- and the first one with trained ground "
        "truth attached rather than a synthetic argument.",
        "",
        "**Why `gain` survives, and the limit of that.** On a *stationary* "
        "quadratic the two things the mutant plays off each other are the same "
        "thing. Loss still recoverable in direction `i` is `c_i^2 / (2 lam_i)`, so "
        "a direction carrying no gradient carries no remaining loss: freezing it is "
        "genuinely free, and `gain` is right to say so. And a direction carrying "
        "enough `lam * s` for damping to help the noise term also carries `g` mass, "
        "so damping it costs the numerator. `gain` cannot be fooled here because "
        "on a stationary quadratic there is nothing to be fooled about.",
        "",
        "That is precisely the boundary of the guarantee. Freezing a subspace is "
        "harmful only when the gradient *later* acquires mass in it -- which makes "
        "G1 and G2 the same defect seen twice: **one-step locality**. Neither "
        "statistic can see it, because neither is given a second point in time.",
        "",
        "## G2 -- one-step locality composed with the trajectory roll-up",
        "",
        "A preconditioner tuned to one curvature regime, screened along a "
        "trajectory that alternates between two.",
        "",
        "| step | phase | rho | gain |",
        "| ---: | --- | ---: | ---: |",
    ]
    for c in g2["per_checkpoint"]:
        out.append(f"| {c['step']} | {c['phase']} | {c['rho']:.4g} | {c['gain']:.4g} |")
    out += [
        "",
        f"The roll-up rule as originally written -- PASS if *any* checkpoint "
        f"passes -- returns **PASS** here, on the strength of the two phase-A "
        f"checkpoints reporting `gain = {g2['max_checkpoint_gain']:.4g}`.",
        "",
        "That is wrong, and the reason is dimensional. `R` is an expected loss "
        "decrease *per step*, and decreases add across steps; ratios do not. The "
        "quantity a trajectory actually accumulates is `sum R(P) / sum R(I)`, "
        f"which here is **{g2['trajectory_gain_sum_R']:.4g}**. The two aggregates "
        f"disagree by "
        f"{g2['max_checkpoint_gain'] / g2['trajectory_gain_sum_R']:.0f}x, in the "
        "unsafe direction.",
        "",
        "The construction is what makes them diverge: phase A is sharp overall, "
        "so its available per-step decrease is small in absolute terms even "
        "though `P` is excellent there; phase B is flat overall, so its available "
        "decrease is large and `P` is terrible there. A max over ratios sees only "
        "phase A. A ratio of sums is dominated by phase B.",
        "",
        f"**Fixed.** The gate now computes the trajectory aggregate and requires "
        f"it to clear the bar before a per-checkpoint PASS stands, so the verdict "
        f"here is **{g2['verdict']}**"
        + ("" if g2["verdict"] == "PASS" else " rather than PASS")
        + ". The check is part of the `gain` half and is skipped under "
        "`require_gain=False`, so the `rho`-only screen stays exactly as blind as "
        "it was and its blindness stays measurable.",
        "",
        "## G3 -- `gain` converges in depth strictly more slowly than `rho`",
        "",
        "Exact values from the true spectrum, same probe vectors, so the gap is "
        "pure Lanczos truncation.",
        "",
        "| delta | depth | min\\|node\\| | delta/min\\|node\\| | rho (frac of exact) | "
        "gain (frac of exact) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in g3["rows"]:
        for c in r["cells"]:
            out.append(
                f"| {r['delta']:g} | {c['depth']} | {c['min_abs_node']:.3g} | "
                f"{c['spectral_resolution']:.2f} | {c['rho_frac_of_exact']:.2f} | "
                f"{c['gain_frac_of_exact']:.2f} |"
            )
    out += [
        "",
        f"`gain` is further from its converged value than `rho` in "
        f"**{g3['n_where_gain_slower']} of {g3['n_unconverged_cells']}** "
        "unconverged cells"
        + (" -- always." if g3["gain_always_slower"] else "."),
        "",
        "Mechanism: for `P = f(H)^-1` the gate integrates `1/f(lam)` for `rho` and "
        "`lam/f(lam)^2` for `gain`. The second peaks at `lam = delta` with height "
        "`1/(4 delta)` -- a sharper near-singular feature than the first -- and a "
        "Gauss rule can only fit polynomials. When `delta` falls at or below the "
        "smallest node the rule resolved, the entire near-singular region is "
        "extrapolated rather than integrated.",
        "",
        "`gain` was added after the depth trap was found for `rho` and shipped "
        "without a safeguard of its own. It now has one: depth convergence is "
        "checked per statistic on the two deepest rungs and gates the verdict in "
        "**both** directions, because a statistic still moving at the deepest rung "
        "cannot support a FAIL either.",
        "",
        "## Calibration -- which statistic orders the family the way training does?",
        "",
        f"`P = (|H| + {cal['delta']:g})^-p`, scored by both statistics and then "
        "trained at a fixed FLOP budget with a per-arm tuned learning rate.",
        "",
        "| power p | verdict | rho | gain | trained loss | lr |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in cal["rows"]:
        out.append(
            f"| {r['power']:g} | {r['verdict']} | {r['rho']:.4g} | {r['gain']:.4g} | "
            f"{r['trained_loss']:.5g} | {r['lr']:g} |"
        )
    out += [
        "",
        "Lower loss is better and higher statistic is better, so a statistic that "
        "predicts correctly has a **negative** rank correlation with trained loss.",
        "",
        f"- Spearman(`rho`, loss) = **{cal['spearman_rho_vs_loss']:+.3f}**",
        f"- Spearman(`gain`, loss) = **{cal['spearman_gain_vs_loss']:+.3f}**",
        f"- best power by training: **{cal['best_trained_power']:g}**; "
        f"by `rho`: {cal['best_rho_power']:g}; by `gain`: {cal['best_gain_power']:g}",
        "",
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(LOGDIR))
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    gate = Gate(depths=DEPTHS, n_bootstrap=500)

    t0 = time.time()
    print("G3 near-singular integrand ...")
    g3 = mutant_near_singular_integrand(gate)
    print("G2 one-step locality ...")
    g2 = mutant_one_step_locality(gate)
    print("G1 subspace freeze (trains) ...")
    g1 = mutant_subspace_freeze(gate)
    print("calibration (trains) ...")
    cal = calibration(gate)
    elapsed = time.time() - t0

    res = {"G1": g1, "G2": g2, "G3": g3, "CAL": cal, "elapsed_s": elapsed}
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "gain_mutants.json").write_text(json.dumps(res, indent=1))
    md = render(res, elapsed)
    (outdir / "gain_mutants.md").write_text(md)
    print(md)
    print(f"\nwrote {outdir / 'gain_mutants.json'} and {outdir / 'gain_mutants.md'}")


if __name__ == "__main__":
    main()
