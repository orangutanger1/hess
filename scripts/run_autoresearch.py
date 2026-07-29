"""End-to-end driver: gate screen, sequential selection, one cost ledger.

    python scripts/run_autoresearch.py                     # synthetic quadratic, ~1 min
    python scripts/run_autoresearch.py --problem mnist     # the MLP this project measures
    python scripts/run_autoresearch.py --admit pass-only   # strict gate admission

Refuses to run unless ``scripts/validate_gate.py`` has recorded a successful
mutant separation.  A gate that has not been shown to separate a known-good
preconditioner from a known-bad one has no demonstrated power, and using it to
allocate a training budget would spend that budget on its noise.
"""

import argparse
import json
import pathlib
import sys
import time

import torch
from dataclasses import replace

from autoresearch import cost as costmod
from autoresearch import report as reportmod
from autoresearch.gate import Gate, Verdict, screen
from autoresearch.select import SuccessiveElimination
from autoresearch.trials import (
    MnistTrial,
    QuadraticTrial,
    default_pool,
    tune_learning_rate,
)

LOGDIR = pathlib.Path("autoresearch/logs")


def require_validated_gate(logdir: pathlib.Path, force: bool) -> dict:
    path = logdir / "gate_mutants.json"
    if not path.exists():
        if force:
            return {"separation": "UNVERIFIED (--force)"}
        raise SystemExit(
            f"{path} not found. Run `python scripts/validate_gate.py` first: the "
            "gate must be shown to separate its mutants before it is used to "
            "reject or accept anything. Pass --force to override, which makes "
            "every verdict below uncertified."
        )
    data = json.loads(path.read_text())
    mutants = [r for r in data["rows"] if r.get("counts_toward_separation", True)]
    ok = all(r["separated"] for r in mutants)
    if not ok and not force:
        raise SystemExit(
            "The recorded mutant validation did not achieve separation. Stage 2 "
            "must not be run against this gate."
        )
    return {
        "separation": "ACHIEVED" if ok else "NOT ACHIEVED (--force)",
        "n_mutants": len(mutants),
        "log": str(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=("quadratic", "mnist"), default="quadratic")
    ap.add_argument("--admit", choices=("not-fail", "pass-only"), default="not-fail")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fdr-q", type=float, default=0.10)
    ap.add_argument("--n-pilot", type=int, default=3)
    ap.add_argument("--max-seeds", type=int, default=12)
    ap.add_argument("--tune-seeds", type=int, default=2,
                    help="seeds per learning-rate grid point; 0 disables tuning")
    ap.add_argument("--no-dsn", action="store_true")
    ap.add_argument("--controls", type=int, default=1,
                    help="fixed-random negative controls in the pool")
    ap.add_argument("--force", action="store_true", help="run without a validated gate")
    ap.add_argument("--out", type=str, default=str(LOGDIR))
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    validation = require_validated_gate(LOGDIR, args.force)
    print(f"gate validation: {validation['separation']}")

    if args.problem == "quadratic":
        trial = QuadraticTrial()
        dim = trial.dim
        depths = (4, 8, 16, 24)
        checkpoints = (0, 40, 90)
        cost_model = costmod.analytic_cost_model(dim)
    else:
        trial = MnistTrial()
        dim = 784 * 96 + 96 + 96 * 10 + 10
        # `lambda_g` is a harmonic mean and Lanczos resolves small eigenvalues
        # last, so a shallow sweep on a real model measures truncation rather
        # than rho -- see the depth_spread diagnostic in gate.py.
        depths = (25, 50, 75, 100)
        checkpoints = (0, 30, 60)
        cost_model = costmod.analytic_cost_model(dim)

    pool = default_pool(dim, include_dsn=not args.no_dsn, n_controls=args.controls)
    baseline = next(c for c in pool if c.is_baseline)

    # ---------------------------------------------------------------- stage 1
    t0 = time.time()
    print(f"\nstage 1: probing {args.problem} trajectory at checkpoints {checkpoints}")
    contexts = trial.probe_contexts(baseline, checkpoints, depths)
    gate = Gate(depths=depths, alpha=args.alpha)
    screenable = [c for c in pool if c.preconditioner is not None]
    reports = screen(gate, [c.preconditioner for c in screenable], contexts)
    by_candidate = {c.name: reports[c.preconditioner.name] for c in screenable}

    ledgers = {
        c.name: costmod.Ledger(
            candidate=c.name,
            stage1_exclusive=cost_model.total(by_candidate[c.name].exclusive_cost),
            stage1_shared_share=cost_model.total(by_candidate[c.name].shared_cost_share),
        )
        if c.name in by_candidate
        else costmod.Ledger(candidate=c.name)
        for c in pool
    }
    stage1_s = time.time() - t0

    admitted, rejected = [], []
    for c in pool:
        r = by_candidate.get(c.name)
        if c.is_baseline:
            admitted.append(c)
            reason = "baseline: admitted regardless, it is the reference"
        elif r is None:
            admitted.append(c)
            reason = "not screenable, admitted unscreened"
        elif r.verdict is Verdict.PASS or (
            args.admit == "not-fail" and r.verdict is not Verdict.FAIL
        ):
            admitted.append(c)
            reason = r.verdict.value
        else:
            rejected.append(c)
            reason = r.verdict.value
        rho = f"{r.rho_ceiling:.4g}" if r else "-"
        gn = f"{r.gain_ceiling:.4g}" if r else "-"
        print(f"  {c.name:34s} rho_ceil {rho:>10s} gain_ceil {gn:>10s}  -> {reason}")

    print(f"  admitted {len(admitted)}, gate-rejected {len(rejected)} "
          f"({stage1_s:.1f}s)")

    # ------------------------------------------------------- stage 2a: tuning
    by_name = {c.name: c for c in admitted}
    if args.tune_seeds > 0:
        # Seeds 10_000+ so tuning never touches a seed the selection phase will
        # later certify a winner on. Choosing the learning rate on the seeds the
        # discovery claim is made from is selection on the test set.
        tune_seeds = [10_000 + i for i in range(args.tune_seeds)]
        print(f"\nstage 2a: learning-rate search on seeds {tune_seeds} "
              f"(disjoint from selection seeds)")
        t_tune = time.time()
        for c in admitted:
            lr, spent, table = tune_learning_rate(c, trial, tune_seeds, cost_model)
            led = ledgers[c.name]
            led.stage2_tuning += spent
            led.n_tuning_trials += len(table) * len(tune_seeds)
            led.chosen_lr = lr
            by_name[c.name] = replace(c, lr=lr)
            best = min(t["mean_score"] for t in table)
            print(f"  {c.name:34s} lr {lr:<8g} best mean {best:.5g} "
                  f"({led.n_tuning_trials} trials, {spent:.3g} FLOPs)")
        print(f"  tuning took {time.time() - t_tune:.1f}s")
    else:
        for c in admitted:
            ledgers[c.name].chosen_lr = c.lr

    # --------------------------------------------------- stage 2b: selection
    print(f"\nstage 2b: sequential selection over {len(admitted)} arms")
    t1 = time.time()
    sel = SuccessiveElimination(
        baseline.name, alpha=args.alpha, fdr_q=args.fdr_q,
        n_pilot=args.n_pilot, max_seeds=args.max_seeds,
    )
    for c in admitted:
        if not c.is_baseline:
            sel.add(c.name)

    def pull(name, seed):
        res = trial.run(by_name[name], seed, cost_model)
        led = ledgers[name]
        led.stage2_training += res.flops
        led.n_trials += 1
        led.trials_flops.append(res.flops)
        return res

    selection = sel.run(pull, verbose=True)
    stage2_s = time.time() - t1
    print(f"  winner: {selection.winner}  survivors: {selection.survivors} "
          f"({selection.stopped_because}, {stage2_s:.1f}s)")

    # ----------------------------------------------------------------- report
    rep = reportmod.build(
        cost_model=cost_model,
        gate_reports=by_candidate,
        selection=selection,
        ledgers=ledgers,
        gate_rejected=[c.name for c in rejected],
        max_seeds=args.max_seeds,
    )
    rep.notes.append(
        f"Gate mutant validation: {validation['separation']} ({validation.get('log', '-')})."
    )
    rep.notes.append(
        f"Wall clock: stage 1 {stage1_s:.1f}s, stage 2 {stage2_s:.1f}s. FLOPs, not "
        "seconds, are the comparison metric; timings are reported only so a run "
        "can be reproduced at a known scale."
    )
    j, m = reportmod.write(rep, args.out)
    print(f"\nwrote {j} and {m}\n")
    print(reportmod.render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
