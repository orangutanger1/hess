# `gain` under attack, and DSN screened at its shipped rank

**Date:** 2026-07-29
**Status:** Two open issues resolved. Both changed the harness; one changed a
previously-reported number.

Follow-up to [`2026-07-29-autoresearch-harness.md`](2026-07-29-autoresearch-harness.md).
Two things were open:

1. `gain` had never been mutant-tested in isolation, and it disagreed with `rho`
   by a reported ~80x on the MNIST trajectory.
2. Stage 1 screened the idealized full-rank `P`, while DSN ships a rank-8 Krylov
   restriction, so no DSN verdict was about DSN.

---

## Issue 1 — three mutants aimed at `gain`

Full log: [`autoresearch/logs/gain_mutants.md`](../../../autoresearch/logs/gain_mutants.md).
Depth data: [`autoresearch/logs/gain_depth_diagnosis.json`](../../../autoresearch/logs/gain_depth_diagnosis.json).

### G3 — `gain` converges in Lanczos depth strictly more slowly than `rho`

This is the one that mattered, and the user's hypothesis was right: `gain` was
added *after* the depth trap was found for `rho` and inherited no safeguard.

For a spectral `P = f(H)^-1` the gate integrates

    rho  needs   1/f(lam)          peaks at lam = 0,      height 1/delta
    gain needs   lam / f(lam)^2    peaks at lam = delta,  height 1/(4 delta)

Both are near-singular for small `delta`. A Gauss rule can only fit polynomials,
and **`gain`'s integrand is the sharper feature**, so it is approximated worse at
every depth short of convergence. Measured on a 400-dimensional flat-tailed
spectrum, against exact values computed from the true spectrum using the same
probe vectors (so the gap is truncation alone, with no sampling difference):

| delta | depth | delta / min\|node\| | `rho` frac of exact | `gain` frac of exact |
| ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 50 | 0.18 | 0.33 | 0.15 |
| 1e-3 | 100 | 0.94 | 0.86 | 0.57 |
| 1e-3 | 200 | 6.29 | 1.00 | 1.00 |
| 1e-2 | 50 | 1.79 | 0.86 | 0.48 |
| 1e-2 | 100 | 9.38 | 1.00 | 0.97 |
| 1e-1 | 25 | 4.67 | 0.96 | 0.65 |
| 1e-1 | 50 | 17.9 | 1.00 | 0.98 |

`gain` is further from its converged value than `rho` in **13 of 13** unconverged
cells — always.

The third column is the mechanism reduced to one number, and it is now a
first-class diagnostic (`Cell.spectral_resolution`). A depth-`k` Gauss rule
samples the spectrum only down to its smallest node; below that the integrand is
*extrapolated* by a polynomial, not integrated. When `delta` falls at or below
that node, the entire near-singular region — which is where all of the
integrand's mass is — lies in the unsampled part.

**Fix, and it is not just a warning this time.** Depth convergence is now
measured per statistic as the relative change between the two *deepest* rungs,
and it gates the verdict in **both directions**. The previous safeguard only
appended a note to the reason string, so a sweep that happened to sit entirely
below the bar still returned a confident FAIL — and on the flat-tail spectrum
`rho` runs 9.2 at depth 5 and 1104 at depth 200, so such a FAIL can be two orders
of magnitude wrong. A statistic still moving at the deepest rung cannot support a
PASS *or* a FAIL.

### G1 — subspace freeze: `gain` holds, `rho` is fooled a third time

`tr(PHP Sigma)` is quadratic in `P`, so damping `P` on the noise subspace drives
the noise term to zero quadratically. The prediction was that a one-step model
would score refusing-to-move as pure profit.

**It did not, and the reason is worth more than the prediction was.** Across nine
configurations that all train worse than SGD — up to 11,432x worse — `gain` was
fooled by **0** and `rho` by **5**, scoring as high as 33.5 on a configuration
that trains 2,264x worse than SGD.

Why `gain` survives, and the exact limit of that: on a *stationary* quadratic the
two things the mutant tries to play off each other are the same thing. Loss still
recoverable in direction `i` is `c_i^2 / (2 lam_i)`, so a direction carrying no
gradient carries no remaining loss and freezing it is genuinely free. And a
direction carrying enough `lam * s` for damping to help the noise term also
carries `g` mass, so damping it costs the numerator. `gain` cannot be fooled here
because on a stationary quadratic there is nothing to be fooled about.

That boundary is the finding. Freezing a subspace is harmful only when the
gradient *later* acquires mass in it — which makes G1 and G2 the same defect seen
twice: **one-step locality**. And it is a fourth strike against `rho`, the first
one with trained ground truth attached rather than a synthetic argument.

### G2 — one-step locality composed with the trajectory roll-up

A preconditioner tuned to one curvature regime, screened along a trajectory
alternating between two:

| step | phase | `rho` | `gain` |
| ---: | --- | ---: | ---: |
| 0 | A (sharp overall, `P` excellent) | 11.08 | 10.84 |
| 1 | B (flat overall, `P` terrible) | 11.08 | 0.0606 |
| 2 | A | 11.08 | 10.84 |
| 3 | B | 11.08 | 0.0606 |

The roll-up rule as originally written — PASS if *any* checkpoint passes —
returned **PASS**, on the strength of the phase-A checkpoints.

That is wrong, and the reason is dimensional. `R` is an expected loss decrease
*per step*; decreases add across steps, ratios do not. The quantity a trajectory
accumulates is `sum R(P) / sum R(I)`, which here is **0.064**. The two aggregates
disagree by **171x**, in the unsafe direction. The construction is what makes
them diverge: phase A has a small available per-step decrease and an excellent
ratio; phase B has a large available decrease and a terrible ratio. A max over
ratios sees only phase A; a ratio of sums is dominated by phase B.

**Fixed.** `GateReport.trajectory_gain` computes the aggregate and a
per-checkpoint PASS must survive it. The check is part of the `gain` half and is
skipped under `require_gain=False`, so the `rho`-only screen stays exactly as
blind as it was and its blindness stays measurable.

### Calibration — which statistic orders candidates the way training does?

The only test that finally settles a statistic. `P = (|H| + 0.01)^-p` for
`p` in `[0, 1.5]`, scored by both statistics and then **actually trained** at a
fixed FLOP budget with a per-arm tuned learning rate:

| p | `rho` | `gain` | trained loss |
| ---: | ---: | ---: | ---: |
| 0 (SGD) | 1 | 1 | **0.00624** |
| 0.25 | 1.397 | 1.458 | 0.01067 |
| 0.5 | 1.064 | 1.324 | 0.00956 |
| 0.75 | 0.806 | 1.172 | 0.195 |
| 1 | 0.660 | 0.946 | 1.521 |
| 1.25 | 0.563 | 0.574 | 4.029 |
| 1.5 | 0.508 | 0.256 | 11.22 |

Spearman against trained loss (a correct statistic is *negatively* correlated):
`rho` **-0.857**, `gain` **-0.750**.

Both are directionally right and **both are wrong at the top**: each ranks
`p = 0.25` best while training says plain SGD wins. `rho` orders the family
slightly better than `gain` here. Neither is an oracle; both are screens. (The
trajectory-aggregate fix downgraded `p = 0.25` from PASS to INDETERMINATE, so the
gate no longer *claims* it beats SGD — but the ranking error remains.)

### Diagnosis of the MNIST `rho`/`gain` gap

Measured with `scripts/diagnose_gain_depth.py` at depths 5 → 150.

**The ~80x figure was mostly an artifact, and the artifact is exactly G3.** At
`delta=1e-3` `gain` does not converge at any checkpoint — at step 39 it runs
4.08, 0.87, 1.46, 2.19, **78.91**, 13.85, 4.41, 6.17 across the depth ladder. The
78.91 that was reported as `gain`'s ceiling is a single spike at depth 75 in a
sequence that is still oscillating at depth 150. Both statistics fail their
convergence check at `delta=1e-3` at every checkpoint, and the reason is
mechanical: at depth 150 the smallest resolved `|lam|` is 4.3e-3, so
`delta=1e-3` sits *below* the resolved spectrum (`delta/min|node| = 0.23`).

**But a genuine, converged disagreement remains, and it is not small.** At
`delta=1e-2` and `delta=1e-1` both statistics converge to better than 7%:

| step | delta | `rho` | `gain` | both converged? |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1e-2 | 0.170 | 0.975 | yes |
| 0 | 1e-1 | 0.555 | 3.792 | yes |
| 20 | 1e-2 | 0.346 | 5.823 | yes |
| 20 | 1e-1 | 0.420 | 3.327 | yes |
| 39 | 1e-2 | 0.493 | 9.658 | yes |
| 39 | 1e-1 | 0.449 | 4.198 | yes |

So: **`rho` says FAIL (0.17–0.56) and `gain` says PASS (3.3–9.7), both
converged, at every checkpoint where either can be trusted.** The honest summary
is that depth explains the *magnitude* of the reported gap — 80x collapses to
roughly 7–20x — and explains nothing about the disagreement itself.

None of the three constructed mutants explains the residual. G1 and G2 both
require non-stationarity or a frozen subspace, and neither is what DSN's
`(|H| + delta)^-1` does. The calibration study is the only evidence bearing on
which to believe, and it says `rho` orders a preconditioner family slightly
better than `gain` on a synthetic quadratic — which is weak evidence, on a
different problem class, and is **not** a resolution. Stated plainly: the
residual disagreement is unexplained. The composite gate rejects on it, which is
the conservative direction, and that is all that can currently be justified.

---

## Issue 2 — DSN screened at rank 8

Full log: [`autoresearch/logs/rank_restricted.md`](../../../autoresearch/logs/rank_restricted.md).

`KrylovRestrictedPreconditioner` implements what DSN applies:

    P_k = W (|W'HW| + delta)^-1 W'  +  c (I - WW')

`W` is a rank-`k` Krylov basis seeded at the gradient. That is not a model of
DSN's subspace — it *is* DSN's subspace: `KrylovBuilder` seeds Lanczos with the
gradient, and at the shipped `k_max <= 10` nothing ever survives Ritz
re-validation, so it rebuilds from scratch every step (negative-result doc §4:
`m_conv = 0` at 23 of 24 pairs). Verified: at `rank = n` the restriction
reproduces the unrestricted form to 1e-13.

Because a `Preconditioner` is the only thing the gate knows about, **both
statistics and every mutant test applied to the restricted form with no change to
`gate.py`** — which is what the original "generalize the predicate, not the
training loop" split was for.

### Correction: full rank is not an upper bound

The task framed the full-rank verdict as "strictly an upper bound" on what the
rank-8 implementation can achieve. **That is wrong, and the measurement shows it
going the other way.**

`P_k` with `c > 0` is itself a full-rank invertible matrix. It is a *member* of
the class the full-rank screen draws from, not a restriction of it. The full-rank
screen evaluates one particular member, `(|H| + delta)^-1`, and DSN applies a
different member. An upper bound would require maximizing the statistic over the
class, which no gate here does — and which this harness already documents as
vacuous, since `P = I` is always feasible so the supremum over all `P` is always
at least 1.

### Result: the verdict flips, FAIL → PASS

| candidate | scope | verdict | `rho` ceil | `gain` ceil | traj. gain | depth conv. |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| dsn(delta=1e-3) | full-rank | INDETERMINATE | 0.462 | 78.91 | 3.82 | 4.698 |
| dsn(delta=1e-2) | full-rank | **FAIL** | 0.515 | 9.848 | 1.981 | 0.279 |
| dsn(delta=1e-1) | full-rank | **FAIL** | 0.571 | 4.196 | 3.757 | 0.005 |
| dsn-rank8(1e-2, c=1) | rank-8 | INDETERMINATE | 2349 | 10.52 | nan | 0.000 |
| dsn-rank8(1e-1, c=0.1) | rank-8 | **PASS** | 2924 | 371.9 | 330.7 | 0.000 |
| dsn-rank8(1e-1, c=1) | rank-8 | **PASS** | 74.76 | 293.5 | 263.2 | 0.000 |
| dsn-rank8(1e-1, c=10) | rank-8 | **PASS** | 1.412 | 16.32 | 13.0 | 0.000 |

The full-rank form FAILs on `rho`. **DSN's shipped rank-8 operator never FAILs,
and at `delta=0.1` it PASSes on both statistics at every complement scaling.**

So the harness's previously-reported DSN rejection was a statement about an
operator DSN does not apply. The direction is explicable after the fact — `P_k`
with `c = 1` leaves the flat tail of the spectrum alone instead of amplifying it
by `1/delta` — but the point is that it flips at all.

**This does not make DSN good.** Stage 2 already measures it training worse per
FLOP than plain SGD, and the negative-result document stands on three independent
mechanism measurements, none of which this touches. What it means is narrower and
still important: *stage 1's rejection was not evidence about DSN.*

### Do not read the rank-restricted `rho` values

`rho` for the rank-8 operator runs to 10^4–10^5 and falls monotonically as `c`
rises. That is not a discovery, it is **mutant G1 firing**: `rho` contains
`tr(P^-1 Sigma) / tr(P Sigma)`, which diverges as `P` is shrunk on one subspace
relative to another, with no reference to what that subspace costs the objective.
A rank-restricted operator *is* a subspace-damping operator, so it lands squarely
in `rho`'s known blind spot — the same one that scored 33.5 on a configuration
training 2,264x worse than SGD. The `gain` column is the trustworthy half here,
and it also clears the bar.

### Two things that are solid

**No truncation error.** Every rank-restricted row has measured depth convergence
`0.000`. A restricted `P` is an explicit operator, so its three `P`-dependent
forms are computed by applying it, and the only quadrature left (`g'Hg`) is a
degree-one polynomial a Gauss rule integrates exactly at any depth. The full-rank
`delta=1e-3` row, by contrast, has convergence 4.698 — it is not a measurement.

**`nan` trajectory gains are the safeguard working.** They mean the
noisy-quadratic model went unbounded at some checkpoint (`neg_frac` is 0.42 at
step 0), so those rows are INDETERMINATE rather than PASS.

Rank itself is not the binding constraint: the rank sweep at `delta=1e-2` moves
non-monotonically and is driven by whether a checkpoint's model went unbounded.
`delta` and `c` both matter more than `k` on this trajectory.

## What changed in the harness

| change | why |
| --- | --- |
| depth convergence gates verdicts in both directions, per statistic | a non-converged FAIL can be two orders of magnitude wrong (G3) |
| `Cell.spectral_resolution` = `delta / min\|node\|` | names *why* a sweep failed to converge |
| `GateReport.trajectory_gain` = `sum R(P) / sum R(I)` | per-checkpoint ratios do not aggregate; decreases do (G2) |
| `KrylovRestrictedPreconditioner` | screen what is shipped, not its upper bound |
| `subspace_damping_preconditioner`, `power_preconditioner`, `flat_tail_quadratic` | carry the new mutants |
| `rank` / `scope` on every gate report | a verdict about one operator must not read as a verdict about another |
| `dsn_candidate()` screens rank-`k_max` | stage 1 and stage 2 must describe one optimizer |

## What is still not established

- **The converged `rho`/`gain` disagreement on MNIST is unexplained.** ~7–20x,
  both statistics converged, at every usable checkpoint.
- **Neither statistic is an oracle.** Both mis-rank the top of a preconditioner
  family against trained ground truth on the calibration problem.
- **One-step locality is unpatched.** The trajectory aggregate fixes the *roll-up*
  arithmetic, not the fact that each checkpoint's statistic is a one-step model
  evaluated at a sparse, equally-weighted sample of the trajectory.
- **`c` is a scalar stand-in for AdamW.** Swept, not measured. The rank-8 PASS
  at `delta=0.1` holds across `c` in {0.1, 1, 10}, which is reassuring but is not
  the same as modelling AdamW.
- **The rank-8 PASS rests on `gain`,** because `rho` is untrustworthy on exactly
  this operator shape (G1). A PASS resting on one of two statistics, where the
  other is known-blind to the construction, is weaker than a PASS from both.
- All measurements remain CPU-tier: a 76k-parameter MNIST MLP and synthetic
  quadratics.

## Reproduce

    python -m pytest tests/test_autoresearch_gate.py

    python scripts/validate_gain.py              # G1-G3 + calibration
    python scripts/diagnose_gain_depth.py        # the MNIST depth ladder
    python scripts/measure_rank_restricted.py    # DSN at its shipped rank
