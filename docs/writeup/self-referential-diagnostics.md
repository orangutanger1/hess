# Self-referential diagnostics in approximate-curvature methods

**Date:** 2026-07-29
**Status:** Methodological finding, fully measured. Independent of any
particular optimizer.

## The claim

A metric or gate computed from the same approximation it is meant to evaluate
cannot detect that approximation's failure. Worse than being merely
underpowered, it is typically **anti-correlated with the truth**: it *improves*
as the approximation degrades, because degradation and optimism have the same
cause.

This is not a subtle statistical effect. It is arithmetic, it is reproducible on
demand, and every one of the six instances below was found *after* someone had
written a gate specifically intended to catch the failure it missed.

The setting is approximate-curvature and subspace-recycling optimizers —
Hessian-free / truncated Newton, GCRO-DR-style Krylov recycling, K-FAC-family
and any method that reuses spectral information across steps. The instances
below come from one project, but nothing in them is specific to it: the
ingredients are a low-rank or stale approximation `Ĥ` of a true operator `H`,
and a diagnostic that happens to be cheap because it is built out of `Ĥ`.

**Cheapness is the tell.** A diagnostic is cheap exactly when it reuses
quantities the approximation already computed. That is the same property that
makes it blind.

---

## Six instances

### 1. A residual measured against a stale projection — 85x optimism, wrong sign

A recycled-Krylov builder retained a basis `U` across steps together with its
curvature images `HU`. The projected curvature `T = W'HW` was assembled from the
Lanczos tridiagonal plus the *carried* `HU` block, and the reported relative
residual `rel_residual` was computed against that `T`.

`HU` came from the previous iterate's Hessian. So `T` was not `W'HW` at the
current point, and the residual was measured against the very object that
staleness corrupts.

Measured, on an ill-conditioned logistic problem, 40 steps, at equal basis width:

| run | mean reported `rel_residual` | drift (tail/head) | reported at final step | **true** residual at final step |
| --- | ---: | ---: | ---: | ---: |
| recycled, healthy (`m_recycle=5, k_max=10`) | 0.1010 | 4.84 | 0.6295 | 0.6897 |
| fresh, equal width (`m_recycle=0, k_max=15`) | 0.0644 | 0.45 | 0.0178 | 0.0178 |
| recycled, **frozen after step 5** (maximal staleness) | **0.0263** | **0.47** | 0.0114 | **0.9752** |

The frozen-basis variant — the exact failure mode the project's Risk 1 was
written to catch — reports the *best* residual of the three. Its reported value
understates its true value by **85x**. The fresh arm's reported and true values
agree exactly, which is what validates the measurement: the entire gap belongs
to the recycled arm, not to measurement noise.

At `cond=1000` the inversion is larger still: reported says the recycled arm is
2x better; a true dense residual says it is 36,000x worse.

The mechanism is one line: `rel_residual` was computed against the same `T` that
staleness corrupts, so a more corrupted `T` produced a smaller reported residual.

### 2. Two convergence gates with zero power against the risk they existed for

Two `pytest` gates were written against Risk 1 (staleness): one bounding the
reported residual of the recycled arm against a fresh arm, one bounding
head-to-tail drift of the reported residual across a run.

Using the table above: the maximally-stale frozen run passes the drift gate
comfortably (0.47 against a bound of 10) **and** passes the residual gate
(0.0263 against a bound of 0.0322) — the gate that the *correct* implementation
fails. The stale variant scored better on both gates than the shipped one.

Two named risks, two gates written specifically to catch them, zero power
against the failure they existed for. The gates were not weak; they were built
from the corrupted quantity, so no amount of tightening could have helped. A
tighter bound on an optimistic number is a tighter bound on nothing.

### 3. A payoff statistic in which the operator does not appear

A generalized payoff gate for a preconditioner `P` was written as

    rho(P) = [ tr(P^-1 Sigma) / tr(P Sigma) ] * (g'Pg)^2 / ||g||^4

with two attractive properties: invariant under `P -> cP`, and exactly 1 at
`P = I`, so the threshold is "does this beat plain SGD" rather than a tuned
constant.

**`H` does not appear anywhere in it.** It enters only through the *definition*
of `P`. So on a synthetic quadratic where `g` and `Sigma` are held fixed while
`H` drifts, `rho` of a frozen `P = H_0^-1` is numerically *identical* before and
after the drift:

- frozen `P = H_0^-1`, `rho` at `t = 0`: **11.0837**
- the same `P`, `rho` at `t = T` after the drift: **11.0837** (identical to
  machine precision)
- the correctly refreshed `P = H_T^-1` at `t = T`: `rho = 1.004e6`

This is not "underpowered against staleness at this sample size". It is
structurally incapable of detecting staleness at any sample size. A statistic
that does not contain the operator cannot report that the operator moved.

The second-order statistic added to cover this — an expected one-step loss
decrease under a noisy quadratic model, `gain(P) = R(P)/R(I)` with
`R(P) = (g'Pg)^2 / (g'PHPg + tr(PHP Sigma))` — scores the same frozen `P` at
**0.01119**. It contains `H`, so it can see the drift.

### 4. A screen that rewards uninformative spread

The same `rho`, swept over a random diagonal `P` drawn from a seed and never
shown `H`, `g` or `Sigma`. 24 draws per row.

| log spread | cond(P) | `rho` median | `rho > 1` | `gain` median | `gain > 1` | composite PASS | **`rho`-only PASS** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.65 | 1.029 | 67% | 1.035 | 58% | 8% | 21% |
| 0.50 | 2.72 | 1.097 | 75% | 1.075 | 58% | 12% | 38% |
| 1.00 | 7.39 | 1.41 | 88% | 1.148 | 54% | 8% | 46% |
| 2.00 | 54.6 | 3.642 | 100% | 0.979 | 46% | **0%** | **79%** |

A preconditioner that by construction carries no curvature information passes
the `rho`-only screen **79% of the time** at the widest spread, and the two
statistics move in *opposite* directions as spread grows.

The mechanism is arithmetic, not sampling: `rho` contains `(g'Pg)^2 / ||g||^4`,
which grows quadratically in `P`'s action on the gradient's support, while the
compensating trace ratio recovers only part of it — an arithmetic-versus-harmonic
asymmetry. `gain` is not fooled because scaling directions up without regard to
their curvature raises `g'PHPg` and `tr(PHP Sigma)` in the denominator.

Same root cause as instance 3, entirely different mechanism: `H` does not appear
in `rho`.

### 5. The blind spot, with trained ground truth attached

Instances 3 and 4 are synthetic arguments. This one is the same defect measured
against actual training.

Mutant construction: damp `P` on a rank-`r` subspace `U` by a factor `epsilon`,
i.e. refuse to move there. `tr(PHP Sigma)` is quadratic in `P`, so damping
drives the noise term to zero quadratically while the numerator loses only the
gradient mass that subspace carries. Nine configurations, each scored by both
statistics and then **actually trained** at a fixed FLOP budget with a tuned
learning rate. Plain SGD reaches 0.0064744.

| rank(U) | epsilon | `rho` ceil | `gain` ceil | trained loss | vs SGD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.1 | 0.774 | 0.498 | 0.17894 | 28x worse |
| 4 | 0.01 | 3.683 | 0.236 | 6.1065 | 943x |
| 4 | 0.001 | **33.47** | 0.213 | 14.657 | **2264x** |
| 8 | 0.1 | 0.622 | 0.578 | 0.23986 | 37x |
| 8 | 0.01 | 1.260 | 0.121 | 8.8001 | 1359x |
| 8 | 0.001 | 9.243 | 0.086 | 43.401 | 6703x |
| 12 | 0.1 | 0.876 | 1.018 | 0.30318 | 47x |
| 12 | 0.01 | 0.813 | 0.222 | 15.011 | 2319x |
| 12 | 0.001 | 2.839 | 0.066 | 74.018 | **11433x** |

All nine train worse than SGD. `rho` is fooled by **5 of 9**, scoring 33.47 on a
configuration that trains 2,264x worse than SGD. `gain` is fooled by **0**.

A rank-restricted operator *is* a subspace-damping operator, which is why `rho`
values for a rank-8 Krylov-restricted preconditioner run to 10^4–10^5 on a real
trajectory and fall monotonically as the complement scaling rises. That is not a
discovery about the operator; it is this blind spot firing on a real candidate.
Any verdict resting on `rho` for a rank-restricted `P` should be discarded.

**Why `gain` survives here, and the exact boundary.** On a *stationary*
quadratic the two things the mutant plays off against each other are the same
thing: loss still recoverable in direction `i` is `c_i^2 / (2 lam_i)`, so a
direction carrying no gradient carries no remaining loss and freezing it is
genuinely free; and a direction carrying enough `lam*s` for damping to help the
noise term also carries `g` mass, so damping it costs the numerator. `gain`
cannot be fooled here because on a stationary quadratic there is nothing to be
fooled about. Freezing a subspace is harmful only when the gradient *later*
acquires mass in it — which neither statistic can see, because neither is given
a second point in time. That is the next instance.

### 6. A per-checkpoint maximum standing in for a trajectory sum — 171x, unsafe direction

The gate's roll-up rule was "PASS if any checkpoint passes". A preconditioner
tuned to one curvature regime, screened along a trajectory that alternates
between two:

| step | phase | `rho` | `gain` |
| ---: | --- | ---: | ---: |
| 0 | A (sharp overall, `P` excellent) | 11.08 | 10.84 |
| 1 | B (flat overall, `P` terrible) | 11.08 | 0.0606 |
| 2 | A | 11.08 | 10.84 |
| 3 | B | 11.08 | 0.0606 |

Max-over-checkpoints returns **PASS**, on the strength of the phase-A rows.

The correct aggregate is **0.0636**, a factor of **171x** in the unsafe
direction. The reason is dimensional, not statistical: `R` is an expected loss
decrease *per step*. Decreases add across steps; ratios do not. The quantity a
trajectory accumulates is `sum R(P) / sum R(I)`, not `max R(P)/R(I)`.

The construction is what makes them diverge: phase A has a small available
per-step decrease and an excellent ratio; phase B has a large available decrease
and a terrible ratio. A max over ratios sees only phase A; a ratio of sums is
dominated by phase B.

This one is not literally self-reference, and it is included because it is what
self-reference degrades into once you patch the first-order version: the gate no
longer reads a corrupted quantity, but it still aggregates a *local* quantity by
a rule that discards the information distinguishing good from bad. Both this and
instance 5 are the same underlying defect — **one-step locality** — seen twice.

---

## What the pattern is, precisely

Let `Ĥ` approximate `H`, and let a diagnostic `D(Ĥ)` be intended to report
`||Ĥ − H||` or something monotone in it.

If `D` is computed *from* `Ĥ` — its residual, its projection, its Ritz values,
its telemetry — then `D` is a function of `Ĥ` alone. It has access to no
information about `H`, so:

1. **It cannot detect a change in `H` that leaves `Ĥ` fixed.** Instance 3 is the
   pure case: `rho` is provably constant before and after arbitrary drift.
2. **It is generally anti-correlated with the truth**, because the same
   degeneration that makes `Ĥ` a worse approximation makes the internal residual
   look smaller. Instance 1 is the pure case: 85x optimism, with the sign
   flipped.
3. **Tightening the threshold does not help.** Instance 2: the stale variant
   passes with more margin than the correct one, so any bound that admits the
   correct implementation also admits the stale one.
4. **Adding a second statistic patches the specific hole and inherits none of
   the first one's hard-won safeguards.** In this project, `gain` was added
   specifically to cover `rho`'s H-blindness, and shipped without the Lanczos
   depth-convergence check that had already been found necessary for `rho`. It
   turned out to converge in depth *strictly more slowly* than `rho` — further
   from its converged value in **13 of 13** unconverged cells — because for
   `P = f(H)^-1` the gate integrates `1/f(lam)` for `rho` (peak `1/delta` at
   `lam=0`) and `lam/f(lam)^2` for `gain` (peak `1/(4 delta)` at `lam=delta`, a
   sharper near-singular feature), and a Gauss rule can only fit polynomials.
   The un-safeguarded reading was wrong by up to 80x. New statistic, same class
   of failure, one layer down.

---

## Countermeasures that were measured to work

Not proposals. Each of these was built, and each caught something.

**Measure against the true object, and pay for it.** The residual was rebuilt as
`||HWy + g||` from curvature images retained under the *current* operator, with
**no quantity crossing a step boundary**. That makes instance 1's failure class
structurally impossible rather than merely absent. It is not free — it costs the
curvature products the stale shortcut was stealing — and paying for it is what
revealed that the recycling saving it was funding never existed.

**Require the diagnostic to contain the object.** `gain` contains `H`; `rho`
does not. That single property is what separates instances 3, 4 and 5. When a
composite gate requires *both* to pass, `gain` covers `rho`'s blind spot in all
three, and the report logs what the verdict *would* have been on `rho` alone —
so the blindness stays a measured quantity in every run rather than a claim in a
document.

**Cross-fit anything estimated from samples.** The noise covariance `Sigma` is
estimated from microbatches drawn from a pool disjoint from the batch defining
`H` and `g`, and cross-fit by halves so only cross-half differences are used:
`u_i = (g_{A_i} − g_{B_i})/sqrt(2)`, giving `E[u_i u_i'] = Sigma` exactly. No
mean is ever computed from the samples it is then evaluated against.

**Mutant-validate the gate against constructions where truth is known, and
refuse to run without it.** Nine mutants including a maximally-stale frozen
basis, an exact optimum, an identity control, and a favorable-shape-but-wrong
answer. The driver script *exits* rather than screening candidates if the
recorded separation log is missing or failed: a gate that has not been shown to
separate a known-good from a known-bad preconditioner has no demonstrated power,
and allocating a training budget with it spends that budget on its noise.

**Attach trained ground truth to at least one mutant.** Instances 3 and 4 are
synthetic arguments about a statistic. Instance 5 is the same argument with an
actually-trained loss beside it, and it is the only one of the three that
settles anything. Synthetic mutants tell you a statistic *can* be fooled; a
trained arm tells you it *is*.

**Check the dimensional consistency of every aggregation.** Instance 6 was a
per-step ratio aggregated by max. Ask what the quantity is per unit of, and
whether that unit adds.

**Gate verdicts on convergence of the diagnostic in both directions.** A
statistic still moving at the deepest quadrature rung cannot support a PASS *or*
a FAIL. The earlier safeguard only appended a warning to the reason string, so a
sweep sitting entirely below the bar still returned a confident FAIL — and on a
flat-tailed spectrum the same statistic runs 9.2 at depth 5 and 1104 at depth
200, so such a FAIL can be two orders of magnitude wrong.

---

## Scope

All of the above was measured on CPU: a 76k-parameter MNIST MLP, synthetic
quadratics of dimension 24–400, and ill-conditioned logistic problems. The
mechanisms are arithmetic — a residual against a stale projection, a statistic
in which `H` does not appear, a per-step ratio aggregated by max — and none of
them depends on problem scale. That the *magnitudes* (85x, 171x, 79%, 2264x) are
specific to these problems is not in dispute; the sign and the mechanism are not.

The finding is stated as: **if you are building an approximate-curvature or
subspace-recycling method, assume every cheap diagnostic you have is blind to
the failure it was written for until you have validated it against a construction
whose answer you know independently.** In this project, six such diagnostics were
built with care and all six were blind.

---

## Reproduce

    python scripts/validate_gate.py       # 9 mutants; exits 1 if they do not separate
    python scripts/validate_gain.py       # instances 5, 6 and the depth result
    python scripts/diagnose_gain_depth.py

    python -m pytest tests/test_krylov_builder.py \
                     tests/test_convergence.py \
                     tests/test_autoresearch_gate.py

Logs: `autoresearch/logs/gate_mutants.md`, `autoresearch/logs/gain_mutants.md`,
`autoresearch/logs/rank_restricted.md`.

The instance-1 and instance-2 tables are reproduced in
`docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md`; the fixes are in
`docs/superpowers/plans/2026-07-28-dsn-plan2-fixes.md`. Instance 1's failure
class is now pinned by
`test_krylov_builder.py::test_a_frozen_recycled_basis_cannot_hide_behind_the_reported_residual`,
which freezes the basis and asserts the reported residual tracks an
independently computed dense one anyway.
