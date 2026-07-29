# Krylov recycling in a trust-region Newton setting

**Date:** 2026-07-29
**Status:** Numerical-linear-algebra finding, measured. Portable to any method
that reuses Krylov information across a sequence of linear systems inside an
optimizer.

Krylov recycling (GCRO-DR, GMRES-DR and relatives) is standard practice for
sequences of slowly-changing linear systems and is essentially unused inside
neural-network training. The obvious motivation is that curvature eigenvectors
are correlated between adjacent optimizer steps, so a basis built at step `t`
should be worth something at step `t+1`.

This document reports two results about doing that inside a **trust-region
Newton** solver:

1. **Deflation-style recycling is break-even by an exact accounting**, not by
   measurement. Refreshing is exactly as expensive as rebuilding. Not refreshing
   is measurably much worse. Both directions are closed.
2. **The augmented (free-augmented-Krylov, "FACA") escape route does not open**
   under a saddle-free damped solve — and the reason is not the obvious one
   (mini-batch noise in the secant pair, worth about 2x). It is that the residual
   the break-even inequality assumes contracts *does not contract*, because the
   solver is deliberately not solving the system that residual belongs to.

Result 1 is arithmetic and transfers unconditionally. Result 2 is measured, and
its scope condition is precise and checkable: it applies to any solver whose
target system differs from the one the recycled quantities' error is measured
against.

---

## 1. Setting

Per optimizer step, solve for a step `d` inside a Krylov subspace built from the
current gradient. The concrete instance measured here builds
`K_k(H, g) = span{g, Hg, ..., H^{k-1} g}` by Lanczos seeded at `g`, deflated
against a retained basis `U`, with full reorthogonalization; assembles
`T = W'HW` with `W = [U, Q]`; and takes the **saddle-free damped** step

    y = -V diag(1 / (|lam_i| + delta)) V' W' g,      d = W y

with `T = V Λ V'`. Negative curvature is handled by absolute value rather than
being clipped away. The step is then clipped to a trust radius and accepted or
undone on an actual-versus-predicted reduction ratio.

The two facts about this setting that drive everything below:

- **The operator is genuinely indefinite.** The `g^2`-weighted mass on negative
  curvature is 0.404 at initialization on the measured problem and 0.01–0.04
  thereafter. This is why the saddle-free absolute value is there.
- **Therefore the solver solves `(|H| + delta) d = -g`, not `H d = -g`.** Every
  quantity that reasons about "the Newton residual `||H d + g||`" is reasoning
  about a system this solver is not solving.

The operator also *drifts*: `H` at step `t+1` is neither `H` at step `t` nor a
small perturbation of it under mini-batching. Drift is what recycling is
supposed to survive.

---

## 2. Result 1 — deflation recycling is break-even by construction

### 2.1 The accounting

Let the builder retain `m` directions `U` from step `t` and build `k` new Lanczos
directions at step `t+1`, for a total width `m + k`.

For `T = W'HW` to be exact at the current iterate — which it must be, or every
downstream quantity including the residual is measuring a different operator (see
[`self-referential-diagnostics.md`](self-referential-diagnostics.md)) — the
carried block needs `H_{t+1} U`. That is **exactly `m` curvature products.**

Those `m` products buy `m` directions of subspace width. Spending the same `m`
products on the Lanczos recurrence instead would buy `m` *new* Krylov directions
at the current operator.

    cost(recycle m) = m products  ->  m carried directions
    cost(build   m) = m products  ->  m fresh directions

**Exact recycling is break-even by construction.** There is no regime, no
conditioning, no drift rate at which the arithmetic changes. Recycling can only
win if the carried directions are *better* than fresh ones of the same count —
and for a Krylov method seeded at the gradient, fresh directions are the optimal
ones reachable from `g` under the residual criterion the method is minimizing, so
"better" has to mean "better than optimal for this criterion" or "reaching
outside the reachable space". Deflation does neither.

Measured, at equal solve tolerance on drifting SPD operators (n = 40/60/80,
cond 1e2–1e4, 3 seeds each): recycled/fresh curvature-product ratio
**1.03–1.11**. Slightly worse than break-even, consistent with the arithmetic
plus bookkeeping.

### 2.2 Not refreshing is not an escape, and the trap is easy to fall into

The obvious dodge is to skip the `m` refresh products and reuse the carried
`H_t U` as if it were `H_{t+1} U`. This appears to work, and the appearance is an
artifact.

In the implementation measured here, the stale shortcut produced a residual that
was optimistic by up to 85x, which tripped the `tau` stopping rule early. The
"saving" was the solver stopping before it had converged, reported as if it had.
Once the residual was made honest, the saving vanished and the ratio settled at
the 1.03–1.11 above.

### 2.3 Which directions may be deflated: the convergence gate

A second, independent constraint. Deflating a direction only shrinks the
effective spectrum if that direction is close to invariant. Ranking heuristics
that select by *usefulness to the step* (largest `|y_i|` contribution, smallest
`|lam|`) select on a different property, and deflating on them steals width from
the new Krylov vectors without shrinking anything.

Measured: with a contribution-based ranking and no convergence gate, the recycled
subspace came out **20–44x worse** than an equal-width fresh basis, and the
measured ordering of three ranking heuristics *reversed* once the residual was
made honest. Relative Ritz residuals of the directions the contribution ranking
retained were 0.1 to 2.8 — not converged by any margin. Largest-`|lam|` ranking
scored 1e-15 to 1e-6, which is the entire reason it measured better: Lanczos
converges to extremal eigenvalues first, so largest-`|lam|` Ritz pairs are the
converged ones.

The fix is the standard one (Saad, *Numerical Methods for Large Eigenvalue
Problems*, ch. 6; the same test governs which harmonic Ritz vectors GMRES-DR and
GCRO-DR retain): admit a direction only if `||H u − θu|| ≤ ritz_tol * |θ|`,
re-tested against the **current** operator each time it is reused. Calibration by
direct measurement of recycled/fresh residual ratio at equal width
(n=40, cond=100, k_max=20, m=8, 3 seeds):

| `ritz_tol` | 1e-6 | **1e-3** | 1e-2 | 5e-2 | 1e-1 | ∞ (no gate) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| recycled/fresh residual ratio | 1.2–2.3 | **0.90–1.21** | 1.23–1.42 | 6.1–13.4 | 11.2–32.5 | 18.5–43.9 |

1e-3 sits at parity with roughly 50x margin to the harmful regime, and the
converged/unconverged populations it separates differ by five orders of
magnitude, so it is not a knife edge. Looser is harmful; tighter over-drops and
narrows the basis.

### 2.4 The consequence for small-`k` regimes

With the gate in place, **nothing qualifies at small `k_max`.** On everything
measured — including a 76k-parameter MNIST MLP — no Ritz pair converges at
`k_max ≤ 10`, so recycling declines to engage and the solver rebuilds from
scratch every step. On a real trajectory, a direction that passed the Ritz test
at step `t` still passed against step `t+1`'s operator at **1 of 24** measured
pairs.

That is the correct behaviour — the alternative is a 20–44x worse subspace — but
it means recycling is *inert* in exactly the operating regime that motivates
subspace methods in the first place, where `k` is small because curvature
products are the dominant cost.

**Portable statement:** deflation-based Krylov recycling inside an optimizer is
break-even at best, is worse than a fresh basis unless a Ritz-convergence gate is
enforced, and the gate that makes it safe also makes it inactive at small basis
width. Any reported saving should be checked for a stopping rule reading an
optimistic residual.

---

## 3. Result 2 — the augmented-Krylov escape and why it does not open

### 3.1 Why augmentation is the right escape route in principle

Deflation removes directions from the Krylov iteration and can only be justified
per-direction. **Augmentation** constructs `W` so that
`span(W) ⊇ K_k(H, g)` *by construction* — the fresh Krylov space is contained in
the recycled one. That makes "never worse than a fresh basis of the same `k`" a
theorem rather than a measurement (Saad, "Analysis of augmented Krylov subspace
methods", SIAM J. Matrix Anal. Appl., 1997; GCRO and GCRO-DR are in this family).

Result 1 does not apply to augmentation, because augmentation is not claiming to
substitute for fresh directions — it is claiming that the carried directions come
*free* relative to a fixed Krylov budget, and that the extra dimensions they add
are worth more than what maintaining them costs. That claim is exactly what a
break-even inequality has to decide.

### 3.2 The break-even inequality

The inequality under test, taken as given from the source excerpt (it is applied
and measured here, not re-derived):

    |Delta_H|  <  (m_eff - 1) * (1 - rho_K) * epsilon_K(k) / ||y_U||

Reading each term, because what the measurement finds is that the terms are not
independent:

| term | meaning | measured how |
| --- | --- | --- |
| `\|Delta_H\|` | how much the operator moved between the two steps, in curvature units: `\|\|w\|\|/\|\|s\|\|` with `w = 2(y_t − H_t s)` — the cost of reusing anything | one extra curvature product per pair, free `s` and `y_t` from the trajectory |
| `m_eff` | how many *genuinely new* dimensions the carried basis contributes: `sum_i \|\|(I − QQ')u_i\|\|^2`, `Q` an undeflated fresh Krylov basis at `t+1` | projection, no extra products |
| `rho_K` | the Krylov residual contraction rate — the inequality assumes `< 1` | true residual `\|\|H(Wy)+g\|\|/\|\|g\|\|` per Lanczos depth |
| `epsilon_K(k)` | absolute residual level the fresh method reaches at depth `k` | same sweep |
| `\|\|y_U\|\|` | how much of the solution the recycled block carries | recycled block of the subspace solution |

Units: `|Delta_H|` is compared rather than the second-difference `delta_hat`
because `epsilon_K(k)/||y_U||` is gradient-units over length-units, i.e.
curvature.

The right-hand side is what reuse buys; the left is what drift costs. Note the
structure: `(m_eff − 1)` is negative when the carried basis contributes less than
one new dimension, and `(1 − rho_K)` is negative when the residual grows. Both
negative gives a positive product and a comparison that "passes" while describing
nothing. Those rows must be flagged as **vacuous**, not counted as open.

### 3.3 Measurement

`scripts/measure_faca_gate.py`. Six consecutive-step pairs `(t, t+1)` per run,
four runs (two seeds, mini-batch and fixed-batch), on a 784-96-tanh-10 MLP,
batch 256, `lr=1e-2`, MNIST. Two secant pairs at every `t`, differing only in the
batch the second gradient is evaluated on — `diff` (what consecutive steps
actually see) and `same` (a common curvature batch, the excerpt's proposed fix) —
differenced against the same `g_t` and the same `H_t s`, so the gap between them
is batch resampling and nothing else. `||y_U||` and `m_eff` are computed with
Ritz re-validation forced **off**, which is the configuration most favourable to
the method (as shipped it fires and drops the basis, so these quantities would
not exist at all).

**The gate is closed or vacuous at every pair, in every configuration, at both
seeds.**

#### Batching is a 2x term, not the killer

`k_max=40`, `m_recycle=8`, seed 0, mini-batch, `|Delta_H|` units:

| t | \|s\| | \|dH\| diff | \|dH\| same | sigma_sec | sigma_Hs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.0 | 0.750 | 0.384 | 0.467 | 0.089 |
| 10 | 4.0 | 0.770 | 0.420 | 0.652 | 0.182 |
| 20 | 4.0 | 0.736 | 0.441 | 0.794 | 0.276 |
| 30 | 2.0 | 2.370 | 0.981 | 1.514 | 0.358 |
| 40 | 2.0 | 1.335 | 0.772 | 1.262 | 0.296 |
| 50 | 2.0 | 1.810 | 1.825 | 1.428 | 0.523 |

Moving the secant onto a common batch removes 40–60% of `|Delta_H|` — not orders
of magnitude. What remains is `L_H*||s||` with `L_H` measured at 0.11–0.49,
stable across step lengths of 1.4–4.0, i.e. behaving like a genuine Lipschitz
constant, and it alone exceeds every right-hand side the same run produces by
**4–90x**.

A measurement note worth carrying: the two candidate readings of "batch noise"
differ by ~5x. `sigma_Hs` (spread of the curvature action `H_B s`, 0.09–0.52) is
the curvature-operator noise, but the term that actually contaminates a
cross-batch secant is `sigma_sec` (0.47–1.53), because `y_t = g_{t+1} − g_t`
picks up the full gradient difference between two batches, which does **not**
shrink with `||s||`. The larger one is the relevant one.

#### Kill 1 — the residual is not a contraction

| t | m_eff | rho_K (last pair) | rho_geo | contracting | eps_k | \|y_U\| | rhs (last-pair) | rhs (geo) | open? |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 2.52 | 0.886 | 0.993 | 54% | 0.874 | 0.760 | 0.199 | 0.0115 | closed |
| 10 | 3.15 | 1.784 | 1.007 | 49% | 1.310 | 0.176 | -12.55 | -0.116 | vacuous |
| 20 | 3.88 | 0.555 | 0.986 | 51% | 0.952 | 0.387 | 3.148 | 0.100 | closed |
| 30 | 4.09 | 0.058 | 0.994 | 46% | 0.785 | 0.165 | 13.87 | 0.088 | closed |
| 40 | 3.09 | 0.352 | 0.986 | 44% | 0.634 | 0.331 | 2.596 | 0.056 | closed |
| 50 | 4.37 | 2.423 | 1.014 | 46% | 2.357 | 0.528 | -21.39 | -0.209 | vacuous |

Geometric-mean rate over the residual history is **0.978–1.014** with only
**41–57%** of consecutive directions contracting at all. `(1 − rho_K)` is
therefore ~0.01 rather than O(1), which crushes the right-hand side by about two
orders of magnitude. Seed 1 reproduces it: `rho_geo` 0.990–1.059, closed or
vacuous at all six pairs.

**This is structural, and the scope condition is exact.** `subspace_newton`
solves the saddle-free damped system `(|H| + delta) d = -g`. The quantity
`||H d + g||` is the residual of a *different* system — one the solver is
deliberately not solving. It has no reason to decrease in `k`, and it doesn't.
The stopping tolerance `tau = 0.1` was never reached; every pair in every
configuration ran to `k_max`.

The trap this creates: **a single adjacent ratio reports open/closed at random.**
Last-pair `rho_K` says open at 3 of 6 pairs; the sequence-level rate says 0 of 6
on identical data. The residual history is a random walk with O(1) excursions —
e.g. at `t=1`: `1.566, 3.496, 21.390, 1.493, 1.272, 4.050, ..., 1.366, 1.210` —
and whether the last two directions happen to fall on the descending side of an
excursion is what decides a single-pair verdict.

#### Kill 2 — the two sides are driven by the same quantity

Fixed-batch run (`--fixed-batch`, `k_max=40`), which removes batch noise from the
secant entirely so `diff` and `same` coincide by construction:

| t | \|s\| | \|dH\| | m_eff | eps_k | rhs (geo) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.0 | 3.7e-1 | 1.10 | 8.8e-1 | 2.6e-3 |
| 10 | 0.769 | 8.8e-4 | 5.5e-2 | 2.3e-4 | -7.7e-5 |
| 20 | 0.269 | 3.3e-6 | 1.5e-3 | 5.7e-5 | 1.7e-6 |
| 30 | 0.108 | 1.5e-6 | 2.5e-4 | 3.1e-5 | 4.9e-7 |
| 40 | 0.046 | 2.3e-6 | 5.5e-5 | 2.2e-5 | 1.8e-7 |
| 50 | 0.023 | 4.7e-6 | 1.7e-5 | 1.6e-5 | 5.0e-8 |

Drift falls by **five orders of magnitude** and the gate is still never open.

This is the more general result and it is independent of kill 1. As the iterate
settles, the carried basis comes to lie *inside* the fresh Krylov space —
`m_eff` collapses to 1.7e-5 of 8 carried directions — so reuse contributes no new
dimensions and `(m_eff − 1)` goes negative; `epsilon_K` collapses at the same
time, so the right-hand side falls faster than `|Delta_H|` does.

**New information and drift are anti-correlated.** Where drift is small there is
nothing to reuse, because the previous basis is already reachable from the
current gradient. Where reuse is worth something (`m_eff` 2.5–4.4) the drift is
large. Removing batch noise removes numerator and denominator together, which is
why the obvious fix cannot work: it is not attacking an independent nuisance
term.

#### Kill 3 — the shipped hyperparameter makes the question unaskable

At `k_max = 8`, the configuration actually shipped, at most one direction is ever
carried, `m_eff` maxes at 0.40, and `(m_eff − 1) < 0` at every pair
unconditionally. The gate is dead for a reason that has nothing to do with the
method. The `k_max = 40` runs are the ones kills 1 and 2 rest on, precisely
because the shipped configuration cannot express the question. Reporting a
closed gate at `k_max = 8` as evidence about the method would be a category
error.

---

## 4. What transfers, and what does not

**Transfers unconditionally:**

- The break-even accounting for deflation recycling (§2.1). It is arithmetic.
- The requirement that only converged Ritz pairs be deflated, re-tested against
  the current operator (§2.3), and the consequence that this makes recycling
  inactive at small basis width (§2.4).
- The measurement discipline in §3.2–§3.3: compare sequence-level contraction
  rates rather than adjacent ratios; flag double-negative "passes" as vacuous
  rather than open; run a zero-drift control to check whether the two sides of a
  break-even inequality are independent.
- The anti-correlation between `m_eff` and drift (§3.3, kill 2). It follows from
  what a Krylov space is, not from anything about neural networks: a basis
  carried from a nearby iterate is nearly reachable from the current gradient,
  and one carried from a distant iterate is stale. There is no configuration
  where it is both new and current.

**Does not transfer, and the scope condition is checkable:**

- Kill 1 is specific to solvers whose target system differs from the one the
  recycled quantities' error is measured against. Here that is the saddle-free
  absolute value: the solver targets `(|H| + delta) d = -g` while the gate reasons
  about `||H d + g||`. **A method built on an actually-contracting Krylov residual
  — an SPD or Gauss-Newton operator with the residual measured against the system
  being solved — is untested here, and this document is not evidence against it.**
  Before building on FACA in such a setting, measure `rho_geo` and `m_eff` on the
  target trajectory first; both are cheap and both are prerequisites.
- All measurements are CPU-tier: a 76k-parameter MLP on MNIST plus synthetic SPD
  operators of dimension 40–80. Nothing here has been run at scale.

---

## 5. Reproduce

    python scripts/measure_faca_gate.py --steps 60 --k-max 8                     # shipped config: gate dead for the trivial reason
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8      # recycling engages
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --fixed-batch
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --seed 1

    python -m pytest tests/test_krylov_builder.py

Measurement only: the script writes nothing into `src/` and contains no solver
code — it reuses `lanczos_iter` and the shipped builder.

Full measurement record:
`docs/superpowers/plans/2026-07-28-faca-gate-measurement.md`. The deflation
accounting and the `ritz_tol` calibration:
`docs/superpowers/plans/2026-07-28-dsn-plan2-fixes.md` (F8, F10).

### References

- Parks, de Sturler et al., *Recycling Krylov subspaces for sequences of linear
  systems*, SIAM J. Sci. Comput. 28(5), 2006 (GCRO-DR)
- Saad, *Analysis of augmented Krylov subspace methods*, SIMAX 1997
- Saad, *Numerical Methods for Large Eigenvalue Problems*, ch. 6 (Ritz
  convergence)
- Morgan, GMRES-DR
- Dauphin et al. 2014, *Identifying and attacking the saddle point problem*
  (the saddle-free absolute value that creates kill 1's scope condition)
