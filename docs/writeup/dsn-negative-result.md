# DSN does not beat AdamW

**Date:** 2026-07-29
**Scope:** CPU tier — a 784-96-tanh-10 MLP (76k parameters) on MNIST, plus
synthetic quadratics and streams. All numerics in float64.
**Verdict:** negative. The mechanism was instrumented at four independent
points and returns nothing at any of them.

Dynamic Subspace Newton (DSN) takes a saddle-free damped Newton step inside a
recycled Krylov subspace `S` and an AdamW step in `S⊥`. Its three novelty
claims were cross-step subspace recycling, a first-order complement, and an
adaptive rank chosen by marginal value.

**As designed and as shipped, DSN does not beat AdamW.** It converges. It is
not buggy — all ten defects found in the fix round are fixed and the suite is
155 passed / 0 xfail. It loses anyway, and the reasons are structural rather
than incidental.

| question | instrument | answer |
| --- | --- | --- |
| Does it win end to end? | `scripts/sanity_mnist.py` | No. MNIST 0.27–0.30 vs AdamW 0.15–0.18. |
| Does curvature preconditioning pay on this trajectory at all? | `scripts/measure_rho.py` | No. The payoff gate caps at 0.717 against a threshold of 1, at every checkpoint. |
| Does cross-step recycling save curvature products? | Plan 2 measurement | No. Break-even by construction; measured 1.03–1.11x a fresh rebuild. |
| Does the augmented-recycling break-even gate open? | `scripts/measure_faca_gate.py` | No. Closed or vacuous at every pair, both seeds, every configuration. |

One thing this document does **not** claim: the pre-registered kill criterion
has not been evaluated. See §6.

---

## 1. Evidence 1 — the end-to-end comparison

The outcome the mechanism findings predict, measured after the fix round rather
than before it, so that the shortfall cannot be attributed to a defect.

| problem | DSN | AdamW |
| --- | --- | --- |
| MNIST MLP, last-10-step mean loss, 3 seeds | 0.2678 / 0.3010 / 0.2805 | 0.1512 / 0.1781 / 0.1713 |
| synthetic stream (batch 32, dim 25, cond 100), 5 seeds | 0.29 – 1.75 | 0.16 – 0.59 |

Configuration: 200 steps, batch 256, `lr=1e-2`,
`KrylovBuilder(k_max=8, m_recycle=4, tau=0.1)`.

The acceptance ratio is valid by construction (same-batch), the trust region is
stable (radius 2.0, `n_shrink` 5–7 of 200), the residual telemetry is a true
residual. DSN converges and loses.

There is a separate FLOP-denominated comparison against plain SGD in §5, which
matters because it is independent of every gate result in this document.

---

## 2. Evidence 2 — the payoff gate never opens

This is the finding that was never on the risk list, and it sits upstream of
everything else.

Instrument: `scripts/measure_rho.py`. Measurement only — nothing in `src/` was
touched and no solver code was written; `lanczos_iter` and `_tridiagonal` are
the library's own, used exactly as `solve.subspace_newton` uses them.

    lambda_g   = ||g||^2 / (g' M^-1 g)
    lambda_Sig = sqrt( tr(M Sigma) / tr(M^-1 Sigma) )
    rho        = (lambda_Sig / lambda_g)^2

The gate opens at `rho >= 1`. Below 1, the gradient is concentrated in
directions sharper than the ones the gradient noise occupies, so dividing by
curvature amplifies noise faster than it amplifies signal.

### 2.1 Which operator this was measured against

**`M = |H| + delta`, DSN's actual damped saddle-free operator — not raw
`H^-1`.** This scoping is forced by the code and is not a convenience.

DSN's exact Hessian here is genuinely indefinite: `neg_frac`, the
`g^2`-weighted mass on negative curvature, is 0.404 at step 0 and 0.01–0.04
thereafter. `H^-1` is therefore not a preconditioner anything here would use,
and `g' H^-1 g` can go negative. `solve.subspace_newton` inverts
`M = |H| + delta` (`damping=1e-3` by default), so `M` is what "curvature
preconditioning" means for this code, and `M` is what was measured. The
undamped indefinite variant appears only as the `neg_frac` diagnostic.

Read the result accordingly: it is a statement about DSN's own operator on
DSN's own trajectory, not about `H^-1` and not about second-order methods in
general.

The same fact — that DSN solves `(|H| + delta) d = -g` and not `H d = -g` —
independently breaks the FACA gate's contraction premise in §4. Two unrelated
measurements each had to confront it.

### 2.2 Estimator, and why `Sigma` is not self-referential

One `lanczos_iter` run per probe vector, `eigh` of its tridiagonal, Gauss
quadrature: `b' f(H) b ≈ ||b||^2 * sum_j V[0,j]^2 * f(lam_j)`. The same run
yields `b' M b` and `b' M^-1 b` at every depth and every `delta` for free.

`Sigma` is estimated from 24 microbatches of size 128, drawn from a pool
**disjoint** from the 2048-sample batch that defines `H` and `g`, and cross-fit
by halves — only cross-half differences are used,

    u_i = (g_{A_i} - g_{B_i}) / sqrt(2),     E[u_i u_i'] = Sigma exactly

so no mean is ever computed from the samples it is then evaluated against. That
constraint is deliberate: it is the failure mode this project had already paid
for at 85x optimism (see
[`self-referential-diagnostics.md`](self-referential-diagnostics.md)).
`Sigma`'s overall scale cancels in `lambda_Sig`, so microbatch size affects only
its shape.

### 2.3 Result (`delta=1e-2`, `k=100`)

Fresh run of the exact shipped `sanity_mnist.py` DSN configuration, seed 0. The
loss trajectory it produces (0.267 at step 80, 0.271 at step 199) sits inside
the measured DSN range above, so the trajectory instrumented is the shipped one.

| step | loss | \|\|g\|\| | neg_frac | lambda_g | lambda_Sig | rho | half A | half B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.3218 | 0.654 | 0.404 | 0.394 | 0.155 | 0.154 | 0.189 | 0.124 |
| 40 | 0.3274 | 0.530 | 0.013 | 0.575 | 0.236 | 0.168 | 0.134 | 0.198 |
| 80 | 0.2672 | 0.346 | 0.035 | 0.277 | 0.184 | 0.443 | 0.573 | 0.350 |
| 120 | 0.2197 | 0.215 | 0.043 | 0.142 | 0.115 | 0.650 | 0.811 | 0.460 |
| 160 | 0.2673 | 0.298 | 0.033 | 0.202 | 0.110 | 0.295 | 0.368 | 0.219 |
| 199 | 0.2710 | 0.415 | 0.025 | 0.362 | 0.146 | 0.162 | 0.175 | 0.150 |

**`lambda_Sig < lambda_g` at every checkpoint measured. Max `rho` over all 72
`(step, delta, k)` cells is 0.717.** It never reaches 1.

`rho` is not static — 0.154 at step 0, peaking near 0.650 around step 120,
falling back to 0.162 by step 199 — but the whole sweep lives in
`[0.045, 0.72]`. There is no phase of training on this problem where the gate
opens.

### 2.4 Convergence and sensitivity

- `delta` in {1e-2, 1e-1}: `rho` stable to ~1% across Lanczos depths
  `k = 25/50/75/100`. Converged.
- `delta = 1e-3` (DSN's shipped default): **not converged** — still drifting at
  `k=100` (step 0: 0.176 → 0.193 → 0.064 → 0.045). Expected: `lambda_g` is a
  harmonic mean, dominated by the smallest eigenvalues, which Lanczos resolves
  last. Those numbers are unreliable in magnitude, and are reported only because
  every one of them is also below 1 and the two converged `delta` values bracket
  them.
- Split-half spread on `Sigma` is ~±30% relative — real sampling noise, far
  short of the ~2x that would be needed to reach 1 anywhere.
- One scope note in DSN's favour that does not rescue it: `lambda_g` and
  `lambda_Sig` are measured on a 2048-sample curvature batch while DSN trains at
  batch 256, so the operator DSN actually sees per step is noisier than the one
  measured. That can only move the real gate further from 1, not closer.

A methods note, recorded because it invalidated a first sweep: the `k`-sweep was
initially slicing eigenvalues out of the depth-100 quadrature rule instead of
building a depth-`k` rule per depth. `eigh` returns nodes in ascending order, so
slicing keeps the most-negative curvature and discards the sharpest directions —
the opposite of a shallower Krylov space. All numbers above are post-fix;
`quadrature()` now builds `T_j` from the recurrence as it yields, at no extra
curvature cost.

---

## 3. Evidence 3 — recycling is break-even by construction

Novelty claim (1) does not survive its own accounting, and this is arithmetic
rather than a measurement:

> Refreshing `m` recycled directions against the current operator costs exactly
> `m` curvature products and saves at most the `m` new Lanczos directions they
> stand in for.

Exact recycling is therefore break-even by construction. Measured at equal
tolerance on drifting SPD operators (n=40/60/80, cond 1e2–1e4, 3 seeds each) it
runs **1.03–1.11x** a fresh rebuild's product count. The savings reported before
the fix round were an artifact: they came from carrying curvature images over
from the previous iterate, whose optimistic residual tripped the `tau` stopping
rule early.

Not refreshing is not an escape. Deflating unconverged Ritz directions was
measured to make the subspace **20–44x worse** than an equal-width fresh one, so
the builder correctly declines — and at `k_max <= 10`, which includes the
shipped MNIST configuration, essentially nothing qualifies and DSN rebuilds from
scratch every step (`reuse_frac` reads 0.00 throughout).

At equal basis width with directions that do qualify, recycling reaches
**parity** with a from-scratch basis. Not an improvement.

This is a conservation law nobody wrote down before implementation. It is
independent of §2 — it would hold even if the payoff gate were wide open. Full
derivation and the portable version of the result:
[`krylov-recycling-in-trust-region-newton.md`](krylov-recycling-in-trust-region-newton.md).

---

## 4. Evidence 4 — the FACA gate: the kills are structural, not batching

The escape route from §3 was *augmented* rather than deflated recycling (FACA),
gated on the break-even inequality

    |Delta_H|  <  (m_eff - 1) * (1 - rho_K) * epsilon_K(k) / ||y_U||

Instrument: `scripts/measure_faca_gate.py`, six consecutive-step pairs per run,
four runs, both seeds. **The gate is closed or vacuous at every pair measured.**

**Batching noise is a minor factor — about 2x.** The excerpt's suspected killer
was that the secant pair `(s, y)` spans two different mini-batches. Moving the
second gradient onto a common batch removes 40–60% of `|Delta_H|`
(0.750 → 0.384, 2.370 → 0.981, and so on) — not orders of magnitude. Its
proposed fix, one extra gradient on a common curvature batch, is measured to be
insufficient by 4–90x.

The real kills are structural, and each closes the gate on its own:

1. **The residual is not a contraction under DSN's solve target.** Geometric-mean
   rate `rho_geo` over the residual history is 0.978–1.014 across every
   configuration and seed, with only 41–57% of consecutive Lanczos directions
   contracting at all. `(1 - rho_K)` is therefore ~0.01 rather than O(1),
   crushing the right-hand side by about two orders of magnitude. This is
   structural, not a tuning failure: `subspace_newton` solves
   `(|H| + delta) d = -g`, so `||H d + g||` is the residual of a system the
   solver is deliberately not solving. It has no reason to decrease in `k`, and
   it doesn't. `tau=0.1` was never reached; every pair in every run hit `k_max`.

2. **New information and drift are anti-correlated, so no regime satisfies
   both.** On a fixed-batch run with zero batch noise, drift falls five orders of
   magnitude (`|dH|` 0.37 → 4.7e-6) and the gate *still* never opens: as the
   iterate settles, the carried basis comes to lie inside the fresh Krylov space,
   `m_eff` collapses to 1.7e-5 of 8 carried directions, and `epsilon_K` collapses
   with it. Both sides of the inequality are driven by the same underlying
   quantity — how much the operator and iterate have moved — so removing batch
   noise removes numerator and denominator together.

3. **`L_H*||s||` alone already violates it.** With the secant on a common batch,
   `|dH|_same` is 0.38–1.83 against a right-hand side of 0.011–0.10. Gap 4–90x.
   `L_H` measures 0.11–0.49, stable across step lengths of 1.4–4.0, i.e. behaving
   like a real Lipschitz constant.

4. **At DSN's shipped `k_max = 8` the gate is mathematically dead regardless of
   any of the above.** At most one direction is ever carried, `m_eff` maxes at
   0.40, so `(m_eff - 1) < 0` at every pair unconditionally. The `k_max = 40`
   runs are the ones points 1–3 rest on, precisely because the shipped
   configuration cannot express the question.

Independent of FACA, and worth carrying: `m_conv = 0` at **23 of 24 pairs** — a
direction passing the Ritz convergence test at step `t` never still passes
against step `t+1`'s operator. The shipped builder discards its basis every step
regardless of what any break-even gate decides. `||y_U||` and `m_eff` only exist
in that measurement because re-validation was forced off, which is the
configuration most favourable to FACA.

Two traps recorded because both would have produced a false positive:
single-pair `rho_K` reports open at 3 of 6 pairs where the sequence-level rate
says 0 of 6 on identical data (lucky landings on one side of an excursion in a
random-walk residual history); and with `m_eff < 1` and `rho_K > 1` both factors
go negative, their product is positive, and the comparison "passes" while
describing nothing. Those rows are flagged as vacuous, not counted.

---

## 5. The rank-`k` gate correction, stated honestly

An earlier report from the screening harness said "full-rank DSN FAILs the
stage-1 gate". **That finding never applied to DSN's shipped operator, and it
has been retracted.**

The harness screened the idealized full-rank form `P = (|H| + delta)^-1`. DSN
applies a rank-`k_max` Krylov restriction with a first-order complement:

    P_k = W (|W'HW| + delta)^-1 W'  +  c (I - WW')

with `W` a rank-`k` Krylov basis seeded at the gradient. The framing under which
the full-rank verdict was reported treated it as "strictly an upper bound" on
what the rank-8 implementation could achieve. That is wrong. `P_k` with `c > 0`
is itself a full-rank invertible matrix: it is a *member* of the class the
full-rank screen draws from, not a restriction of it. An upper bound would
require maximizing the statistic over the class, which no gate here does — and
which would be vacuous anyway, since `P = I` is always feasible so the supremum
is always at least 1.

Screening what DSN actually applies flips the verdict:

| candidate | scope | verdict | `rho` ceil | `gain` ceil | traj. gain | depth conv. |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| dsn(delta=1e-3) | full-rank | INDETERMINATE | 0.462 | 78.91 | 3.82 | 4.698 |
| dsn(delta=1e-2) | full-rank | **FAIL** | 0.515 | 9.848 | 1.981 | 0.279 |
| dsn(delta=1e-1) | full-rank | **FAIL** | 0.571 | 4.196 | 3.757 | 0.005 |
| dsn-rank8(1e-2, c=1) | rank-8 | INDETERMINATE | 2349 | 10.52 | nan | 0.000 |
| dsn-rank8(1e-1, c=0.1) | rank-8 | **PASS** | 2924 | 371.9 | 330.7 | 0.000 |
| dsn-rank8(1e-1, c=1) | rank-8 | **PASS** | 74.76 | 293.5 | 263.2 | 0.000 |
| dsn-rank8(1e-1, c=10) | rank-8 | **PASS** | 1.412 | 16.32 | 13.0 | 0.000 |

DSN's shipped rank-8 operator never FAILs, and at `delta=0.1` it PASSes on both
statistics at every complement scaling. The stage-1 rejection of DSN was not
evidence about DSN and should not have been reported as if it were.

**Correcting this did not rescue DSN.** Three independent reasons:

1. **Stage 2's own training-based FLOP measurement still shows DSN losing, and
   it does not consult the gate.** On the harness's quadratic trial (dim 24,
   cond 100, batch 16, held-out eval, analytic FLOP model), both arms at their
   separately tuned learning rate (`lr=0.01` for both), 3 seeds, run until an
   equal **4e7-FLOP budget** is exhausted:

   | arm | steps taken in budget | held-out loss (3 seeds) |
   | --- | ---: | ---: |
   | sgd | 17,362 | 0.00691 / 0.00657 / 0.00619 (mean 0.00656) |
   | dsn(delta=1e-3, k_max=8) | 1,432 | 0.01222 / 0.01188 / 0.01321 (mean 0.01244) |

   DSN gets 12.1x fewer steps for the same FLOPs and ends **1.90x worse**. At
   equal *step* count instead (400 steps each), DSN spends 12.2x the FLOPs and
   still ends 3.1x worse (0.0193 vs 0.0062). Neither number depends on any gate
   verdict.

2. **The rank-8 PASS rests on `gain` alone, and `rho` is known-blind to exactly
   this operator shape.** The rank-restricted `rho` values run to 10^4–10^5 and
   fall monotonically as `c` rises. That is not a discovery; it is the `rho`
   blind spot firing. `rho` contains `tr(P^-1 Sigma) / tr(P Sigma)`, which
   diverges as `P` is shrunk on one subspace relative to another with no
   reference to what that subspace costs the objective — the same blind spot that
   scored 33.5 on a synthetic configuration training 2,264x worse than SGD. A
   rank-restricted operator *is* a subspace-damping operator. A PASS resting on
   one of two statistics, where the other is known-blind to the construction, is
   weaker than a PASS from both.

3. **The end-to-end result against AdamW (§1) and the three mechanism
   measurements (§2–§4) are untouched by any of this.** The stage-1 gate is a
   cheap pre-flight screen. It was never load-bearing for the conclusion.

What the correction does establish is narrower and still important: a verdict
about one operator must not be read as a verdict about another, and the harness
now carries a `scope` field on every gate report so it cannot be.

---

## 6. Which pre-registered risk fired, and what is not claimed

The design named two risks and said the CPU sanity phase would expose both.

**Risk 1 — "recycled eigenvectors go stale faster than cross-step curvature
correlation predicts."** Fired, and the pre-registration understated it. The
failure is not a *rate*. It is that exact refresh is break-even by an exact
accounting (§3), and that both sides of the augmented-recycling inequality are
driven by the same underlying quantity, so no regime satisfies both (§4). A
rate-based risk can be fixed by a faster refresh; a conservation law cannot.

**Risk 2 — "Adam's per-coordinate scaling in the complement conflicts with the
trust region inside `S`."** Fired, and was worse than described: Lanczos is
seeded with the deflated gradient, so `||W'g|| / ||g|| = 1.0000` every step, the
complement carried *exactly* zero first-order descent, and it was not scaled by
the trust-region clip. A collapsing region degraded DSN into a zero-descent
method taking unit-norm steps. Diagnosed as F4, fixed. **Not fatal** —
correcting it did not make DSN win.

**Risk 3 — unnamed, and the one that decides the case.** Whether curvature
preconditioning pays *at all* in this regime was never posed as a risk; the
design assumed it and asked only whether the subspace could be found cheaply
enough. §2 measures the assumption and finds it false on this trajectory at
every checkpoint. The dependency runs Risk 3 → Risk 1, Risk 2: a cheaper or
fresher subspace does not help if the preconditioner it builds has negative
expected value. That ordering is why fixing all ten defects did not change the
outcome.

### What is not claimed

- **The pre-registered kill criterion has not been evaluated.** It is defined
  against FLOPs-to-target-loss on the CIFAR-10 or WikiText tiers, under a
  16-trial tuning budget, 3 seeds, non-overlapping error bars. Neither GPU tier
  has been run. Nothing above fires that criterion and nothing above should be
  reported as if it had. The recommendation that follows from the evidence is to
  not spend the ~35 GPU-hours on the current mechanism — but that is a
  recommendation, and the decision is the project owner's.
- **Not** that DSN would fail on CIFAR-10 or WikiText. `rho`'s own trajectory
  dependence (0.045 → 0.72 → 0.16 within a single run) is evidence that it is
  problem- and phase-dependent.
- **Not** that curvature preconditioning fails for neural networks generally.
  The measurement is a gate evaluated on one trajectory, not a theorem.
- **Not** that the `delta=1e-3` `rho` numbers are quantitatively right. They are
  not converged at `k=100`, and are reported only because they are bracketed and
  also below 1.
- **Not** that FACA is unsound. It was measured on a codebase whose saddle-free
  damped solve violates the gate's contraction premise. A method built on an
  actually-contracting Krylov residual is untested here.

---

## 7. Reproduce

    python -m pytest                                    # 155 passed

    python scripts/sanity_mnist.py --optimizer adamw
    python scripts/sanity_mnist.py

    python scripts/measure_rho.py --steps 200 --k 100 --micro 24 --out rho.json

    python scripts/measure_faca_gate.py --steps 60 --k-max 8                        # shipped config
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --fixed-batch
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --seed 1

    python scripts/measure_rank_restricted.py           # DSN at its shipped rank

The measurement scripts download MNIST, are not part of the test suite, and none
of them touches `src/`.

The §5 equal-FLOP comparison, which is not a checked-in script:

```python
import torch
from dataclasses import replace
from autoresearch.trials import QuadraticTrial, sgd_candidate, dsn_candidate
from autoresearch.cost import analytic_cost_model

torch.set_default_dtype(torch.float64)
t = QuadraticTrial(max_steps=200_000)          # so the FLOP budget binds, not the step cap
cm = analytic_cost_model(t.dim)
for cand in (replace(sgd_candidate(), lr=0.01), replace(dsn_candidate(), lr=0.01)):
    rs = [t.run(cand, s, cm) for s in (0, 1, 2)]
    print(cand.name, sum(r.score for r in rs) / 3, [r.steps for r in rs], rs[0].flops)
```

---

## Appendix — the defect history (F1–F10)

This is here rather than in the body because it is not the result. It is what
makes the result trustworthy: the implementation was wrong in ten identifiable
ways, all ten were found, all ten were fixed, and DSN lost anyway. Every number
in §1–§5 was measured after the fix round.

Suite went from 61 passed / 2 `xfail(strict=True)` to 79 passed / 0 xfail at the
end of the fix round, and 155 passed today. Neither xfail gate was relaxed — the
defects they pinned were fixed, and in one case the claim the gate asserted
turned out to be false and was replaced by the honest one.

### The defects

| # | defect | fix |
| --- | --- | --- |
| F1 | DSN diverged on mini-batch MNIST (loss 4.7–9.2 vs AdamW's 0.15–0.18) while the trust radius collapsed to 1e-12 and `n_shrink` hit 102–117 of 200 | consequence of F2 |
| F2 | the acceptance ratio spanned two mini-batches — `predicted` from step `t`'s batch scored against step `t+1`'s loss | same-batch reference: re-evaluate the same closure at the trial point. One extra forward pass, no extra curvature products, zero sampling noise in the ratio by construction |
| F3 | (analysis) lr and subspace under-resolution ruled out as the cause | no code change |
| F4 | `g ∈ span(W)` unconditionally (`\|\|W'g\|\|/\|\|g\|\| = 1.0000`), so the complement carried *exactly* zero first-order descent — and was not scaled by the trust-region clip, so a collapsing region produced a zero-descent method taking unit-norm steps | the radius now bounds the whole step; the two parts are orthogonal, so one common factor bounds both |
| F5 | (negative result) removing the trust region is not the fix — frozen at 1e6 the loss explodes to 1416 by step 40 | trust region kept; only its ratio input was broken |
| F6 | telemetry `k=12` vs `k_max=8` read as a bug | documentation: `k` is total basis width, `k_max` new vectors plus recycled columns |
| F7 | unbounded trust-radius growth (to 1.4e45 on a fixed-batch run — already an overflow in float32) | `trust_max`, defaulting to `1e3 * trust_radius` |
| F8 | `HU` carried over from the previous iterate, so `T ≠ W'HW` and `rel_residual` was measured against the same stale `T` that staleness corrupts — reported 85x optimistic, and *improving* as the subspace degraded | `HU` recomputed against the current operator every build; `newton_residual` is now `\|\|HWy + g\|\|` from retained curvature images, with no quantity crossing a step boundary; retained directions re-*validated*, not just re-projected |
| F9 | a zero gradient still produced a momentum-driven complement step, walking off an exact optimum | complement skipped when the subspace is empty and `\|\|g\|\| < stationary_eps` |
| F10 | (found while fixing F8) deflating *unconverged* Ritz pairs produced a subspace 20–44x worse than an equal-width fresh basis; the measured ordering of the three `rank_by` heuristics reversed once the metric was honest | a convergence gate `\|\|Hu − θu\|\| ≤ ritz_tol·\|θ\|` with `ritz_tol=1e-3`, calibrated by direct measurement (parity at 1e-3, 50x margin to the harmful regime) |

### Two things the fix round established that the defect list does not

**A gate can assert something false.** The F8 regression gate demanded that the
recycled arm beat an equal-width fresh basis by 2x on the reported residual. It
does not, and it never did. With both quantities computed exactly, the honest
claim is parity, and parity is what the replacement gates assert. Likewise
`test_recycling_reduces_hvps_on_a_slowly_changing_operator` demanded a 20%
saving and its docstring called that "the point of the whole method"; it was
replaced by a budget bound that compares products at equal accuracy instead of
comparing which arm stopped early.

**Fixing the metric exposed the defect underneath it.** F10 was invisible while
F8's optimistic residual was in place. This is the general pattern the companion
document is about: a diagnostic computed inside the approximation it evaluates
hides the failures of everything downstream of it too.
See [`self-referential-diagnostics.md`](self-referential-diagnostics.md).

Two thresholds were recalibrated during the fix round, both flagged in the test
docstrings: the mini-batch `n_shrink` bound (an aspiration written against a
broken implementation, replaced by the measured equilibrium — a working trust
region on a genuinely noisy objective binds on 144 of 200 steps and shrinks on
48–57), and the residual-drift bound, because the quantity being bounded changed
from an optimistic residual to a true one.

### Source documents

The unedited contemporaneous records, kept for provenance:

- `docs/superpowers/specs/2026-07-27-dynamic-subspace-newton-design.md` — design and pre-registration
- `docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md` — F1–F9 as first diagnosed
- `docs/superpowers/plans/2026-07-28-dsn-plan2-fixes.md` — the fix round, plus F10
- `docs/superpowers/plans/2026-07-28-faca-gate-measurement.md` — the full FACA measurement
- `docs/superpowers/plans/2026-07-29-negative-result.md` — the first pass at this document
