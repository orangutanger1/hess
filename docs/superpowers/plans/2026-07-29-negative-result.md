# DSN — negative result

**Date:** 2026-07-29
**Status:** Measured. Recommends against spending the GPU budget.

The design document pre-registered that "a negative result is a real result and
gets written up as one," and that if the mechanism failed, the most valuable
output would be *why* — specifically, which of two named risks fired. This is
that writeup.

The short version: **neither named risk is the reason.** Both were real, both
were measured, and one of them (Risk 1) turned out to be stronger than its
pre-registration predicted — but the thing that closes the case is a third
condition that was never on the risk list, sits upstream of both, and holds at
every point of the trajectory measured.

---

## 1. What can and cannot be claimed

Stated precisely, because the pre-registration was written to stop the bar from
moving and that cuts both ways.

**The pre-registered kill criterion has not been evaluated.** It is defined
against FLOPs-to-target-loss on the CIFAR-10 or WikiText tiers, under a 16-trial
tuning budget, 3 seeds, non-overlapping error bars. Neither GPU tier has been
run. Nothing below fires that criterion, and nothing below should be reported as
if it had.

**What has been established is what P2 was designed to establish.** The design
put a CPU sanity phase in front of the GPU spend for exactly this purpose —
"P2 is designed to expose both [named risks] on CPU, before any money is spent."
P2 did its job, and then some: it exposed both named risks and a third, more
fundamental one. The evidence below is CPU-tier, on a 784-96-tanh-10 MLP
(76k parameters) on MNIST, plus synthetic problems.

**Whether to spend the ~35 GPU-hours anyway is a judgment call, and it is the
project owner's.** The case against is section 6. It is not a claim that the
GPU tiers would fail; it is a claim about what the mechanism is doing on the
problems where it has been instrumented.

---

## 2. Evidence 1 — the payoff gate never opens

This is the finding that was not pre-registered, and it is the one that matters
most.

Instrument: `scripts/measure_rho.py`. Measurement only — nothing in `src/` was
touched and no solver code was written. The formulas are taken as given from the
P0 excerpt and are not re-derived here.

    lambda_g   = ||g||^2 / (g' M^-1 g)
    lambda_Sig = sqrt( tr(M Sigma) / tr(M^-1 Sigma) )
    rho        = (lambda_Sig / lambda_g)^2

The gate opens at `rho >= 1`. Below 1, the gradient is concentrated in
directions sharper than the ones the gradient noise occupies, so dividing by
curvature amplifies noise faster than it amplifies signal, and curvature
preconditioning does not pay by this criterion.

### 2.1 Setup

There are no checkpoints or logs in the repo, so this is a fresh run of the
exact `scripts/sanity_mnist.py` DSN configuration: seed 0, 200 steps, batch 256,
`lr=1e-2`, `KrylovBuilder(k_max=8, m_recycle=4, tau=0.1)`. The loss trajectory
it produces (0.267 at step 80, 0.271 at step 199, on the fixed 2048-sample
curvature pool) sits inside the README's measured DSN range of 0.2678–0.3010, so
the trajectory being instrumented is the shipped one, not a variant.

### 2.2 Which operator — a decision forced by the code

The excerpt writes `H` and assumes `H^-1` exists. **DSN's exact Hessian here is
indefinite**: `neg_frac`, the `g^2`-weighted mass on negative curvature, is 0.40
at step 0 and 0.01–0.04 thereafter. So `H^-1` is not a preconditioner anything
here would use, and `g' H^-1 g` can go negative.

DSN does not invert `H`. It inverts the saddle-free damped operator
`M = |H| + delta` (`solve.subspace_newton`, `damping=1e-3` by default). `M` is
what "curvature preconditioning" means for this code, so `M` is what was
measured. The undamped indefinite variant is reported only as the `neg_frac`
diagnostic.

This is not a side note. The same fact — that DSN solves `(|H| + delta) d = -g`
and not `H d = -g` — is independently what breaks the FACA gate's contraction
assumption in section 4. Two unrelated measurements each had to confront it.

### 2.3 Estimator

One `lanczos_iter` run per vector, `eigh` of its tridiagonal, Gauss quadrature:

    b' f(H) b  ~=  ||b||^2 * sum_j V[0,j]^2 * f(lam_j)

The same run yields `b' M b` and `b' M^-1 b` together, at every depth and every
`delta`, for free. No new solver code — `lanczos_iter` and `_tridiagonal` are
the library's own, used exactly as `solve.subspace_newton` uses them.

### 2.4 Sigma, and why it is not self-referential

`Sigma` is estimated from 24 microbatches of size 128, drawn from a pool
**disjoint** from the 2048-sample batch that defines `H` and `g`. It is
cross-fit by halves: only cross-half differences are used,

    u_i = (g_{A_i} - g_{B_i}) / sqrt(2),     E[u_i u_i'] = Sigma exactly

so no mean is ever computed from the samples it is then evaluated against. That
constraint is deliberate — it is the F8/F10 failure mode (a metric computed
against the same object it evaluates, which then improves as the object
degrades), and this project has already been burned by it once at 85x. `Sigma`'s
overall scale cancels in `lambda_Sig`, so the microbatch size affects only its
shape.

### 2.5 Result (`delta=1e-2`, `k=100`)

| step | loss | \|\|g\|\| | neg_frac | lambda_g | lambda_Sig | rho | half A | half B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.3218 | 0.654 | 0.404 | 0.394 | 0.155 | 0.154 | 0.189 | 0.124 |
| 40 | 0.3274 | 0.530 | 0.013 | 0.575 | 0.236 | 0.168 | 0.134 | 0.198 |
| 80 | 0.2672 | 0.346 | 0.035 | 0.277 | 0.184 | 0.443 | 0.573 | 0.350 |
| 120 | 0.2197 | 0.215 | 0.043 | 0.142 | 0.115 | 0.650 | 0.811 | 0.460 |
| 160 | 0.2673 | 0.298 | 0.033 | 0.202 | 0.110 | 0.295 | 0.368 | 0.219 |
| 199 | 0.2710 | 0.415 | 0.025 | 0.362 | 0.146 | 0.162 | 0.175 | 0.150 |

**`lambda_Sig < lambda_g` at every checkpoint measured.** Max `rho` over all 72
`(step, delta, k)` cells is **0.717**. It never reaches 1.

`rho` is not static — it dips low early, peaks around 0.65 near step 120, and
falls again. But the whole trajectory lives in `[0.045, 0.72]`. There is no
phase of training on this problem where the gate opens.

### 2.6 Convergence and sensitivity

- `delta` in {1e-2, 1e-1}: `rho` stable to ~1% across Lanczos depths
  `k = 25/50/75/100`. Converged.
- `delta = 1e-3` (DSN's shipped default): **not converged** — still drifting at
  `k=100` (step 0: 0.176 → 0.193 → 0.064 → 0.045). Expected, and the reason is
  understood: `lambda_g` is a harmonic mean, so it is dominated by the smallest
  eigenvalues, which Lanczos resolves last. Those numbers are unreliable in
  magnitude. Two things keep them from mattering: every one of them is also
  `< 1`, and the two converged `delta` values bracket them.
- Split-half spread on `Sigma` is ~±30% relative — real sampling noise, and far
  short of the ~2x that would be needed to reach 1 anywhere.

### 2.7 Methods note

First bug hit and fixed mid-run: the `k`-sweep was slicing eigenvalues out of
the depth-100 quadrature rule instead of building a depth-`k` rule per depth.
`eigh` returns nodes in ascending order, so slicing keeps the most-negative
curvature and discards the sharpest directions — the opposite of a shallower
Krylov space. That made the first sweep meaningless. All numbers above are
post-fix, and `quadrature()` now builds `T_j` from the recurrence as it yields,
at no extra curvature cost.

### 2.8 Reading

Gradient concentration in sharp directions is present on this trajectory,
throughout training, and it is the sign the excerpt's corrected analysis
predicts. By this criterion, curvature preconditioning has nothing to buy here.

One scope note in DSN's favour, which does not rescue it: `lambda_g` and
`lambda_Sig` are measured on a 2048-sample curvature batch, while DSN trains at
batch 256. The operator DSN actually sees per step is noisier than the one
measured. That can only move the real gate further from 1, not closer.

---

## 3. Evidence 2 — recycling is break-even by construction

Novelty claim (1) — cross-step subspace recycling — does not survive its own
accounting, and this is arithmetic rather than a measurement:

> Refreshing `m` recycled directions against the current operator costs exactly
> `m` curvature products and saves at most the `m` new Lanczos directions they
> stand in for.

Exact recycling is therefore break-even by construction. Measured at equal
tolerance on drifting SPD operators it runs **1.03–1.11x** a fresh rebuild's
product count. The savings reported before Plan 2 were an artifact: they came
from carrying curvature images over from the previous iterate, whose optimistic
residual tripped the `tau` stopping rule early.

Not refreshing is not an escape. Deflating unconverged directions was measured
to make the subspace **20–44x worse** than an equal-width fresh one, so the
builder correctly declines — and at `k_max <= 10`, which includes the shipped
MNIST configuration, essentially nothing qualifies and DSN rebuilds from scratch
every step.

At equal basis width with directions that do qualify, recycling reaches
**parity** with a from-scratch basis. Not an improvement.

This is a conservation law nobody wrote down before implementation. It is
independent of everything in section 2 — it would hold even if the payoff gate
were wide open.

---

## 4. Evidence 3 — the FACA break-even gate never opens

Full measurement in
[`2026-07-28-faca-gate-measurement.md`](2026-07-28-faca-gate-measurement.md);
not duplicated here. The relevant conclusions:

The escape route from section 3 was augmented rather than deflated recycling,
gated on

    |Delta_H|  <  (m_eff - 1) * (1 - rho_K) * epsilon_K(k) / ||y_U||

**The gate is closed or vacuous at every pair measured**, across every
configuration and both seeds. The suspected killer — different mini-batches
across the secant pair — is worth about **2x** and is *not* what closes it.
Three other things each close it alone:

1. **`rho_K` is not a contraction.** Geometric-mean rate 0.978–1.014; only
   41–57% of consecutive Lanczos directions contract at all. `(1 - rho_K)` is
   ~0.01 rather than O(1), crushing the right-hand side ~100x. Structural, not
   tuning: `subspace_newton` solves `(|H| + delta) d = -g`, so `||H d + g||` is
   the residual of a system the solver is deliberately not solving. It has no
   reason to decrease in `k`, and it doesn't. `tau=0.1` was never reached; every
   pair in every run hit `k_max`.
2. **`m_eff` and `|Delta_H|` are anti-correlated.** On a fixed-batch run with
   zero batch noise, drift falls five orders of magnitude (`|dH|` 0.37 →
   4.7e-6) and the gate *still* never opens: the carried basis comes to lie
   inside the fresh Krylov space, `m_eff` collapses to 1.7e-5 of 8 carried
   directions, and `epsilon_K` collapses with it. Removing batch noise removes
   numerator and denominator together.
3. **`L_H*||s||` alone already violates it.** Common-batch secant:
   `|dH|_same` 0.38–1.83 against `rhs_geo` 0.011–0.10. Gap 4–90x.

Independent of FACA: `m_conv = 0` at **23 of 24 pairs** — a direction passing
the Ritz test at step `t` never still passes against step `t+1`'s operator. The
shipped builder discards its basis every step regardless of what any break-even
gate decides. `||y_U||` and `m_eff` only exist in that measurement because
re-validation was forced off, which is the configuration most favorable to FACA.

Two traps worth carrying forward, because both would have produced a false
positive:

- **Single-pair `rho_K` reports open/closed at random.** Last-pair ratio says
  open at 3/6 pairs; the sequence-level rate says 0/6 on identical data. The
  "open" rows are lucky landings on one side of an excursion.
- **Sign artifact.** With `m_eff < 1` and `rho_K > 1`, both factors go negative,
  their product is positive, and the comparison "passes" while describing
  nothing. The script flags these rather than counting them.

---

## 5. Evidence 4 — end-to-end, DSN loses

For completeness, the outcome the mechanism findings predict:

| problem | DSN | AdamW |
| --- | --- | --- |
| MNIST MLP, last-10-step mean loss, 3 seeds | 0.2678 / 0.3010 / 0.2805 | 0.1512 / 0.1781 / 0.1713 |
| synthetic stream (batch 32, dim 25, cond 100), 5 seeds | 0.29 – 1.75 | 0.16 – 0.59 |

DSN converges. It does not win. The acceptance ratio is valid by construction
(same-batch), the trust region is sane, the residual telemetry is a true
residual — all of the F1–F10 defects are fixed — and it still does not win.
That is the point of reporting these numbers *after* the fix round rather than
before: the loss is not attributable to a bug.

---

## 6. Which pre-registered risk fired

The design named two, and said P2 would expose both on CPU.

**Risk 1 — "recycled eigenvectors go stale faster than cross-step curvature
correlation predicts."** Fired, and the pre-registration understated it. The
failure is not a *rate* — it is not that staleness outruns correlation. It is
that exact refresh is break-even by an exact accounting (section 3), and that
the two sides of the augmented-recycling inequality are driven by the same
underlying quantity, so no regime satisfies both (section 4). A rate-based risk
can be fixed by a faster refresh. A conservation law cannot.

**Risk 2 — "Adam's per-coordinate scaling in the complement conflicts with the
trust region inside `S`."** Fired, and was worse than described: because Lanczos
is seeded with the deflated gradient, `||W'g|| / ||g|| = 1.0000` every step, so
the complement carried *exactly* zero first-order descent, and it was not scaled
by the trust-region clip. A collapsing region degraded DSN into a zero-descent
method taking unit-norm steps. Diagnosed as F4, fixed in Plan 2. **Not fatal** —
it was an implementation consequence of a design decision, and correcting it did
not make DSN win.

**Risk 3 — unnamed, and the one that decides the case.** Whether curvature
preconditioning pays *at all* in this regime was never posed as a risk. The
design assumed it and asked only whether the subspace could be found cheaply
enough. Section 2 measures the assumption directly and finds it false on this
trajectory at every checkpoint, with `rho` capping at 0.717 against a threshold
of 1.

The dependency runs Risk 3 → Risk 1, Risk 2. A cheaper or fresher subspace does
not help if the preconditioner it builds has negative expected value. That
ordering is why fixing all ten F-defects did not change the outcome.

---

## 7. What is not claimed

- **Not** that DSN would fail on CIFAR-10 or WikiText. Those were not run.
  `rho` is measured on one 76k-parameter MLP on MNIST, one seed, one
  hyperparameter setting. Its trajectory dependence (0.045 → 0.72 → 0.16 within
  a single run) is itself evidence that it is problem- and phase-dependent.
- **Not** that curvature preconditioning fails for neural networks generally.
  The measurement is a gate evaluated on one trajectory, not a theorem.
- **Not** that the `delta=1e-3` numbers are quantitatively right. They are not
  converged at `k=100`, and are reported only because they are bracketed and
  also below 1.
- **Not** that FACA is unsound. It was measured on this codebase, whose
  saddle-free damped solve violates the gate's contraction premise. A method
  built on an actually-contracting Krylov residual is untested here.

---

## 8. What survives

Three results are worth keeping independent of DSN's fate:

1. **Recycling's break-even conservation law** (section 3), and the measured
   anti-correlation between reuse volume and operator drift (section 4). This
   applies to every Krylov-in-training method, not just this one, and it is
   arithmetic plus a clean fixed-batch measurement rather than a benchmark.
2. **The self-referential-metric failure**, which is the strongest cautionary
   result the project produced and is fully measured: `rel_residual` was
   computed against the same stale `T` that staleness corrupts, so it *improved*
   as the subspace degraded — 85x optimism — and a **maximally-stale
   frozen-basis variant scored better on both staleness gates than the correct
   implementation**. Two named risks, two gates written specifically to catch
   them, zero power against the failure they existed for.
3. **The payoff gate as a pre-flight instrument** (section 2). It costs a few
   Lanczos runs at a handful of checkpoints and no solver code, and it answers
   "can curvature pay here" before a sweep rather than after. On this problem it
   would have answered in minutes what the full implementation answered in a
   project.

The recommendation that follows from all of it: **do not spend the GPU budget on
the current mechanism.** Not because the pre-registered criterion fired — it has
not been evaluated — but because the mechanism has been instrumented at three
independent points and returns nothing at all three. That is a decision for the
project owner, not a conclusion of this document.

---

## 9. Reproduce

    python scripts/measure_rho.py --steps 200 --k 100 --micro 24 --out rho.json

    python scripts/measure_faca_gate.py --steps 60 --k-max 8
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --fixed-batch
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --seed 1

    python scripts/sanity_mnist.py --optimizer adamw
    python scripts/sanity_mnist.py

Both measurement scripts download MNIST and are not part of the test suite.
Neither touches `src/`.
