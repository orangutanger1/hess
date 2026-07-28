# Task 9 method-level findings: the trust region under mini-batch noise

**Status: ruled DEFER-TO-PLAN-2.** These are findings about the *method*
(`src/dsn/optimizer.py`), not about the test suite. Per the user's ruling
on the Task 9 fix round, none of them are fixed on this branch — no
Task 1-8 source file was touched to produce or address this document. They
are recorded here so Plan 2 starts from a correct diagnosis instead of
rediscovering it.

All full-suite pytest gates pass on this branch, including the new
mini-batch trust-region test, which is marked
`@pytest.mark.xfail(strict=True)` specifically because it reproduces the
defect described below and is expected to start passing only once Plan 2
fixes it.

## F1 — DSN diverges on mini-batch MNIST while AdamW converges

Across 3 seeds, 200 steps each (`scripts/sanity_mnist.py`-equivalent
mini-batch training, small MLP, `Linear(784,96) -> Tanh -> Linear(96,10)`):

| seed | DSN last-10-step mean loss | AdamW last-10-step mean loss | final `n_shrink` (of 200) | final `trust_radius` |
|---|---|---|---|---|
| 1 | 9.2297 | 0.1571 | 117 | 7.3e-12 |
| 2 | 5.8713 | 0.1512 | 109 | 3.0e-08 |
| 3 | 4.6986 | 0.1781 | 102 | 1.9e-06 |

DSN's loss rises by roughly an order of magnitude or more over training
while AdamW converges normally, in every seed tried. This is not
seed-specific noise; it is systematic.

## F2 — The lagged rho is the operative cause (isolated by single-variable experiment)

Root cause isolated by changing exactly one thing: replacing the lagged
acceptance ratio (`rho` computed from the *previous* step's `predicted`
against the *current* step's `loss_now`, which under mini-batching are
losses on two different batches) with a same-batch ratio (recomputing the
reference loss on the same batch the prediction was made for). Holding
everything else fixed:

| variant | final loss | trust_radius | n_shrink |
|---|---|---|---|
| lagged rho (current code) | 9.6429 | 5.8e-11 | 112 |
| same-batch rho | 0.1712 | stable at 2.0 | 7 |
| `torch.optim.AdamW` reference | 0.1692 | n/a | n/a |

The same-batch-rho variant matches the AdamW reference almost exactly
(0.1712 vs 0.1692) and keeps a stable, sane trust radius. This single
variable swap is sufficient to eliminate the divergence, which pins the
defect specifically on how `rho` is computed
(`DSN._update_trust_radius`, `src/dsn/optimizer.py:83-95`), not on the
subspace construction, the damping, or the complement mechanism in
general.

## F3 — lr and the under-resolved subspace are ruled out as sole causes; severity is hyperparameter-dependent

Two alternative explanations were checked and ruled out as the primary
cause:

- **Learning rate.** A fixed mini-batch (no per-step resampling) at the
  same `lr` and `k_max` reaches loss 0.0000 -- i.e. the step size itself is
  not the problem; it is specifically the *combination* of a per-step
  changing objective with the lagged ratio.
- **Under-resolved subspace** (`k_max` too small to represent curvature,
  see F6). With the lag left intact, raising `k_max` from 8 to 16 at
  `lr=1e-2` still converges, to 0.197 -- better, but the defect's
  *severity* is hyperparameter-dependent, not evidence that a bigger
  subspace fixes the underlying mechanism. The lagged-rho defect (F2) is
  the dominant, structural cause; subspace size modulates how bad it gets.
  The decisive evidence against under-resolution is not the `k_max` sweep
  but the *converging* run: the fixed-batch run that reaches loss 0.0000
  also sits at `rel_residual` 0.744-0.769 and never reaches `tau` either.
  An under-resolved subspace is therefore compatible with convergence on
  this problem, so under-resolution cannot be what distinguishes the
  diverging mini-batch run from the converging fixed-batch one. The only
  variable that does distinguish them is the per-step change of objective,
  i.e. F2.

## F4 — `g ∈ span(W)` always: the complement carries zero first-order descent

This is the deepest finding, and the original Task 9 report's framing --
"removing even a handful of directions from the update is enough to break
convergence" -- understated it. The actual mechanism is stronger and more
specific than "a few directions removed":

By construction, `src/dsn/lanczos.py:44,49` seeds the Lanczos basis with
the deflated gradient itself: `r = g - U @ (U.T @ g)`, normalized to become
the first new Krylov vector. Since `W = concat(U, Q)` and `Q`'s first
column is (proportional to) `g`'s component orthogonal to `U`, the entire
gradient `g` -- not most of it, all of it -- lies in `span(W)`, every
step, unconditionally. Measured: `‖W^T g‖ / ‖g‖ = 1.0000` at every step.

Consequently `d_comp = project_out(adam_step, W)`, which is by
construction orthogonal to `span(W)`, is *also* orthogonal to `g` itself,
every step: measured `cos(d_comp, -g) = -0.0000`. `d_comp` carries **zero**
first-order descent, not "reduced" descent -- exactly zero, always. All of
the update's first-order descent lives in `d_sub` alone.

This matters because of what the trust-region clip does next
(`src/dsn/optimizer.py:133-138`): when `‖d_sub‖ > trust_radius`, `d_sub` is
scaled by `s = trust_radius / ‖d_sub‖ -> 0` as the region collapses, but
`d_comp` is **not** scaled -- it is added at essentially unit norm
regardless of `s`. So as the trust region collapses, `g^T d -> 0` (the
step's descent guarantee vanishes) while `‖d_comp‖` stays ~1.0. The trust
region does not degrade DSN toward "AdamW with a few directions missing"
or toward a small-step first-order method; it degrades DSN toward **a
zero-descent method that keeps taking unit-norm steps**, which is exactly
why the loss *rises* under collapse rather than merely stalling or
slowing down.

Isolation experiment, `Δ` (trust radius) pinned at `1e-12` so the clip
factor is effectively always active:

| variant | final loss |
|---|---|
| complement projection ON (current code) | 6.4602 (diverges) |
| complement projection OFF (`d_comp` skipped entirely) | 0.1692 (matches AdamW reference to four decimals) |

Turning the projection off entirely, at the same pinned trust radius,
recovers AdamW's own performance almost exactly. This confirms the
mechanism precisely: it is not the small subspace, not the AdamW
complement in general, and not the trust region's existence -- it is
specifically the unscaled, zero-descent complement step being added on
top of a collapsed, near-zero subspace step.

## F5 — Removing the trust region is NOT the fix

A naive reading of F1-F4 might suggest simply deleting or disabling the
trust-region clip. This was tested and is explicitly wrong. Freezing
`trust_radius` at `1e6` (i.e. effectively never clipping) under the same
mini-batch training: loss explodes to 1416 by step 40, final loss 40.2,
`step_norm_subspace` reaching 265. Without a trust region, the raw
(unclipped) Newton step in an 8-12-dimensional subspace built from a
single noisy mini-batch's curvature is itself unstable and produces huge,
destructive steps.

**The trust region is load-bearing.** The defect is specifically in its
acceptance-ratio computation (F2), not in the trust-region mechanism's
existence. Any Plan 2 fix must keep a trust region and repair how `rho` is
computed under a changing objective (e.g. a same-batch reference per F2,
a held-out validation batch, or a variance-reduced / EMA'd ratio) -- not
remove the safeguard.

## F6 — `k=12` vs `k_max=8` is by design, not a defect

Telemetry `k` reports the *total* basis width `W.shape[1]`, which is
`k_max` new Lanczos vectors plus up to `m_recycle` recycled columns
concatenated on top (`src/dsn/subspace/krylov.py:83`: `W = st.Q if m == 0
else torch.cat([self.U, st.Q], dim=1)`; `src/dsn/optimizer.py:157`:
`k=res.W.shape[1]`). At `k_max=8, m_recycle=4` the reported total is
correctly 12 once recycling is warmed up. This is not a bug -- recycling
is designed to *add to*, not replace, the new-vector budget. The
brief text for `scripts/sanity_mnist.py` ("DSN prints a small `k`, well
under `k_max=8`") describes `k_max` itself, not the total width the
telemetry field reports; that expectation was miswritten, not the code.

## F7 — Unbounded trust-radius growth, no upper clamp

`src/dsn/optimizer.py:93-94` grows `trust_radius` by `trust_grow` (default
2.0x) whenever `rho > 0.75`, with no upper bound anywhere in
`_update_trust_radius`. Measured:

- Gate 3's own full-batch run (`test_trust_region_does_not_collapse`):
  grows from 1.0 to 1.342e8 over 40 steps.
- Fixed-batch (non-resampled) MNIST training: grows to roughly 1.4e45 over
  200 steps.

This is harmless in both cases observed so far because `‖d_sub‖` shrinks
in lockstep as the iterate nears a optimum (the clip never engages once
the radius is enormous relative to the actual Newton step), but it is a
latent issue worth flagging for Plan 2: the radius carries no usable upper
bound or memory of a sane scale, so after any transient spike it cannot
promptly re-engage a reasonable clip. In `float64` the observed magnitudes
are nowhere near dangerous -- 1.4e45 is some 263 orders of magnitude below
the `float64` limit. The concrete risk is precision, not `float64`
overflow: `float32` overflows around 3.4e38, so the fixed-batch MNIST run's
1.4e45 would already have overflowed there.

## F8 — Recycling degrades the TRUE residual while telemetry reports the reverse

Mechanism: `HU` is carried over from the PREVIOUS iterate's Hessian
(`krylov.py:113-117, 144-145`), so `T` is not `WᵀHW` at the current point,
and `newton_residual` measures against that stale `T` rather than the true
operator. This is the quantitative form of the Risk-1 staleness Task 9 set
out to detect. Deferred to the follow-up plan by ruling, pinned by the
xfail gate. This finding has two distinct parts, which must not be
blurred together — one needs a true dense residual to see, the other does
not.

**Claim 1 — telemetry reports the reverse of ground truth (unequal
widths, the fix-round-1 comparison).** At `m_recycle=5, k_max=10`
(recycled, width 15) vs `m_recycle=0, k_max=10` (fresh, width 10) — the
comparison as it existed after fix round 1, before it was re-pointed —
telemetry reports the recycled arm 3.5x *better* (reported 0.1010 vs
0.3572), while an independently measured true dense residual says the
recycled arm is 3x *worse* (true 1.078 vs 0.357). At cond=1000 the same
inversion is far larger: reported says recycled is 2x better (0.0722 vs
0.1444), true says it is 36000x worse (5269.7 vs 0.144). The fresh arm's
reported (0.3572) and true (0.357) values agree, which is what validates
the measurement — the fresh arm is exact by construction whenever
`reuse_frac == 0` (`base.py:20-23`), so the entire reported-vs-true gap
belongs to the recycled arm, not to measurement noise. The optimism
reaches 27x-36000x, which exceeds `base.py`'s own documented "order of
magnitude or more" by a wide margin — that docstring understates the
effect.

**Claim 2 — what the xfail gate actually pins (equal widths, the current
comparison).** `tests/test_convergence.py::test_recycling_stays_fresh_rather_than_going_stale`
was re-pointed at an equal-basis-width comparison (`m_recycle=5,k_max=10`
vs `m_recycle=0,k_max=15`, both width 15 once warmed up) to isolate
recycling quality from the basis-width confound in Claim 1's comparison.
At equal width, recycling is worse even by its own optimistic *reported*
metric, with no true-residual computation needed: recycled 0.1010 vs
fresh 0.0644. This is a weaker but more robust statement than Claim 1 — it
needs no dense-Hessian probe, which is why the gate can assert it
directly (computing a true dense residual inside the test was ruled out:
no extra curvature products in the test). Recycling also used fewer HVPs
doing it (400 vs the fresh arm's 591 over 40 steps), so the reversal is
not explained by recycling being given a smaller budget. The gate fails
today for this reason and flips green the day the recycled block of `T`
is recomputed against the current step's Hessian instead of carried over
from the previous step's.

Recycling is not broken in the sense of producing garbage steps — DSN
still converges on the fixed-batch problems this suite exercises
(`test_dsn_reaches_lower_loss_than_adamw_on_logistic_regression`,
`test_trains_a_small_mlp_on_synthetic_data`, both passing). The claim is
narrower: recycling does not improve the residual over a fresh basis of
the same width on this problem family — true by Claim 1's ground truth,
and even by the optimistic reported metric per Claim 2 — and the reported
(`rel_residual`) telemetry hides the true-residual gap because it is
computed against the same stale `T` that produced the bad recycled
directions in the first place, rather than against the true operator.

## F9 — A zero gradient still produces a step, walking off an exact minimum

When `‖g‖` is exactly zero the builder short-circuits and returns an empty
subspace (`src/dsn/subspace/krylov.py:76-77`, `k=0`, no curvature
products), so `d_sub` is zero — but the AdamW complement is *not*
short-circuited. `AdamWState.step` is still called, its first moment `m`
still carries the previous steps' gradients, and `project_out` against a
zero-width `W` is the identity, so the full momentum-driven AdamW
displacement is applied on top of a zero Newton step.

Measured on `test_lagged_rho_accounts_for_trust_region_clipping`'s exact
quadratic (`n=10, cond=50, lr=1e-3`): iteration 11 is the first step whose
Newton step fits inside the grown trust radius and it lands on `x*`
exactly, so iteration 12 sees `‖g‖ = 1.3e-15`, reports `k=0`, and applies
`‖d_comp‖ = 2.455e-3`. Iteration 13 then measures `‖g‖ = 2.6e-2` — the
optimizer has moved back off the optimum it had reached, and the loss it
had minimized is no longer minimal.

This is the same class of defect as F1-F8 (method-level, not an
implementation slip): a first-order complement with momentum has no notion
of "the subspace method has already converged", and DSN has no stationarity
check anywhere that would stop it. It is benign on the problems this suite
exercises because the excursion is small and immediately re-corrected on
the next step (iteration 13's subspace step is 2.455e-3, exactly undoing
it), but it means an exact optimum is not a fixed point of the iteration.
How the sequence behaves over many steps at stationarity was not measured
and is not claimed here. Plan 2 should either damp the complement by the same trust-region
logic that scales `d_sub`, or gate the complement on `‖g‖` being above a
stationarity threshold. Note that this defect is *not* what F4 describes:
F4 is about the complement carrying zero first-order descent while the
gradient is nonzero; F9 is about it carrying a nonzero step while the
gradient is zero.

## Where this is tested on this branch

- `tests/test_convergence.py::test_trust_region_does_not_collapse` --
  passes; this is the regime (fixed full-batch objective) in which the
  lagged rho is valid, confirming Task 8's death-spiral fix holds
  independently.
- `tests/test_convergence.py::test_trust_region_collapses_under_minibatch_noise`
  -- new, `xfail(strict=True)`. Reproduces the F1/F2 collapse with a
  synthetic (non-MNIST, no download) mini-batch stream. Currently fails
  with `n_shrink=99` (want <30) and `trust_radius=0.000122` (want >1e-2)
  over 200 steps, batch size 32. `strict=True` means this test flips the
  suite to failing the day someone "fixes" this without actually fixing
  it, and flips to passing the day the acceptance-ratio computation is
  corrected.
- `tests/test_convergence.py::test_recycling_stays_fresh_rather_than_going_stale`
  -- `xfail(strict=True)` as of the Task 9 fix round 2 (previously passed
  for the wrong reason, see the fix round 2 report). Reproduces F8's
  Claim 2 (reported-residual reversal at equal basis width, no true
  residual involved): `mean_recycled=0.1010` vs `mean_fresh=0.0644`, want
  `mean_recycled < 0.0322`. Flips to passing the day the recycled block of
  `T` is recomputed against the current step's Hessian instead of carried
  over from the previous step's.
- `tests/test_convergence.py::test_recycled_residual_does_not_drift_upward_across_a_run`
  -- passes. Split out of the test above by the final whole-branch review,
  which found it dead code there: the xfailed assertion raised before it in
  both normal and `--runxfail` mode, so it was evaluated in neither. It is
  the drift half of Risk 1 (`mean_tail <= 10 * mean_head`, measured
  `mean_head=0.0338`, `mean_tail=0.1634`, bound 0.33774).

**These gates are not staleness detectors, and Plan 2 must not treat them
as such.** Freezing `builder.U`/`HU` after step 5 and leaving them frozen
for the remaining 35 steps -- i.e. maximally stale recycling, the exact
failure Risk 1 was written to catch -- does not trip either gate. Measured
on the same 40-step ill-conditioned logistic run the gates use:

| run | mean reported `rel_residual` | drift (tail/head) | reported at final step | TRUE residual at final step |
|---|---|---|---|---|
| recycled, healthy (`m_recycle=5,k_max=10`) | 0.1010 | 4.84 | 0.6295 | 0.6897 |
| fresh, equal width (`m_recycle=0,k_max=15`) | 0.0644 | 0.45 | 0.0178 | 0.0178 |
| recycled, frozen after step 5 | **0.0263** | **0.47** | 0.0114 | **0.9752** |

The frozen-stale run passes the drift gate comfortably (0.47 <= 10) and it
passes the F8 gate too (0.0263 < 0.5 x 0.0644 = 0.0322) -- the gate the
*correct* implementation fails. So the maximally-stale variant scores
better on both gates than the shipped one. Its reported residual (0.0114)
understates its true residual (0.9752) by 85x, while the fresh arm's
reported and true values agree exactly (0.0178 vs 0.0178), which is what
validates the measurement.

This is F8's mechanism taken to its limit: `rel_residual` is computed
against the same `T` that staleness corrupts, so the metric *improves* as
the subspace degrades. Any gate built on reported `rel_residual` is
therefore blind to staleness by construction, no matter how it is
strengthened. Detecting staleness requires a measurement against the true
operator, which these gates deliberately do not make (the ruling was: no
extra curvature products in tests). Plan 2 should assume Risk 1 is
currently **uncovered**, not covered by these two gates.

## Recommendation for Plan 2

1. Fix `rho`'s definition to be valid under a changing objective (F2) --
   this is the root cause and the single highest-leverage change.
2. Do not remove the trust region as part of that fix (F5) -- it is load
   bearing; only its acceptance-ratio input is broken.
3. Consider clamping `trust_radius` to some problem-scaled maximum (F7) --
   independent of F2, cheap, and removes a latent numerical risk.
4. `k_max` vs total basis width (F6) is a documentation fix only (correct
   the `scripts/sanity_mnist.py` docstring expectation), not a code change.
5. Recompute the recycled block of `T` against the current step's Hessian
   rather than carrying `HU` over from the previous step (F8). Until this
   lands, recycling should not be assumed to help accuracy on any given
   step -- its only demonstrated benefit is spending fewer HVPs (F8: 400
   vs 591 over 40 steps at equal width), not a better subspace.
6. Stop the complement from stepping at stationarity (F9) -- either scale
   `d_comp` by the same trust-region factor `s` that scales `d_sub`, or gate
   it on `‖g‖` exceeding a stationarity threshold. Cheap and independent of
   F2.
