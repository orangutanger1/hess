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
promptly re-engage a reasonable clip, and in a longer or less benign run
this is one `float64` overflow away from becoming a real problem (`float32`
overflows around 3.4e38 -- the fixed-batch MNIST run's 1.4e45 would
already have overflowed in `float32`).

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

## Recommendation for Plan 2

1. Fix `rho`'s definition to be valid under a changing objective (F2) --
   this is the root cause and the single highest-leverage change.
2. Do not remove the trust region as part of that fix (F5) -- it is load
   bearing; only its acceptance-ratio input is broken.
3. Consider clamping `trust_radius` to some problem-scaled maximum (F7) --
   independent of F2, cheap, and removes a latent numerical risk.
4. `k_max` vs total basis width (F6) is a documentation fix only (correct
   the `scripts/sanity_mnist.py` docstring expectation), not a code change.
