# Plan 2: fixing F1–F9, plus F10 found on the way

Follow-up to `2026-07-27-dsn-core-task9-findings.md`, which is treated here as
ground truth and not re-litigated. Every defect it records is fixed. One further
defect (F10) surfaced while fixing F8 and is fixed too.

**Suite: 79 passed, 0 xfail** (from 61 passed / 2 xfail). Both
`xfail(strict=True)` gates were removed — not by relaxing them, but because the
defects they pinned are fixed and the claims they asserted turned out, in one
case, to be false. That case is F8, and it is the most important thing in this
document.

---

## Headline result: F1/F2/F5, mini-batch MNIST

The defect the findings doc led with, re-measured at its exact settings
(200 steps, lr 1e-2, k_max 8, m_recycle 4, batch 256):

| seed | DSN before | DSN after | AdamW | `trust_radius` before | after | `n_shrink` before | after |
|---|---|---|---|---|---|---|---|
| 1 | 9.2297 | **0.2678** | 0.1512 | 7.3e-12 | 2.0 | 117 | **6** |
| 2 | 5.8713 | **0.3010** | 0.1781 | 3.0e-8 | 2.0 | 109 | **5** |
| 3 | 4.6986 | **0.2805** | 0.1713 | 1.9e-6 | 2.0 | 102 | **7** |

This reproduces the findings doc's own F2 prediction for a same-batch ratio
(`n_shrink=7`, radius stable at 2.0) almost exactly. DSN converges; it does not
beat AdamW here, for reasons stated under "What is still true" below.

---

## Fixes

### F1 / F2 / F5 — the acceptance ratio spanned two mini-batches

**Chosen: same-batch reference.** After computing the step, DSN re-evaluates the
*same closure* at the trial point and forms `rho` from that. One extra forward
pass per step, no extra curvature products, and zero sampling noise in the ratio
by construction. `_pending` and the lagged ratio are gone.

Problem class → method → source: noisy acceptance ratio in finite-sum trust
region → require the ratio to be evaluated on a consistent sample →
inexact-restoration trust region for finite sums (Bellavia, Krejić & Morini) and
progressive-sampling subsampled Newton (Bollapragada, Byrd & Nocedal); STORM
(Chen, Menickelly & Scheinberg, *Math. Prog.* 2018) states the same requirement
as a probabilistic-accuracy condition. STORM's adaptive batch growth was
considered and rejected as architecturally unavailable: DSN receives a closure
and does not own the data loader, so it cannot resample.

The trust region was kept, per F5. Two further changes were needed before the
same-batch ratio actually produced convergence, and both are corrections of the
same kind of error the findings doc identified — *comparing a model against
something other than what the model describes*:

1. **The step is now rejected when the ratio says it made things worse.** A
   trust region that only shrinks its radius after a bad step does nothing about
   the bad step, which has already been applied. Before this, the radius stayed
   healthy under mini-batching but the loss still diverged (11.3 at 200 steps).
   Conn/Gould/Toint Alg. 6.1.1 and STORM both make acceptance conditional.
   `trust_accept=0.0` is the default: undo only steps that demonstrably raised
   the loss on their own batch. New telemetry field `n_reject`.
2. **`rho` scores the subspace step alone.** `predicted` models `d_sub`; the
   complement contributes no first-order term (F4) but its unmodeled curvature
   `d_compᵀ H d_comp` is positive, so including it biased every ratio downward
   and shrank the region for a reason the model never claimed to cover. `d_sub`
   is applied and scored on its own; the complement is applied afterwards
   regardless. A collapsing region therefore degrades DSN toward plain AdamW,
   which is the intended floor.

Regression test: `test_convergence.py::test_trust_region_survives_minibatch_noise`
(replaces the xfail'd `test_trust_region_collapses_under_minibatch_noise`).

*Threshold recalibrated, deliberately and with measurement.* The old test
demanded `n_shrink < 30` of 200. That was an aspiration written against a broken
implementation and never measured on a working one. A trust region doing its job
on a genuinely noisy objective shrinks often: on that synthetic stream the region
**binds on 144 of 200 steps** and shrinks on 48–57 across five seeds, a ~27%
rate that does not move when `damping` is swept from 1e-4 to 1.0. What separates
fixed from broken is whether the radius survives and whether the loss diverges,
so the test now asserts those directly (radius > 1e-2, final loss < 3x initial)
and keeps a shrink bound at 80, calibrated to measured behavior.

### F4 — `g ∈ span(W)` always, so the complement carries zero first-order descent

**Chosen: option 2 (documentation), plus the structural fix for the harm.**

Verified against `solve.py` as the prompt required: the residual accounting does
already cover the whole gradient. `newton_residual` measures `‖H(Wy) + g‖`, and
since `g ∈ span(W)`, the out-of-span part of that residual comes only from `HWy`
leaking out — no part of `g` is unaccounted for. So the subspace step is
genuinely responsible for 100% of the first-order descent, and no redesign of
what AdamW accumulates is needed. Option 1 was rejected on inspection: feeding
AdamW `g` projected out of `W` would feed it exactly zero, since `g ∈ span(W)`.

What F4 got right was the *harm*, and that needed code. The trust region scaled
`d_sub` but not `d_comp`, so a collapsing radius drove the descent-carrying part
to zero while the zero-descent part stayed at unit norm — which is why the loss
rose under collapse instead of stalling. **The radius now bounds the whole
step** (Conn/Gould/Toint §6.1: the constraint is `‖d‖ ≤ Δ`). The two parts are
orthogonal by construction, so `‖d‖² = ‖d_sub‖² + ‖d_comp‖²` exactly and one
common factor bounds both.

Regression test: `test_optimizer.py::test_the_complement_is_clipped_by_the_trust_region_too`.

### F7 — no upper clamp on the trust radius

`trust_max`, defaulting to `1e3 * trust_radius` so the bound scales with whatever
problem scale the caller set. Conn/Gould/Toint §6.1. Regression test:
`test_optimizer.py::test_trust_radius_is_clamped_from_above`.

### F8 — recycled directions stale; telemetry structurally blind to it

**Chosen: option 2, implemented at the root rather than at the display.** Three
changes, none of which leaves a stale quantity anywhere:

1. `HU` is recomputed as `H_now @ U` at the top of every `build` — GCRO-DR's
   `C = A_new U` (Parks, de Sturler et al., SIAM J. Sci. Comput. 28(5), 2006).
   `T` is now exactly `WᵀHW` at the current iterate at every `reuse_frac`.
2. `lanczos_iter` retains `HQ`, the curvature images it already computed and
   used to discard. `newton_residual` is now `‖HW y + g‖` — a true residual
   built from images under the current operator, with no `T` and no `beta_next`
   leak estimate. **No quantity in it crosses a step boundary**, so the class of
   bug where the metric improves as the subspace degrades is structurally
   impossible, not merely absent.
3. The retained basis is re-*validated*, not just re-projected. A direction that
   was a converged Ritz pair at the previous iterate need not still be one here;
   the curvature image is now exact but the direction is what went stale.
   Directions failing the convergence test against the current operator are
   dropped, and if none survive, the step rebuilds from scratch — the prompt's
   "fall back to fresh rather than recycle stale directions" path. This is free:
   `HU` is already computed.

This was load-bearing, not belt-and-braces. On the suite's own ill-conditioned
logistic gate, with (1) and (2) but not (3), DSN reached loss 1.4e-3 while
`m_recycle=0` reached 0.0. With (3) both reach 2.4e-7 — DSN beats AdamW's
1.3e-4 by three orders of magnitude, and the gate
`test_dsn_reaches_lower_loss_than_adamw_on_logistic_regression` passes again.

Regression tests, all in `test_krylov_builder.py`:

- `test_recycled_reported_residual_equals_the_true_dense_residual` — inverts the
  old `..._is_optimistic_vs_true_dense_residual`, which asserted a gap of up to
  10x as correct behavior.
- `test_projected_curvature_is_exact_even_on_a_recycled_step` — `T == WᵀHW`
  against a drifting operator.
- `test_a_frozen_recycled_basis_cannot_hide_behind_the_reported_residual` —
  freezes the basis (maximal staleness, the case the findings doc showed passed
  *every* existing gate while understating its true residual by 85x) and asserts
  the reported residual tracks an independently computed dense one anyway.

**The old F8 gate asserted something false, and this is the part worth reading.**
`test_recycling_stays_fresh_rather_than_going_stale` demanded the recycled arm
beat an equal-width fresh basis by 2x on the reported residual. It does not, and
it never did. With both quantities exact, the honest and defensible claim is
*parity*, and parity is what the replacement gates assert
(`test_recycling_matches_a_fresh_basis_of_equal_width`, and at the optimizer
level `test_convergence.py::test_recycling_never_costs_dsn_accuracy_versus_an_equal_width_fresh_basis`).

**Recycling does not save curvature products.** Refreshing `m` directions costs
exactly `m` products and saves at most the `m` new Lanczos directions they stand
in for, so exact recycling is break-even by construction. Measured at equal
tolerance, recycled/fresh product ratios were 1.03–1.11 across n=40/60/80,
cond 1e2–1e4, 3 seeds each. The saving measured before Plan 2 was an artifact of
the stale shortcut: the optimistic residual tripped the `tau` stopping rule
early. `test_recycling_reduces_hvps_on_a_slowly_changing_operator` (which
demanded a 20% saving, and whose docstring called it "the point of the whole
method") is replaced by
`test_recycling_costs_no_more_hvps_than_a_fresh_rebuild_at_equal_accuracy`, a
1.25x budget bound — and, unlike the old one, both arms are given enough `k_max`
to actually reach `tau`, so it compares products at equal accuracy instead of
comparing which arm stopped early.

### F9 — a zero gradient still produced a momentum-driven step

The complement is skipped when the subspace is empty *and* `‖g‖ <
stationary_eps` (default 1e-10, matching `lanczos_iter`'s breakdown tolerance,
which is what makes the builder return `k=0` in the first place). Traced on the
exact quadratic the findings doc used: step 10 lands on `x*`, step 11 sees
`‖g‖ = 1.6e-15`, reports `k=0`, applies `‖d_comp‖ = 0`, and steps 12+ do not
move. An exact optimum is now a fixed point. Regression test:
`test_optimizer.py::test_an_exact_optimum_is_a_fixed_point_of_the_iteration`.

### F3, F6 — no code change required

F3 is an analysis, already correct. F6 is a documentation fix; the
`scripts/sanity_mnist.py` docstring is corrected.

### API contract violations

All four fixed against `torch.optim.Optimizer`'s contract:

- `state_dict()` / `load_state_dict()` round-trip the AdamW moments, the trust
  radius and bound, the telemetry counters, and the builder's recycled basis
  (via new `KrylovBuilder.state_dict` / `load_state_dict`, duck-typed so a
  third-party builder without them still works). Tested on the resumed
  *trajectory*, not on attributes, so a field added later and not checkpointed
  still fails the test.
- Multiple param groups: per-group `lr` and `weight_decay` are expanded over the
  elements each group owns. `add_param_group` extends the complement's moments
  with zeros and resets the recycled basis.
- `weight_decay` lives in the param group, where schedulers and `state_dict()`
  can see it; rewriting `param_groups[i]["weight_decay"]` now changes behavior.
- `step(closure=None)` matches the base signature and raises a specific
  `ValueError` rather than silently taking a different step. DSN cannot form a
  curvature-vector product from a pre-populated `.grad`, so requiring the
  closure is inherent; only the silent-signature-mismatch part was fixable.

---

## F10 — deflating unconverged Ritz pairs (found while fixing F8)

Making the residual honest exposed a second defect underneath it. With the
metric no longer optimistic, the shipped `rank_by="contribution"` ranking
produced a recycled subspace **20–44x worse** than an equal-width fresh basis,
and the measured ordering of the three `rank_by` heuristics *reversed* — the
ranking `test_large_eig_ranking_is_the_worst_of_the_three` pinned as worst
measured best.

Problem class → method → source: which spectral directions may be deflated →
only converged Ritz pairs, tested by Ritz residual → Saad, *Numerical Methods
for Large Eigenvalue Problems*, ch. 6; the same convergence test governs which
harmonic Ritz vectors GMRES-DR (Morgan) and GCRO-DR retain.

Root cause: the ranking heuristics select by *usefulness to the step*, which is
a different property from *being close to an invariant direction*. Deflating a
direction that is not near-invariant does not shrink the effective spectrum; it
only steals width from the new Krylov vectors. Measured relative Ritz residuals
of the directions the shipped ranking retained: **0.1 to 2.8** — not converged
by any margin. `large_eig` scored 1e-15 to 1e-6, which is the entire reason it
measured better; Lanczos converges to extremal eigenvalues first, so
largest-|λ| Ritz pairs are the converged ones.

Fix: a convergence gate, `‖H u − θu‖ ≤ ritz_tol · |θ|` with `ritz_tol=1e-3`,
applied both when directions are retained and again against the current operator
when they are reused. Ranking then chooses among directions deflation theory
admits, rather than overriding it. The `rank_by` argument the old test made
about *step accuracy* (the Newton step weights by 1/λ) remains correct — it was
never an argument about deflatability.

`ritz_tol=1e-3` is calibrated by direct measurement of the objective, not by
theory alone. Recycled/fresh residual ratio at equal width, n=40 cond=100
k_max=20 m=8, three seeds:

| `ritz_tol` | 1e-6 | **1e-3** | 1e-2 | 5e-2 | 1e-1 | ∞ (no gate) |
|---|---|---|---|---|---|---|
| ratio | 1.2–2.3 | **0.90–1.21** | 1.23–1.42 | 6.1–13.4 | 11.2–32.5 | 18.5–43.9 |

1e-3 sits at parity with ~50x margin to the harmful regime, and the converged /
unconverged populations it separates differ by five orders of magnitude, so it
is not a knife edge. Looser is harmful; tighter over-drops and narrows the basis.

Consequence, stated plainly: **recycling now declines to engage on problems
where no Ritz pair converges** — roughly `k_max ≤ 10` on everything measured,
including the MNIST MLP, where `reuse_frac` reads 0.00 throughout. That is the
correct behavior (the alternative is a 20–44x worse subspace), but it means
`m_recycle` is inert in DSN's small-`k_max` operating regime.

Regression tests: `test_only_converged_ritz_directions_are_ever_deflated`
(parametrized over all three rankings) and
`test_nothing_is_recycled_when_no_ritz_pair_has_converged`. These replace
`test_large_eig_ranking_is_the_worst_of_the_three`, whose claim does not survive
honest measurement: with the gate in place all three rankings land within noise
of each other and of the equal-width fresh baseline, so no ordering among them
is assertable.

---

## What is still true, and what it would take to change it

- **Recycling buys nothing measurable right now.** Parity at equal width, ~5%
  more curvature products, inert at small `k_max`. Its previously-measured
  benefit was the stale shortcut. Recovering a real advantage needs *augmented*
  Krylov — construct `W` so that `span(W) ⊇ K_k(H, g)` by construction, which
  makes "never worse than a fresh basis of the same `k`" a theorem rather than a
  measurement (Saad, "Analysis of augmented Krylov subspace methods", SIMAX
  1997; GCRO/GCRO-DR are in this family). Deflation, which is what is
  implemented, has no such guarantee: it removes directions from the Krylov
  iteration and can only be justified per-direction. Not implemented, and it is
  a redesign of the builder core rather than a patch.
- **DSN converges under mini-batching but does not beat AdamW there** (MNIST
  0.27–0.30 vs 0.15–0.18). The ratio is valid now; the *curvature* is still a
  single mini-batch's. The subsampled-Newton literature's answer is a sample
  size that grows with the iteration, which DSN cannot do without owning the
  data loader.
- **Two thresholds were recalibrated, both flagged above and in the test
  docstrings**: the mini-batch `n_shrink` bound (aspiration → measured
  equilibrium) and the residual-drift bound in
  `test_recycled_residual_does_not_drift_upward_across_a_run` (10x → 25x,
  because the quantity being bounded changed from an optimistic residual to a
  true one; head/tail moved from 0.0338/0.1634 to 0.0814/0.8852). That gate was
  never a staleness detector, per the findings doc; staleness is now pinned
  directly by the frozen-basis test.

## Final verification

`python -m pytest` — **79 passed**, 0 failed, 0 xfail, 7.45s.

MNIST comparison run once at the end of the batch, three seeds each, table at
the top of this document.
