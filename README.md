# DSN — Dynamic Subspace Newton

Second-order optimization restricted to the smallest dynamically-identified
subspace that predicts the true Newton step, with AdamW in the orthogonal
complement.

## Status: negative result

The mechanism was instrumented at three independent points and returns nothing
at all three.

| question | instrument | answer |
|---|---|---|
| Does curvature preconditioning pay on this trajectory? | `scripts/measure_rho.py` | No. The payoff gate caps at 0.717 against a threshold of 1, at every checkpoint. |
| Does cross-step recycling save curvature products? | Plan 2 | No. Break-even by construction; measured 1.03–1.11x a fresh rebuild. |
| Does the augmented-recycling break-even gate open? | `scripts/measure_faca_gate.py` | No. Closed or vacuous at every pair, both seeds. |

DSN converges and does not beat AdamW. All ten F-defects found in Plan 2 are
fixed, so the shortfall is not attributable to a bug.

The pre-registered kill criterion is defined on FLOPs-to-target-loss on the
CIFAR-10 and WikiText tiers, neither of which was run, so it **has not been
evaluated**. The writeup recommends against spending that budget and explains
why; that decision is the project owner's.

- Result:  `docs/superpowers/plans/2026-07-29-negative-result.md`
- Design:  `docs/superpowers/specs/2026-07-27-dynamic-subspace-newton-design.md`
- Plan:    `docs/superpowers/plans/2026-07-27-dsn-core.md`
- Plan 2:  `docs/superpowers/plans/2026-07-28-dsn-plan2-fixes.md`
- Task 9 findings (F1–F9): `docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md`
- FACA gate measurement:   `docs/superpowers/plans/2026-07-28-faca-gate-measurement.md`

## Install

    pip install -e ".[dev]"

## Test

    python -m pytest -q

All numerical tests run in float64; see `tests/conftest.py`.

## Use

    from dsn import DSN
    from dsn.subspace import KrylovBuilder

    opt = DSN(model.parameters(), lr=1e-2,
              builder=KrylovBuilder(k_max=8, m_recycle=4, tau=0.1))

    def closure():
        return loss_fn(model(x), y)

    opt.step(closure)          # closure is re-evaluated per curvature product
    print(opt.telemetry)       # k, n_hvp, rel_residual, reuse_frac, ...

`closure` must rebuild the loss graph on every call.

## Scripts

None of these are part of the test suite — they download MNIST, and none of
them import anything from `src/dsn` that the suite does not already cover.
`torchvision` comes with `.[dev]`.

### Demo

    python scripts/sanity_mnist.py --optimizer adamw    # reference
    python scripts/sanity_mnist.py                      # DSN

CPU smoke run. Both optimizers converge; measured last-10-step mean loss over
three seeds at the script's defaults: DSN 0.2678 / 0.3010 / 0.2805, AdamW
0.1512 / 0.1781 / 0.1713.

### Measurement

Instruments for the two gates in "Status" above. Measurement only — neither
touches `src/` or contains solver code; both reuse `lanczos_iter` and the
optimizer as shipped.

    python scripts/measure_rho.py --steps 200 --k 100 --micro 24 --out rho.json

    python scripts/measure_faca_gate.py --steps 60 --k-max 8                        # shipped config
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8          # recycling engages
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --fixed-batch
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --seed 1

At the shipped `--k-max 8` the FACA gate is dead for a trivial reason (at most
one direction is ever carried, so `m_eff - 1 < 0` always); the `k_max=40` runs
are the ones the verdict rests on.

## Cost per step

One closure evaluation for the gradient, one per curvature-vector product, and
one more after the step to evaluate the trust-region acceptance ratio on the
same batch. With `m_recycle > 0` a further `m` curvature products per step go to
re-projecting the recycled basis against the current operator.

## Known limitations

### Numerical

**Curvature preconditioning does not pay on this problem, at any point in
training.** Measured directly rather than inferred from a loss curve. With

    lambda_g   = ||g||^2 / (g' M^-1 g)
    lambda_Sig = sqrt( tr(M Sigma) / tr(M^-1 Sigma) )
    rho        = (lambda_Sig / lambda_g)^2

the gate opens at `rho >= 1`. (This `rho` is the payoff gate, unrelated to the
trust-region acceptance ratio also called `rho` below and in
`optimizer.py`.) Below 1, the gradient sits in directions sharper than the ones
the gradient noise occupies, so dividing by curvature amplifies noise faster
than signal.

On DSN's own MNIST trajectory `lambda_Sig < lambda_g` at every checkpoint, and
`rho` maxes at **0.717** over all 72 `(step, delta, k)` cells. It is not static
— at the converged `delta=1e-2` it runs 0.154 at step 0, peaks at 0.650 near
step 120, and falls back to 0.162 by step 199 — but the whole sweep lives in
`[0.045, 0.72]` and never crosses. Split-half spread on `Sigma` is ~±30%
relative, far short of the ~2x that would be needed to reach 1 anywhere.

This is upstream of everything below. A cheaper or fresher subspace does not
help if the preconditioner it builds has negative expected value, which is why
fixing all ten F-defects did not change the outcome.

Measured against `M = |H| + delta`, not `H`: the exact Hessian here is
indefinite (`neg_frac`, the `g^2`-weighted mass on negative curvature, is 0.404
at step 0), so `H^-1` is not a preconditioner and `g' H^-1 g` can go negative.
`M` is what `solve.subspace_newton` inverts, so `M` is what "curvature
preconditioning" means for this code. The same fact independently breaks the
FACA gate's contraction premise — `||H d + g||` is the residual of a system the
solver is deliberately not solving, so it has no reason to decrease in `k`, and
it doesn't.

Converged for `delta` in {1e-2, 1e-1} (stable to ~1% across `k` =
25/50/75/100). `delta=1e-3`, the shipped default, is not converged at `k=100` —
expected, since `lambda_g` is a harmonic mean and Lanczos resolves the smallest
eigenvalues last — but it is bracketed by the converged values and is also
below 1 everywhere.

**Recycling (`m_recycle > 0`) does not reduce curvature-product count, and often
does not engage at all.** Both follow from paying for it honestly, and neither
is a bug:

- Refreshing `m` recycled directions against the current operator costs exactly
  `m` products and saves at most the `m` new Lanczos directions they stand in
  for, so exact recycling is break-even by construction. Measured at equal
  tolerance on drifting SPD operators it runs 1.03–1.11x a fresh rebuild's
  product count. The savings reported before Plan 2 came from carrying the
  curvature images over from the previous iterate, whose optimistic residual
  tripped the `tau` stopping rule early.
- A direction is deflated only once its Ritz pair has converged, re-checked
  against the *current* operator every step. On problems where the iterate moves
  quickly, or where `k_max` is too small for any Ritz pair to converge (roughly
  `k_max <= 10` on the problems measured, including the MNIST MLP in
  `scripts/sanity_mnist.py`), nothing qualifies and DSN rebuilds from scratch.
  Deflating unconverged directions was measured to make the subspace 20–44x
  worse than an equal-width fresh one, so declining is the correct behavior, not
  a missed opportunity.

At equal basis width with directions that do qualify, recycling reaches parity
with a from-scratch basis — not an improvement. Recovering a real advantage
would need *augmented* Krylov (span W ⊇ the fresh Krylov space by construction,
Saad, SIMAX 1997) rather than deflation; that is not implemented.

**DSN converges under mini-batch closures but does not beat AdamW on them.** On
MNIST it reaches 0.27–0.30 against AdamW's 0.15–0.18. The acceptance ratio is
valid (same-batch by construction), but the curvature still comes from a single
mini-batch, and the subsampled-Newton literature is clear that this needs a
sample size that grows with the iteration. DSN cannot do that: it does not own
the data loader and cannot resample, so the batch size is the caller's to
choose. On a deliberately harsh synthetic stream (batch 32, dimension 25,
condition 100) DSN ends between 0.29 and 1.75 across five seeds against AdamW's
0.16–0.59.

**The trust region governs the Newton step, not the complement.** `d_sub` is
scored by `rho` and undone when it raises the loss on its own batch; the AdamW
complement is applied either way, so a collapsing trust region degrades DSN
toward plain AdamW rather than toward a stalled iterate. The whole step is still
bounded by the radius.

**`rel_residual` is a true residual** — `‖H d + g‖ / ‖g‖` against the current
operator — at every `reuse_frac`. Before Plan 2 it was computed against the
projected curvature `T`, whose recycled block came from the previous iterate, so
it understated the truth by up to 85x and *improved* as the subspace degraded.

### API

- All parameters must share one dtype and device (the constructor raises
  otherwise; the AdamW complement keeps a single flat moment vector).
- **`step()` requires a closure.** The signature matches
  `torch.optim.Optimizer.step(closure=None)`, but DSN cannot form a
  curvature-vector product from a pre-populated `.grad`, so passing nothing
  raises rather than silently taking a different step.
- Multiple param groups are supported (per-group `lr` and `weight_decay` are
  honored), but all parameters are still optimized as one flat vector, and the
  AdamW complement keeps a single global step counter `t` rather than a
  per-parameter one.
- `weight_decay` is applied *after* the subspace step rather than before it as
  `torch.optim.AdamW` does, so the acceptance ratio scores the Newton step
  alone. The difference is second order in `lr * weight_decay`.
