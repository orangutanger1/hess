# DSN — Dynamic Subspace Newton

Second-order optimization restricted to the smallest dynamically-identified
subspace that predicts the true Newton step, with AdamW in the orthogonal
complement.

- Design: `docs/superpowers/specs/2026-07-27-dynamic-subspace-newton-design.md`
- Plan:   `docs/superpowers/plans/2026-07-27-dsn-core.md`
- Plan 2: `docs/superpowers/plans/2026-07-28-dsn-plan2-fixes.md`

**Result: negative.** `docs/superpowers/plans/2026-07-29-negative-result.md` —
the mechanism was instrumented at three independent points (does curvature
preconditioning pay; is recycling break-even; does the augmented-recycling gate
open) and returns nothing at all three. The pre-registered kill criterion is
defined on GPU tiers that were never run, so it has not been evaluated; the
document recommends against running them and explains why. The two supporting
measurements are `scripts/measure_rho.py` and `scripts/measure_faca_gate.py`.

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

## Demo script

    python scripts/sanity_mnist.py --optimizer adamw    # reference
    python scripts/sanity_mnist.py                      # DSN

`scripts/sanity_mnist.py` is a CPU smoke run on MNIST (it downloads data, so it
is not part of the test suite; `torchvision` comes with `.[dev]`). Both
optimizers converge; measured last-10-step mean loss over three seeds at the
script's defaults: DSN 0.2678 / 0.3010 / 0.2805, AdamW 0.1512 / 0.1781 / 0.1713.

## Cost per step

One closure evaluation for the gradient, one per curvature-vector product, and
one more after the step to evaluate the trust-region acceptance ratio on the
same batch. With `m_recycle > 0` a further `m` curvature products per step go to
re-projecting the recycled basis against the current operator.

## Known limitations

### Numerical

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
