# DSN — Dynamic Subspace Newton

Second-order optimization restricted to the smallest dynamically-identified
subspace that predicts the true Newton step, with AdamW in the orthogonal
complement.

- Design: `docs/superpowers/specs/2026-07-27-dynamic-subspace-newton-design.md`
- Plan:   `docs/superpowers/plans/2026-07-27-dsn-core.md`

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

    python scripts/sanity_mnist.py --optimizer adamw    # converging reference
    python scripts/sanity_mnist.py                      # DSN -- expected to diverge

`scripts/sanity_mnist.py` is a CPU smoke run on MNIST (it downloads data, so it
is not part of the test suite; `torchvision` comes with `.[dev]`). It uses a
mini-batch closure, so **the default `--optimizer dsn` run is expected to
diverge** — that is the documented limitation below, not a bug in the script.
It prints the same warning at startup.

## Known limitations

### Numerical

DSN's trust region is currently only valid under a **fixed, full-batch**
closure. Under real mini-batch training the acceptance ratio `rho` --
`(prev_loss - loss_now) / predicted` -- compares the loss of two
*different* batches, which is not a meaningful trust-region ratio. On a
small MLP trained on MNIST with mini-batches, this causes `trust_radius`
to collapse (observed down to ~1e-11) and the loss to diverge, while
`torch.optim.AdamW` on the same data converges normally. Root cause,
isolation experiments, and why removing the trust region is not the fix
are documented in
[`docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md`](docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md).
This is tracked by `tests/test_convergence.py::test_trust_region_collapses_under_minibatch_noise`,
marked `xfail(strict=True)`, and is deferred to Plan 2 -- **do not use DSN
with mini-batch closures until this is fixed.**

Recycling (`m_recycle > 0`) carries `HU` over from the previous iterate's
Hessian, so the recycled block of the projected curvature `T` is stale at
the current point, and the reported `rel_residual` telemetry (computed
against that stale `T`) hides it. Two independent measurements on the same
ill-conditioned logistic problem: at unequal basis width (recycled width
15 vs a fresh, from-scratch width-10 subspace), telemetry reports the
recycled arm 3.5x *better* while an independently measured true dense
residual says it is 3x *worse* (true 1.078 vs 0.357; the fresh arm's
reported and true values agree at 0.357, since a from-scratch subspace's
`rel_residual` is exact by construction, which is what validates the
measurement). At *equal* basis width (both width 15), recycling is worse
even by its own optimistic reported metric alone, no true residual needed:
0.1010 (recycled) vs 0.0644 (fresh). Recycling is not broken in the sense
of producing garbage steps -- DSN still converges on the fixed-batch
problems in this suite -- but it should not currently be assumed to
improve subspace accuracy; its only demonstrated benefit is spending fewer
curvature-vector products, not a better subspace. Details in
[`docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md`](docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md)
(F8), tracked by
`tests/test_convergence.py::test_recycling_stays_fresh_rather_than_going_stale`,
also `xfail(strict=True)`.

When the gradient is exactly zero DSN still moves: the builder returns an
empty subspace (`k=0`), but the AdamW complement's momentum term is not zero,
so a momentum-driven step is still applied and walks the iterate back off an
exact minimum (measured `‖d_comp‖ = 2.5e-3` immediately after landing on `x*`
of an exact quadratic). See F9 in the findings doc.

### API

`DSN` subclasses `torch.optim.Optimizer` but does not yet honor the whole
contract. In addition to the numerical limitations above:

- **`state_dict()` / `load_state_dict()` do not round-trip DSN's state.** This
  is the worst of these, because it looks like it worked: `state_dict()`
  returns `{'state': {}, 'param_groups': [...]}` with none of the AdamW
  moments (`m`, `v`, `t`), the trust radius, the pending prediction, or the
  recycled basis (`builder.U` / `HU`) in it. Loading into a fresh `DSN` and
  resuming silently restarts from zero state — measured after 5 steps, a
  "resumed" run had `adam.t = 0` (was 5) and `trust_radius = 1.0` (was 16.0),
  and its next step diverged from the un-resumed run by 0.134 in parameter
  space. Checkpoint-and-resume is not supported.
- **Multiple param groups are not supported.** All parameters are optimized as
  one flat vector using `param_groups[0]["lr"]`; a per-group `lr` is ignored.
  A single-group `lr_scheduler` does work (covered by
  `tests/test_optimizer.py::test_lr_scheduler_changes_the_lr_the_optimizer_actually_uses`).
- **`weight_decay` never reaches the param group.** It is stored on the
  optimizer and applied by `DSN._apply`, so it is absent from
  `param_groups[i]` and from `state_dict()`, and anything that inspects or
  rewrites the group's `weight_decay` will have no effect.
- **`step()` requires a closure**, unlike `torch.optim.Optimizer.step(closure=None)`.
  The closure is re-evaluated once per curvature-vector product.
- All parameters must share one dtype and device (the constructor raises
  otherwise; the AdamW complement keeps a single flat moment vector).
