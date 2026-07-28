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

## Known limitations

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
