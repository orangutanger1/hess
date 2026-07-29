import pytest
import torch

from dsn import DSN
from dsn.subspace import KrylovBuilder


def ill_conditioned_logistic(n_samples=200, d=25, cond=100.0, seed=0):
    torch.manual_seed(seed)
    scale = torch.logspace(0, torch.log10(torch.tensor(cond)), d)
    X = torch.randn(n_samples, d) * scale
    w_true = torch.randn(d)
    y = (X @ w_true > 0).to(torch.get_default_dtype())
    return X, y


def logistic_loss(w, X, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y)


def test_dsn_reaches_lower_loss_than_adamw_on_logistic_regression():
    X, y = ill_conditioned_logistic()
    d = X.shape[1]

    w_a = torch.zeros(d, requires_grad=True)
    w_d = torch.zeros(d, requires_grad=True)

    adam = torch.optim.AdamW([w_a], lr=1e-1, weight_decay=0.0)
    dsn = DSN([w_d], lr=1e-1, weight_decay=0.0, trust_radius=10.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    for _ in range(60):
        adam.zero_grad()
        logistic_loss(w_a, X, y).backward()
        adam.step()
        dsn.step(lambda: logistic_loss(w_d, X, y))

    with torch.no_grad():
        assert logistic_loss(w_d, X, y) < logistic_loss(w_a, X, y)


def test_recycling_never_costs_dsn_accuracy_versus_an_equal_width_fresh_basis():
    """F8 at the optimizer level: recycling must not make DSN's steps worse.

    Replaces the xfail(strict=True) ``test_recycling_stays_fresh_rather_than_going_stale``,
    which demanded the recycled arm beat an equal-width fresh basis by 2x on
    the reported residual. That claim was never true; what made it *look*
    plausible was that `rel_residual` was computed against the same stale `T`
    that corrupted the step, so the metric improved as the subspace degraded
    (85x optimism measured at maximal staleness). Both quantities are exact
    now, so the honest and defensible claim is parity, and this asserts parity
    on the two things a caller actually gets: the residual DSN achieves and the
    loss it reaches.

    The residual comparison here is between two real DSN runs on the same
    problem, so the arms follow different trajectories; the bound is loose
    enough for that and tight enough to catch a recycled arm that is
    systematically worse, which is what the defect produced.
    """
    X, y = ill_conditioned_logistic()
    d = X.shape[1]

    def run(m_recycle, k_max):
        w = torch.zeros(d, requires_grad=True)
        dsn = DSN([w], lr=1e-1, weight_decay=0.0, trust_radius=10.0,
                  builder=KrylovBuilder(k_max=k_max, m_recycle=m_recycle,
                                        tau=1e-2, damping=1e-4))
        residuals = []
        for _ in range(40):
            dsn.step(lambda: logistic_loss(w, X, y))
            residuals.append(dsn.telemetry.rel_residual)
        with torch.no_grad():
            return residuals, float(logistic_loss(w, X, y))

    residuals, loss_recycled = run(m_recycle=5, k_max=10)
    residuals_fresh, loss_fresh = run(m_recycle=0, k_max=15)

    mean_recycled = sum(residuals) / len(residuals)
    mean_fresh = sum(residuals_fresh) / len(residuals_fresh)

    assert mean_recycled <= 2.0 * mean_fresh, (
        f"F8: recycling degrades the residual against an equal-width fresh "
        f"basis: mean_recycled={mean_recycled:.4f} mean_fresh={mean_fresh:.4f}"
    )
    assert loss_recycled <= 10.0 * max(loss_fresh, 1e-12), (
        f"F8: recycling degrades the loss DSN reaches: "
        f"recycled={loss_recycled:.3e} fresh={loss_fresh:.3e}"
    )


def test_recycled_residual_does_not_drift_upward_across_a_run():
    """Risk 1 from the spec, drift half: reuse high while residual climbs.

    The recycled arm alone, over 40 steps: the mean reported residual of the
    last ten steps must not exceed 25x that of the first ten.

    Recalibrated in Plan 2, from a 10x bound, because the quantity being
    bounded changed meaning. `rel_residual` is now the TRUE residual against
    the current operator rather than one computed against a stale `T`, and the
    true value is both larger and more mobile than the optimistic one this
    bound was originally fitted to (measured head/tail 0.0338/0.1634 = 4.8x
    before, 0.0814/0.8852 = 10.9x after). The gate is also no longer
    load-bearing for staleness, which is now pinned directly and far more
    sharply by
    ``test_krylov_builder.py::test_a_frozen_recycled_basis_cannot_hide_behind_the_reported_residual``
    -- that one compares the reported residual against an independently
    computed dense one, which is the measurement the Task 9 findings doc
    identified as the only kind that can detect staleness at all.
    """
    X, y = ill_conditioned_logistic()
    d = X.shape[1]

    w = torch.zeros(d, requires_grad=True)
    dsn = DSN([w], lr=1e-1, weight_decay=0.0, trust_radius=10.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))
    residuals = []
    for _ in range(40):
        dsn.step(lambda: logistic_loss(w, X, y))
        residuals.append(dsn.telemetry.rel_residual)

    mean_head = sum(residuals[:10]) / 10
    mean_tail = sum(residuals[-10:]) / 10

    assert mean_tail <= 25 * mean_head, (
        f"recycled residual drifted upward across the run: "
        f"mean_head={mean_head:.4f} mean_tail={mean_tail:.4f} "
        f"(want tail <= {25 * mean_head:.5f})"
    )


def test_trust_region_does_not_collapse():
    """Risk 2 from the spec: complement fighting the trust region."""
    X, y = ill_conditioned_logistic()
    w = torch.zeros(X.shape[1], requires_grad=True)
    dsn = DSN([w], lr=1e-1, weight_decay=0.0, trust_radius=1.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    for _ in range(40):
        dsn.step(lambda: logistic_loss(w, X, y))

    assert dsn.telemetry.n_shrink < 30
    assert dsn.telemetry.trust_radius > 1e-6


def test_trust_region_survives_minibatch_noise():
    """F1/F2 regression: the trust region under a *changing* objective.

    ``test_trust_region_does_not_collapse`` above always re-evaluates the same
    fixed closure, which is exactly the regime in which the pre-Plan-2 lagged
    rho was a valid acceptance ratio -- so it could not exercise this failure
    mode by construction. This test draws a fresh random mini-batch from a fixed
    synthetic population every step (no dataset download -- tests may not
    download data) and reproduces the regime the Task 9 report measured on real
    MNIST.

    Measured before the fix: trust_radius 1.22e-4 and falling, n_shrink 99 of
    200, and the loss rising by an order of magnitude. Measured after: the
    radius holds at 0.125-0.25 across 5 seeds and the loss no longer runs away.

    On the `n_shrink` bound. The original version of this test demanded
    `n_shrink < 30`, which was an aspiration written against a broken
    implementation, never a measurement of a working one. A trust region that
    is doing its job on a genuinely noisy objective shrinks often: measured
    here, the region binds on 144 of 200 steps and shrinks on 48-57 of them
    across 5 seeds, a ~27% rate that is ordinary trust-region equilibrium and
    does not move with `damping` (swept 1e-4 to 1.0). What distinguishes fixed
    from broken is not the shrink count but whether the radius survives and
    whether the loss diverges, so those are what this asserts, with the shrink
    count kept as a bound calibrated to measured behavior. See the Plan 2 doc,
    F1/F2.
    """
    gen = torch.Generator().manual_seed(0)
    n_samples, d, cond, batch_size = 2000, 25, 100.0, 32
    scale = torch.logspace(0, torch.log10(torch.tensor(cond)), d)
    X_full = torch.randn(n_samples, d, generator=gen) * scale
    w_true = torch.randn(d, generator=gen)
    y_full = (X_full @ w_true > 0).to(torch.get_default_dtype())

    w = torch.zeros(d, requires_grad=True)
    dsn = DSN([w], lr=1e-1, weight_decay=0.0, trust_radius=1.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    with torch.no_grad():
        first = float(logistic_loss(w, X_full, y_full))

    for _ in range(200):
        idx = torch.randint(0, n_samples, (batch_size,), generator=gen)
        Xb, yb = X_full[idx], y_full[idx]
        dsn.step(lambda: logistic_loss(w, Xb, yb))

    with torch.no_grad():
        last = float(logistic_loss(w, X_full, y_full))

    n_shrink = dsn.telemetry.n_shrink
    trust_radius = dsn.telemetry.trust_radius

    assert trust_radius > 1e-2, (
        f"trust region collapsed under mini-batch noise: "
        f"trust_radius={trust_radius:.3g} (want >1e-2, was 1.22e-4 before the fix)"
    )
    # The headline defect: the loss RISING by an order of magnitude over
    # training. 3x the starting loss is far above anything measured after the
    # fix (worst of 5 seeds: 1.75 against a 0.693 start) and far below the
    # 7x-13x rise the defect produced.
    assert last < 3.0 * first, (
        f"loss diverged under mini-batch noise: {first:.4f} -> {last:.4f}"
    )
    assert n_shrink < 80, (
        f"trust region shrinking far more than its measured equilibrium: "
        f"n_shrink={n_shrink} of 200 (measured 48-57 across 5 seeds)"
    )


def test_trains_a_small_mlp_on_synthetic_data():
    """No network access: synthetic classification, not MNIST."""
    torch.manual_seed(0)
    X = torch.randn(128, 10)
    y = (X[:, 0] * X[:, 1] > 0).long()
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)
    )
    dsn = DSN(model.parameters(), lr=1e-2, weight_decay=0.0,
              builder=KrylovBuilder(k_max=6, m_recycle=3, tau=0.1))

    def closure():
        return torch.nn.functional.cross_entropy(model(X), y)

    first = float(closure().detach())
    for _ in range(50):
        dsn.step(closure)
    last = float(closure().detach())

    assert last < first
    assert dsn.telemetry.n_fallback == 0
