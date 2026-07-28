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
    dsn = DSN([w_d], lr=1e-1, weight_decay=0.0, damping=1e-4, trust_radius=10.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    for _ in range(60):
        adam.zero_grad()
        logistic_loss(w_a, X, y).backward()
        adam.step()
        dsn.step(lambda: logistic_loss(w_d, X, y))

    with torch.no_grad():
        assert logistic_loss(w_d, X, y) < logistic_loss(w_a, X, y)


def test_recycling_stays_fresh_rather_than_going_stale():
    """Risk 1 from the spec: reuse high while residual climbs means staleness.

    Two gates, both empirically demonstrated to be capable of failing under a
    genuinely degraded scenario -- see the Task 9 fix-round report and
    docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md for the
    corruption experiments that trip each one. The plan's original text used
    ``reuse_frac > 0.2``, which is `m_recycle / (m_recycle + j)` with
    ``j <= k_max``; at m_recycle=5, k_max=10 the arithmetic floor is
    5/15 = 0.333, already above 0.2 regardless of whether recycling helps at
    all -- that assertion could not fail and is not used here.

    1. Recycling must pay for itself: at equal curvature-vector-product
       budget (both configurations exhaust k_max=10 new Lanczos vectors every
       step on this problem, so the budget really is equal), the mean
       rel_residual with recycling must beat a matched from-scratch
       (m_recycle=0) rebuild by a real margin, not merely be numerically
       present. Measured: 0.1010 (recycled) vs 0.3572 (fresh), a 3.54x edge;
       the 0.5x threshold leaves 1.77x of headroom below where recycling
       actually lands today. Continuously replacing the recycled state with
       random, wrongly-scaled vectors (simulating permanently-stale
       recycling) drives the mean to 1.063 -- 6x over the fresh baseline --
       and trips this assertion, proving it is not vacuous.
    2. The residual must not climb by more than an order of magnitude from
       the run's first ten steps to its last ten. Measured drift is 4.84x
       (mean head 0.033774, mean tail 0.163374); the 10x ceiling leaves
       2.07x of headroom above that and is not an arbitrary round number --
       it is the same "order of magnitude" bound `SubspaceResult.rel_residual`
       documents (src/dsn/subspace/base.py) for how optimistic the recycled-
       surrogate residual can be relative to the true dense one once
       reuse_frac > 0, which Task 9's watch items showed is every step here
       after the first.
    """
    X, y = ill_conditioned_logistic()
    d = X.shape[1]

    def run(m_recycle):
        w = torch.zeros(d, requires_grad=True)
        dsn = DSN([w], lr=1e-1, weight_decay=0.0, damping=1e-4, trust_radius=10.0,
                  builder=KrylovBuilder(k_max=10, m_recycle=m_recycle, tau=1e-2, damping=1e-4))
        residuals = []
        for _ in range(40):
            dsn.step(lambda: logistic_loss(w, X, y))
            residuals.append(dsn.telemetry.rel_residual)
        return residuals

    residuals = run(m_recycle=5)
    residuals_fresh = run(m_recycle=0)

    mean_recycled = sum(residuals) / len(residuals)
    mean_fresh = sum(residuals_fresh) / len(residuals_fresh)
    mean_head = sum(residuals[:10]) / 10
    mean_tail = sum(residuals[-10:]) / 10

    assert mean_recycled < 0.5 * mean_fresh
    assert mean_tail <= 10 * mean_head


def test_trust_region_does_not_collapse():
    """Risk 2 from the spec: complement fighting the trust region."""
    X, y = ill_conditioned_logistic()
    w = torch.zeros(X.shape[1], requires_grad=True)
    dsn = DSN([w], lr=1e-1, weight_decay=0.0, damping=1e-4, trust_radius=1.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    for _ in range(40):
        dsn.step(lambda: logistic_loss(w, X, y))

    assert dsn.telemetry.n_shrink < 30
    assert dsn.telemetry.trust_radius > 1e-6


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect, deferred to Plan 2 (see "
        "docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md F1/F2/F5): "
        "the lagged acceptance ratio rho = (prev_loss - loss_now) / predicted "
        "compares the loss of two DIFFERENT mini-batches whenever the objective "
        "changes between steps, so it is not a valid trust-region ratio outside "
        "a fixed full-batch objective. n_shrink climbs on nearly every step and "
        "trust_radius collapses. This is expected to fail today and should flip "
        "to passing the day the acceptance-ratio computation is fixed to use a "
        "same-batch reference (e.g. re-evaluating the previous batch's loss "
        "before advancing, or a variance-reduced / EMA'd ratio)."
    ),
)
def test_trust_region_collapses_under_minibatch_noise():
    """Risk 2 in the regime Gate 3 cannot reach: a *changing* objective.

    ``test_trust_region_does_not_collapse`` above always re-evaluates the
    same fixed closure, which is exactly the regime in which the lagged rho
    is a valid acceptance ratio -- so it cannot exercise this failure mode by
    construction (see the Task 9 fix-round report: 1.34e8 final trust_radius,
    1.34e14x headroom over its own 1e-6 threshold). This test draws a fresh
    random mini-batch from a fixed synthetic population every step (no
    dataset download -- tests may not download data) and reproduces the
    collapse measured on real MNIST in the original Task 9 report: trust
    region falling and n_shrink climbing without bound.
    """
    gen = torch.Generator().manual_seed(0)
    n_samples, d, cond, batch_size = 2000, 25, 100.0, 32
    scale = torch.logspace(0, torch.log10(torch.tensor(cond)), d)
    X_full = torch.randn(n_samples, d, generator=gen) * scale
    w_true = torch.randn(d, generator=gen)
    y_full = (X_full @ w_true > 0).to(torch.get_default_dtype())

    w = torch.zeros(d, requires_grad=True)
    dsn = DSN([w], lr=1e-1, weight_decay=0.0, damping=1e-4, trust_radius=1.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    for _ in range(200):
        idx = torch.randint(0, n_samples, (batch_size,), generator=gen)
        Xb, yb = X_full[idx], y_full[idx]
        dsn.step(lambda: logistic_loss(w, Xb, yb))

    n_shrink = dsn.telemetry.n_shrink
    trust_radius = dsn.telemetry.trust_radius
    assert n_shrink < 30 and trust_radius > 1e-2, (
        f"trust region collapsed under mini-batch noise: "
        f"n_shrink={n_shrink} (want <30), trust_radius={trust_radius:.3g} (want >1e-2)"
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
