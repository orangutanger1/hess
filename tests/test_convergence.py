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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect, deferred to Plan 2 (see "
        "docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md F8): "
        "HU is carried over from the PREVIOUS iterate's Hessian "
        "(src/dsn/subspace/krylov.py:113-117,144-145), so the recycled block "
        "of T is not W'HW at the current point, and rel_residual (computed "
        "against that stale T, see src/dsn/subspace/base.py) is optimistic "
        "for the recycled arm but exact for the fresh arm. At equal basis "
        "width (both arms width 15: m_recycle=5,k_max=10 vs m_recycle=0,"
        "k_max=15) the recycled arm's reported residual (0.1010) is WORSE "
        "than the fresh arm's (0.0644), not the 2x-better this assertion "
        "demands. Expected to fail today and should flip to passing the day "
        "the recycled block of T is recomputed against the current step's "
        "Hessian instead of carried over from the previous step's."
    ),
)
def test_recycling_stays_fresh_rather_than_going_stale():
    """Risk 1 from the spec: reuse high while residual climbs means staleness.

    Compares recycling (m_recycle=5, k_max=10 -> basis width 15 once warmed
    up) against a from-scratch rebuild at the SAME basis width
    (m_recycle=0, k_max=15), so the comparison isolates recycling quality
    from basis width. An earlier version of this test compared m_recycle=5
    against m_recycle=0,k_max=10 (width 15 vs width 10), which confounded
    "recycling helps" with "a wider basis helps" -- see the Task 9 fix
    round 2 report.

    This asserts the recycled arm should beat the equal-width fresh
    baseline by 2x on the reported (T-relative, not true-Hessian-relative)
    residual. It does not: mean_recycled=0.1010 vs mean_fresh=0.0644 --
    fresh is *better* than recycled at equal width, not worse. The
    assertion fails, by design, and this test is xfail(strict=True): it
    pins a real defect (F8 in
    docs/superpowers/plans/2026-07-27-dsn-core-task9-findings.md) rather
    than asserting a false claim as green.

    Root cause (F8): `HU` is carried over from the *previous* iterate's
    Hessian (src/dsn/subspace/krylov.py:113-117,144-145), so the recycled
    block of `T` is not `W'HW` at the current point, and `rel_residual`
    measures against that stale `T`. The fresh arm has reuse_frac == 0,
    where `rel_residual` is exact by construction
    (src/dsn/subspace/base.py); the recycled arm's is not. Computing a true
    dense residual inside this test to quantify that gap directly was
    explicitly ruled out (no extra curvature products in the test); F8
    reports an independently-measured true-residual comparison instead.

    n_hvp is NOT equal between the two arms at these widths, so this is an
    equal-WIDTH comparison, not an equal-budget one: recycled uses 10 new
    Lanczos products on every one of the 40 steps (400 total; tau=1e-2 is
    never satisfied early once recycling is warmed up), while fresh at
    k_max=15 mostly uses 15, occasionally fewer near convergence, totaling
    591. Recycling does *worse* on the reported metric while also spending
    fewer HVPs -- the two numbers move in opposite directions from what
    recycling is supposed to buy.

    The residual-drift half of Risk 1 (residual must not climb more than
    10x from the run's first ten steps to its last ten, measured on the
    recycled arm alone) used to be a second assertion in this test, where
    it was dead code -- the assertion below raises first, in both normal
    and --runxfail mode, so it was never evaluated in either. It now lives
    in ``test_recycled_residual_does_not_drift_upward_across_a_run`` as a
    passing test, which is the only way it provides live regression
    coverage.
    """
    X, y = ill_conditioned_logistic()
    d = X.shape[1]

    def run(m_recycle, k_max):
        w = torch.zeros(d, requires_grad=True)
        dsn = DSN([w], lr=1e-1, weight_decay=0.0, trust_radius=10.0,
                  builder=KrylovBuilder(k_max=k_max, m_recycle=m_recycle, tau=1e-2, damping=1e-4))
        residuals = []
        for _ in range(40):
            dsn.step(lambda: logistic_loss(w, X, y))
            residuals.append(dsn.telemetry.rel_residual)
        return residuals

    residuals = run(m_recycle=5, k_max=10)
    residuals_fresh = run(m_recycle=0, k_max=15)

    mean_recycled = sum(residuals) / len(residuals)
    mean_fresh = sum(residuals_fresh) / len(residuals_fresh)

    assert mean_recycled < 0.5 * mean_fresh, (
        f"F8: recycled residual not better than an equal-width fresh basis: "
        f"mean_recycled={mean_recycled:.4f} (want < {0.5 * mean_fresh:.4f} "
        f"i.e. < half of mean_fresh={mean_fresh:.4f})"
    )


def test_recycled_residual_does_not_drift_upward_across_a_run():
    """Risk 1 from the spec, drift half: reuse high while residual climbs.

    The recycled arm alone, over 40 steps: the mean reported residual of
    the last ten steps must not exceed 10x that of the first ten. This is
    the multiplicative bound Task 9 fix round 1 introduced (replacing an
    unfailable additive `+0.5` slack) under the "STRENGTHEN ALL THREE"
    ruling. It lived as a second assertion inside
    ``test_recycling_stays_fresh_rather_than_going_stale`` until the final
    review found it dead: that test's first assertion is xfail(strict=True)
    and raises before this one is reached, in both normal and --runxfail
    mode, so this bound was evaluated in neither. Split out here so it
    actually runs.

    Measured on this configuration: mean_head=0.0338, mean_tail=0.1634,
    bound 0.33774 -- the tail is 4.8x the head, so the 10x bound has real
    headroom while still catching a regression that lets the residual run
    away. Note this gate is NOT a staleness detector: freezing the recycled
    basis mid-run leaves it passing (see the Task 9 findings doc).
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

    assert mean_tail <= 10 * mean_head, (
        f"recycled residual drifted upward across the run: "
        f"mean_head={mean_head:.4f} mean_tail={mean_tail:.4f} "
        f"(want tail <= {10 * mean_head:.5f})"
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
    dsn = DSN([w], lr=1e-1, weight_decay=0.0, trust_radius=1.0,
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
