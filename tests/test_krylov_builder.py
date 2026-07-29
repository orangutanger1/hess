import pytest
import torch

from dsn.subspace import KrylovBuilder


def spd_matrix(n, cond=100.0):
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    lam = torch.logspace(0, torch.log10(torch.tensor(cond)), n)
    return Q @ torch.diag(lam) @ Q.T


def test_basis_is_orthonormal():
    n = 20
    A = spd_matrix(n)
    b = KrylovBuilder(k_max=6, m_recycle=0, tau=0.0)
    r = b.build(lambda v: A @ v, torch.randn(n))

    torch.testing.assert_close(
        r.W.T @ r.W, torch.eye(r.W.shape[1]), atol=1e-9, rtol=0
    )

    # m_recycle=0 above never exercises the deflated [U, Q] path (build's
    # `W = st.Q if m == 0 else torch.cat([self.U, st.Q], dim=1)` branch).
    # Extend with a recycling builder and a second build so the combined
    # basis's orthonormality is actually covered, not just the single-Q case.
    # k_max=15 (not 6) because recycling only engages once some Ritz pair has
    # converged to `ritz_tol`, and at k_max=6 on this spectrum none has.
    b2 = KrylovBuilder(k_max=15, m_recycle=4, tau=0.0)
    b2.build(lambda v: A @ v, torch.randn(n))
    r2 = b2.build(lambda v: A @ v, torch.randn(n))

    assert r2.reuse_frac > 0.0
    torch.testing.assert_close(
        r2.W.T @ r2.W, torch.eye(r2.W.shape[1]), atol=1e-9, rtol=0
    )


def test_loose_tolerance_uses_fewer_hvps_than_tight():
    n = 30
    A = spd_matrix(n)
    g = torch.randn(n)

    loose = KrylovBuilder(k_max=25, m_recycle=0, tau=0.5).build(lambda v: A @ v, g)
    tight = KrylovBuilder(k_max=25, m_recycle=0, tau=1e-6).build(lambda v: A @ v, g)

    assert loose.n_hvp < tight.n_hvp
    assert loose.rel_residual <= 0.5


def test_stops_as_soon_as_tolerance_is_met():
    n = 30
    A = spd_matrix(n)
    g = torch.randn(n)
    r = KrylovBuilder(k_max=25, m_recycle=0, tau=0.1, damping=0.0).build(
        lambda v: A @ v, g
    )

    assert r.rel_residual <= 0.1
    assert r.n_hvp < 25


def test_marginal_gain_rule_stops_early():
    n = 40
    A = spd_matrix(n)
    g = torch.randn(n)

    patient = KrylovBuilder(k_max=30, m_recycle=0, tau=0.0,
                            eps_marginal=0.0).build(lambda v: A @ v, g)
    impatient = KrylovBuilder(k_max=30, m_recycle=0, tau=0.0,
                              eps_marginal=0.05).build(lambda v: A @ v, g)

    assert impatient.n_hvp < patient.n_hvp


def test_recycling_costs_no_more_hvps_than_a_fresh_rebuild_at_equal_accuracy():
    """Recycling must stay budget-competitive with rebuilding from scratch.

    This assertion replaces `test_recycling_reduces_hvps_on_a_slowly_changing_operator`,
    which demanded a 20% *saving* and is false under honest curvature
    accounting. Refreshing m recycled directions against the current operator
    costs exactly m products and saves at most the m new Lanczos directions
    they stand in for, so exact recycling is break-even by construction; the
    saving the old test measured came from the stale-`HU` shortcut, whose
    optimistic residual tripped `tau` early (Plan 2 doc, F8/F10). Both arms are
    given k_max large enough to actually reach `tau`, so this compares products
    spent at EQUAL accuracy rather than which arm stopped early.

    Measured ratios: 1.03-1.11 across n=40/60/80, cond 1e2-1e4, 3 seeds. The
    1.25 bound has real headroom while still failing if the deflation stops
    paying for itself -- e.g. with the Ritz convergence gate removed, the
    retained directions are not near-invariant, deflation shrinks nothing, and
    the recycled arm runs to k_max on every step.
    """
    n = 40
    A = spd_matrix(n)
    g = torch.randn(n)

    fresh = KrylovBuilder(k_max=n, m_recycle=0, tau=1e-3)
    recyc = KrylovBuilder(k_max=n, m_recycle=8, tau=1e-3)

    fresh_hvp, recyc_hvp, fresh_res, recyc_res = [], [], [], []
    for step in range(5):
        drift = 1e-3 * step
        A_t = A + drift * torch.eye(n)
        g_t = g + drift * torch.randn(n)
        rf = fresh.build(lambda v: A_t @ v, g_t)
        rr = recyc.build(lambda v: A_t @ v, g_t)
        fresh_hvp.append(rf.n_hvp)
        recyc_hvp.append(rr.n_hvp)
        fresh_res.append(rf.rel_residual)
        recyc_res.append(rr.rel_residual)

    # Equal accuracy is the premise of the comparison, so assert it holds.
    assert max(fresh_res[1:]) < 2e-3 and max(recyc_res[1:]) < 2e-3

    assert sum(recyc_hvp[1:]) <= 1.25 * sum(fresh_hvp[1:]), (
        f"recycling is no longer budget-competitive: "
        f"recycled={sum(recyc_hvp[1:])} fresh={sum(fresh_hvp[1:])}"
    )


def test_recycling_matches_a_fresh_basis_of_equal_width():
    """F8's live gate: recycling must not be WORSE than fresh at equal width.

    Replaces `test_convergence.py::test_recycling_stays_fresh_rather_than_going_stale`,
    which was xfail(strict=True) and asserted recycling should be 2x *better*.
    That claim is not true and never was: what the pre-Plan-2 code actually did
    was report a residual against the same stale `T` that corrupted the step,
    making a degrading subspace look like an improving one (85x optimism at
    maximal staleness). With `T` and the residual both exact, the defensible
    claim is parity, and parity is what this asserts.

    This gate discriminates. Run against the pre-Plan-2 builder -- or against
    this one with the Ritz convergence gate removed, so unconverged directions
    get deflated -- the recycled arm's residual is 20-30x the fresh arm's here,
    far outside the 1.25x bound.
    """
    n = 40
    A = spd_matrix(n)
    g = torch.randn(n)

    recyc = KrylovBuilder(k_max=20, m_recycle=8, tau=1e-3)   # width 28 once warm
    fresh = KrylovBuilder(k_max=28, m_recycle=0, tau=1e-3)   # width 28 always

    recyc_res, fresh_res = [], []
    for step in range(5):
        drift = 1e-3 * step
        A_t = A + drift * torch.eye(n)
        g_t = g + drift * torch.randn(n)
        rr = recyc.build(lambda v: A_t @ v, g_t)
        rf = fresh.build(lambda v: A_t @ v, g_t)
        recyc_res.append(rr.rel_residual)
        fresh_res.append(rf.rel_residual)
        if step:
            # Both arms stop at `tau`, so width is an outcome, not a constant.
            # Require them to stay comparable: a materially narrower or wider
            # recycled basis would make the residual comparison meaningless.
            assert abs(rr.W.shape[1] - rf.W.shape[1]) <= 2, (
                f"widths diverged: recycled={rr.W.shape[1]} fresh={rf.W.shape[1]}"
            )

    mean_recycled = sum(recyc_res[1:]) / 4
    mean_fresh = sum(fresh_res[1:]) / 4

    assert mean_recycled <= 1.25 * mean_fresh, (
        f"recycled subspace is worse than an equal-width fresh one: "
        f"mean_recycled={mean_recycled:.4e} mean_fresh={mean_fresh:.4e}"
    )


def test_reuse_fraction_is_zero_on_first_step_then_positive():
    n = 25
    A = spd_matrix(n)
    # k_max=15: recycling engages only once a Ritz pair has converged to
    # `ritz_tol`, which on this spectrum needs roughly 12+ Lanczos directions.
    b = KrylovBuilder(k_max=15, m_recycle=4, tau=1e-3)

    first = b.build(lambda v: A @ v, torch.randn(n))
    second = b.build(lambda v: A @ v, torch.randn(n))

    assert first.reuse_frac == 0.0
    assert second.reuse_frac > 0.0


def test_recycled_basis_never_exceeds_m_recycle():
    n = 25
    A = spd_matrix(n)
    b = KrylovBuilder(k_max=10, m_recycle=3, tau=1e-8)
    for _ in range(4):
        b.build(lambda v: A @ v, torch.randn(n))

    # `== 3`, not `<= 3`: `topk(score, min(m_recycle, score.numel()))` in
    # `_recycle` makes the width <= m_recycle by construction, so `<= 3` cannot
    # fail -- a regression would raise out of `topk` rather than return a wider
    # basis. 3 is the value actually reachable here (score.numel() is the basis
    # width, always >= 3 at k_max=10), so asserting it is a live check.
    assert b.U is not None and b.U.shape[1] == 3


def test_reset_clears_recycled_state():
    n = 20
    A = spd_matrix(n)
    b = KrylovBuilder(k_max=6, m_recycle=4, tau=1e-3)
    b.build(lambda v: A @ v, torch.randn(n))
    b.reset()

    assert b.U is None
    assert b.build(lambda v: A @ v, torch.randn(n)).reuse_frac == 0.0


def test_zero_gradient_returns_empty_subspace():
    n = 10
    A = spd_matrix(n)
    r = KrylovBuilder(k_max=5, m_recycle=0).build(lambda v: A @ v, torch.zeros(n))

    assert r.W.shape[1] == 0
    assert r.n_hvp == 0


def test_k_max_zero_returns_empty_subspace_not_none():
    """k_max=0 means lanczos_iter yields nothing, so `best` never gets assigned
    inside build's loop. Task 8 relies on build returning a valid empty
    SubspaceResult in this case (DSN(builder=KrylovBuilder(k_max=0, ...)) must
    degrade to AdamW, not raise AttributeError on `None.y`)."""
    n = 10
    A = spd_matrix(n)
    r = KrylovBuilder(k_max=0, m_recycle=0).build(lambda v: A @ v, torch.randn(n))

    assert r is not None
    assert r.W.shape[1] == 0
    assert r.n_hvp == 0


def test_recycled_reported_residual_equals_the_true_dense_residual():
    """F8 regression: `rel_residual` must be the TRUE residual at any reuse_frac.

    Before Plan 2 this test asserted the opposite -- that the reported value is
    optimistic by up to 10x -- because it was computed against `T`, and `T`'s
    recycled block was assembled from the previous iterate's `HU`. Plan 2
    re-projects `HU` against the current operator and computes the residual from
    the curvature images `HW`, so reported and true must now agree to numerical
    precision on the recycled step exactly as they always did on the fresh one.

    Same setup as the version it replaces: an IDENTICAL A and g on every step,
    so operator drift cannot explain any gap.
    """
    n = 40
    A = spd_matrix(n)
    g = torch.randn(n)

    b = KrylovBuilder(k_max=20, m_recycle=8, tau=1e-3)

    r0 = b.build(lambda v: A @ v, g)
    assert r0.reuse_frac == 0.0
    true0 = float((A @ (r0.W @ r0.y) + g).norm() / g.norm())
    assert abs(true0 - r0.rel_residual) < 1e-8

    r1 = b.build(lambda v: A @ v, g)
    assert r1.reuse_frac > 0.0
    true1 = float((A @ (r1.W @ r1.y) + g).norm() / g.norm())

    assert abs(true1 - r1.rel_residual) <= 1e-8 * max(1.0, true1), (
        f"F8: reported residual {r1.rel_residual:.6e} does not match the true "
        f"dense residual {true1:.6e} at reuse_frac={r1.reuse_frac:.2f}"
    )


def test_projected_curvature_is_exact_even_on_a_recycled_step():
    """The root of F8: `T` must be W'HW at the CURRENT iterate, not a surrogate.

    Uses a drifting operator so a carried-over `HU` would be measurably wrong:
    under the pre-Plan-2 builder the recycled block of `T` came from the
    previous step's Hessian, so this comparison failed on the drift term.
    """
    n = 30
    A = spd_matrix(n)
    b = KrylovBuilder(k_max=20, m_recycle=6, tau=1e-6)

    b.build(lambda v: A @ v, torch.randn(n))
    A_next = A + 0.5 * torch.eye(n)          # the operator moved between steps
    r = b.build(lambda v: A_next @ v, torch.randn(n))

    assert r.reuse_frac > 0.0
    torch.testing.assert_close(r.T, r.W.T @ A_next @ r.W, atol=1e-9, rtol=0)


def test_a_frozen_recycled_basis_cannot_hide_behind_the_reported_residual():
    """F8's limiting case: maximal staleness must show up in the telemetry.

    The Task 9 findings doc measured a run whose recycled basis was frozen
    mid-training -- the worst possible staleness -- reporting `rel_residual`
    0.0114 against a true residual of 0.9752, an 85x understatement, and
    scoring *better* on every reported-residual gate than the healthy run. That
    is only possible when the metric is computed against the same stale
    quantity that corrupts the step. Freeze the basis here and assert the
    reported residual tracks the truth anyway.
    """
    n = 30
    A = spd_matrix(n)
    g = torch.randn(n)
    b = KrylovBuilder(k_max=20, m_recycle=5, tau=1e-8)

    b.build(lambda v: A @ v, g)
    b._recycle = lambda res, HW: None        # freeze U at step 0's basis
    frozen = b.U.clone()

    for step in range(1, 6):
        A_t = A + 0.3 * step * torch.eye(n)  # drift away from the frozen basis
        r = b.build(lambda v: A_t @ v, g)
        true = float((A_t @ (r.W @ r.y) + g).norm() / g.norm())
        assert abs(true - r.rel_residual) <= 1e-8 * max(1.0, true), (
            f"step {step}: reported {r.rel_residual:.6e} vs true {true:.6e}"
        )

    torch.testing.assert_close(b.U, frozen, atol=0, rtol=0)


@pytest.mark.parametrize("rank_by", ["contribution", "small_eig", "large_eig"])
def test_only_converged_ritz_directions_are_ever_deflated(rank_by):
    """F10 regression: every retained direction must be a converged Ritz pair.

    Deflation is justified only for a Ritz vector close to an invariant
    direction of H. Ranking heuristics select by usefulness-to-the-step, which
    is a different property, so without a convergence gate the shipped
    "contribution" ranking retained directions with relative Ritz residual
    0.1-2.8 -- not converged by any margin -- and the recycled subspace came out
    ~30x worse than an equal-width fresh one. Assert the gate holds for every
    ranking, since it is what makes the ranking choice safe.

    This test replaces `test_large_eig_ranking_is_the_worst_of_the_three`, whose
    claim does not survive honest measurement: that ordering was read off
    n_hvp, and n_hvp was driven by a `tau` rule fed the optimistic
    stale-`T` residual. With exact curvature and this gate in place, all three
    rankings land within noise of each other and of the equal-width fresh
    baseline (Plan 2 doc, F10), so no ordering among them is assertable here.
    """
    n = 40
    for seed in range(3):
        torch.manual_seed(seed)
        A = spd_matrix(n)
        g = torch.randn(n)
        b = KrylovBuilder(k_max=20, m_recycle=8, tau=1e-3, rank_by=rank_by)

        for step in range(5):
            drift = 1e-3 * step
            A_t = A + drift * torch.eye(n)
            g_t = g + drift * torch.randn(n)
            b.build(lambda v: A_t @ v, g_t)

            if b.U is None:
                continue
            # Rayleigh quotient and Ritz residual of each retained direction,
            # measured against the operator that produced it.
            AU = A_t @ b.U
            theta = (b.U * AU).sum(dim=0)
            rel = (AU - b.U * theta).norm(dim=0) / theta.abs()
            assert float(rel.max()) <= b.ritz_tol, (
                f"seed {seed} step {step} {rank_by}: deflated an unconverged "
                f"Ritz pair, worst relative residual {float(rel.max()):.3e} "
                f"> ritz_tol {b.ritz_tol}"
            )


def test_nothing_is_recycled_when_no_ritz_pair_has_converged():
    """The "fall back to a fresh rebuild" path: an empty converged set drops U.

    A single Lanczos direction cannot produce a converged Ritz pair on an
    operator with a spread spectrum, so nothing is deflatable and the next step
    must rebuild from scratch rather than deflate against a direction that is
    not near-invariant.
    """
    n = 30
    A = spd_matrix(n)
    b = KrylovBuilder(k_max=1, m_recycle=4, tau=1e-9)

    b.build(lambda v: A @ v, torch.randn(n))

    assert b.U is None
    assert b.build(lambda v: A @ v, torch.randn(n)).reuse_frac == 0.0


def test_unknown_rank_by_raises_value_error():
    with pytest.raises(ValueError):
        KrylovBuilder(rank_by="bogus")
