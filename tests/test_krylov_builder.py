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
    b2 = KrylovBuilder(k_max=6, m_recycle=8, tau=0.0)
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


def test_recycling_reduces_hvps_on_a_slowly_changing_operator():
    """The point of the whole method: reuse across steps must actually pay."""
    n = 40
    A = spd_matrix(n)
    g = torch.randn(n)

    fresh = KrylovBuilder(k_max=20, m_recycle=0, tau=1e-3)
    recyc = KrylovBuilder(k_max=20, m_recycle=8, tau=1e-3)

    fresh_hvp, recyc_hvp = [], []
    for step in range(5):
        drift = 1e-3 * step
        A_t = A + drift * torch.eye(n)
        g_t = g + drift * torch.randn(n)
        fresh_hvp.append(fresh.build(lambda v: A_t @ v, g_t).n_hvp)
        recyc_hvp.append(recyc.build(lambda v: A_t @ v, g_t).n_hvp)

    # A bare `<` is satisfiable by rank_by="large_eig" -- the ranking the plan
    # explicitly says is wrong -- on every seed measured (72-76 out of an
    # 80 fresh-baseline ceiling). The default ("contribution") measures
    # 46-60 across the same seeds, so a real margin (not knife-edge: worst
    # observed is 60/80 = 0.75) is required to actually discriminate.
    assert sum(recyc_hvp[1:]) < 0.8 * sum(fresh_hvp[1:])


def test_reuse_fraction_is_zero_on_first_step_then_positive():
    n = 25
    A = spd_matrix(n)
    b = KrylovBuilder(k_max=8, m_recycle=4, tau=1e-3)

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


def test_recycled_reported_residual_is_optimistic_vs_true_dense_residual():
    """rel_residual (base.py's docstring) is measured against the recycled
    surrogate curvature T, not the true operator, so it understates the real
    error once reuse_frac > 0. Pin the gap directly: rebuild with an
    IDENTICAL A and g on every step (so operator drift cannot explain any of
    it -- this isolates the surrogate effect from staleness), take the
    returned basis and coefficients, and compute the TRUE dense residual
    ||A(Wy)+g|| / ||g|| by hand.

    Measured under this file's fixed seed (conftest resets to 0): step 0
    (reuse_frac=0) reports 8.826e-3 and the true residual matches to 1e-10;
    step 1 (reuse_frac=0.42) reports 9.37e-4 but the true residual is
    4.40e-3, a 4.7x gap. Across other seeds (see the fix report) this gap
    reaches ~60-100x, so the bound below (10x) has real headroom above the
    4.7x this seed produces while still catching a regression that widens
    the gap meaningfully.
    """
    n = 40
    A = spd_matrix(n)
    g = torch.randn(n)

    b = KrylovBuilder(k_max=20, m_recycle=8, tau=1e-3)

    r0 = b.build(lambda v: A @ v, g)
    assert r0.reuse_frac == 0.0
    d0 = r0.W @ r0.y
    true0 = float((A @ d0 + g).norm() / g.norm())
    # No recycling yet: T is the fresh Lanczos tridiagonal, so reported and
    # true residual must agree to numerical precision.
    assert abs(true0 - r0.rel_residual) < 1e-8

    r1 = b.build(lambda v: A @ v, g)
    assert r1.reuse_frac > 0.0
    d1 = r1.W @ r1.y
    true1 = float((A @ d1 + g).norm() / g.norm())

    # The reported residual is optimistic (smaller than the truth) ...
    assert r1.rel_residual < true1
    # ... but not by an unbounded amount: pin the gap with headroom.
    ratio = true1 / r1.rel_residual
    assert ratio < 10.0


@pytest.mark.parametrize("rank_by", ["contribution", "small_eig"])
def test_large_eig_ranking_is_the_worst_of_the_three(rank_by):
    """rank_by="large_eig" is the ranking the plan explicitly names as wrong:
    the Newton step weights each eigendirection by 1/lambda, so keeping the
    *largest* |lambda| retains the directions the step least depends on. Both
    other rankings must beat it, and a bare hvp-reduction assertion does not
    discriminate -- large_eig also reduces hvps versus a fresh rebuild. Assert
    the separation directly.

    Measured sums of n_hvp over steps 1-4 on the slowly-drifting operator,
    out of an 80 fresh-rebuild ceiling, seeds 0-2 at this configuration:

        contribution  50, 46, 59
        small_eig     53, 50, 54
        large_eig     76, 72, 72

    Worst observed ratio against large_eig is 59/72 = 0.819 (contribution)
    and 54/72 = 0.750 (small_eig), so the 0.9 bound has real headroom without
    being knife-edge.

    Deliberately NOT asserted: contribution < small_eig. Across 6 seeds at two
    configurations the contribution/small_eig ratio ranged 0.755 to 1.250 --
    the default ranking does not reliably beat small_eig on this benchmark,
    so an assertion to that effect would be seed-dependent. See the comment in
    src/dsn/subspace/krylov.py:_recycle.
    """
    n = 40

    for seed in range(3):
        torch.manual_seed(seed)
        A = spd_matrix(n)
        g = torch.randn(n)

        candidate = KrylovBuilder(k_max=20, m_recycle=8, tau=1e-3, rank_by=rank_by)
        large_eig = KrylovBuilder(
            k_max=20, m_recycle=8, tau=1e-3, rank_by="large_eig"
        )

        candidate_hvp, large_hvp = [], []
        for step in range(5):
            drift = 1e-3 * step
            A_t = A + drift * torch.eye(n)
            g_t = g + drift * torch.randn(n)
            candidate_hvp.append(candidate.build(lambda v: A_t @ v, g_t).n_hvp)
            large_hvp.append(large_eig.build(lambda v: A_t @ v, g_t).n_hvp)

        candidate_sum = sum(candidate_hvp[1:])
        large_sum = sum(large_hvp[1:])
        assert candidate_sum < 0.9 * large_sum, (
            f"seed {seed}: {rank_by}={candidate_sum} large_eig={large_sum}"
        )


def test_unknown_rank_by_raises_value_error():
    with pytest.raises(ValueError):
        KrylovBuilder(rank_by="bogus")
