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

    assert sum(recyc_hvp[1:]) < sum(fresh_hvp[1:])


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

    assert b.U is not None and b.U.shape[1] <= 3


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
