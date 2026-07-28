import torch

from dsn import DSN
from dsn.subspace import KrylovBuilder


def quadratic_problem(n=12, cond=50.0, seed=0):
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    lam = torch.logspace(0, torch.log10(torch.tensor(cond)), n)
    A = Q @ torch.diag(lam) @ Q.T
    b = torch.randn(n)
    x = torch.zeros(n, requires_grad=True)
    return A, b, x


def test_degenerates_to_adamw_when_subspace_is_empty():
    """k=0 must reproduce AdamW exactly, or every comparison is meaningless."""
    torch.manual_seed(0)
    p_ref = torch.randn(8, requires_grad=True)
    p_dsn = p_ref.detach().clone().requires_grad_(True)
    target = torch.randn(8)

    ref = torch.optim.AdamW([p_ref], lr=1e-2, weight_decay=0.0, eps=1e-8)
    dsn = DSN([p_dsn], lr=1e-2, weight_decay=0.0,
              builder=KrylovBuilder(k_max=0, m_recycle=0))

    for _ in range(20):
        ref.zero_grad()
        (p_ref - target).pow(2).sum().backward()
        ref.step()
        dsn.step(lambda: (p_dsn - target).pow(2).sum())

    torch.testing.assert_close(p_dsn.detach(), p_ref.detach(), rtol=0, atol=1e-11)


def test_full_rank_step_equals_exact_newton_on_a_quadratic():
    """The central correctness invariant of the whole method."""
    n = 10
    A, b, x = quadratic_problem(n)

    dsn = DSN([x], lr=1.0, weight_decay=0.0, damping=0.0, trust_radius=1e9,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))
    x0 = x.detach().clone()
    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    newton = -torch.linalg.solve(A, A @ x0 + b)
    torch.testing.assert_close(x.detach() - x0, newton, rtol=1e-6, atol=1e-8)


def test_solves_a_quadratic_in_one_step():
    n = 10
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1.0, weight_decay=0.0, damping=0.0, trust_radius=1e9,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))

    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    torch.testing.assert_close(A @ x.detach() + b, torch.zeros(n),
                               atol=1e-6, rtol=0)


def test_complement_step_is_orthogonal_to_the_subspace():
    n = 15
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-2, builder=KrylovBuilder(k_max=4, m_recycle=2, tau=0.1))
    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    W = dsn._last_W
    torch.testing.assert_close(
        W.T @ dsn._last_d_complement, torch.zeros(W.shape[1]), atol=1e-10, rtol=0
    )


def test_trust_radius_clips_a_large_subspace_step():
    n = 10
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-3, damping=0.0, trust_radius=1e-3,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))
    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    assert dsn.telemetry.step_norm_subspace <= 1e-3 + 1e-12


def test_telemetry_is_populated():
    n = 15
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-2, builder=KrylovBuilder(k_max=5, m_recycle=2, tau=0.1))
    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
    t = dsn.telemetry

    assert 0 < t.k <= 5
    assert t.n_hvp >= 1
    assert 0.0 <= t.rel_residual
    assert t.n_fallback == 0


def test_non_finite_curvature_falls_back_to_adamw_without_crashing():
    x = torch.tensor([1.0], requires_grad=True)

    class Exploding:
        def build(self, matvec, g):
            raise FloatingPointError("simulated non-finite curvature")

        def reset(self):
            pass

    dsn = DSN([x], lr=1e-2, builder=Exploding())
    dsn.step(lambda: (x**2).sum())

    assert dsn.telemetry.n_fallback == 1
    assert torch.isfinite(x).all()


def test_beats_adamw_on_an_ill_conditioned_quadratic():
    """Second order should win decisively where first order is known to struggle."""
    n = 20
    A, b, _ = quadratic_problem(n, cond=1e4)

    x_a = torch.zeros(n, requires_grad=True)
    x_d = torch.zeros(n, requires_grad=True)
    adam = torch.optim.AdamW([x_a], lr=1e-1, weight_decay=0.0)
    dsn = DSN([x_d], lr=1e-1, weight_decay=0.0, damping=1e-6, trust_radius=1e3,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-6))

    for _ in range(30):
        adam.zero_grad()
        (0.5 * x_a @ A @ x_a + b @ x_a).backward()
        adam.step()
        dsn.step(lambda: 0.5 * x_d @ A @ x_d + b @ x_d)

    f = lambda z: 0.5 * z @ A @ z + b @ z
    assert f(x_d.detach()) < f(x_a.detach())
