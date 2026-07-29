import torch

from dsn.lanczos import lanczos_iter
from dsn.solve import newton_residual, subspace_newton


def spd_matrix(n):
    A = torch.randn(n, n)
    return A @ A.T + n * torch.eye(n)


def test_full_rank_undamped_solve_is_exact_newton():
    n = 6
    H = spd_matrix(n)
    g = torch.randn(n)

    y, _, _ = subspace_newton(H, g, damping=0.0)

    torch.testing.assert_close(y, -torch.linalg.solve(H, g))


def test_negative_curvature_is_reflected_not_followed():
    """With eigenvalue -2 and gradient component +1, saddle-free steps downhill."""
    T = torch.diag(torch.tensor([-2.0, 4.0]))
    Wg = torch.tensor([1.0, 1.0])

    y, _, _ = subspace_newton(T, Wg, damping=0.0, saddle_free=True)

    torch.testing.assert_close(y, torch.tensor([-0.5, -0.25]))


def test_saddle_free_disabled_clamps_instead():
    T = torch.diag(torch.tensor([-2.0, 4.0]))
    Wg = torch.tensor([1.0, 1.0])

    y, _, _ = subspace_newton(T, Wg, damping=1.0, saddle_free=False)

    # clamp_min(0) leaves the first direction with denominator = damping = 1
    torch.testing.assert_close(y, torch.tensor([-1.0, -0.2]))


def test_damping_shrinks_the_step():
    n = 5
    H = spd_matrix(n)
    g = torch.randn(n)

    undamped, _, _ = subspace_newton(H, g, damping=0.0)
    damped, _, _ = subspace_newton(H, g, damping=10.0)

    assert damped.norm() < undamped.norm()


def test_residual_matches_explicit_norm_for_krylov_basis():
    n = 12
    A = spd_matrix(n)
    g = torch.randn(n)
    state = list(lanczos_iter(lambda v: A @ v, g, k_max=5))[-1]

    T = torch.diag(state.alphas)
    i = torch.arange(state.betas.numel())
    T[i, i + 1] = state.betas
    T[i + 1, i] = state.betas

    Wg = state.Q.T @ g
    y, _, _ = subspace_newton(T, Wg, damping=1e-3)
    d = state.Q @ y

    torch.testing.assert_close(
        newton_residual(state.HQ, g, y),
        (A @ d + g).norm(),
        rtol=1e-8,
        atol=1e-10,
    )


def test_residual_matches_explicit_norm_for_a_deflated_basis():
    """The case the old (T, Wg, y, beta_next) form could only estimate.

    With deflation, ``H U`` leaves the span in directions ``beta_next`` does not
    track, so the in-span-plus-one-leak formula understated the residual. The
    HW form has no such gap: assert it against the dense truth on the combined
    basis [U, Q], which is where F8's optimism used to live.
    """
    n = 14
    A = spd_matrix(n)
    g = torch.randn(n)
    U, _ = torch.linalg.qr(torch.randn(n, 3))
    state = list(lanczos_iter(lambda v: A @ v, g, k_max=4, U=U))[-1]

    W = torch.cat([U, state.Q], dim=1)
    HW = torch.cat([A @ U, state.HQ], dim=1)
    y, _, _ = subspace_newton(W.T @ A @ W, W.T @ g, damping=1e-3)

    torch.testing.assert_close(
        newton_residual(HW, g, y),
        (A @ (W @ y) + g).norm(),
        rtol=1e-8,
        atol=1e-10,
    )


def test_residual_falls_to_zero_when_krylov_space_is_complete():
    n = 7
    A = spd_matrix(n)
    g = torch.randn(n)
    state = list(lanczos_iter(lambda v: A @ v, g, k_max=n))[-1]

    T = state.Q.T @ A @ state.Q
    Wg = state.Q.T @ g
    y, _, _ = subspace_newton(T, Wg, damping=0.0)

    assert newton_residual(state.HQ, g, y) < 1e-8 * g.norm()
