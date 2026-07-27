import torch

from dsn.lanczos import lanczos_iter


def tridiag(state):
    T = torch.diag(state.alphas)
    if state.betas.numel():
        i = torch.arange(state.betas.numel())
        T[i, i + 1] = state.betas
        T[i + 1, i] = state.betas
    return T


def sym_matrix(n):
    A = torch.randn(n, n)
    return (A + A.T) / 2


def test_full_run_recovers_all_eigenvalues():
    n = 8
    A = sym_matrix(n)
    b = torch.randn(n)
    state = list(lanczos_iter(lambda v: A @ v, b, k_max=n))[-1]

    got = torch.linalg.eigvalsh(tridiag(state))

    torch.testing.assert_close(got, torch.linalg.eigvalsh(A), rtol=1e-6, atol=1e-8)


def test_basis_is_orthonormal():
    n = 10
    A = sym_matrix(n)
    state = list(lanczos_iter(lambda v: A @ v, torch.randn(n), k_max=6))[-1]

    QtQ = state.Q.T @ state.Q

    torch.testing.assert_close(QtQ, torch.eye(QtQ.shape[0]), atol=1e-10, rtol=0)


def test_basis_is_orthogonal_to_deflation_space():
    n = 12
    A = sym_matrix(n)
    U, _ = torch.linalg.qr(torch.randn(n, 3))
    state = list(lanczos_iter(lambda v: A @ v, torch.randn(n), k_max=5, U=U))[-1]

    torch.testing.assert_close(
        U.T @ state.Q, torch.zeros(3, state.Q.shape[1]), atol=1e-10, rtol=0
    )


def test_tridiagonal_matches_explicit_projection():
    n = 9
    A = sym_matrix(n)
    state = list(lanczos_iter(lambda v: A @ v, torch.randn(n), k_max=4))[-1]

    torch.testing.assert_close(tridiag(state), state.Q.T @ A @ state.Q,
                               atol=1e-9, rtol=0)


def test_yields_one_state_per_direction_and_counts_hvps():
    n = 10
    A = sym_matrix(n)
    states = list(lanczos_iter(lambda v: A @ v, torch.randn(n), k_max=5))

    assert [s.Q.shape[1] for s in states] == [1, 2, 3, 4, 5]
    assert [s.n_hvp for s in states] == [1, 2, 3, 4, 5]


def test_breakdown_stops_early_on_invariant_subspace():
    """A rank-2 operator exhausts its Krylov space after 2 directions."""
    n = 8
    V, _ = torch.linalg.qr(torch.randn(n, 2))
    A = V @ torch.diag(torch.tensor([3.0, -1.0])) @ V.T
    b = V @ torch.randn(2)

    states = list(lanczos_iter(lambda v: A @ v, b, k_max=6))

    assert states[-1].Q.shape[1] == 2


def test_zero_seed_vector_yields_nothing():
    n = 5
    states = list(lanczos_iter(lambda v: v, torch.zeros(n), k_max=3))
    assert states == []
