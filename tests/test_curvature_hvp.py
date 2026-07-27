import torch

from dsn.curvature import hvp


def test_hvp_on_quadratic_equals_matrix_product():
    """For f(x) = 0.5 x'Ax + b'x the Hessian is exactly A."""
    A = torch.randn(6, 6)
    A = A @ A.T
    b = torch.randn(6)
    x = torch.randn(6, requires_grad=True)
    v = torch.randn(6)

    got = hvp(lambda: 0.5 * x @ A @ x + b @ x, [x], v)

    torch.testing.assert_close(got, A @ v)


def test_hvp_matches_dense_autograd_hessian_nonquadratic():
    x = torch.randn(5, requires_grad=True)
    c = torch.randn(5)

    def f(z):
        return (torch.sin(z) * c).sum() + (z**4).sum()

    dense = torch.autograd.functional.hessian(f, x.detach())
    v = torch.randn(5)

    got = hvp(lambda: f(x), [x], v)

    torch.testing.assert_close(got, dense @ v)


def test_hvp_matches_finite_differences():
    x = torch.randn(4, requires_grad=True)
    A = torch.randn(4, 4)
    A = A @ A.T

    def f(z):
        return 0.5 * z @ A @ z + (z**3).sum()

    v = torch.randn(4)
    eps = 1e-6

    def grad_at(z):
        zz = z.clone().requires_grad_(True)
        return torch.autograd.grad(f(zz), zz)[0]

    fd = (grad_at(x.detach() + eps * v) - grad_at(x.detach() - eps * v)) / (2 * eps)

    got = hvp(lambda: f(x), [x], v)

    torch.testing.assert_close(got, fd, rtol=1e-5, atol=1e-6)


def test_hvp_across_multiple_parameter_tensors():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(),
                                torch.nn.Linear(4, 1))
    params = list(model.parameters())
    x = torch.randn(8, 3)
    n = sum(p.numel() for p in params)
    v = torch.randn(n)

    got = hvp(lambda: model(x).pow(2).mean(), params, v)

    assert got.shape == (n,)
    assert torch.isfinite(got).all()
