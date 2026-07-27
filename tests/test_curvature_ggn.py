import torch

from dsn.curvature import ggnvp, hvp


def mse(z, y):
    return 0.5 * (z - y).pow(2).sum() / z.shape[0]


def test_ggn_equals_hessian_for_linear_model():
    """A model linear in its parameters has GGN exactly equal to the Hessian."""
    model = torch.nn.Linear(4, 3, bias=False)
    x, y = torch.randn(10, 4), torch.randn(10, 3)
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    v = torch.randn(n)

    got = ggnvp(model, mse, x, y, v)
    expected = hvp(lambda: mse(model(x), y), params, v)

    torch.testing.assert_close(got, expected)


def test_ggn_is_psd_where_hessian_is_not():
    """Nonlinear model: GGN stays PSD, the raw Hessian need not be."""
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
    )
    x, y = torch.randn(16, 3), torch.randn(16, 2)
    n = sum(p.numel() for p in model.parameters())

    for _ in range(20):
        v = torch.randn(n)
        assert torch.dot(v, ggnvp(model, mse, x, y, v)) >= -1e-10


def test_ggn_is_symmetric():
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(),
                                torch.nn.Linear(5, 2))
    x, y = torch.randn(12, 3), torch.randn(12, 2)
    n = sum(p.numel() for p in model.parameters())
    u, v = torch.randn(n), torch.randn(n)

    uGv = torch.dot(u, ggnvp(model, mse, x, y, v))
    vGu = torch.dot(v, ggnvp(model, mse, x, y, u))

    torch.testing.assert_close(uGv, vGu)


def test_ggn_works_with_cross_entropy():
    model = torch.nn.Linear(5, 4)
    x = torch.randn(20, 5)
    y = torch.randint(0, 4, (20,))
    n = sum(p.numel() for p in model.parameters())
    v = torch.randn(n)

    got = ggnvp(model, torch.nn.functional.cross_entropy, x, y, v)

    assert got.shape == (n,)
    assert torch.isfinite(got).all()
    assert torch.dot(v, got) >= -1e-10
