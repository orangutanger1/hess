import torch

from dsn.utils import flat_grad, flatten, unflatten_like


def test_flatten_unflatten_roundtrip():
    params = [torch.randn(3, 4), torch.randn(5), torch.randn(2, 2, 2)]
    vec = flatten(params)
    assert vec.shape == (12 + 5 + 8,)
    back = unflatten_like(vec, params)
    assert len(back) == len(params)
    for a, b in zip(back, params):
        assert a.shape == b.shape
        torch.testing.assert_close(a, b)


def test_flatten_preserves_order():
    params = [torch.zeros(2), torch.ones(3)]
    torch.testing.assert_close(
        flatten(params), torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    )


def test_flat_grad_matches_autograd():
    x = torch.randn(4, requires_grad=True)
    y = torch.randn(3, requires_grad=True)
    loss = (x**2).sum() + 3.0 * (y**3).sum()
    got = flat_grad(loss, [x, y])
    expected = torch.cat([2 * x.detach(), 9 * y.detach() ** 2])
    torch.testing.assert_close(got, expected)
