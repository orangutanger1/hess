import pytest
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


@pytest.mark.parametrize("n_vec", [3, 8])
def test_unflatten_like_reports_a_length_mismatch_in_both_directions(n_vec):
    """A too-*short* vector used to raise `RuntimeError: shape '[5]' is invalid
    for input of size 3` out of `view_as`, because the length check sat after
    the loop and never ran. Both directions must give the friendly message.
    """
    params = [torch.zeros(5)]
    with pytest.raises(ValueError, match=f"vector has {n_vec} elements, params need 5"):
        unflatten_like(torch.randn(n_vec), params)


def test_flat_grad_matches_autograd():
    x = torch.randn(4, requires_grad=True)
    y = torch.randn(3, requires_grad=True)
    loss = (x**2).sum() + 3.0 * (y**3).sum()
    got = flat_grad(loss, [x, y])
    expected = torch.cat([2 * x.detach(), 9 * y.detach() ** 2])
    torch.testing.assert_close(got, expected)
