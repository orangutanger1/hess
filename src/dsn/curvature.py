"""Curvature-vector products.

Two operators are available:

``hvp``    exact Hessian-vector product via double backward. Can be indefinite.
``ggnvp``  Gauss-Newton-vector product, always positive semi-definite.

Both return a flat vector in the same ordering as ``params``.
"""

from typing import Callable, Sequence

import torch

from .utils import flat_grad, flatten


def hvp(
    loss_fn: Callable[[], torch.Tensor],
    params: Sequence[torch.Tensor],
    v: torch.Tensor,
) -> torch.Tensor:
    """Exact Hessian-vector product ``H @ v``.

    ``loss_fn`` must rebuild its graph on every call, because the graph is
    consumed by the second backward pass.
    """
    loss = loss_fn()
    g = flat_grad(loss, params, create_graph=True)
    return flat_grad((g * v).sum(), params)
