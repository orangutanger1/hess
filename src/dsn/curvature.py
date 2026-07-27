"""Curvature-vector products.

Two operators are available:

``hvp``    exact Hessian-vector product via double backward. Can be indefinite.
``ggnvp``  Gauss-Newton-vector product, always positive semi-definite.

Both return a flat vector in the same ordering as ``params``.
"""

from typing import Callable, Sequence

import torch

from .utils import flat_grad, flatten, unflatten_like


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


def ggnvp(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Gauss-Newton-vector product ``J' H_z J @ v``, always PSD.

    Computed forward-over-reverse in three stages:
      1. ``Jv``        push v through the model Jacobian      (forward mode)
      2. ``H_z (Jv)``  apply the loss Hessian in output space (forward mode)
      3. ``J' (...)``  pull back through the Jacobian         (reverse mode)
    """
    names = [n for n, _ in model.named_parameters()]
    pdict = {n: p.detach() for n, p in model.named_parameters()}
    vdict = dict(zip(names, unflatten_like(v, [pdict[n] for n in names])))

    def f(pd):
        return torch.func.functional_call(model, pd, (x,))

    z, jv = torch.func.jvp(f, (pdict,), (vdict,))
    _, hjv = torch.func.jvp(torch.func.grad(lambda zz: loss_fn(zz, y)), (z,), (jv,))
    _, vjp_fn = torch.func.vjp(f, pdict)
    out = vjp_fn(hjv)[0]
    return flatten([out[n] for n in names])
