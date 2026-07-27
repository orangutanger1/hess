"""Conversion between a flat parameter vector and a list of parameter tensors.

Parameter ordering is fixed by the caller (normally ``list(model.parameters())``)
and every function here assumes that same order. Mixing orders produces silently
wrong curvature.
"""

from typing import Sequence

import torch


def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate tensors into a single 1-D vector."""
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_like(
    vec: torch.Tensor, params: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    """Split a flat vector into tensors shaped like ``params``."""
    out, i = [], 0
    for p in params:
        n = p.numel()
        out.append(vec[i : i + n].view_as(p))
        i += n
    if i != vec.numel():
        raise ValueError(f"vector has {vec.numel()} elements, params need {i}")
    return out


def flat_grad(
    loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    create_graph: bool = False,
) -> torch.Tensor:
    """Gradient of ``loss`` w.r.t. ``params``, returned as one flat vector.

    Parameters that do not affect ``loss`` contribute zeros rather than raising,
    so that models with conditionally-unused branches still work.
    """
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, allow_unused=True
    )
    return flatten(
        [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]
    )
