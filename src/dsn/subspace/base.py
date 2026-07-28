"""Interface every subspace-construction strategy implements.

Backends A (Krylov), B (gradient history) and C (randomized sketch) all satisfy
this, so swapping them is a config change rather than a rewrite.
"""

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


@dataclass
class SubspaceResult:
    """One step's subspace and the in-subspace Newton step.

    W            (n, r) orthonormal basis; r may be 0
    T            (r, r) symmetric projected curvature W'HW
    y            (r,)   coefficients, so the in-subspace step is W @ y
    rel_residual ‖H d + g‖ / ‖g‖, computed against `T`, the recycled
                 surrogate curvature, not the true operator H. Exact when
                 reuse_frac == 0, since T is then the fresh Lanczos
                 tridiagonal. Optimistic when reuse_frac > 0: the retained
                 Ritz pairs have not converged, so the recycled blocks of T
                 (built from HU = W @ (T @ Vk), free of extra curvature
                 products) only approximate W'HW, and the true dense
                 residual can exceed the reported one by an order of
                 magnitude or more.
    n_hvp        curvature-vector products consumed this step
    reuse_frac   fraction of the basis carried over from the previous step
    """

    W: torch.Tensor
    T: torch.Tensor
    y: torch.Tensor
    rel_residual: float
    n_hvp: int
    reuse_frac: float


class SubspaceBuilder(Protocol):
    def build(
        self, matvec: Callable[[torch.Tensor], torch.Tensor], g: torch.Tensor
    ) -> SubspaceResult:
        """Build a subspace for this step and solve the Newton system inside it."""
        ...

    def reset(self) -> None:
        """Drop any state carried between steps."""
        ...
