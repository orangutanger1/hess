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
    T            (r, r) symmetric projected curvature W'HW, exact at the
                 current iterate whatever `reuse_frac` is
    y            (r,)   coefficients, so the in-subspace step is W @ y
    rel_residual ‖H d + g‖ / ‖g‖ for d = W @ y, computed from the curvature
                 images H W under the CURRENT operator, not from `T`. It is
                 therefore a true residual at every `reuse_frac`. This is a
                 change from the behavior documented here before Plan 2: the
                 residual used to be assembled from `T` plus a single Lanczos
                 leak term, so a stale recycled block corrupted the step and
                 the metric in the same direction and the reported value
                 *improved* as the subspace degraded (Task 9 findings doc,
                 F8; 85x optimism measured at maximal staleness). No quantity
                 in the current computation crosses a step boundary.
    n_hvp        curvature-vector products consumed this step, including the
                 m_recycle products spent re-projecting the recycled basis
                 against this step's operator
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
