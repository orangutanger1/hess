"""First-order method for the orthogonal complement of the curvature subspace.

DSN takes a Newton step inside the subspace it has curvature for, and an AdamW
step everywhere else. This module owns the second half.
"""

import torch


class AdamWState:
    """Flat-vector AdamW moments.

    Matches ``torch.optim.AdamW`` step-for-step. Weight decay is decoupled and
    applied by the optimizer, not here, because it must not be projected.
    """

    def __init__(
        self,
        n: int,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        device=None,
        dtype=None,
    ):
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = torch.zeros(n, device=device, dtype=dtype)
        self.v = torch.zeros(n, device=device, dtype=dtype)

    def step(self, g: torch.Tensor, lr: float) -> torch.Tensor:
        """Return the AdamW displacement for gradient ``g`` (already negated)."""
        self.t += 1
        self.m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
        self.v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
        m_hat = self.m / (1.0 - self.b1**self.t)
        v_hat = self.v / (1.0 - self.b2**self.t)
        return -lr * m_hat / (v_hat.sqrt() + self.eps)


def project_out(d: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``d`` lying in the span of orthonormal ``W``."""
    if W.shape[1] == 0:
        return d
    return d - W @ (W.T @ d)
