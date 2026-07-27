"""Newton solve restricted to a subspace, with saddle-free damping."""

import torch


def subspace_newton(
    T: torch.Tensor,
    Wg: torch.Tensor,
    damping: float,
    saddle_free: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the damped Newton system inside a subspace.

    ``T`` is W'HW and ``Wg`` is W'g for an orthonormal basis W. The step is
    ``d = W @ y``.

    With ``saddle_free`` the eigenvalues are replaced by their absolute values,
    so directions of negative curvature descend instead of ascending toward the
    saddle. Otherwise they are clamped at zero and carried by the damping alone.

    Returns ``(y, eigenvalues, eigenvectors)``; the spectrum is returned because
    the caller needs it for recycling and telemetry, and it is already computed.
    """
    lam, V = torch.linalg.eigh(0.5 * (T + T.T))
    denom = (lam.abs() if saddle_free else lam.clamp_min(0.0)) + damping
    return -(V @ ((V.T @ Wg) / denom)), lam, V


def newton_residual(
    T: torch.Tensor,
    Wg: torch.Tensor,
    y: torch.Tensor,
    beta_next: torch.Tensor,
) -> torch.Tensor:
    """Norm of the Newton residual ``H d + g`` for ``d = W y``.

    Exact for an undeflated Krylov basis, where the residual splits into an
    in-span part ``T y + Wg`` and one orthogonal leak of size
    ``beta_next * y[-1]``. With deflation it is an estimate: the curvature image
    of the recycled space can leave the span in a direction ``beta_next`` does
    not track.
    """
    inside = T @ y + Wg
    leak = beta_next * y[-1]
    return torch.sqrt(inside.dot(inside) + leak * leak)
