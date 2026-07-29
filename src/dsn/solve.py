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
    HW: torch.Tensor,
    g: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Norm of the Newton residual ``H d + g`` for ``d = W y``.

    Computed from the curvature images ``HW`` (column ``i`` is ``H w_i``)
    directly, so it is the *true* residual against the operator that produced
    those images, with no in-span/leak split and no dependence on the projected
    curvature ``T``.

    This replaces an earlier ``(T, Wg, y, beta_next)`` form that summed the
    in-span part ``T y + Wg`` with a single orthogonal leak ``beta_next*y[-1]``.
    That form was exact only for an undeflated Krylov basis: with deflation the
    curvature image of the recycled block leaves the span in directions
    ``beta_next`` does not track, and -- worse -- it measured against the same
    ``T`` that a stale recycled block corrupts, so the reported residual
    *improved* as the subspace degraded (Task 9 findings doc, F8). Measuring
    against ``HW`` removes that blind spot by construction: no quantity in this
    expression comes from a previous iterate.
    """
    return (HW @ y + g).norm()
