"""Lanczos tridiagonalization with full reorthogonalization and deflation.

Yields its state after every direction so the caller can stop adaptively.
"""

from dataclasses import dataclass
from typing import Callable, Iterator

import torch


@dataclass
class LanczosState:
    """State after ``j`` Lanczos directions.

    Q         (n, j) orthonormal basis, orthogonal to the deflation space U
    HQ        (n, j) curvature images H q_i, retained as computed
    alphas    (j,)   diagonal of Q'HQ
    betas     (j-1,) off-diagonal of Q'HQ
    beta_next scalar norm of the next unnormalized Lanczos vector
    n_hvp     number of curvature-vector products consumed so far

    ``HQ`` costs no extra curvature products -- each column is the ``matvec``
    output the recurrence already computed, kept instead of discarded after the
    orthogonalization consumes it. Retaining it is what lets the caller form the
    true residual ``H (W y) + g`` rather than an in-span estimate, which is the
    only measurement that can detect a stale recycled block (see the Task 9
    findings doc, F8).
    """

    Q: torch.Tensor
    HQ: torch.Tensor
    alphas: torch.Tensor
    betas: torch.Tensor
    beta_next: torch.Tensor
    n_hvp: int


def lanczos_iter(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    k_max: int,
    U: torch.Tensor | None = None,
    tol: float = 1e-10,
) -> Iterator[LanczosState]:
    """Build a Krylov basis for ``matvec`` seeded at ``b``, deflated against ``U``.

    Yields a ``LanczosState`` after each new direction. Stops early on breakdown
    (the Krylov space is exhausted, which is a success, not a failure).
    """
    deflate = U is not None and U.shape[1] > 0

    r = b - U @ (U.T @ b) if deflate else b.clone()
    beta = r.norm()
    if beta < tol:
        return

    q_list = [r / beta]
    hq_list: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []

    for j in range(k_max):
        w = matvec(q_list[j])
        hq_list.append(w)
        alpha = torch.dot(w, q_list[j])
        alphas.append(alpha)

        w = w - alpha * q_list[j]
        if j > 0:
            w = w - betas[j - 1] * q_list[j - 1]
        if deflate:
            w = w - U @ (U.T @ w)
        for q in q_list:  # full reorthogonalization
            w = w - torch.dot(w, q) * q

        beta_next = w.norm()

        yield LanczosState(
            Q=torch.stack(q_list, dim=1),
            HQ=torch.stack(hq_list, dim=1),
            alphas=torch.stack(alphas),
            betas=torch.stack(betas) if betas else w.new_zeros(0),
            beta_next=beta_next,
            n_hvp=j + 1,
        )

        if beta_next < tol or j == k_max - 1:
            return
        betas.append(beta_next)
        q_list.append(w / beta_next)
