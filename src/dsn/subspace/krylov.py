"""Adaptive-rank Krylov subspace with cross-step recycling (backend A).

Standard Hessian-free optimization throws its Krylov basis away every step and
pays 20-100 curvature products to rebuild an equivalent one. Curvature
eigenvectors are strongly correlated between adjacent steps, so this builder
keeps the most useful converged Ritz vectors and deflates against them.

The retained directions are *re-projected against the current step's operator*
before use: ``HU`` is recomputed as ``H_now @ U`` at the top of every ``build``,
costing ``m_recycle`` curvature products. This is the price GCRO-DR pays for a
sequence of systems with a changing operator (Parks, de Sturler et al.,
*Recycling Krylov Subspaces for Sequences of Linear Systems*, SIAM J. Sci.
Comput. 28(5), 2006, which recomputes ``C = A_new U`` per system). Carrying
``HU`` over instead -- what this builder used to do -- makes the projected
curvature ``T`` something other than ``WᵀHW`` at the current iterate, and the
error is invisible to any metric computed against that same ``T`` (Task 9
findings doc, F8).

**Recycling does not save curvature products, and paying for it honestly is why.**
Refreshing m recycled directions costs exactly m products, and deflating them
saves at most the m new Lanczos directions they stand in for, so exact recycling
is break-even by construction; measured on drifting SPD operators at equal
tolerance it runs 1.03-1.11x the fresh rebuild's product count (n=40/60/80,
cond 1e2-1e4, 3 seeds each). The earlier measured saving came from the
stale-``HU`` shortcut, whose optimistic residual tripped the ``tau`` stopping
rule early. What recycling does buy is *parity at equal width* with a
from-scratch basis while carrying curvature information across steps; see the
Plan 2 doc (F8, F10) for the full comparison.
"""

from typing import Callable

import torch

from ..lanczos import lanczos_iter
from ..solve import newton_residual, subspace_newton
from .base import SubspaceResult


def _empty_result(g: torch.Tensor) -> SubspaceResult:
    """The zero-width subspace result: no basis, no HVPs, nothing recycled."""
    empty = g.new_zeros(g.numel(), 0)
    return SubspaceResult(empty, g.new_zeros(0, 0), g.new_zeros(0), 0.0, 0, 0.0)


def _tridiagonal(alphas: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    T = torch.diag(alphas)
    if betas.numel():
        i = torch.arange(betas.numel(), device=betas.device)
        T[i, i + 1] = betas
        T[i + 1, i] = betas
    return T


class KrylovBuilder:
    def __init__(
        self,
        k_max: int = 16,
        m_recycle: int = 8,
        tau: float = 0.1,
        eps_marginal: float = 0.0,
        damping: float = 1e-3,
        saddle_free: bool = True,
        rank_by: str = "contribution",
        ritz_tol: float = 1e-3,
    ):
        if rank_by not in ("contribution", "small_eig", "large_eig"):
            raise ValueError(f"unknown rank_by: {rank_by}")
        self.k_max = k_max
        self.m_recycle = m_recycle
        self.tau = tau
        self.eps_marginal = eps_marginal
        self.damping = damping
        self.saddle_free = saddle_free
        self.rank_by = rank_by
        # Deflation is only justified for a CONVERGED Ritz pair: an unconverged
        # Ritz vector is not close to an invariant direction of H, so removing
        # it from the Krylov iteration does not shrink the effective spectrum,
        # it only steals width from the new directions (Saad, *Numerical Methods
        # for Large Eigenvalue Problems*, ch. 6; the same convergence test
        # governs which harmonic Ritz vectors GMRES-DR/GCRO-DR retain). A
        # direction is retained only if ||H u - theta u|| <= ritz_tol * |theta|.
        # Measured separation on the drifting-SPD benchmark: converged
        # directions score 1e-15..1e-6, unconverged ones 0.1..2.8, so 1e-3 sits
        # in a five-order-of-magnitude gap rather than on a knife edge.
        self.ritz_tol = ritz_tol
        # Only the basis U crosses a step boundary. Its curvature image is
        # deliberately NOT stored: it is recomputed against the current
        # operator inside every `build`, which is the whole point of F8's fix.
        self.U: torch.Tensor | None = None
        self._last_eig: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        self.U = None
        self._last_eig = None

    def state_dict(self) -> dict:
        """The state that must survive a checkpoint: the recycled basis."""
        return {"U": None if self.U is None else self.U.detach().clone()}

    def load_state_dict(self, state: dict) -> None:
        U = state.get("U")
        self.U = None if U is None else U.detach().clone()
        self._last_eig = None

    def build(
        self, matvec: Callable[[torch.Tensor], torch.Tensor], g: torch.Tensor
    ) -> SubspaceResult:
        """Build this step's subspace, stopping at `tau` or on negligible marginal gain.

        `eps_marginal=0.0` (the default) disables the marginal-gain rule entirely, so
        `tau` is the only stopping criterion; this matters because `rel_residual` is
        not monotonically decreasing in Krylov dimension, so a nonzero `eps_marginal`
        can trigger on an ordinary transient upward blip, not just genuine plateauing.
        """
        gnorm = g.norm()
        m = self.U.shape[1] if self.U is not None else 0

        if gnorm == 0:
            # Deliberate: this early return does NOT clear U, unlike the
            # k_max == 0 path, which falls through to `_recycle` with a
            # zero-width result and drops it. A zero-gradient step carries no
            # new curvature information, so the basis built at the previous
            # iterate is still the best available one for the next step;
            # throwing it away would force a full rebuild for no gain. It also
            # spends no curvature products: the HU refresh below is skipped.
            return _empty_result(g)

        # Re-project the recycled basis against THIS step's operator before
        # trusting it (GCRO-DR's `C = A_new U`). Costs m curvature products and
        # is what makes T below equal W'HW at the current iterate rather than a
        # surrogate assembled from the previous one.
        HU = None
        n_refresh = 0
        if m and self.k_max > 0:
            HU = torch.stack([matvec(self.U[:, i]) for i in range(m)], dim=1)
            n_refresh = m
            # Re-validate, not just re-project. A direction that was a converged
            # Ritz pair at the previous iterate need not still be one here: the
            # curvature image is now exact, but the DIRECTION is what went
            # stale. Deflating a direction that is no longer near-invariant
            # steals width from the new Krylov vectors, which is what made
            # recycling lose to a fresh basis on fast-moving problems. The test
            # is free -- HU is already computed.
            keep = self._converged(self.U, HU)
            if not bool(keep.all()):
                if not bool(keep.any()):
                    self.U = HU = None
                    m = 0
                else:
                    self.U = self.U[:, keep]
                    HU = HU[:, keep]
                    m = int(keep.sum())

        best = None
        best_HW = None
        prev_rel = float("inf")

        for st in lanczos_iter(matvec, g, self.k_max, U=self.U):
            W = st.Q if m == 0 else torch.cat([self.U, st.Q], dim=1)
            HW = st.HQ if m == 0 else torch.cat([HU, st.HQ], dim=1)
            T = self._assemble(st, m, HU)
            Wg = W.T @ g
            y, lam, V = subspace_newton(T, Wg, self.damping, self.saddle_free)
            rel = float(newton_residual(HW, g, y) / gnorm)

            best = SubspaceResult(
                W=W, T=T, y=y, rel_residual=rel, n_hvp=st.n_hvp + n_refresh,
                reuse_frac=m / W.shape[1],
            )
            best_HW = HW
            self._last_eig = (lam, V)

            if rel <= self.tau or (
                self.eps_marginal > 0.0
                and (prev_rel - rel) < self.eps_marginal
            ):
                break
            prev_rel = rel

        if best is None:
            # Lanczos produced no direction (the deflated gradient broke down).
            # The refresh products were still spent, so report them rather than
            # letting an empty result understate the step's cost.
            best = _empty_result(g)
            best = SubspaceResult(
                best.W, best.T, best.y, best.rel_residual, n_refresh, best.reuse_frac
            )

        self._recycle(best, best_HW)
        return best

    def _converged(self, U: torch.Tensor, HU: torch.Tensor) -> torch.Tensor:
        """Mask of directions whose Ritz pair has converged against ``HU``'s operator.

        ``theta_i`` is the Rayleigh quotient ``u_i' H u_i`` and the test is
        ``||H u_i - theta_i u_i|| <= ritz_tol * |theta_i|``. The denominator is
        floored at a fixed fraction of the spectral scale so a Ritz value at or
        near zero -- which is never safely deflatable -- cannot report a
        spuriously small relative residual.
        """
        theta = (U * HU).sum(dim=0)
        scale = theta.abs().max()
        if not (scale > 0):
            return torch.zeros_like(theta, dtype=torch.bool)
        denom = torch.maximum(theta.abs(), 1e-8 * scale)
        return (HU - U * theta).norm(dim=0) <= self.ritz_tol * denom

    def _assemble(self, st, m: int, HU: torch.Tensor | None) -> torch.Tensor:
        """Build W'HW from the Lanczos tridiagonal plus the refreshed H U blocks.

        Every block is exact at the current iterate. ``Tqq`` is exact because
        ``Q`` is orthogonal to ``U``, so ``QᵀHQ = Qᵀ(PHP)Q`` for the deflation
        projector ``P = I - UUᵀ``, and the deflated recurrence tridiagonalizes
        ``PHP``. ``Tuu`` and ``Tqu`` are exact because ``HU`` was recomputed
        against this step's operator in ``build``.
        """
        Tqq = _tridiagonal(st.alphas, st.betas)
        if m == 0:
            return Tqq
        Tuu = self.U.T @ HU
        Tuu = 0.5 * (Tuu + Tuu.T)
        Tqu = st.Q.T @ HU
        return torch.cat(
            [torch.cat([Tuu, Tqu.T], dim=1), torch.cat([Tqu, Tqq], dim=1)], dim=0
        )

    def _recycle(self, res: SubspaceResult, HW: torch.Tensor | None) -> None:
        """Retain the most useful CONVERGED Ritz vectors for the next step.

        Only the basis is retained. Their curvature images are not: they would
        be images under *this* step's Hessian, and `build` recomputes them under
        the next step's operator instead.

        Two filters, in this order:

        1. **Convergence** (hard gate). A Ritz pair is deflatable only once
           ``||H u - theta u|| <= ritz_tol * |theta|``. Ranking heuristics alone
           select by usefulness-to-the-step, which is a different property from
           being close to an invariant direction, and deflating a direction that
           is not near-invariant costs Krylov width while shrinking nothing.
           Measured (F10 in the Plan 2 doc): with this gate absent, the shipped
           "contribution" ranking retained directions whose relative Ritz
           residual was 0.1-2.8 -- not converged by any margin -- and the
           resulting recycled subspace was *worse* than a fresh basis of the
           same width. The residual images ``H u_i - theta_i u_i`` come from
           ``HW``, so the gate costs no curvature products.
        2. **Usefulness** (ranking), applied among the survivors only.

        If nothing has converged, ``U`` is dropped and the next step rebuilds
        from scratch -- the "fall back to fresh rather than recycle stale
        directions" path.
        """
        if self.m_recycle == 0 or res.W.shape[1] == 0 or HW is None:
            self.U = None
            return

        lam, V = self._last_eig
        scale = lam.abs().max()
        if not (scale > 0):
            self.U = None
            return

        # Ritz residual per candidate direction u_i = W v_i, theta_i = lam_i.
        # Floor the denominator at a fixed fraction of the spectral scale: a
        # Ritz value at (or near) zero is not safely deflatable, and dividing by
        # it would report a spuriously small residual for exactly that case.
        ritz_res = (HW @ V - res.W @ (V * lam)).norm(dim=0)
        converged = ritz_res <= self.ritz_tol * torch.maximum(
            lam.abs(), 1e-8 * scale
        )
        n_ok = int(converged.sum())
        if n_ok == 0:
            self.U = None
            return

        if self.rank_by == "contribution":
            # Ritz-coordinate magnitude of the step: how much each eigendirection
            # actually moves the parameters. Ranking by |lambda| instead would
            # retain the directions the Newton step least depends on, since it
            # weights each direction by 1/lambda. That argument is about step
            # ACCURACY and remains correct; it is not an argument about which
            # directions are deflatable, which is what filter 1 decides. Both
            # rankings are applied only to converged candidates, so the
            # heuristic now chooses among directions that deflation theory
            # admits rather than overriding it.
            score = (V.T @ res.y).abs()
        elif self.rank_by == "small_eig":
            score = -lam.abs()
        else:
            score = lam.abs()

        score = torch.where(converged, score, score.new_full((), -float("inf")))
        keep = torch.topk(score, min(self.m_recycle, n_ok)).indices
        self.U = res.W @ V[:, keep]
