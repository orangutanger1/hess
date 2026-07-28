"""Adaptive-rank Krylov subspace with cross-step recycling (backend A).

Standard Hessian-free optimization throws its Krylov basis away every step and
pays 20-100 curvature products to rebuild an equivalent one. Curvature
eigenvectors are strongly correlated between adjacent steps, so this builder
keeps the most useful Ritz vectors, deflates against them, and spends its
products only on directions it does not already have.
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
        self.U: torch.Tensor | None = None
        self.HU: torch.Tensor | None = None
        self._last_eig: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        self.U = None
        self.HU = None
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
            # Deliberate: this early return does NOT clear U/HU, unlike the
            # k_max == 0 path, which falls through to `_recycle` with a
            # zero-width result and drops them. A zero-gradient step carries no
            # new curvature information, so the basis built at the previous
            # iterate is still the best available one for the next step;
            # throwing it away would force a full rebuild for no gain.
            return _empty_result(g)

        best = None
        prev_rel = float("inf")

        for st in lanczos_iter(matvec, g, self.k_max, U=self.U):
            W = st.Q if m == 0 else torch.cat([self.U, st.Q], dim=1)
            T = self._assemble(st, m)
            Wg = W.T @ g
            y, lam, V = subspace_newton(T, Wg, self.damping, self.saddle_free)
            rel = float(newton_residual(T, Wg, y, st.beta_next) / gnorm)

            best = SubspaceResult(
                W=W, T=T, y=y, rel_residual=rel, n_hvp=st.n_hvp,
                reuse_frac=m / W.shape[1],
            )
            self._last_eig = (lam, V)

            if rel <= self.tau or (
                self.eps_marginal > 0.0
                and (prev_rel - rel) < self.eps_marginal
            ):
                break
            prev_rel = rel

        if best is None:
            best = _empty_result(g)

        self._recycle(best)
        return best

    def _assemble(self, st, m: int) -> torch.Tensor:
        """Build W'HW from the Lanczos tridiagonal plus the stored H U blocks."""
        Tqq = _tridiagonal(st.alphas, st.betas)
        if m == 0:
            return Tqq
        Tuu = self.U.T @ self.HU
        Tuu = 0.5 * (Tuu + Tuu.T)
        Tqu = st.Q.T @ self.HU
        return torch.cat(
            [torch.cat([Tuu, Tqu.T], dim=1), torch.cat([Tqu, Tqq], dim=1)], dim=0
        )

    def _recycle(self, res: SubspaceResult) -> None:
        """Retain the most useful Ritz vectors for the next step.

        Their curvature images come from H W ~= W T, which is already computed,
        so refreshing HU costs no extra curvature products.
        """
        if self.m_recycle == 0 or res.W.shape[1] == 0:
            self.U = self.HU = None
            return

        lam, V = self._last_eig
        if self.rank_by == "contribution":
            # Ritz-coordinate magnitude of the step: how much each eigendirection
            # actually moves the parameters. Ranking by |lambda| instead would
            # retain the directions the Newton step least depends on, since it
            # weights each direction by 1/lambda.
            #
            # The same 1/lambda argument predicts the ordering of the two
            # eigenvalue heuristics: "small_eig" keeps the directions the step
            # weights most, so it should be the better of the two, and it is.
            # Measured (slowly-drifting SPD operator, sum of n_hvp over steps
            # 1-4, lower is better; 6 seeds x 2 configurations, n=30/k_max=15/
            # m_recycle=6 and n=40/k_max=20/m_recycle=8): small_eig beat
            # large_eig in 12 of 12 runs. Note that "contribution" does NOT
            # beat "small_eig" reliably on this benchmark -- ratios ranged
            # 0.755 to 1.250 across the same 12 runs -- so only the
            # small_eig < large_eig ordering is asserted in
            # tests/test_krylov_builder.py.
            score = (V.T @ res.y).abs()
        elif self.rank_by == "small_eig":
            score = -lam.abs()
        else:
            score = lam.abs()

        keep = torch.topk(score, min(self.m_recycle, score.numel())).indices
        Vk = V[:, keep]
        self.U = res.W @ Vk
        self.HU = res.W @ (res.T @ Vk)
