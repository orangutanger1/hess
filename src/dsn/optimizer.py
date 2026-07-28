"""Dynamic Subspace Newton.

Each step: build the smallest subspace whose Newton residual meets a tolerance,
solve the Newton system inside it, and take an AdamW step in the orthogonal
complement so no coordinate is left unoptimized.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from .complement import AdamWState, project_out
from .curvature import hvp
from .subspace import KrylovBuilder
from .utils import flat_grad, flatten, unflatten_like


@dataclass
class Telemetry:
    """Per-step diagnostics.

    ``n_fallback`` and ``n_shrink`` are cumulative and deliberately visible: a
    run that silently spent most of its steps in the AdamW fallback path must be
    distinguishable from one that did not.
    """

    k: int = 0
    n_hvp: int = 0
    rel_residual: float = 0.0
    reuse_frac: float = 0.0
    trust_radius: float = 0.0
    rho: float = float("nan")
    step_norm_subspace: float = 0.0
    step_norm_complement: float = 0.0
    n_fallback: int = 0
    n_shrink: int = 0


class DSN(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        damping: float = 1e-3,
        trust_radius: float = 1.0,
        trust_grow: float = 2.0,
        trust_shrink: float = 0.5,
        builder=None,
    ):
        params = list(params)
        super().__init__(params, dict(lr=lr))
        self._params: Sequence[torch.Tensor] = [
            p for group in self.param_groups for p in group["params"]
        ]
        n = sum(p.numel() for p in self._params)

        self.lr = lr
        self.weight_decay = weight_decay
        self.damping = damping
        self.trust_radius = trust_radius
        self.trust_grow = trust_grow
        self.trust_shrink = trust_shrink
        self.builder = builder if builder is not None else KrylovBuilder(damping=damping)
        self.adam = AdamWState(
            n, betas=betas, eps=eps,
            device=self._params[0].device, dtype=self._params[0].dtype,
        )

        self.telemetry = Telemetry(trust_radius=trust_radius)
        self._pending: tuple[float, float] | None = None  # (prev_loss, predicted)
        self._last_W: torch.Tensor | None = None
        self._last_d_complement: torch.Tensor | None = None

    @torch.no_grad()
    def _add_to_params(self, d: torch.Tensor) -> None:
        for p, dp in zip(self._params, unflatten_like(d, self._params)):
            p.add_(dp)

    def _update_trust_radius(self, loss_now: float) -> float:
        """Lagged acceptance ratio: compare the last step's prediction to reality."""
        rho = float("nan")
        if self._pending is not None:
            prev_loss, predicted = self._pending
            if predicted > 0:
                rho = (prev_loss - loss_now) / predicted
                if rho < 0.25:
                    self.trust_radius *= self.trust_shrink
                    self.telemetry.n_shrink += 1
                elif rho > 0.75:
                    self.trust_radius *= self.trust_grow
        return rho

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Take one optimization step.

        ``closure`` must rebuild the loss graph on every call; it is invoked once
        for the gradient and once per curvature-vector product.
        """
        loss = closure()
        g = flat_grad(loss, self._params, create_graph=False)
        rho = self._update_trust_radius(float(loss.detach()))

        n_hvp = 0

        def matvec(v: torch.Tensor) -> torch.Tensor:
            nonlocal n_hvp
            n_hvp += 1
            return hvp(closure, self._params, v)

        try:
            res = self.builder.build(matvec, g)
            if not torch.isfinite(res.y).all():
                raise FloatingPointError("non-finite subspace solution")
        except (FloatingPointError, torch._C._LinAlgError):
            self.telemetry = Telemetry(
                n_hvp=n_hvp,
                rel_residual=float("nan"),
                n_fallback=self.telemetry.n_fallback + 1,
                n_shrink=self.telemetry.n_shrink,
                trust_radius=self.trust_radius,
                rho=rho,
            )
            self.builder.reset()
            d = self.adam.step(g, self.lr)
            self._apply(d)
            self._pending = None
            return loss

        d_sub = res.W @ res.y if res.W.shape[1] else torch.zeros_like(g)
        norm = d_sub.norm()
        s = 1.0
        if norm > self.trust_radius:
            s = self.trust_radius / norm
            d_sub = d_sub * s

        d_comp = project_out(self.adam.step(g, self.lr), res.W)

        # Predicted reduction of the quadratic model, scaled by the trust-region
        # clip factor `s` so a clipped step is judged against the reduction it
        # actually produces, not the reduction the unclipped Newton step would
        # have produced (s == 1.0 when no clipping occurred).
        Wg = res.W.T @ g if res.W.shape[1] else g.new_zeros(0)
        predicted = (
            -(s * Wg.dot(res.y) + 0.5 * s * s * res.y.dot(res.T @ res.y))
            if res.W.shape[1]
            else 0.0
        )
        self._pending = (float(loss.detach()), float(predicted))

        self._last_W = res.W
        self._last_d_complement = d_comp
        self.telemetry = Telemetry(
            k=res.W.shape[1],
            n_hvp=n_hvp,
            rel_residual=res.rel_residual,
            reuse_frac=res.reuse_frac,
            trust_radius=self.trust_radius,
            rho=rho,
            step_norm_subspace=float(d_sub.norm()),
            step_norm_complement=float(d_comp.norm()),
            n_fallback=self.telemetry.n_fallback,
            n_shrink=self.telemetry.n_shrink,
        )

        self._apply(d_sub + d_comp)
        return loss

    @torch.no_grad()
    def _apply(self, d: torch.Tensor) -> None:
        """Apply the update plus decoupled weight decay.

        Weight decay is applied to the parameters directly rather than folded
        into ``g``, and is never projected, matching AdamW semantics.
        """
        if self.weight_decay:
            for p in self._params:
                p.mul_(1.0 - self.lr * self.weight_decay)
        self._add_to_params(d)
