"""Dynamic Subspace Newton.

Each step: build the smallest subspace whose Newton residual meets a tolerance,
solve the Newton system inside it, and take an AdamW step in the orthogonal
complement so no coordinate is left unoptimized.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .complement import AdamWState, project_out
from .curvature import hvp
from .subspace import KrylovBuilder
from .utils import flat_grad, unflatten_like


@dataclass
class Telemetry:
    """Per-step diagnostics.

    ``n_fallback``, ``n_shrink`` and ``n_reject`` are cumulative and
    deliberately visible: a run that silently spent most of its steps in the
    AdamW fallback path, or rejecting, must be distinguishable from one that did
    not.

    ``rho`` is *this* step's acceptance ratio, measured on this step's own
    closure. It used to be the previous step's ratio, which is what made it
    meaningless under a changing objective (findings doc F1/F2).

    ``step_norm_subspace`` and ``step_norm_complement`` report what was actually
    applied, so a rejected subspace step (``n_reject`` incremented) reports
    ``step_norm_subspace == 0.0``.
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
    n_reject: int = 0


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
        trust_max: float | None = None,
        trust_accept: float = 0.0,
        stationary_eps: float = 1e-10,
        builder=None,
    ):
        params = list(params)
        # `lr` and `weight_decay` both live in the param group, which is where
        # torch.optim's contract says a hyperparameter belongs: schedulers write
        # there, `state_dict()` serializes from there, and anything inspecting
        # the optimizer reads from there.
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        self.trust_radius = trust_radius
        self.trust_grow = trust_grow
        self.trust_shrink = trust_shrink
        # Trust-region hygiene (Conn, Gould & Toint, *Trust Region Methods*,
        # SIAM 2000, §6.1): the radius must be bounded above by some Delta_max.
        # Without it the radius grows geometrically forever on a well-predicted
        # run -- measured 1.4e45 over 200 fixed-batch steps, which already
        # overflows float32 -- and after any transient spike it cannot promptly
        # re-engage a sane clip. Defaults to 1000x the initial radius, so the
        # bound scales with whatever problem scale the caller set.
        self.trust_max = trust_max if trust_max is not None else 1e3 * trust_radius
        # A trust region that never rejects is not a trust region: shrinking the
        # radius after a bad step does nothing about the bad step, which has
        # already been applied. Conn/Gould/Toint Alg. 6.1.1 and STORM both make
        # acceptance conditional on rho exceeding a threshold. `trust_accept=0`
        # is the weakest useful form -- undo only steps that demonstrably raised
        # the loss on their own batch -- which keeps the AdamW complement's
        # progress on every step the model got directionally right.
        self.trust_accept = trust_accept
        # Below this gradient norm, with an empty subspace, the iterate is
        # treated as stationary and the complement does not step (findings doc
        # F9). Matches `lanczos_iter`'s breakdown tolerance, which is what makes
        # the builder return k=0 in the first place.
        self.stationary_eps = stationary_eps

        if builder is not None and damping != 1e-3:
            raise ValueError(
                "damping configures the default KrylovBuilder only; "
                "when builder= is supplied, set damping on the builder."
            )
        self.builder = builder if builder is not None else KrylovBuilder(damping=damping)
        # No self.damping: it was never read. `damping` reaches the solver only
        # through the builder, so storing it here would advertise a knob that
        # does nothing whenever a builder is supplied.

        self._betas = betas
        self._eps = eps
        self.adam: AdamWState | None = None
        self._params: Sequence[torch.Tensor] = []
        self._rebuild_params()

        self.telemetry = Telemetry(trust_radius=trust_radius)
        self._last_W: torch.Tensor | None = None
        self._last_d_complement: torch.Tensor | None = None

    # ---------------------------------------------------------------- params

    def _rebuild_params(self) -> None:
        """Refresh the flat parameter view and size the complement's moments.

        Called on construction and after ``add_param_group``. Parameters are
        optimized as one flat vector across *all* groups; per-group ``lr`` and
        ``weight_decay`` are honored by expanding each group's value over the
        elements it owns.
        """
        params = [p for group in self.param_groups for p in group["params"]]
        if any(
            p.dtype != params[0].dtype or p.device != params[0].device
            for p in params
        ):
            raise ValueError(
                "DSN requires all parameters to share one dtype and device: the "
                "AdamW complement keeps a single flat moment vector built from "
                "the first parameter's dtype/device."
            )
        self._params = params
        n = sum(p.numel() for p in params)

        if self.adam is None:
            self.adam = AdamWState(
                n, betas=self._betas, eps=self._eps,
                device=params[0].device, dtype=params[0].dtype,
            )
        elif n != self.adam.m.numel():
            # A group was added: extend the moments with zeros for the new
            # elements, which is the state torch.optim.AdamW would give a
            # freshly-added parameter.
            grow = n - self.adam.m.numel()
            pad = self.adam.m.new_zeros(grow)
            self.adam.m = torch.cat([self.adam.m, pad])
            self.adam.v = torch.cat([self.adam.v, pad.clone()])

    def add_param_group(self, param_group) -> None:
        super().add_param_group(param_group)
        # `Optimizer.__init__` calls this before DSN's own attributes exist;
        # skip until construction has set them up.
        if getattr(self, "builder", None) is None:
            return
        self._rebuild_params()
        # The recycled basis was built for a narrower flat vector; its rows no
        # longer line up with the parameters.
        self.builder.reset()

    def _group_values(self, key: str) -> torch.Tensor | float:
        """Per-element view of a param-group hyperparameter.

        Single-group (the common case) short-circuits to the scalar, so the
        per-step cost of multi-group support is only paid by multi-group users.
        """
        if len(self.param_groups) == 1:
            return float(self.param_groups[0][key])
        ref = self._params[0]
        return torch.cat([
            torch.full((p.numel(),), float(group[key]),
                       device=ref.device, dtype=ref.dtype)
            for group in self.param_groups for p in group["params"]
        ])

    @property
    def weight_decay(self) -> float:
        """The first group's decay. Kept for callers that read it off DSN."""
        return float(self.param_groups[0]["weight_decay"])

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict:
        """``torch.optim.Optimizer.state_dict()`` plus everything DSN carries.

        The base implementation serializes ``param_groups`` and the (empty)
        per-parameter ``state`` dict, which for DSN is none of the state that
        matters: the AdamW moments, the trust radius, and the recycled basis all
        live outside it. Resuming from a base-class state_dict silently
        restarted from zero -- see the README's API limitations before Plan 2.
        """
        sd = super().state_dict()
        sd["dsn"] = {
            "adam_m": self.adam.m.detach().clone(),
            "adam_v": self.adam.v.detach().clone(),
            "adam_t": self.adam.t,
            "trust_radius": self.trust_radius,
            "trust_max": self.trust_max,
            "telemetry": vars(self.telemetry).copy(),
            "builder": (
                self.builder.state_dict()
                if hasattr(self.builder, "state_dict") else None
            ),
        }
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        sd = dict(state_dict)
        dsn = sd.pop("dsn", None)
        super().load_state_dict(sd)
        self._rebuild_params()
        if dsn is None:
            return
        self.adam.m = dsn["adam_m"].detach().clone().to(self.adam.m)
        self.adam.v = dsn["adam_v"].detach().clone().to(self.adam.v)
        self.adam.t = int(dsn["adam_t"])
        self.trust_radius = float(dsn["trust_radius"])
        self.trust_max = float(dsn["trust_max"])
        self.telemetry = Telemetry(**dsn["telemetry"])
        if dsn["builder"] is not None and hasattr(self.builder, "load_state_dict"):
            self.builder.load_state_dict(dsn["builder"])

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def _add_to_params(self, d: torch.Tensor) -> None:
        for p, dp in zip(self._params, unflatten_like(d, self._params)):
            p.add_(dp)

    def _update_trust_radius(self, reduction: float, predicted: float) -> float:
        """Same-batch acceptance ratio: measured reduction over predicted.

        ``reduction`` and ``predicted`` must come from the SAME closure -- the
        same mini-batch -- or the ratio measures sampling noise rather than
        model quality. That is the whole content of findings F1/F2: the previous
        lagged form compared the model built on batch t against the loss of
        batch t+1, so under mini-batch training `rho` was noise, `n_shrink`
        climbed on nearly every step, and the radius collapsed to ~1e-11.

        The finite-sum trust-region literature converges on the same invariant
        from two directions: inexact-restoration trust region for finite sums
        (Bellavia, Krejic & Morini) and progressive-sampling subsampled Newton
        (Bollapragada, Byrd & Nocedal) both require the acceptance test to be
        evaluated on a *consistent* sample rather than a noise-free one, and
        STORM (Chen, Menickelly & Scheinberg, Math. Prog. 2018) requires the
        estimates entering the ratio to be sufficiently accurate at the current
        iterate. Re-evaluating the same closure after the step is the cheapest
        construction that satisfies this: one extra forward pass, no extra
        curvature products, and zero sampling noise in the ratio by
        construction.
        """
        rho = float("nan")
        # With the shipped saddle-free solver this gate never falls through:
        # predicted = sum_i c_i^2 [s/(|lam_i|+d) - 0.5 s^2 lam_i/(|lam_i|+d)^2]
        # is > 0 unless y == 0 or W is empty. For lam < 0 both terms are
        # positive; for lam > 0 the bracket factors to
        # s/(lam+d) * [1 - 0.5 s lam/(lam+d)] >= 0.5 s/(lam+d) > 0 since
        # s <= 1. So `rho` staying nan is unreachable with KrylovBuilder --
        # but it IS reachable for a third-party SubspaceBuilder, which the
        # Protocol in subspace/base.py invites, so the gate stays.
        if predicted > 0:
            rho = reduction / predicted
            if rho < 0.25:
                self.trust_radius *= self.trust_shrink
                self.telemetry.n_shrink += 1
            elif rho > 0.75:
                self.trust_radius = min(
                    self.trust_radius * self.trust_grow, self.trust_max
                )
        return rho

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor:
        """Take one optimization step.

        ``closure`` must rebuild the loss graph on every call; it is invoked once
        for the gradient, once per curvature-vector product, and once more after
        the step to evaluate the trust-region acceptance ratio on the same batch
        the step was planned for.

        Unlike ``torch.optim.Optimizer.step``, the closure is required rather
        than optional: DSN cannot form a curvature-vector product from a
        pre-populated ``.grad`` alone. The signature matches the base class so
        that callers passing ``closure=`` positionally or by keyword work, and
        the omission is reported rather than silently mis-stepping.
        """
        if closure is None:
            raise ValueError(
                "DSN.step() requires a closure: it re-evaluates the loss once "
                "per curvature-vector product and once more for the "
                "trust-region acceptance ratio, which a pre-populated .grad "
                "cannot provide."
            )
        loss = closure()
        g = flat_grad(loss, self._params, create_graph=False)
        gnorm = float(g.norm())
        lr = self._group_values("lr")

        n_hvp = 0

        def matvec(v: torch.Tensor) -> torch.Tensor:
            nonlocal n_hvp
            n_hvp += 1
            return hvp(closure, self._params, v)

        try:
            res = self.builder.build(matvec, g)
            if not torch.isfinite(res.y).all():
                raise FloatingPointError("non-finite subspace solution")
        except (FloatingPointError, torch.linalg.LinAlgError):
            self.telemetry = Telemetry(
                n_hvp=n_hvp,
                rel_residual=float("nan"),
                n_fallback=self.telemetry.n_fallback + 1,
                n_shrink=self.telemetry.n_shrink,
                n_reject=self.telemetry.n_reject,
                trust_radius=self.trust_radius,
                rho=float("nan"),
            )
            self.builder.reset()
            d = self.adam.step(g, lr)
            self._last_W = None
            self._last_d_complement = None
            self._apply(d)
            return loss

        k = res.W.shape[1]
        d_sub = res.W @ res.y if k else torch.zeros_like(g)

        if k == 0 and gnorm < self.stationary_eps:
            # Stationarity guard (findings doc F9). The builder returns an empty
            # subspace at a vanishing gradient, but AdamW's first moment still
            # carries the previous steps' gradients, and `project_out` against a
            # zero-width basis is the identity -- so the full momentum-driven
            # displacement used to be applied on top of a zero Newton step and
            # walked the iterate straight back off an exact minimum. An optimum
            # must be a fixed point of the iteration.
            d_comp = torch.zeros_like(g)
        else:
            d_comp = project_out(self.adam.step(g, lr), res.W)

        # The trust region bounds the WHOLE step, not just its subspace part
        # (Conn, Gould & Toint, §6.1: the constraint is ||d|| <= Delta). Scaling
        # only `d_sub` left `d_comp` at essentially unit norm while the subspace
        # step was clipped toward zero, so a collapsing region degraded DSN
        # toward a method that takes unit-norm steps carrying no first-order
        # descent at all (findings doc F4). The two parts are orthogonal by
        # construction -- `d_comp` is projected out of span(W) and `d_sub` lies
        # in it -- so ||d||^2 = ||d_sub||^2 + ||d_comp||^2 exactly and one
        # common factor bounds both.
        norm = (d_sub + d_comp).norm()
        s = 1.0
        if norm > self.trust_radius:
            s = float(self.trust_radius / norm)
            d_sub, d_comp = d_sub * s, d_comp * s

        # Predicted reduction of the quadratic model, scaled by the clip factor
        # `s` so a clipped step is judged against the reduction it actually
        # produces (s == 1.0 when no clipping occurred). The complement
        # contributes no first-order term: g lies in span(W) by construction
        # (findings doc F4), so g'd_comp is exactly zero and the model's linear
        # part is carried entirely by the subspace step.
        Wg = res.W.T @ g if k else g.new_zeros(0)
        predicted = float(
            -(s * Wg.dot(res.y) + 0.5 * s * s * res.y.dot(res.T @ res.y))
        ) if k else 0.0

        # The acceptance ratio must compare the model against the reality of the
        # step the model describes -- the same invariant as F1/F2, applied to
        # the step decomposition rather than to the batch. `predicted` models
        # `d_sub` only; the complement contributes no first-order term (g lies
        # in span(W), so g'd_comp == 0) but its unmodeled curvature
        # d_comp' H d_comp is positive and would bias every ratio downward,
        # shrinking the region for a reason the model never claimed to cover.
        # So `d_sub` is applied and scored on its own, and the complement --
        # AdamW's own step, which the quadratic model does not describe -- is
        # applied afterwards regardless. A collapsing region therefore degrades
        # DSN toward plain AdamW, which is the intended floor (findings F5).
        prev = [p.detach().clone() for p in self._params] if predicted > 0 else None
        self._add_to_params(d_sub)

        # Same-batch acceptance ratio: re-evaluate the SAME closure at the
        # trial point. No gradient is needed, so this is one forward pass.
        with torch.no_grad():
            loss_mid = float(closure())
        rho = self._update_trust_radius(float(loss.detach()) - loss_mid, predicted)

        if prev is not None and rho <= self.trust_accept:
            with torch.no_grad():
                for p, old in zip(self._params, prev):
                    p.copy_(old)
            self.telemetry.n_reject += 1
            d_sub = torch.zeros_like(d_sub)

        self._apply(d_comp)

        self._last_W = res.W
        self._last_d_complement = d_comp
        self.telemetry = Telemetry(
            k=k,
            n_hvp=n_hvp,
            rel_residual=res.rel_residual,
            reuse_frac=res.reuse_frac,
            trust_radius=self.trust_radius,
            rho=rho,
            step_norm_subspace=float(d_sub.norm()),
            step_norm_complement=float(d_comp.norm()),
            n_fallback=self.telemetry.n_fallback,
            n_shrink=self.telemetry.n_shrink,
            n_reject=self.telemetry.n_reject,
        )
        return loss

    @torch.no_grad()
    def _apply(self, d: torch.Tensor) -> None:
        """Apply the update plus decoupled weight decay.

        Weight decay is applied to the parameters directly rather than folded
        into ``g``, and is never projected, matching AdamW semantics. Unlike
        AdamW it is applied *after* the subspace step rather than before, so the
        acceptance ratio scores the Newton step alone and is not perturbed by a
        regularizer the quadratic model does not describe. The difference is
        second order in ``lr * weight_decay``.
        """
        for group in self.param_groups:
            wd = group["weight_decay"]
            if wd:
                factor = 1.0 - group["lr"] * wd
                for p in group["params"]:
                    p.mul_(factor)
        self._add_to_params(d)
