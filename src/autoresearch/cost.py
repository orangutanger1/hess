"""FLOP accounting for both stages.

The design document fixes the headline metric as loss versus FLOPs, not loss
versus steps, "because a method that halves step count at 4x per-step cost is a
regression, and step-count plots hide exactly that".  The same argument applies
one level up: a *search* that screens forty candidates to find one winner has
spent those FLOPs, and a ledger that reports only the winner's training cost
hides exactly that.  So every count here is tagged with the stage that paid it,
and ``report.py`` refuses to print a total that omits either.

Multipliers
-----------

FLOPs are expressed as multiples of one forward pass:

    forward     1x       (measured, or 2 * n_params per sample for dense layers)
    gradient    3x       forward + backward, backward ~= 2x forward
    curvature   6x       double-backward: a graph-retaining forward+backward,
                         then a second backward through it

These are the standard estimates and they are *estimates*.  ``calibrate`` exists
because the design's P3 acceptance criterion is agreement with
``torch.utils.flop_counter.FlopCounterMode`` within 5%, and the only way to meet
that is to measure the model in question rather than assume the multipliers.
Use ``calibrate`` whenever the model is available; the analytic path is for
synthetic problems where no ``nn.Module`` exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from .probes import ProbeCost


@dataclass(frozen=True)
class CostModel:
    """FLOPs per operation, as a function of batch size."""

    fwd_flops_per_sample: float
    grad_multiplier: float = 3.0
    hvp_multiplier: float = 6.0
    #: How the multipliers were obtained.  Printed in every report, because a
    #: FLOP total whose provenance is unstated is a number nobody can check.
    provenance: str = "analytic (2 * n_params/sample, 3x grad, 6x curvature)"

    def forward(self, batch: float) -> float:
        return self.fwd_flops_per_sample * batch

    def grad(self, batch: float) -> float:
        return self.fwd_flops_per_sample * batch * self.grad_multiplier

    def hvp(self, batch: float) -> float:
        return self.fwd_flops_per_sample * batch * self.hvp_multiplier

    def total(self, cost: ProbeCost) -> float:
        t = 0.0
        for bs, n in cost.forwards.items():
            t += self.forward(bs) * n
        for bs, n in cost.grads.items():
            t += self.grad(bs) * n
        for bs, n in cost.hvps.items():
            t += self.hvp(bs) * n
        return t


def analytic_cost_model(n_params: int) -> CostModel:
    """``2 * n_params`` multiply-adds per sample -- the dense-layer convention."""
    return CostModel(
        fwd_flops_per_sample=2.0 * n_params,
        provenance=f"analytic: 2 * {n_params} params/sample, 3x grad, 6x curvature",
    )


def calibrate(
    model: torch.nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    params: Sequence[torch.Tensor],
    batch: int,
) -> CostModel:
    """Measure forward / gradient / curvature FLOPs with ``FlopCounterMode``.

    ``loss_fn`` must rebuild its graph on every call.  Returns a ``CostModel``
    whose multipliers are the measured ratios, so downstream totals are measured
    rather than assumed.  Falls back to the analytic model, with the reason
    recorded in ``provenance``, if ``FlopCounterMode`` is unavailable or returns
    zero (it counts only the ops it knows; a model built from unrecognized ops
    would otherwise silently report a free forward pass).
    """
    n_params = sum(p.numel() for p in params)
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:  # pragma: no cover - depends on torch build
        m = analytic_cost_model(n_params)
        return CostModel(
            m.fwd_flops_per_sample,
            m.grad_multiplier,
            m.hvp_multiplier,
            provenance=f"analytic fallback: FlopCounterMode unavailable ({exc})",
        )

    from dsn.curvature import hvp
    from dsn.utils import flat_grad

    def count(fn) -> float:
        counter = FlopCounterMode(display=False)
        with counter:
            fn()
        return float(counter.get_total_flops())

    fwd = count(lambda: loss_fn())
    grad = count(lambda: flat_grad(loss_fn(), params))
    v = torch.randn(n_params, dtype=params[0].dtype, device=params[0].device)
    curv = count(lambda: hvp(loss_fn, params, v))

    if fwd <= 0:
        m = analytic_cost_model(n_params)
        return CostModel(
            m.fwd_flops_per_sample,
            m.grad_multiplier,
            m.hvp_multiplier,
            provenance="analytic fallback: FlopCounterMode counted 0 forward FLOPs",
        )

    return CostModel(
        fwd_flops_per_sample=fwd / batch,
        grad_multiplier=grad / fwd,
        hvp_multiplier=curv / fwd,
        provenance=(
            f"measured with FlopCounterMode at batch {batch}: "
            f"fwd {fwd:.4g}, grad {grad / fwd:.3f}x, curvature {curv / fwd:.3f}x"
        ),
    )


@dataclass
class Ledger:
    """Stage-tagged FLOP totals for one candidate.

    ``search_shared`` is this candidate's share of probe work that served the
    whole pool.  ``search_pool`` is the *entire* pool's stage-1 cost, carried so
    that a survivor can be charged the fully-loaded price of the search that
    found it -- see ``report.fully_loaded``.
    """

    candidate: str
    stage1_exclusive: float = 0.0
    stage1_shared_share: float = 0.0
    #: Learning-rate search. Search cost, not training cost -- it buys a
    #: hyperparameter, not a result -- but it is spent on training runs, so it
    #: lives in stage 2 and is broken out rather than folded in.
    stage2_tuning: float = 0.0
    stage2_training: float = 0.0
    n_tuning_trials: int = 0
    n_trials: int = 0
    trials_flops: list[float] = field(default_factory=list)
    chosen_lr: float | None = None

    @property
    def stage1(self) -> float:
        return self.stage1_exclusive + self.stage1_shared_share

    @property
    def stage2(self) -> float:
        return self.stage2_tuning + self.stage2_training

    @property
    def marginal_total(self) -> float:
        """Search + training attributable to this candidate alone."""
        return self.stage1 + self.stage2

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "stage1_exclusive": self.stage1_exclusive,
            "stage1_shared_share": self.stage1_shared_share,
            "stage1": self.stage1,
            "stage2_tuning": self.stage2_tuning,
            "stage2_training": self.stage2_training,
            "stage2": self.stage2,
            "n_tuning_trials": self.n_tuning_trials,
            "n_trials": self.n_trials,
            "chosen_lr": self.chosen_lr,
            "marginal_total": self.marginal_total,
        }
