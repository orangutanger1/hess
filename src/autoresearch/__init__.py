"""Two-stage screening harness for optimizer preconditioner candidates.

Stage 1 (``gate``)   cheap probes decide whether a preconditioner is favorable
                     at a point on an existing trajectory -- no training runs.
Stage 2 (``select``) surviving candidates get an adaptive seed budget under a
                     family-wise error budget that holds as the pool grows.

Cost from both stages lands in one ledger (``cost``, ``report``).  A search
ledger that reports only training FLOPs flatters whichever candidate got lucky
early, which is the accounting error this package refuses to make.
"""

from .cost import CostModel, Ledger, analytic_cost_model, calibrate
from .gate import Gate, GateReport, Verdict, screen
from .preconditioners import (
    IdentityPreconditioner,
    OperatorPreconditioner,
    Preconditioner,
    SpectralPreconditioner,
    dense_preconditioner,
    diagonal_preconditioner,
    dsn_preconditioner,
    fixed_random_preconditioner,
    frozen_spectral_preconditioner,
    identity_spectral,
    newton_preconditioner,
)
from .probes import ProbeContext, ProbeCost, build_probe_context, cross_fit, quadrature
from .select import (
    AlphaLedger,
    SelectionReport,
    SuccessiveElimination,
    TrialResult,
    anytime_p_value,
    benjamini_hochberg,
    benjamini_yekutieli,
    confidence_radius,
    sequential_halving,
)
from .synthetic import blocked_quadratic, favorable_pair, unfavorable_quadratic

__all__ = [
    "AlphaLedger",
    "CostModel",
    "Gate",
    "GateReport",
    "IdentityPreconditioner",
    "Ledger",
    "SelectionReport",
    "SuccessiveElimination",
    "TrialResult",
    "OperatorPreconditioner",
    "Preconditioner",
    "ProbeContext",
    "ProbeCost",
    "SpectralPreconditioner",
    "Verdict",
    "analytic_cost_model",
    "anytime_p_value",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "blocked_quadratic",
    "build_probe_context",
    "calibrate",
    "confidence_radius",
    "cross_fit",
    "dense_preconditioner",
    "diagonal_preconditioner",
    "dsn_preconditioner",
    "favorable_pair",
    "fixed_random_preconditioner",
    "frozen_spectral_preconditioner",
    "identity_spectral",
    "newton_preconditioner",
    "quadrature",
    "screen",
    "sequential_halving",
    "unfavorable_quadratic",
]
