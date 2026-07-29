"""Stage 1 -- the cheap screen.

Question: *at this point on this trajectory, is there evidence that
preconditioning with ``P`` is favorable?*  Answered from a handful of Lanczos
solves and cross-fitted gradient samples.  No training runs, no new solver code,
nothing written into ``src/dsn``.

The predicate is generalized; the training loop is not touched.  A new candidate
preconditioner form means a new ``Preconditioner`` subclass and nothing else.

Two statistics
--------------

**1. ``rho`` -- the project's own P0 payoff gate, generalized off ``M``.**
``measure_rho.py`` fixes ``M = |H| + delta`` and reports

    lambda_g   = ||g||^2 / (g' M^-1 g)
    lambda_Sig = sqrt( tr(M Sigma) / tr(M^-1 Sigma) )
    rho        = (lambda_Sig / lambda_g)^2

Written in terms of the preconditioner ``P = M^-1`` those become

    lambda_g   = ||g||^2 / (g'Pg)
    lambda_Sig = sqrt( tr(P^-1 Sigma) / tr(P Sigma) )
    rho(P)     = [ tr(P^-1 Sigma) / tr(P Sigma) ] * (g'Pg)^2 / ||g||^4

which is defined for any SPD ``P``, reduces to the shipped formula at
``P = (|H| + delta)^-1``, and has two properties worth stating because the
verdict rule leans on both:

* ``rho`` is invariant under ``P -> cP``.  The learning rate absorbs scale, so
  the gate cannot be gamed by rescaling a candidate.
* ``rho(I) = 1`` **exactly**.  The threshold at 1 is therefore not a tuned
  constant: it is literally "does this beat plain SGD by this criterion".

**2. ``gain`` -- an H-aware companion, because ``rho`` alone is H-blind.**
Note what is *not* in the ``rho`` expression above: ``H``.  It enters only
through the definition of ``P``.  So ``rho`` measures whether ``P``'s shape
matches the ``(g, Sigma)`` geometry, and it cannot tell whether ``P`` still
inverts the curvature it claims to invert.  A preconditioner frozen from an
earlier checkpoint can therefore score well on ``rho`` while being wrong about
``H``, which is precisely the failure mode this project already hit once: a
maximally-stale frozen basis scored *better* than the correct implementation on
both staleness gates written to catch it (negative-result doc, section 8.2).
Shipping a generalized ``rho`` as the only screen would rebuild that blind spot
at the harness level.

So the gate also evaluates the expected one-step loss decrease of preconditioned
SGD under a noisy quadratic model.  For ``x <- x - eta P (g + noise)`` with
``E[noise noise'] = Sigma``,

    E[dL] = -eta g'Pg + (eta^2/2) [ g'PHPg + tr(P H P Sigma) ]

minimized over ``eta`` at

    R(P) = (g'Pg)^2 / ( g'PHPg + tr(P H P Sigma) )
    gain(P) = R(P) / R(I)

``gain`` is also scale-invariant in ``P`` and also exactly 1 at ``P = I``.  It
*is* sensitive to the scale of ``Sigma``, which ``rho`` is not -- see
``ProbeContext.noise_scale``.

Both are necessary conditions.  A candidate PASSes only if both pass; it FAILs
if either fails.  ``rho_only`` verdicts are logged alongside, so the H-blindness
above is a *measured* claim in every report rather than an asserted one.

**They can disagree, and on the one real trajectory measured here they do.**  At
Lanczos depths where *both* statistics pass their convergence check
(``delta`` in {1e-2, 1e-1}; ``delta=1e-3`` converges at no checkpoint), DSN's
MNIST trajectory gives ``rho`` 0.17-0.57 and ``gain`` 0.98-9.7.  An earlier
report of "``gain`` runs 4-79" was wrong: the 79 was a single spike at depth 75
in a sequence still oscillating at depth 150, i.e. the depth trap, and it is what
motivated the per-statistic convergence gate below.  The residual disagreement is
smaller and real.

The conjunction rejects, which is the conservative direction, but the
disagreement is **not resolved** by this harness and should not be reported as if
it were.  What is known about it:

* they are different criteria.  ``rho`` asks whether the gradient sits in
  directions sharper than the noise occupies; ``gain`` asks whether the one-step
  expected decrease of the noisy quadratic model improves.  Neither implies the
  other.
* ``gain`` is a *one-step, local* model, evaluated at a sparse sample of the
  trajectory.  ``GateReport.trajectory_gain`` aggregates it correctly across
  checkpoints (decreases add; ratios do not), but aggregating correctly is not
  the same as the per-checkpoint model being right.
* neither is an oracle.  Calibrated against actual training on a preconditioner
  family ``P = (|H| + delta)^-p``, both order it broadly correctly and **both
  mis-rank the top**, claiming ``p = 0.25`` beats plain SGD when training says it
  does not (Spearman vs trained loss: ``rho`` -0.86, ``gain`` -0.75).
* ``rho`` has three independent demonstrated blind spots -- staleness, random
  spread, and subspace damping -- and the third is the one that matters for any
  rank-restricted candidate.  See ``scripts/validate_gain.py``.

Screening a form versus screening an implementation
---------------------------------------------------

A ``Preconditioner`` is the only thing this module knows about, so screening
DSN's *shipped* rank-``k_max`` operator rather than the full-rank form it
approximates needs no change here -- see ``KrylovRestrictedPreconditioner``.  Use
it.  The two are **different operators, and the full-rank one is not an upper
bound on the restricted one**: ``P_k`` with a positive complement scaling is
itself full-rank and invertible, so it is another member of the same class rather
than a restriction of the member being screened.  Measured on the MNIST
trajectory, the verdict flips from FAIL (full rank) to PASS (rank 8).

``GateReport.rank`` is ``None`` for an unrestricted form and every report
carries a ``scope`` of ``full-rank`` or ``rank-k``, so a verdict about one
operator is never read as a verdict about the other.

Ceiling and margin
------------------

Following the project's own reporting idiom ("rho maxes at 0.717 against a
threshold of 1"), each statistic is swept over the estimator's nuisance
parameter -- Lanczos depth -- and reported at both ends:

* **ceiling** = the largest value over the sweep, with the *upper* end of its
  confidence interval.  Rejecting here is selection-safe: even at the most
  favorable estimator setting and the top of the interval, it does not reach 1.
* **floor** = the smallest value over the sweep, with the *lower* end of its
  interval.  Accepting here is selection-safe for the same reason in reverse.

Anything in between is ``INDETERMINATE``, which is a real answer and is counted
as one.  Maximizing over a sweep and then testing at the maximum would inflate
the false-pass rate by exactly the amount the sweep is wide.

What the intervals do and do not cover
--------------------------------------

The bootstrap resamples the cross-fit vectors ``u_i``, so the intervals cover
sampling error in ``Sigma`` and nothing else.  ``g``, ``H`` and the Lanczos
truncation are conditioned on.  Truncation is handled by the depth sweep rather
than by the interval; ``g`` is a single sample and its error is *not* quantified
here.  Stated because an interval whose coverage is not stated will be read as
covering everything.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from .preconditioners import Forms, Preconditioner
from .probes import ProbeContext, ProbeCost

THRESHOLD = 1.0
"""Both statistics are normalized so that plain SGD scores exactly 1.0."""

DEFAULT_MIN_EFFECT = 0.05
"""Relative margin over SGD a candidate must clear, on top of ``THRESHOLD``.

Two reasons it is not zero.

*Practical.*  A preconditioner that buys 1% is not worth a training budget.
Stage 2 costs orders of magnitude more than stage 1, and a screen that forwards
everything technically above 1.0 forwards the entire pool.

*Numerical, and this one is not optional.*  ``P = I`` and any scalar multiple of
it score **exactly** 1.0 by construction, so with a zero margin the verdict is
decided by whether floating-point rounding lands at ``1 + 1e-16`` or
``1 - 1e-16``.  Worse, those candidates have a degenerate bootstrap -- every
resample gives the same number -- so the confidence interval has zero width and
that dust is reported as a *certified* pass.  A gate that certifies the identity
matrix is a gate that certifies doing nothing.  ``min_effect=0`` is still
available and is the bare P0 predicate; it is not the default.
"""


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


def _combine(*verdicts: Verdict) -> Verdict:
    """Necessary-condition combination: any FAIL fails, all PASS passes."""
    if any(v is Verdict.FAIL for v in verdicts):
        return Verdict.FAIL
    if all(v is Verdict.PASS for v in verdicts):
        return Verdict.PASS
    return Verdict.INDETERMINATE


# ------------------------------------------------------------- the statistics


def _rho_from(gPg: float, gg: float, uPu: np.ndarray, uPinvu: np.ndarray) -> float:
    tr_P = float(uPu.mean())
    tr_Pinv = float(uPinvu.mean())
    if tr_P <= 0 or gg <= 0:
        return float("nan")
    return (tr_Pinv / tr_P) * (gPg**2) / (gg**2)


def _R(num_sq: float, quad: float, noise: float, noise_scale: float) -> float:
    """``(g'Pg)^2 / (g'PHPg + tr(PHP Sigma))``; ``nan`` if the model is unbounded.

    A non-positive denominator means the quadratic model predicts unbounded
    decrease along ``Pg``.  That is a statement about the model breaking down on
    indefinite curvature, not about the candidate being infinitely good, so it
    is reported as undefined and can never produce a PASS.
    """
    denom = quad + noise_scale * noise
    if not math.isfinite(denom) or denom <= 0:
        return float("nan")
    return num_sq**2 / denom


def expected_decreases(f: Forms, idx: np.ndarray | None = None) -> tuple[float, float]:
    """``(R(P), R(I))`` -- the two *unnormalized* expected one-step decreases.

    Returned separately as well as as a ratio because they are the quantities
    that **add across steps**.  A trajectory accumulates decreases, not ratios,
    so the trajectory-level comparison is ``sum R(P) / sum R(I)`` and not the
    max, mean or any other function of the per-checkpoint ratios.  Those differ
    whenever the trajectory is non-stationary, and they differ by a lot: see the
    G2 mutant, where the per-checkpoint max says 10.84 and the ratio of sums says
    0.064.
    """
    n = len(f.uPu)
    if idx is None:
        idx = np.arange(n)
    uPHPu = np.asarray(f.uPHPu, dtype=float)[idx]
    uHu = np.asarray(f.uHu, dtype=float)[idx]
    RP = _R(f.gPg, f.gPHPg, float(uPHPu.mean()), f.noise_scale)
    RI = _R(f.gg, f.gHg, float(uHu.mean()), f.noise_scale)
    return RP, RI


def _gain_from(f: Forms, idx: np.ndarray) -> float:
    RP, RI = expected_decreases(f, idx)
    if not (math.isfinite(RP) and math.isfinite(RI)) or RI <= 0:
        return float("nan")
    return RP / RI


def statistics(f: Forms, idx: np.ndarray | None = None) -> tuple[float, float]:
    """``(rho, gain)`` from one set of forms, optionally on a resampled index set."""
    n = len(f.uPu)
    if idx is None:
        idx = np.arange(n)
    uPu = np.asarray(f.uPu, dtype=float)[idx]
    uPinvu = np.asarray(f.uPinvu, dtype=float)[idx]
    return _rho_from(f.gPg, f.gg, uPu, uPinvu), _gain_from(f, idx)


# --------------------------------------------------------------------- result


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    se: float
    #: The bar this interval is judged against: ``THRESHOLD * (1 + min_effect)``.
    bar: float = THRESHOLD

    @property
    def margin(self) -> float:
        """``(point - bar) / se`` -- distance to the gate in its own noise."""
        if not math.isfinite(self.se) or self.se <= 0:
            return float("inf") if self.point > self.bar else float("-inf")
        return (self.point - self.bar) / self.se

    def as_dict(self) -> dict:
        d = asdict(self)
        d["margin"] = self.margin
        return d


@dataclass
class Cell:
    """One (candidate, checkpoint, Lanczos depth) evaluation."""

    depth: int
    rho: Interval
    gain: Interval
    unbounded_model: bool
    #: Unnormalized expected one-step decreases, ``R(P)`` and ``R(I)``.  These
    #: are what a trajectory sums; the ratio is only meaningful per checkpoint.
    R_P: float = float("nan")
    R_I: float = float("nan")
    #: ``scale / min|node|`` for a damped spectral candidate, else ``nan``.
    #:
    #: The mechanism behind the depth trap, reduced to one number.  A Gauss rule
    #: of depth ``k`` samples the spectrum only down to its smallest node; below
    #: that, ``1/(|lam| + delta)`` and ``lam/(|lam| + delta)^2`` are being
    #: *extrapolated* by a polynomial, not integrated.  When ``delta`` sits at or
    #: below the smallest resolved node, the near-singular part of the integrand
    #: -- which is where all of its mass is -- lies entirely in the unsampled
    #: region.
    #:
    #: Measured on the MNIST MLP at depth 150: smallest resolved ``|lam|`` is
    #: 4.3e-3, so ``delta=1e-3`` gives a ratio of 0.23 (unresolved, and both
    #: statistics fail their convergence check at every checkpoint) while
    #: ``delta=1e-2`` gives 2.3 and ``delta=1e-1`` gives 23 (both converged).
    spectral_resolution: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "depth": self.depth,
            "rho": self.rho.as_dict(),
            "gain": self.gain.as_dict(),
            "unbounded_model": self.unbounded_model,
            "R_P": self.R_P,
            "R_I": self.R_I,
            "spectral_resolution": self.spectral_resolution,
        }


@dataclass
class CheckpointResult:
    candidate: str
    spec: str
    step: int
    label: str
    cells: list[Cell]
    neg_frac: float
    n_cross_fit: int
    noise_scale: float
    verdict: Verdict
    rho_verdict: Verdict
    gain_verdict: Verdict
    #: Verdict this checkpoint would have returned on ``rho`` alone.  Logged so
    #: the H-blindness of ``rho`` is measured in every run, not assumed.
    rho_only_verdict: Verdict
    reason: str

    def _end(self, attr: str, which: str) -> Interval:
        vals = [getattr(c, attr) for c in self.cells]
        finite = [v for v in vals if math.isfinite(v.point)]
        if not finite:
            return Interval(float("nan"), float("nan"), float("nan"), float("nan"))
        return max(finite, key=lambda v: v.point) if which == "ceiling" else min(
            finite, key=lambda v: v.point
        )

    @property
    def rho_ceiling(self) -> Interval:
        return self._end("rho", "ceiling")

    @property
    def rho_floor(self) -> Interval:
        return self._end("rho", "floor")

    @property
    def gain_ceiling(self) -> Interval:
        return self._end("gain", "ceiling")

    @property
    def gain_floor(self) -> Interval:
        return self._end("gain", "floor")

    def depth_spread(self, attr: str) -> float:
        """Relative spread of a statistic across the whole Lanczos depth sweep.

        Descriptive only.  A large spread is expected whenever the sweep starts
        shallow, because the shallow end *is* truncation error; that says
        nothing about whether the deep end has settled.  Use
        ``depth_convergence`` for the decision and this for the picture.
        """
        vals = [getattr(c, attr).point for c in self.cells]
        finite = [v for v in vals if math.isfinite(v) and v != 0]
        if len(finite) < 2:
            return float("nan")
        return (max(finite) - min(finite)) / abs(max(finite, key=abs))

    def depth_convergence(self, attr: str) -> float:
        """Relative change between the two *deepest* rungs of the sweep.

        This is the quantity a verdict may rest on.  ``lambda_g`` is a harmonic
        mean, dominated by the smallest eigenvalues, which Lanczos resolves
        last; for a spectral ``P = f(H)^-1`` the gate's integrands are

            rho  needs   1/f(lam)              peaks at lam = 0,   height 1/delta
            gain needs   lam / f(lam)^2        peaks at lam = delta, height 1/(4 delta)

        Both are near-singular for small ``delta`` and therefore badly
        approximated by any polynomial, which is all a Gauss rule can do.
        **``gain``'s integrand is the sharper of the two, so ``gain`` converges
        strictly more slowly than ``rho``.**  Measured on a 400-dimensional
        flat-tailed spectrum: at ``delta=1e-3`` and depth 100, ``rho`` sits at
        0.84x its converged value while ``gain`` sits at 0.55x; at
        ``delta=1e-1`` and depth 25, ``rho`` is at 0.97x and ``gain`` at 0.65x.

        ``gain`` was added after the depth trap was found for ``rho`` and
        inherited no safeguard of its own.  This is that safeguard, and it is
        deliberately applied per statistic rather than to the pair, because the
        two do not converge together.
        """
        vals = [getattr(c, attr).point for c in self.cells]
        if len(vals) < 2:
            return float("nan")  # unassessable, not converged -- see `reason`
        a, b = vals[-2], vals[-1]
        if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
            return float("nan")
        return abs(b - a) / abs(b)

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "spec": self.spec,
            "step": self.step,
            "label": self.label,
            "neg_frac": self.neg_frac,
            "rho_depth_spread": self.depth_spread("rho"),
            "gain_depth_spread": self.depth_spread("gain"),
            "rho_depth_convergence": self.depth_convergence("rho"),
            "gain_depth_convergence": self.depth_convergence("gain"),
            "n_cross_fit": self.n_cross_fit,
            "noise_scale": self.noise_scale,
            "verdict": self.verdict.value,
            "rho_verdict": self.rho_verdict.value,
            "gain_verdict": self.gain_verdict.value,
            "rho_only_verdict": self.rho_only_verdict.value,
            "reason": self.reason,
            "rho_ceiling": self.rho_ceiling.as_dict(),
            "rho_floor": self.rho_floor.as_dict(),
            "gain_ceiling": self.gain_ceiling.as_dict(),
            "gain_floor": self.gain_floor.as_dict(),
            "cells": [c.as_dict() for c in self.cells],
        }


@dataclass
class GateReport:
    """Trajectory-level verdict for one candidate, plus its stage-1 cost."""

    candidate: str
    spec: str
    #: ``None`` means the verdict is about the full-rank operator.
    #:
    #: That is a statement about *that* operator and nothing else.  If the
    #: implementation applies a rank-restricted version, screen the restriction
    #: (``KrylovRestrictedPreconditioner``): the two are different operators and
    #: **neither bounds the other**.  Measured on the MNIST trajectory, the
    #: verdict flips from FAIL at full rank to PASS at rank 8.
    rank: int | None
    checkpoints: list[CheckpointResult]
    verdict: Verdict
    rho_only_verdict: Verdict
    reason: str
    #: ``sum R(P) / sum R(I)`` over checkpoints at the deepest sweep rung.  The
    #: trajectory-level counterpart of ``gain``; equals it when there is one
    #: checkpoint, diverges from the per-checkpoint max whenever the trajectory
    #: is non-stationary.  Checkpoints are a sparse, equally-weighted sample of
    #: the trajectory, so this is a Riemann proxy, not an integral.
    trajectory_gain: float = float("nan")
    exclusive_cost: ProbeCost = field(default_factory=ProbeCost)
    shared_cost_share: ProbeCost = field(default_factory=ProbeCost)

    @property
    def rho_ceiling(self) -> float:
        vals = [c.rho_ceiling.point for c in self.checkpoints]
        finite = [v for v in vals if math.isfinite(v)]
        return max(finite) if finite else float("nan")

    @property
    def gain_ceiling(self) -> float:
        vals = [c.gain_ceiling.point for c in self.checkpoints]
        finite = [v for v in vals if math.isfinite(v)]
        return max(finite) if finite else float("nan")

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "spec": self.spec,
            "rank": self.rank,
            "scope": "rank-%d" % self.rank if self.rank else "full-rank",
            "verdict": self.verdict.value,
            "rho_only_verdict": self.rho_only_verdict.value,
            "reason": self.reason,
            "rho_ceiling": self.rho_ceiling,
            "gain_ceiling": self.gain_ceiling,
            "trajectory_gain": self.trajectory_gain,
            "exclusive_cost": self.exclusive_cost.as_dict(),
            "shared_cost_share": self.shared_cost_share.as_dict(),
            "checkpoints": [c.as_dict() for c in self.checkpoints],
        }


def trajectory_gain(results: Sequence[CheckpointResult], depth: int) -> float:
    """``sum R(P) / sum R(I)`` over checkpoints -- the aggregate a run accumulates.

    Not the mean of the per-checkpoint ratios and not their max.  ``R`` is an
    expected loss decrease per step; decreases add, ratios do not.  Returns
    ``nan`` if any checkpoint's model is unbounded, since a broken model cannot
    be summed into a total.
    """
    num = den = 0.0
    for r in results:
        cell = next((c for c in r.cells if c.depth == depth), None)
        if cell is None or not (math.isfinite(cell.R_P) and math.isfinite(cell.R_I)):
            return float("nan")
        num += cell.R_P
        den += cell.R_I
    return num / den if den > 0 else float("nan")


# ----------------------------------------------------------------- the gate


class Gate:
    """The stage-1 predicate.  Stateless apart from its configuration.

    ``alpha`` is the two-sided miscoverage of the bootstrap interval used for a
    single checkpoint decision.  It is *not* the family-wise error rate of the
    search -- stage 1 is a screen whose job is to be cheap, and stage 2 owns the
    error budget over the candidate pool.  Reporting them as one number would
    make the search look more rigorous than it is.
    """

    def __init__(
        self,
        depths: Sequence[int],
        alpha: float = 0.05,
        n_bootstrap: int = 2000,
        seed: int = 0,
        require_gain: bool = True,
        min_effect: float = DEFAULT_MIN_EFFECT,
        depth_tol: float = 0.10,
        min_spectral_resolution: float = 1.0,
    ):
        if not depths:
            raise ValueError("gate needs at least one Lanczos depth")
        if min_effect < 0:
            raise ValueError("min_effect must be non-negative")
        self.depths = tuple(sorted({int(d) for d in depths}))
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.require_gain = require_gain
        self.min_effect = min_effect
        # Relative change allowed between the two deepest rungs before a
        # statistic is treated as converged enough to decide anything.
        self.depth_tol = depth_tol
        # `delta / min|node|` below which a damped spectral candidate's
        # integrand is being extrapolated. Diagnostic, not a gate: the depth
        # convergence check is what decides, and this says why it failed.
        self.min_spectral_resolution = min_spectral_resolution

    @property
    def bar(self) -> float:
        """The value both statistics must clear."""
        return THRESHOLD * (1.0 + self.min_effect)

    # -------------------------------------------------------------- internals

    def _interval(self, point: float, draws: np.ndarray) -> Interval:
        draws = draws[np.isfinite(draws)]
        if not math.isfinite(point) or draws.size < 2:
            return Interval(point, float("nan"), float("nan"), float("nan"), self.bar)
        lo, hi = np.quantile(draws, [self.alpha / 2, 1 - self.alpha / 2])
        return Interval(
            point, float(lo), float(hi), float(draws.std(ddof=1)), self.bar
        )

    def _cell(self, f: Forms, rng: np.random.Generator,
              resolution: float = float("nan")) -> Cell:
        n = len(f.uPu)
        rho_p, gain_p = statistics(f)
        # One resampled index set per replicate, shared by numerator and
        # denominator and by both statistics.  Resampling them independently
        # would break the pairing that makes `gain` a ratio of two estimates on
        # the *same* noise sample, and would widen every interval for free.
        idx = rng.integers(0, n, size=(self.n_bootstrap, n))
        draws = np.array([statistics(f, i) for i in idx], dtype=float)
        RP, RI = expected_decreases(f)
        return Cell(
            depth=f.depth,
            rho=self._interval(rho_p, draws[:, 0]),
            gain=self._interval(gain_p, draws[:, 1]),
            unbounded_model=not math.isfinite(gain_p),
            R_P=RP,
            R_I=RI,
            spectral_resolution=resolution,
        )

    def _verdict_from(
        self, cells: list[Cell], attr: str, convergence: float
    ) -> tuple[Verdict, str]:
        bar = self.bar
        vals = [getattr(c, attr) for c in cells]
        finite = [v for v in vals if math.isfinite(v.point)]
        if not finite:
            return Verdict.INDETERMINATE, f"{attr}: undefined at every depth"

        # Convergence in depth gates the verdict in BOTH directions, and the
        # second direction is the one that was missing.  A statistic still
        # moving at the deepest rung can be arbitrarily far from its converged
        # value, so it cannot support a PASS -- and it cannot support a FAIL
        # either, even when every depth in the sweep happens to land on the same
        # side of the bar.  Measured: on a flat-tailed spectrum `rho` runs 9.2 at
        # depth 5 and 1104 at depth 200, so a shallow sweep would have returned
        # a confident FAIL two orders of magnitude below the truth.
        if len(cells) < 2:
            return (
                Verdict.INDETERMINATE,
                f"{attr}: depth convergence UNASSESSABLE (sweep has one depth); "
                "a verdict needs at least two rungs to compare",
            )
        if not math.isfinite(convergence):
            return (
                Verdict.INDETERMINATE,
                f"{attr}: depth convergence undefined at the deepest rungs",
            )
        if convergence > self.depth_tol:
            return (
                Verdict.INDETERMINATE,
                f"{attr}: NOT CONVERGED IN DEPTH -- {convergence:.0%} relative "
                f"change between depths {self.depths[-2]} and {self.depths[-1]} "
                f"(tolerance {self.depth_tol:.0%}); deepen the sweep before "
                "reading either end",
            )
        ceiling = max(finite, key=lambda v: v.point)
        floor = min(finite, key=lambda v: v.point)
        if math.isfinite(ceiling.hi) and ceiling.hi < bar:
            return (
                Verdict.FAIL,
                f"{attr} ceiling {ceiling.point:.4g} (CI hi {ceiling.hi:.4g}) "
                f"< {bar:g}",
            )
        if math.isfinite(floor.lo) and floor.lo > bar:
            return (
                Verdict.PASS,
                f"{attr} floor {floor.point:.4g} (CI lo {floor.lo:.4g}) "
                f"> {bar:g}",
            )
        if len(finite) < len(vals):
            return (
                Verdict.INDETERMINATE,
                f"{attr} undefined at {len(vals) - len(finite)}/{len(vals)} depths",
            )
        return (
            Verdict.INDETERMINATE,
            f"{attr} interval straddles {bar:g} "
            f"(floor lo {floor.lo:.4g}, ceiling hi {ceiling.hi:.4g})",
        )

    # ----------------------------------------------------------------- public

    def evaluate_checkpoint(
        self, precond: Preconditioner, ctx: ProbeContext
    ) -> CheckpointResult:
        rng = np.random.default_rng(
            abs(hash((self.seed, precond.name, ctx.step, ctx.label))) % (2**63)
        )
        g_rules, _ = ctx.rules()
        scale = getattr(precond, "scale", None)
        cells = []
        for d in self.depths:
            res = float("nan")
            if scale is not None and scale > 0:
                min_node = float(g_rules[d].nodes.abs().min())
                res = scale / min_node if min_node > 0 else float("inf")
            cells.append(self._cell(precond.forms(ctx, d), rng, res))

        probe = CheckpointResult(
            candidate=precond.name, spec=precond.spec, step=ctx.step,
            label=ctx.label, cells=cells, neg_frac=0.0, n_cross_fit=0,
            noise_scale=0.0, verdict=Verdict.INDETERMINATE,
            rho_verdict=Verdict.INDETERMINATE, gain_verdict=Verdict.INDETERMINATE,
            rho_only_verdict=Verdict.INDETERMINATE, reason="",
        )
        conv_rho = probe.depth_convergence("rho")
        conv_gain = probe.depth_convergence("gain")

        rho_v, rho_why = self._verdict_from(cells, "rho", conv_rho)
        gain_v, gain_why = self._verdict_from(cells, "gain", conv_gain)
        combined = _combine(rho_v, gain_v) if self.require_gain else rho_v
        why = f"{rho_why}; {gain_why}"
        if any(c.unbounded_model for c in cells):
            why += "; quadratic model unbounded at >=1 depth (indefinite curvature)"
        res = cells[-1].spectral_resolution
        if math.isfinite(res) and res < self.min_spectral_resolution:
            why += (
                f"; damping sits at {res:.2f}x the smallest resolved curvature "
                f"(want > {self.min_spectral_resolution:g}) -- the spectral "
                "integrand is extrapolated below that node, not integrated"
            )

        result = CheckpointResult(
            candidate=precond.name,
            spec=precond.spec,
            step=ctx.step,
            label=ctx.label,
            cells=cells,
            neg_frac=ctx.neg_frac(),
            n_cross_fit=ctx.n_cross_fit,
            noise_scale=ctx.noise_scale,
            verdict=combined,
            rho_verdict=rho_v,
            gain_verdict=gain_v,
            rho_only_verdict=rho_v,
            reason=why,
        )
        return result

    def evaluate(
        self, precond: Preconditioner, contexts: Sequence[ProbeContext]
    ) -> GateReport:
        """Screen one candidate over a whole trajectory.

        Roll-up rule, and the asymmetry is deliberate:

        * **PASS** if it passes at *any* checkpoint.  The question is whether
          preconditioning pays *at some point in training*, and this project has
          already measured a ``rho`` that moves by 16x within one run
          (0.045 -> 0.72 -> 0.16).  A candidate that pays only mid-training is
          still a candidate.
        * **FAIL** only if it fails at *every* checkpoint.  This is the DSN
          finding's shape, and it is the only rejection strong enough to spend
          no training FLOPs on.
        * ``INDETERMINATE`` otherwise -- which is most of the time, and is the
          honest answer when a few checkpoints of a cheap probe disagree.
        """
        if not contexts:
            raise ValueError("evaluate needs at least one probe context")
        results = [self.evaluate_checkpoint(precond, c) for c in contexts]

        def rollup(key) -> tuple[Verdict, str]:
            vs = [key(r) for r in results]
            if all(v is Verdict.FAIL for v in vs):
                return Verdict.FAIL, f"FAIL at all {len(vs)} checkpoints"
            if any(v is Verdict.PASS for v in vs):
                steps = [r.step for r, v in zip(results, vs) if v is Verdict.PASS]
                return Verdict.PASS, f"PASS at checkpoint(s) {steps}"
            n_fail = sum(v is Verdict.FAIL for v in vs)
            return (
                Verdict.INDETERMINATE,
                f"no checkpoint passes; {n_fail}/{len(vs)} fail outright",
            )

        verdict, reason = rollup(lambda r: r.verdict)
        rho_only, rho_reason = rollup(lambda r: r.rho_only_verdict)

        # A PASS from "passes at any checkpoint" is not enough on its own.
        # Expected decreases ADD across steps, so a candidate that is excellent
        # where the per-step decrease is small and terrible where it is large can
        # top the per-checkpoint ratio while losing badly over the trajectory.
        # Measured on the G2 mutant: per-checkpoint max 10.84, ratio of sums
        # 0.064 -- a 170x disagreement, in the unsafe direction.
        traj = trajectory_gain(results, self.depths[-1])
        # Guarded by `require_gain`: this check is part of the `gain` half, and
        # `require_gain=False` must remain exactly the H-blind `rho`-only screen
        # so that what it misses stays measurable rather than partly patched.
        if self.require_gain and verdict is Verdict.PASS:
            if not math.isfinite(traj):
                verdict = Verdict.INDETERMINATE
                reason += "; trajectory-level decrease undefined"
            elif traj <= self.bar:
                verdict = Verdict.INDETERMINATE
                reason += (
                    f"; but sum R(P)/sum R(I) over the trajectory is {traj:.4g} "
                    f"(<= {self.bar:g}) -- the per-checkpoint pass does not survive "
                    "aggregation, which is what a trajectory actually accumulates"
                )
            else:
                reason += f"; trajectory sum R(P)/sum R(I) = {traj:.4g}"

        exclusive = ProbeCost()
        for c in contexts:
            exclusive = exclusive + precond.extra_cost(c)

        return GateReport(
            candidate=precond.name,
            spec=precond.spec,
            rank=getattr(precond, "rank", None),
            checkpoints=results,
            trajectory_gain=traj,
            verdict=verdict,
            rho_only_verdict=rho_only,
            reason=f"{reason} | rho-only would say {rho_only.value}: {rho_reason}",
            exclusive_cost=exclusive,
        )


def screen(
    gate: Gate,
    candidates: Sequence[Preconditioner],
    contexts: Sequence[ProbeContext],
) -> dict[str, GateReport]:
    """Screen a pool of candidates against a shared set of probe contexts.

    The shared quadrature cost is paid once for the whole pool and then split
    evenly across the candidates that consumed it.  Even splitting is a
    convention and is recorded as one: the marginal cost of screening the second
    spectral candidate at a checkpoint is genuinely zero, so any attribution is
    arbitrary.  What is *not* arbitrary is that the total must appear somewhere
    in the ledger, because a search cost that vanishes when divided is the exact
    accounting error this harness exists to prevent.
    """
    reports = {c.name: gate.evaluate(c, contexts) for c in candidates}
    shared = ProbeCost()
    for ctx in contexts:
        shared = shared + ctx.shared_cost
    share = shared.scaled(1.0 / len(reports)) if reports else ProbeCost()
    for r in reports.values():
        r.shared_cost_share = share
    return reports
