"""Preconditioner forms the stage-1 gate knows how to evaluate.

A candidate optimizer is screened through the preconditioner ``P`` it applies to
the gradient: the step it would take is ``d = -eta * P g``.  ``P = I`` is plain
SGD, ``P = (|H| + delta)^-1`` is what ``dsn.solve.subspace_newton`` inverts, and
anything in between is a candidate.

The gate needs exactly four quadratic forms per probe vector ``v``:

    v'Pv        how much of ``v`` the preconditioner keeps
    v'P^-1v     the dual form, which is what makes ``rho`` scale-invariant
    v'PHPv      the second-order cost of stepping along ``Pv``
    v'Hv        the same, unpreconditioned -- the SGD baseline

Two ways to get them, and the split is the whole reason this module exists:

**Spectral** (``P = f(H)^-1``).  All four forms are values of *some* function of
``H`` at the same probe vector, so all four come out of the single Gauss
quadrature rule the ``ProbeContext`` already built.  Screening the entire
spectral family -- every ``delta``, every ``f`` -- costs **zero** additional
curvature products.  This is the ``measure_rho.py`` machinery, generalized from
one hard-coded ``f`` to any ``f``.

**Operator** (``P`` given as a linear map).  ``v'Pv`` and ``v'P^-1v`` are direct
applications, but ``v'PHPv`` needs ``H`` applied to the *rotated* vector ``Pv``,
which no rule for ``v`` can supply.  Cost: one curvature product per probe
vector, per candidate.  Diagonal, dense, fixed-random and *frozen* candidates
all land here.

Adding a new preconditioner form means adding a subclass here.  It does not mean
touching ``gate.py``, and it never means touching a training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .probes import ProbeContext, ProbeCost, QuadRule


@dataclass(frozen=True)
class Forms:
    """The four quadratic forms, for ``g`` and for each cross-fit vector ``u_i``.

    ``u_*`` entries are per-vector so the bootstrap in ``gate.py`` can resample
    them.  Resampling a pre-averaged trace would report a confidence interval of
    width zero, which is the most flattering possible mistake.
    """

    depth: int
    gg: float
    gPg: float
    gPinvg: float
    gPHPg: float
    gHg: float
    uPu: tuple[float, ...]
    uPinvu: tuple[float, ...]
    uPHPu: tuple[float, ...]
    uHu: tuple[float, ...]
    #: ``B_micro / B_train``: converts the measured microbatch covariance to the
    #: covariance of the gradient the screened optimizer actually sees.
    noise_scale: float

    def __post_init__(self) -> None:
        n = len(self.uPu)
        if not (len(self.uPinvu) == len(self.uPHPu) == len(self.uHu) == n):
            raise ValueError("cross-fit form arrays must be the same length")


class Preconditioner:
    """Base class: a named ``P`` the gate can evaluate at a probe context."""

    name: str = "unnamed"
    #: Human-readable form, logged verbatim so a result is reproducible from it.
    spec: str = ""
    #: Subspace rank this ``P`` is restricted to; ``None`` means full rank.
    #:
    #: Load-bearing for how a verdict may be read.  A verdict about the
    #: full-rank form and a verdict about a rank-restricted implementation of it
    #: are verdicts about **two different operators**, and neither bounds the
    #: other -- a restricted ``P`` with a positive complement scaling is itself
    #: full-rank and invertible, so it is a different member of the same class
    #: rather than a restriction of the member being screened.  Measured on the
    #: MNIST trajectory the verdict flips from FAIL (full rank) to PASS (rank 8).
    #: Reports label the two differently so nothing downstream conflates them.
    rank: int | None = None

    def forms(self, ctx: ProbeContext, depth: int) -> Forms:
        raise NotImplementedError

    def extra_cost(self, ctx: ProbeContext) -> ProbeCost:
        """Cost *beyond* the context's shared probe cost, for this candidate."""
        return ProbeCost()

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """``P v``.  Only required for candidates that are also trained."""
        raise NotImplementedError(f"{self.name} cannot be applied to a vector")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name!r}>"


# ------------------------------------------------------------------- spectral


def _h_forms(rule: QuadRule) -> float:
    return rule.apply(lambda lam: lam)


class SpectralPreconditioner(Preconditioner):
    """``P = f(H)^-1`` for a positive spectral function ``f``.

    ``f`` must be positive on the whole measured spectrum, including its
    negative part.  ``(|lam| + delta)`` is positive by construction, which is
    why DSN's saddle-free damped operator is the natural member of this family;
    a plain ``max(lam, 0)`` is not, and is offered only with an explicit floor.

    All four forms at once, from one rule:

        v'Pv     = ||v||^2 sum_j w_j / f(lam_j)
        v'P^-1v  = ||v||^2 sum_j w_j f(lam_j)
        v'PHPv   = ||v||^2 sum_j w_j lam_j / f(lam_j)^2
        v'Hv     = ||v||^2 sum_j w_j lam_j
    """

    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], name: str,
                 spec: str = "", scale: float | None = None):
        self.f = f
        self.name = name
        self.spec = spec or name
        # The damping scale, when the family has one. Not decoration: the gate
        # compares it against the smallest curvature the quadrature actually
        # resolved, and a damping that sits *below* that node means the
        # near-singular part of the integrand is being extrapolated rather than
        # integrated. See `Cell.spectral_resolution`.
        self.scale = scale

    def _four(self, rule: QuadRule) -> tuple[float, float, float, float]:
        fv = self.f(rule.nodes)
        if bool((fv <= 0).any()):
            raise ValueError(
                f"{self.name}: spectral function is non-positive somewhere on the "
                f"measured spectrum (min f = {float(fv.min()):.4g}); P = f(H)^-1 is "
                "then not a preconditioner and every form below is meaningless"
            )
        w, lam, n2 = rule.weights, rule.nodes, rule.norm2
        return (
            n2 * float((w / fv).sum()),
            n2 * float((w * fv).sum()),
            n2 * float((w * lam / fv**2).sum()),
            n2 * float((w * lam).sum()),
        )

    def forms(self, ctx: ProbeContext, depth: int) -> Forms:
        g_rules, u_rules = ctx.rules()
        gPg, gPinvg, gPHPg, gHg = self._four(g_rules[depth])
        per_u = [self._four(r[depth]) for r in u_rules]
        return Forms(
            depth=depth,
            gg=float(ctx.g.dot(ctx.g)),
            gPg=gPg,
            gPinvg=gPinvg,
            gPHPg=gPHPg,
            gHg=gHg,
            uPu=tuple(q[0] for q in per_u),
            uPinvu=tuple(q[1] for q in per_u),
            uPHPu=tuple(q[2] for q in per_u),
            uHu=tuple(q[3] for q in per_u),
            noise_scale=ctx.noise_scale,
        )


def dsn_preconditioner(delta: float) -> SpectralPreconditioner:
    """``P = (|H| + delta)^-1`` -- the operator ``subspace_newton`` inverts.

    Not ``H^-1``.  The exact Hessian of the models this project measures is
    indefinite (``neg_frac`` 0.40 at step 0 on the MNIST MLP), so ``H^-1`` is
    not a preconditioner and ``g'H^-1 g`` can go negative.  Screening ``H^-1``
    would be screening an operator no code here applies.
    """
    return SpectralPreconditioner(
        lambda lam: lam.abs() + delta,
        name=f"dsn(delta={delta:g})",
        spec=f"P = (|H| + {delta:g})^-1, saddle-free damped, as solve.subspace_newton",
        scale=delta,
    )


def newton_preconditioner(floor: float) -> SpectralPreconditioner:
    """``P = max(H, floor)^-1`` -- textbook damped Newton, clamped to stay SPD."""
    return SpectralPreconditioner(
        lambda lam: lam.clamp_min(floor),
        name=f"newton(floor={floor:g})",
        spec=f"P = max(H, {floor:g})^-1",
        scale=floor,
    )


def identity_spectral() -> SpectralPreconditioner:
    """``P = I``, routed through the spectral path.

    Exists to cross-check the two paths against each other: this and
    ``IdentityPreconditioner`` must produce identical statistics, and any drift
    between them is a bug in the quadrature rather than in a candidate.
    """
    return SpectralPreconditioner(
        lambda lam: torch.ones_like(lam), name="identity[spectral]", spec="P = I"
    )


# ------------------------------------------------------------------- operator


class OperatorPreconditioner(Preconditioner):
    """``P`` supplied as a pair of linear maps ``v -> Pv`` and ``v -> P^-1 v``.

    Costs one curvature product per probe vector, because ``v'PHPv`` needs ``H``
    at ``Pv`` and no rule built at ``v`` contains that.  ``v'Hv`` still comes
    free from the shared rule.
    """

    def __init__(
        self,
        apply: Callable[[torch.Tensor], torch.Tensor],
        apply_inverse: Callable[[torch.Tensor], torch.Tensor],
        name: str,
        spec: str = "",
    ):
        self._apply = apply
        self._apply_inv = apply_inverse
        self.name = name
        self.spec = spec or name

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        return self._apply(v)

    def extra_cost(self, ctx: ProbeContext) -> ProbeCost:
        c = ProbeCost()
        c.add_hvp(ctx.curv_batch, ctx.n_cross_fit + 1)
        return c

    def _three(self, v: torch.Tensor, ctx: ProbeContext) -> tuple[float, float, float]:
        Pv = self._apply(v)
        Pinv_v = self._apply_inv(v)
        HPv = ctx.matvec(Pv)
        return float(v.dot(Pv)), float(v.dot(Pinv_v)), float(Pv.dot(HPv))

    def forms(self, ctx: ProbeContext, depth: int) -> Forms:
        g_rules, u_rules = ctx.rules()
        gPg, gPinvg, gPHPg = self._three(ctx.g, ctx)
        per_u = [self._three(ui, ctx) for ui in ctx.u]
        return Forms(
            depth=depth,
            gg=float(ctx.g.dot(ctx.g)),
            gPg=gPg,
            gPinvg=gPinvg,
            gPHPg=gPHPg,
            gHg=_h_forms(g_rules[depth]),
            uPu=tuple(q[0] for q in per_u),
            uPinvu=tuple(q[1] for q in per_u),
            uPHPu=tuple(q[2] for q in per_u),
            uHu=tuple(_h_forms(r[depth]) for r in u_rules),
            noise_scale=ctx.noise_scale,
        )


class IdentityPreconditioner(OperatorPreconditioner):
    """``P = I`` -- plain SGD, the control every statistic is normalized against.

    Both gate statistics are *defined* so that this candidate scores exactly
    1.0.  That is not a coincidence to be tolerated; it is the calibration, and
    ``tests/test_autoresearch_gate.py`` asserts it to machine precision.  A gate
    that awards a pass to the identity is a gate that would award a pass to
    doing nothing.
    """

    def __init__(self) -> None:
        super().__init__(lambda v: v, lambda v: v, name="identity", spec="P = I (SGD)")


def diagonal_preconditioner(d: torch.Tensor, name: str) -> OperatorPreconditioner:
    """``P = diag(d)`` with ``d > 0``.  Covers Adam-style and fixed-random controls."""
    if bool((d <= 0).any()):
        raise ValueError(f"{name}: diagonal preconditioner needs strictly positive entries")
    dd = d.detach()
    return OperatorPreconditioner(
        lambda v: dd * v,
        lambda v: v / dd,
        name=name,
        spec=f"P = diag(d), n={dd.numel()}, cond={float(dd.max() / dd.min()):.4g}",
    )


def fixed_random_preconditioner(
    n: int, seed: int, log_spread: float = 2.0, like: torch.Tensor | None = None
) -> OperatorPreconditioner:
    """A fixed, data-independent random diagonal ``P``.

    The negative control for stage 2.  It has the *shape* of a preconditioner --
    per-coordinate scaling, well-conditioned, positive -- and carries no
    information about ``H``, ``g`` or ``Sigma``.  Anything that cannot be
    separated from this is not being separated on curvature.

    ``log_spread`` sets the condition number: entries are ``exp(U(-s, s))``, so
    ``cond(P) = exp(2s)``.
    """
    gen = torch.Generator().manual_seed(seed)
    ref = like if like is not None else torch.zeros(n)
    u = torch.rand(n, generator=gen, dtype=torch.float64).to(ref.dtype).to(ref.device)
    d = torch.exp((2.0 * u - 1.0) * log_spread)
    p = diagonal_preconditioner(d, name=f"fixed-random(seed={seed})")
    p.spec = f"P = diag(exp(U(-{log_spread},{log_spread}))), seed={seed}, data-independent"
    return p


def dense_preconditioner(P: torch.Tensor, name: str, spec: str = "") -> OperatorPreconditioner:
    """``P`` given as an explicit symmetric positive-definite matrix.

    For synthetic problems where the exact answer is known and must be checked
    against, not for anything at scale.
    """
    Pd = P.detach()
    sym = 0.5 * (Pd + Pd.T)
    lam = torch.linalg.eigvalsh(sym)
    if float(lam.min()) <= 0:
        raise ValueError(f"{name}: dense preconditioner is not positive definite")
    Pinv = torch.linalg.inv(sym)
    return OperatorPreconditioner(
        lambda v: sym @ v,
        lambda v: Pinv @ v,
        name=name,
        spec=spec or f"dense P, n={sym.shape[0]}, cond={float(lam.max() / lam.min()):.4g}",
    )


# --------------------------------------------------------------------- frozen


def frozen_spectral_preconditioner(
    snapshot: ProbeContext,
    f: Callable[[torch.Tensor], torch.Tensor],
    rank: int,
    name: str,
    tail_node: str = "mean",
) -> OperatorPreconditioner:
    """``P`` captured at ``snapshot`` and then applied unchanged, forever.

    The staleness mutant.  Built from a rank-``rank`` Lanczos basis ``Q`` of the
    snapshot operator: inside ``span(Q)`` it is ``Q f(T)^-1 Q'``, and on the
    orthogonal complement -- where the snapshot measured nothing -- it is a
    scalar ``1/f(lam_ref)``.

    The complement scaling is an arbitrary choice and is stated rather than
    hidden: ``tail_node="mean"`` uses the quadrature-weighted mean node,
    ``"max"`` the largest.  It cannot be avoided, because a rank-``k`` snapshot
    genuinely contains no information about the other ``n - k`` directions, and
    pretending otherwise would be the same self-referential error the rest of
    this harness is built to exclude.

    This is not a strawman.  It is what a Krylov basis carried across steps
    without re-projection *is* -- the exact configuration that scored 85x
    optimistic on this project's own residual telemetry (findings F8/F10).
    """
    from dsn.lanczos import lanczos_iter
    from dsn.subspace.krylov import _tridiagonal

    Q = None
    T = None
    for st in lanczos_iter(snapshot.matvec, snapshot.g, rank):
        Q, T = st.Q, _tridiagonal(st.alphas, st.betas)
    if Q is None or T is None:
        raise RuntimeError("frozen snapshot: Lanczos broke down immediately")

    lam, V = torch.linalg.eigh(0.5 * (T + T.T))
    fv = f(lam)
    if bool((fv <= 0).any()):
        raise ValueError(f"{name}: spectral function non-positive on the snapshot spectrum")

    if tail_node == "mean":
        ref = lam.abs().mean()
    elif tail_node == "max":
        ref = lam.abs().max()
    else:  # pragma: no cover - guarded input
        raise ValueError(f"unknown tail_node {tail_node!r}")
    tail = 1.0 / float(f(ref.reshape(1))[0])

    QV = Q @ V  # (n, rank) eigenbasis of the snapshot's projected curvature
    inv_f = (1.0 / fv).to(Q.dtype)
    f_vals = fv.to(Q.dtype)

    def apply(v: torch.Tensor) -> torch.Tensor:
        c = QV.T @ v
        return QV @ (inv_f * c) + tail * (v - QV @ c)

    def apply_inverse(v: torch.Tensor) -> torch.Tensor:
        c = QV.T @ v
        return QV @ (f_vals * c) + (v - QV @ c) / tail

    return OperatorPreconditioner(
        apply,
        apply_inverse,
        name=name,
        spec=(
            f"frozen at step {snapshot.step}: rank-{rank} spectral snapshot, "
            f"tail 1/f({tail_node} |lam|) = {tail:.4g}, never refreshed"
        ),
    )


def spectral_family(deltas: Sequence[float]) -> list[SpectralPreconditioner]:
    """The DSN preconditioner at several damping values -- one Lanczos cost total."""
    return [dsn_preconditioner(d) for d in deltas]


def subspace_damping_preconditioner(
    U: torch.Tensor, epsilon: float, name: str
) -> OperatorPreconditioner:
    """``P = (I - UU') + epsilon UU'`` -- identity, except ``span(U)`` is damped.

    The mutant that targets ``gain``'s denominator directly.  ``gain`` is

        R(P) = (g'Pg)^2 / ( g'PHPg + tr(P H P Sigma) )

    and the noise term ``tr(PHP Sigma)`` is quadratic in ``P``.  Shrinking ``P``
    on the subspace where ``Sigma`` lives therefore drives that term toward zero
    *quadratically* while the numerator only loses whatever gradient mass ``U``
    happens to carry.  As ``epsilon -> 0``, ``R(P)`` diverges and ``gain`` reports
    an arbitrarily good preconditioner.

    What it actually is: an optimizer that has been told never to move in
    ``span(U)``.  The one-step model scores that as pure profit because refusing
    to move costs no variance and the model is only asked about one step.  Over a
    trajectory it is a permanent floor on the loss.

    ``epsilon = 0`` is rejected -- a singular ``P`` has no ``P^-1``, so ``rho``
    is undefined and the mutant would be untestable rather than merely bad.
    """
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0, 1]; a singular P has no rho")
    Ud = U.detach()

    def apply(v: torch.Tensor) -> torch.Tensor:
        c = Ud.T @ v
        return v - (1.0 - epsilon) * (Ud @ c)

    def apply_inverse(v: torch.Tensor) -> torch.Tensor:
        c = Ud.T @ v
        return v + (1.0 / epsilon - 1.0) * (Ud @ c)

    return OperatorPreconditioner(
        apply,
        apply_inverse,
        name=name,
        spec=(
            f"P = (I - UU') + {epsilon:g} UU', rank(U) = {Ud.shape[1]}: "
            "refuses to move in span(U)"
        ),
    )


def power_preconditioner(power: float, delta: float) -> SpectralPreconditioner:
    """``P = (|H| + delta)^-power`` -- a one-knob family through SGD and Newton.

    ``power = 0`` is plain SGD, ``power = 1`` is DSN's damped saddle-free
    operator, ``power > 1`` over-preconditions.  A family rather than a point,
    because a statistic is calibrated by checking that it *orders* candidates
    the way real training does, not by checking one candidate.
    """
    if power < 0:
        raise ValueError("power must be non-negative")
    return SpectralPreconditioner(
        lambda lam: (lam.abs() + delta) ** power,
        name=f"power(p={power:g},delta={delta:g})",
        spec=f"P = (|H| + {delta:g})^-{power:g}",
        scale=delta,
    )


class KrylovRestrictedPreconditioner(Preconditioner):
    """DSN's *actual* operator: the damped inverse restricted to a rank-k subspace.

    Every verdict this harness produced for DSN before this class existed was a
    verdict about ``(|H| + delta)^-1`` at **full rank**.  That is an upper bound
    on what a rank-``k`` implementation can achieve, and a full-rank PASS does
    not imply a rank-restricted one.  This is the thing DSN ships:

        P_k = W (|W'HW| + delta)^-1 W'  +  c (I - WW')

    with ``W`` an orthonormal basis of the ``k``-step Krylov space ``K_k(H, g)``.
    That subspace is not a model of DSN's -- it *is* DSN's.  ``KrylovBuilder``
    seeds Lanczos with the gradient, and at the shipped ``k_max <= 10`` nothing
    ever survives Ritz re-validation, so it rebuilds from scratch every step
    (negative-result doc, section 4: ``m_conv = 0`` at 23 of 24 pairs).

    The complement
    --------------

    ``c`` is what the optimizer does in the ``n - k`` directions the subspace
    does not cover.  DSN puts AdamW there, whose per-coordinate scaling is
    data-dependent and is *not* a scalar -- so ``c`` is an explicit,
    acknowledged approximation and the default ``c = 1`` models it as plain SGD.
    Two reasons that is the right default rather than ``c = 0``:

    * ``c = 0`` makes ``P`` singular, so ``P^-1`` does not exist and ``rho`` is
      undefined.  The candidate would be untestable rather than merely bad.
    * ``c = 0`` is also not what DSN does.  DSN always steps in the complement;
      reporting a subspace-only operator would understate it.

    Because ``c`` is a modelling choice and not a measurement, sweep it.  A
    verdict that flips between ``c = 0.1`` and ``c = 10`` is a verdict about the
    complement, not about the subspace.
    """

    def __init__(self, delta: float, rank: int, complement: float = 1.0,
                 name: str | None = None):
        if rank < 1:
            raise ValueError("rank must be at least 1")
        if complement <= 0:
            raise ValueError(
                "complement must be positive; c = 0 makes P singular, so rho is "
                "undefined and the candidate cannot be screened at all"
            )
        self.delta = delta
        self.rank = rank
        self.complement = complement
        self.name = name or f"dsn-rank{rank}(delta={delta:g},c={complement:g})"
        self.spec = (
            f"P = W (|W'HW| + {delta:g})^-1 W' + {complement:g} (I - WW'), "
            f"W = rank-{rank} Krylov basis seeded at g -- DSN's shipped operator"
        )
        self._cache: dict[int, OperatorPreconditioner] = {}

    def _for(self, ctx: ProbeContext) -> OperatorPreconditioner:
        key = id(ctx)
        if key not in self._cache:
            Q, T = ctx.krylov(self.rank)
            lam, V = torch.linalg.eigh(0.5 * (T + T.T))
            QV = Q @ V
            inv = (1.0 / (lam.abs() + self.delta)).to(Q.dtype)
            fwd = (lam.abs() + self.delta).to(Q.dtype)
            c = self.complement

            def apply(v: torch.Tensor) -> torch.Tensor:
                a = QV.T @ v
                return QV @ (inv * a) + c * (v - QV @ a)

            def apply_inverse(v: torch.Tensor) -> torch.Tensor:
                a = QV.T @ v
                return QV @ (fwd * a) + (v - QV @ a) / c

            self._cache[key] = OperatorPreconditioner(
                apply, apply_inverse, name=self.name, spec=self.spec
            )
        return self._cache[key]

    def apply(self, v: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError(
            "a Krylov-restricted P is defined relative to a probe context; "
            "use forms(ctx, depth)"
        )

    def extra_cost(self, ctx: ProbeContext) -> ProbeCost:
        return self._for(ctx).extra_cost(ctx)

    def forms(self, ctx: ProbeContext, depth: int) -> Forms:
        return self._for(ctx).forms(ctx, depth)
