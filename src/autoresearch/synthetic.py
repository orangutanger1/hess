"""Synthetic quadratics with closed-form gate statistics.

Used for two things and nothing else:

1. **Validating the estimator.** Every quantity the gate computes from Lanczos
   quadrature and cross-fit sampling has a closed form here, so the estimator
   can be checked against the truth rather than against itself.  (Checking an
   estimator against itself is the F8/F10 failure this project already paid
   for.)
2. **Carrying the mutants.** A gate that has not been shown to separate a known
   good preconditioner from a known bad one is an unvalidated gate, and running
   it against real candidates would launder its own noise into a verdict.

Construction
------------

Everything is diagonal in one fixed eigenbasis and then rotated by a random
orthogonal ``Q``:

    H = Q diag(lam) Q',   g = Q c,   Sigma = Q diag(s) Q'

The rotation makes the Krylov space non-degenerate (a diagonal operator seeded
with a coordinate vector has a one-dimensional Krylov space, which would make
the depth sweep vacuous) and changes no statistic, since every quantity the gate
computes is a trace or a quadratic form and therefore basis-independent.

Coordinates are split into three blocks, which is what makes the statistics
steerable:

    G   carries the gradient and no noise
    N1  carries noise, moderately sharp
    N2  carries noise, and is the block that *drifts*

Because ``g`` and ``Sigma`` are held fixed while ``lam`` drifts, ``rho`` of any
fixed ``P`` is **constant in time by construction** -- it is a function of
``(P, g, Sigma)`` only.  That is not a quirk of these numbers; it is the reason
``gate.py`` carries a second, H-aware statistic, and ``test_gate_mutants``
asserts the invariance to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .probes import ProbeContext, ProbeCost


@dataclass(frozen=True)
class QuadraticProblem:
    """A quadratic with known ``H``, ``g`` and gradient-noise covariance."""

    lam: torch.Tensor  # (n,) eigenvalues, all positive
    c: torch.Tensor  # (n,) gradient coefficients in the eigenbasis
    s: torch.Tensor  # (n,) noise variance per eigendirection
    Q: torch.Tensor  # (n, n) orthogonal
    noise_scale: float = 1.0
    name: str = "quadratic"

    @property
    def n(self) -> int:
        return int(self.lam.numel())

    def H(self) -> torch.Tensor:
        return self.Q @ torch.diag(self.lam) @ self.Q.T

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.Q @ (self.lam * (self.Q.T @ v))

    def g(self) -> torch.Tensor:
        return self.Q @ self.c

    def rotate_diag(self, d: torch.Tensor) -> torch.Tensor:
        """A diagonal-in-eigenbasis operator, expressed in ambient coordinates."""
        return self.Q @ torch.diag(d) @ self.Q.T

    # ------------------------------------------------------- closed-form truth

    def exact_rho(self, p: torch.Tensor) -> float:
        """``rho`` for ``P = Q diag(p) Q'``, computed exactly from the spectrum."""
        gg = float((self.c**2).sum())
        gPg = float((p * self.c**2).sum())
        trP = float((p * self.s).sum())
        trPinv = float((self.s / p).sum())
        return (trPinv / trP) * (gPg**2) / (gg**2)

    def exact_gain(self, p: torch.Tensor) -> float:
        """``gain`` for ``P = Q diag(p) Q'``, computed exactly from the spectrum."""
        ns = self.noise_scale
        gg = float((self.c**2).sum())
        gPg = float((p * self.c**2).sum())
        RP = gPg**2 / (
            float((p**2 * self.lam * self.c**2).sum())
            + ns * float((p**2 * self.lam * self.s).sum())
        )
        RI = gg**2 / (
            float((self.lam * self.c**2).sum()) + ns * float((self.lam * self.s).sum())
        )
        return RP / RI

    def exact_R(self, p: torch.Tensor) -> float:
        """``R(P) = (g'Pg)^2 / (g'PHPg + tr(PHP Sigma))`` -- the unnormalized form."""
        ns = self.noise_scale
        gPg = float((p * self.c**2).sum())
        return gPg**2 / (
            float((p**2 * self.lam * self.c**2).sum())
            + ns * float((p**2 * self.lam * self.s).sum())
        )

    def newton_diag(self) -> torch.Tensor:
        """``P* = H^-1`` in the eigenbasis -- the P0-derivation optimum."""
        return 1.0 / self.lam

    def exact_spectral_statistics(
        self, f, g: torch.Tensor, u: Sequence[torch.Tensor], noise_scale: float | None = None
    ) -> tuple[float, float]:
        """``(rho, gain)`` for ``P = f(H)^-1`` computed from the *true* spectrum.

        Uses the same probe vectors the quadrature would, so the difference
        between this and ``gate.statistics`` is **purely Lanczos truncation** --
        no sampling error, no different ``Sigma`` draw.  That separation is the
        whole point: it lets "the estimator has not converged in depth" be
        distinguished from "the statistic really is that value".
        """
        ns = self.noise_scale if noise_scale is None else noise_scale
        fv = f(self.lam)
        if bool((fv <= 0).any()):
            raise ValueError("spectral function must be positive on the spectrum")

        def forms(v):
            a = (self.Q.T @ v) ** 2
            return (
                float((a / fv).sum()),          # v'Pv
                float((a * fv).sum()),          # v'P^-1 v
                float((a * self.lam / fv**2).sum()),  # v'PHPv
                float((a * self.lam).sum()),    # v'Hv
            )

        gPg, gPinvg, gPHPg, gHg = forms(g)
        per_u = [forms(v) for v in u]
        n = len(per_u)
        gg = float(g.dot(g))
        rho = (sum(q[1] for q in per_u) / sum(q[0] for q in per_u)) * gPg**2 / gg**2
        RP = gPg**2 / (gPHPg + ns * sum(q[2] for q in per_u) / n)
        RI = gg**2 / (gHg + ns * sum(q[3] for q in per_u) / n)
        return rho, RP / RI

    def ceiling_R(self) -> float:
        """``sup_P R(P)`` over *all* symmetric ``P``, by Cauchy-Schwarz.

        With ``C = g g' + Sigma``,

            R(P) = tr(P g g')^2 / tr(P H P C)  <=  (g'H^-1 g)(g'C^-1 g)

        Reported as a sanity bound rather than as a gate: because ``P = I`` is
        feasible, this ceiling is ``>= R(I)`` always, so a *gate* built on it
        would pass everything.  Its use here is the opposite -- in the noiseless
        limit it is attained exactly at ``P = H^-1``, which is how the "exact
        optimum" mutant is verified to be the optimum rather than merely good.
        """
        ns = self.noise_scale
        gHinvg = float((self.c**2 / self.lam).sum())
        # C is diagonal in the eigenbasis only up to the rank-one gg' term;
        # Sherman-Morrison keeps it exact:  g'C^-1 g = a / (1 + a),  a = g'D^-1 g.
        sd = ns * self.s
        live = self.c**2 > 0
        if bool((sd[live] <= 0).any()):
            # A gradient direction with no noise sends `a` to infinity, and the
            # ratio to its limit of 1.  Handled rather than guarded against,
            # because it is the noiseless limit -- exactly the regime in which
            # this bound is tight and therefore the one worth checking.
            gCinvg = 1.0
        else:
            a = float((self.c[live] ** 2 / sd[live]).sum())
            gCinvg = a / (1.0 + a)
        return gHinvg * gCinvg

    # ------------------------------------------------------------- probe input

    def exact_cross_fit(self) -> list[torch.Tensor]:
        """Deterministic vectors whose empirical second moment is *exactly* Sigma.

        ``u_i = sqrt(m * s_i) * Q e_i`` over the ``m`` directions carrying noise
        gives ``(1/m) sum_i u_i u_i' = Q diag(s) Q'`` with no sampling error at
        all.  That removes the estimator's Monte-Carlo noise so a closed-form
        comparison tests the *estimator*, not the draw.

        Sampling noise is not thereby swept away -- ``sampled_cross_fit`` exists
        for that -- it is isolated, so the two error sources can be checked one
        at a time.
        """
        idx = torch.nonzero(self.s > 0, as_tuple=False).flatten()
        m = int(idx.numel())
        if m == 0:
            raise ValueError("problem has no noise directions")
        out = []
        for i in idx.tolist():
            e = torch.zeros_like(self.c)
            e[i] = (m * float(self.s[i])) ** 0.5
            out.append(self.Q @ e)
        return out

    def sampled_cross_fit(self, h: int, seed: int) -> list[torch.Tensor]:
        """``h`` genuine draws ``u ~ N(0, Sigma)``, for exercising the bootstrap."""
        gen = torch.Generator().manual_seed(seed)
        root = self.s.sqrt()
        return [
            self.Q
            @ (root * torch.randn(self.n, generator=gen, dtype=self.c.dtype))
            for _ in range(h)
        ]

    def context(
        self,
        depths,
        *,
        exact_noise: bool = True,
        h: int = 16,
        seed: int = 0,
        step: int = 0,
        label: str = "",
    ) -> ProbeContext:
        u = self.exact_cross_fit() if exact_noise else self.sampled_cross_fit(h, seed)
        return ProbeContext(
            matvec=self.matvec,
            g=self.g(),
            u=u,
            depths=tuple(depths),
            noise_scale=self.noise_scale,
            curv_batch=1,
            micro_batch=1,
            step=step,
            label=label or self.name,
            shared_cost=ProbeCost(),
        )


def _orthogonal(n: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=gen, dtype=dtype)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def blocked_quadratic(
    lam_g: float,
    lam_n1: float,
    lam_n2: float,
    *,
    block: int = 4,
    noise_var: float = 1.0,
    spread: float = 1.15,
    seed: int = 0,
    noise_scale: float = 1.0,
    g_noise_frac: float = 0.01,
    name: str = "blocked",
    dtype: torch.dtype | None = None,
) -> QuadraticProblem:
    """Three blocks of ``block`` eigendirections: gradient, noise, drifting noise.

    ``spread`` gives each block a geometric spread of eigenvalues so that no two
    eigenvalues coincide.  Distinct eigenvalues matter: a repeated eigenvalue
    collapses the Krylov space, which would make the depth sweep -- and
    therefore the ceiling/floor rule -- vacuous without saying so.

    ``g_noise_frac`` puts a small share of the gradient's squared norm on the
    noise blocks.  Setting it to zero makes the blocks exactly orthogonal, and a
    Krylov basis seeded at ``g`` then *provably cannot see the noise blocks at
    all* -- so a low-rank snapshot of that basis is neither stale nor fresh, it
    is simply blind, and testing staleness against it measures nothing.  Real
    gradients are not orthogonal to their own noise; 1% is enough to make the
    Krylov space full-support while leaving every statistic within a few percent
    of the exactly-blocked values.
    """
    dt = dtype if dtype is not None else torch.get_default_dtype()
    n = 3 * block
    f = torch.tensor([spread ** (i - (block - 1) / 2) for i in range(block)], dtype=dtype or dt)
    lam = torch.cat([lam_g * f, lam_n1 * f, lam_n2 * f]).to(dt)
    if not 0.0 <= g_noise_frac < 1.0:
        raise ValueError("g_noise_frac must be in [0, 1)")
    c = torch.cat(
        [
            torch.full((block,), ((1.0 - g_noise_frac) / block) ** 0.5),
            torch.full((2 * block,), (g_noise_frac / (2 * block)) ** 0.5),
        ]
    ).to(dt)
    s = torch.cat([torch.zeros(block), torch.full((2 * block,), noise_var)]).to(dt)
    return QuadraticProblem(
        lam=lam,
        c=c,
        s=s,
        Q=_orthogonal(n, seed, dt),
        noise_scale=noise_scale,
        name=name,
    )


def favorable_pair(seed: int = 0, **kw) -> tuple[QuadraticProblem, QuadraticProblem]:
    """``(t=0, t=T)`` problems sharing ``g``, ``Sigma`` and eigenbasis; ``H`` drifts.

    At ``t=0`` the gradient sits in a flat block and the noise in sharp ones, so
    curvature preconditioning pays and ``P = H_0^-1`` passes both statistics.

    Between ``t=0`` and ``t=T`` the second noise block sharpens by ``1e5``.
    ``g`` and ``Sigma`` do not move, so:

    * ``rho`` of the *frozen* ``P = H_0^-1`` is numerically identical at both
      times -- it cannot see the drift, because ``H`` does not appear in it;
    * ``gain`` of that same frozen ``P`` collapses to ~0.01, because the
      preconditioner still amplifies a direction whose curvature grew by five
      orders of magnitude, so preconditioned gradient noise explodes;
    * ``gain`` of the correctly-refreshed ``P = H_T^-1`` stays far above 1.

    That triple is the whole justification for the composite gate, and it is
    reproduced as a logged measurement in ``scripts/validate_gate.py``.
    """
    t0 = blocked_quadratic(0.1, 10.0, 0.01, seed=seed, name="drift@t0", **kw)
    tT = blocked_quadratic(0.1, 10.0, 1000.0, seed=seed, name="drift@tT", **kw)
    return t0, tT


def flat_tail_quadratic(
    n: int = 400,
    lo: float = 1e-4,
    hi: float = 1e2,
    seed: int = 0,
    noise_scale: float = 1.0,
    name: str = "flat-tail",
) -> QuadraticProblem:
    """A spectrum shaped like a real network's: a few sharp directions, a long
    nearly-flat tail, and the gradient and noise spread across all of it.

    Built to stress the *estimator* rather than the predicate.  For a spectral
    ``P = (|lam| + delta)^-1`` the gate integrates ``1/f`` (for ``rho``) and
    ``lam/f^2`` (for ``gain``); both are near-singular when ``delta`` sits inside
    the flat tail, and a Gauss rule can only fit polynomials.  The tail here runs
    six decades below the top of the spectrum, so at any practical Lanczos depth
    the smallest quadrature node is far above the smallest eigenvalue and the
    near-singular region is extrapolated rather than integrated.

    ``exact_spectral_statistics`` gives the converged answer for comparison.
    """
    dt = torch.get_default_dtype()
    lam = torch.logspace(
        torch.log10(torch.tensor(lo)).item(),
        torch.log10(torch.tensor(hi)).item(),
        n,
    ).to(dt)
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(n, generator=gen, dtype=dt)
    c = c / c.norm()
    s = (torch.rand(n, generator=gen, dtype=dt) + 0.5)
    return QuadraticProblem(
        lam=lam, c=c, s=s, Q=_orthogonal(n, seed, dt),
        noise_scale=noise_scale, name=name,
    )


def unfavorable_quadratic(seed: int = 0, **kw) -> QuadraticProblem:
    """Gradient in the sharp block, noise in the flat ones -- the DSN situation.

    Curvature preconditioning has nothing to buy here, and ``P = H^-1`` should
    be rejected.  This is the measured shape of DSN's own MNIST trajectory
    (``rho`` capped at 0.717 against a threshold of 1), reproduced synthetically
    so the gate's rejection behavior can be tested without MNIST.
    """
    return blocked_quadratic(10.0, 0.1, 0.05, seed=seed, name="unfavorable", **kw)
