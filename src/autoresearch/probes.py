"""Cheap curvature and gradient-noise probes shared by every stage-1 gate call.

Everything here is *measurement*: no training runs, no solver code, no writes
into ``src/dsn``. The only curvature primitive is ``dsn.lanczos.lanczos_iter``,
used exactly as ``solve.subspace_newton`` and ``scripts/measure_rho.py`` use it.

Two probe families, and the whole cost of stage 1 is the sum of them:

``quadrature``      one Lanczos run per seed vector, turned into a Gauss
                    quadrature rule for ``b' f(H) b``.  One run serves *every*
                    spectral function ``f`` at *every* truncation depth, so
                    screening ten spectral preconditioner candidates costs the
                    same curvature products as screening one.
``cross_fit``       paired cross-half microbatch gradient differences, whose
                    outer product is an unbiased estimator of the mini-batch
                    gradient covariance with no self-referential mean.

The cross-fit construction is not a stylistic choice.  A covariance centered on
a mean computed from the same samples it is then evaluated against is the
F8/F10 failure mode this project has already been burned by at 85x (see
``docs/superpowers/plans/2026-07-29-negative-result.md`` section 2.4), so the
estimator is built so that failure mode cannot occur.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch

from dsn.lanczos import lanczos_iter
from dsn.subspace.krylov import _tridiagonal

Matvec = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------- cost counter


@dataclass
class ProbeCost:
    """Curvature/gradient work, kept as counts and batch sizes rather than FLOPs.

    Counting operations rather than FLOPs keeps the cost model (which needs a
    model architecture to be meaningful) out of the probe layer.  ``cost.py``
    turns these into FLOPs.
    """

    #: ``{batch_size: count}`` for full gradient evaluations.
    grads: dict[int, int] = field(default_factory=dict)
    #: ``{batch_size: count}`` for curvature-vector products.
    hvps: dict[int, int] = field(default_factory=dict)
    #: ``{batch_size: count}`` for loss-only forward passes.
    forwards: dict[int, int] = field(default_factory=dict)

    def add_grad(self, batch_size: int, n: int = 1) -> None:
        self.grads[batch_size] = self.grads.get(batch_size, 0) + n

    def add_hvp(self, batch_size: int, n: int = 1) -> None:
        self.hvps[batch_size] = self.hvps.get(batch_size, 0) + n

    def add_forward(self, batch_size: int, n: int = 1) -> None:
        self.forwards[batch_size] = self.forwards.get(batch_size, 0) + n

    def __add__(self, other: "ProbeCost") -> "ProbeCost":
        out = ProbeCost()
        for src in (self, other):
            for bs, n in src.grads.items():
                out.add_grad(bs, n)
            for bs, n in src.hvps.items():
                out.add_hvp(bs, n)
            for bs, n in src.forwards.items():
                out.add_forward(bs, n)
        return out

    def scaled(self, factor: float) -> "ProbeCost":
        """Fractional share of this cost, for amortizing a shared probe.

        Counts become fractional.  That is deliberate: rounding a shared cost to
        integers before attributing it is how shared search cost goes missing
        from a ledger, which is the failure this harness exists to prevent.
        """
        out = ProbeCost()
        out.grads = {bs: n * factor for bs, n in self.grads.items()}  # type: ignore[assignment]
        out.hvps = {bs: n * factor for bs, n in self.hvps.items()}  # type: ignore[assignment]
        out.forwards = {bs: n * factor for bs, n in self.forwards.items()}  # type: ignore[assignment]
        return out

    def total_hvps(self) -> float:
        return sum(self.hvps.values())

    def as_dict(self) -> dict:
        return {
            "grads": {str(k): v for k, v in self.grads.items()},
            "hvps": {str(k): v for k, v in self.hvps.items()},
            "forwards": {str(k): v for k, v in self.forwards.items()},
        }


# ----------------------------------------------------------- Gauss quadrature


@dataclass(frozen=True)
class QuadRule:
    """A ``k``-node Gauss quadrature rule for ``b' f(H) b``.

        b' f(H) b  ~=  ||b||^2 * sum_j w_j f(lam_j),     sum_j w_j == 1

    with ``(lam, V) = eigh(T_k)`` from a ``k``-step Lanczos run seeded at ``b``
    and ``w_j = V[0, j]^2``.  One rule evaluates *any* ``f``, so the entire
    spectral preconditioner family is screened at the cost of the single
    Lanczos run that produced the rule.
    """

    nodes: torch.Tensor
    weights: torch.Tensor
    norm2: float
    depth: int

    def apply(self, f: Callable[[torch.Tensor], torch.Tensor]) -> float:
        return self.norm2 * float((self.weights * f(self.nodes)).sum())

    def negative_mass(self) -> float:
        """Weight on negative curvature -- the ``neg_frac`` diagnostic.

        Above ~0 the exact Hessian is indefinite, ``H^-1`` is not a
        preconditioner, and any statistic that assumes it exists is describing
        something other than what the optimizer does.
        """
        return float(self.weights[self.nodes < 0].sum())


def quadrature(
    matvec: Matvec,
    b: torch.Tensor,
    depths: Sequence[int],
    cost: ProbeCost | None = None,
    batch_size: int = 1,
) -> dict[int, QuadRule]:
    """Quadrature rules for ``b' f(H) b`` at each Lanczos depth in ``depths``.

    A depth-``j`` rule must come from an ``eigh`` of the ``j``-by-``j``
    tridiagonal ``T_j``, **not** from a subset of the depth-``k`` rule's nodes.
    ``eigh`` returns nodes in ascending eigenvalue order, so slicing them keeps
    the most-negative curvature and discards the sharpest directions, which is
    the opposite of a shallower Krylov space.  That bug made the first `rho`
    sweep in this project meaningless (negative-result doc, section 2.7).
    Building ``T_j`` from the recurrence as Lanczos yields it costs no extra
    curvature products, so there is no reason to ever take the shortcut.

    On Krylov breakdown before the deepest requested depth the space is
    exhausted and the deepest *reached* rule is exact; it is reused for the
    missing depths rather than reporting a hole.  Note that the reached depth
    need not itself be a requested depth -- breaking down at 6 with
    ``depths=(4, 8)`` must fall back to the depth-6 rule, not to nothing.  The
    fallback is only ever applied upward (missing depths are strictly deeper
    than the one reached), so it never substitutes a deeper rule for a shallower
    request, which would be the meaningless direction.
    """
    want = sorted({int(d) for d in depths if d > 0})
    if not want:
        raise ValueError("depths must contain at least one positive Lanczos depth")

    norm2 = float(b.dot(b))

    def rule_of(state) -> QuadRule:
        T = _tridiagonal(state.alphas, state.betas)
        lam, V = torch.linalg.eigh(0.5 * (T + T.T))
        return QuadRule(
            nodes=lam.double(),
            weights=(V[0, :] ** 2).double(),
            norm2=norm2,
            depth=state.n_hvp,
        )

    rules: dict[int, QuadRule] = {}
    last = None
    for st in lanczos_iter(matvec, b, max(want)):
        last = st
        if cost is not None:
            cost.add_hvp(batch_size)
        if st.n_hvp in want:
            rules[st.n_hvp] = rule_of(st)
    if last is None:
        raise RuntimeError("Lanczos broke down immediately: ||b|| below tol")
    missing = [j for j in want if j not in rules]
    if missing:
        exhausted = rule_of(last)
        for j in missing:
            rules[j] = exhausted
    return rules


# ------------------------------------------------------------- cross-fit noise


def cross_fit(microbatch_grads: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Paired cross-half differences ``u_i = (g_Ai - g_Bi) / sqrt(2)``.

    For independent microbatch gradients with common mean, ``E[u_i u_i'] =
    Sigma`` exactly, where ``Sigma`` is the covariance of a single microbatch
    gradient.  No mean is ever estimated, so no statistic built on ``u`` can be
    evaluated against a mean computed from the samples it is testing.

    Requires an even number of microbatches; an odd one is a caller bug, not
    something to silently truncate, because a dropped microbatch changes the
    effective sample size that every downstream confidence interval assumes.
    """
    n = len(microbatch_grads)
    if n < 2 or n % 2 != 0:
        raise ValueError(f"need an even number of microbatches >= 2, got {n}")
    h = n // 2
    root2 = 2.0**0.5
    return [(microbatch_grads[i] - microbatch_grads[h + i]) / root2 for i in range(h)]


# ------------------------------------------------------------- probe context


@dataclass
class ProbeContext:
    """Everything a gate evaluation needs at one point on one trajectory.

    Built once per checkpoint and shared by every candidate screened there.  The
    quadrature rules are the expensive part -- ``(h + 1) * max(depths)``
    curvature products -- and they are computed once no matter how many spectral
    candidates are screened.  ``ProbeCost`` records that as a *shared* cost so
    the ledger in ``report.py`` can amortize it rather than double-count it or,
    worse, drop it.

    ``noise_scale`` converts the measured microbatch covariance to the
    covariance of the gradient the optimizer being screened actually sees:

        Sigma_B = Sigma_1 / B   =>   Sigma_train = Sigma_micro * (B_micro/B_train)

    ``rho`` is invariant to this scale (it cancels), which is exactly why it is
    easy to forget.  The noisy-quadratic ``gain`` statistic is **not** invariant
    to it -- it weighs signal ``g'PHPg`` against noise ``tr(PHPSigma)`` -- so
    getting it wrong silently rescales the second half of the gate.
    """

    matvec: Matvec
    g: torch.Tensor
    u: list[torch.Tensor]
    depths: tuple[int, ...]
    noise_scale: float = 1.0
    curv_batch: int = 1
    micro_batch: int = 1
    step: int = -1
    label: str = ""

    #: Shared probe cost -- paid once per checkpoint, used by every candidate.
    shared_cost: ProbeCost = field(default_factory=ProbeCost)
    _g_rules: dict[int, QuadRule] | None = None
    _u_rules: list[dict[int, QuadRule]] | None = None
    _krylov: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.u:
            raise ValueError("ProbeContext needs at least one cross-fit vector")
        self.depths = tuple(sorted({int(d) for d in self.depths}))

    # The rules are built lazily and exactly once.  A candidate that needs no
    # spectral information still triggers them, because the SGD baseline term
    # `g'Hg + tr(H Sigma)` in `gain` needs curvature either way.
    def rules(self) -> tuple[dict[int, QuadRule], list[dict[int, QuadRule]]]:
        if self._g_rules is None:
            self._g_rules = quadrature(
                self.matvec, self.g, self.depths, self.shared_cost, self.curv_batch
            )
            self._u_rules = [
                quadrature(
                    self.matvec, ui, self.depths, self.shared_cost, self.curv_batch
                )
                for ui in self.u
            ]
        assert self._u_rules is not None
        return self._g_rules, self._u_rules

    def krylov(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Q, T)`` for the rank-``rank`` Krylov space seeded at ``g``.

        This is the subspace DSN itself builds: ``KrylovBuilder`` seeds Lanczos
        with the (deflated) gradient and rebuilds from scratch every step at the
        shipped ``k_max <= 10``, because nothing survives the Ritz re-validation
        (negative-result doc, section 4: ``m_conv = 0`` at 23 of 24 pairs).  So a
        fresh ``k``-step Krylov space at the current ``g`` is not an
        approximation of DSN's subspace -- it is DSN's subspace.

        Cached per rank.  The curvature products are charged to ``shared_cost``
        once, like the quadrature rules, since every rank-restricted candidate at
        this checkpoint re-uses the same basis prefix.
        """
        from dsn.subspace.krylov import _tridiagonal

        if rank not in self._krylov:
            last = None
            for st in lanczos_iter(self.matvec, self.g, rank):
                last = st
                self.shared_cost.add_hvp(self.curv_batch)
            if last is None:
                raise RuntimeError("Krylov restriction: Lanczos broke down at ||g||")
            self._krylov[rank] = (last.Q, _tridiagonal(last.alphas, last.betas))
        return self._krylov[rank]

    @property
    def n_cross_fit(self) -> int:
        return len(self.u)

    def neg_frac(self, depth: int | None = None) -> float:
        """``g^2``-weighted mass on negative curvature at this checkpoint."""
        g_rules, _ = self.rules()
        d = self.depths[-1] if depth is None else depth
        return g_rules[d].negative_mass()


def build_probe_context(
    loss_fn: Callable[[], torch.Tensor],
    params: Sequence[torch.Tensor],
    microbatch_losses: Iterable[Callable[[], torch.Tensor]],
    depths: Sequence[int],
    *,
    curv_batch: int,
    micro_batch: int,
    train_batch: int,
    step: int = -1,
    label: str = "",
) -> ProbeContext:
    """Assemble a ``ProbeContext`` from a curvature closure and microbatch closures.

    ``loss_fn`` defines ``H`` and ``g`` and must rebuild its graph on every call
    (``hvp`` consumes it).  The microbatch closures must be drawn from a pool
    **disjoint** from ``loss_fn``'s batch: sharing samples makes ``Sigma``
    correlated with ``g``, and every ceiling reported from it optimistic.
    """
    from dsn.curvature import hvp
    from dsn.utils import flat_grad

    cost = ProbeCost()

    g = flat_grad(loss_fn(), params, create_graph=False).detach()
    cost.add_grad(curv_batch)

    micro = []
    for mb in microbatch_losses:
        micro.append(flat_grad(mb(), params).detach())
        cost.add_grad(micro_batch)

    ctx = ProbeContext(
        matvec=lambda v: hvp(loss_fn, params, v),
        g=g,
        u=cross_fit(micro),
        depths=tuple(depths),
        noise_scale=micro_batch / train_batch,
        curv_batch=curv_batch,
        micro_batch=micro_batch,
        step=step,
        label=label,
    )
    ctx.shared_cost = cost
    return ctx
