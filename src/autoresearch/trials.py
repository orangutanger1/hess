"""Candidates, and the training trials stage 2 spends its budget on.

A *candidate* is two things bolted together: a ``Preconditioner`` that stage 1
can screen without training anything, and an optimizer that stage 2 can actually
run.  Keeping them in one object is what stops the two stages from silently
screening one thing and training another.

Trials are budgeted in **FLOPs, not steps**.  A fixed step count structurally
favours whichever method does the most work per step, and a fixed wall-clock box
structurally penalizes methods that pay setup cost early and win later --
precisely the class under test here.  Every trial therefore runs until its FLOP
budget is spent and reports the loss it reached, so a method that halves step
count at 4x per-step cost shows up as the regression it is.

No new optimizer is implemented here.  DSN is used as shipped; the two baselines
are plain SGD and a *fixed-random* preconditioner whose diagonal is drawn once
from a seed and never looks at the data.  The random control is the important
one: it has the shape of a preconditioner -- per-coordinate scaling, positive,
well-conditioned -- and carries no curvature information at all, so anything the
harness cannot separate from it is not being separated on curvature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import torch

from .cost import CostModel, analytic_cost_model
from .preconditioners import (
    IdentityPreconditioner,
    KrylovRestrictedPreconditioner,
    Preconditioner,
    fixed_random_preconditioner,
)
from .probes import ProbeCost
from .select import TrialResult


#: A diverged run is scored at this multiple of the loss at initialization.
#: Capping rather than returning ``inf`` or ``nan`` keeps the sub-Gaussian
#: machinery in stage 2 well-defined, and it errs in the safe direction: a capped
#: score *understates* how bad a divergent configuration is, so a candidate can
#: never be eliminated for diverging more spectacularly than another.
_DIVERGENCE_CAP = 10.0


def _cap(score: float, cap: float) -> tuple[float, bool]:
    if not math.isfinite(score) or score > cap:
        return cap, True
    return score, False


# ------------------------------------------------------------------ optimizer


class PreconditionedSGD(torch.optim.Optimizer):
    """``x <- x - lr * P g`` over the flat parameter vector.

    ``P = None`` is plain SGD.  Deliberately minimal: no momentum, no weight
    decay, no per-coordinate state.  The point of the baselines is that the
    *only* difference between them is ``P``, so any separation the harness finds
    between them is attributable to ``P`` and to nothing else.
    """

    def __init__(self, params, lr: float, precond: Preconditioner | None = None):
        params = list(params)
        super().__init__(params, dict(lr=lr))
        self.precond = precond
        self._params = [p for gr in self.param_groups for p in gr["params"]]

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        flat = torch.cat([
            (torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
            for p in self._params
        ])
        if self.precond is not None:
            flat = self.precond.apply(flat)
        lr = self.param_groups[0]["lr"]
        i = 0
        for p in self._params:
            n = p.numel()
            p.add_(flat[i : i + n].view_as(p), alpha=-lr)
            i += n
        return loss


# ------------------------------------------------------------------ candidate


@dataclass
class Candidate:
    """A preconditioner form, screenable by stage 1 and trainable by stage 2."""

    name: str
    #: How stage 1 screens it.  ``None`` means "not screenable" -- the baseline
    #: arm is admitted to stage 2 regardless, because it *is* the reference the
    #: discovery claims are made against.
    preconditioner: Preconditioner | None
    build: Callable[[Sequence[torch.Tensor], float], torch.optim.Optimizer]
    lr: float
    #: DSN owns its own closure protocol (it re-evaluates for every curvature
    #: product); the SGD family reads a pre-populated ``.grad``.
    needs_closure: bool = False
    is_baseline: bool = False
    notes: str = ""
    #: Learning rates the tuning phase may choose between.  Comparing optimizers
    #: at one shared learning rate measures which of them happens to like that
    #: learning rate; the design document fixes an *equal tuning budget* per
    #: optimizer for exactly this reason, and the tuning trials are charged to
    #: the ledger like any other search cost.
    lr_grid: tuple[float, ...] = ()


def make_optimizer(cand: Candidate, params: Sequence[torch.Tensor]):
    return cand.build(params, cand.lr)


def run_step(
    cand: Candidate,
    opt: torch.optim.Optimizer,
    loss_fn: Callable[[], torch.Tensor],
    cost: ProbeCost,
    batch: int,
) -> float:
    """One optimizer step, with its cost charged to ``cost``.

    Costs are read from the optimizer rather than assumed.  DSN's curvature
    count varies per step -- the builder stops adaptively, and the fallback path
    spends none at all -- so a fixed per-step estimate would misprice exactly the
    runs where the fallback fired.
    """
    if cand.needs_closure:
        loss = opt.step(loss_fn)
        tel = getattr(opt, "telemetry", None)
        n_hvp = int(getattr(tel, "n_hvp", 0))
        cost.add_grad(batch)  # the closure's gradient
        cost.add_hvp(batch, n_hvp)
        cost.add_forward(batch)  # same-batch acceptance ratio
        return float(loss.detach())

    opt.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    opt.step()
    cost.add_grad(batch)
    return float(loss.detach())


# --------------------------------------------------------------------- trials


def tune_learning_rate(
    cand: Candidate,
    trial,
    seeds: Sequence[int],
    cost_model: CostModel | None = None,
) -> tuple[float, float, list[dict]]:
    """Pick a learning rate for ``cand`` on seeds disjoint from the evaluation set.

    Returns ``(best_lr, flops_spent, table)``.  Three properties matter and each
    is a way this is usually got wrong:

    * **Tuning seeds are disjoint from selection seeds.**  Choosing the learning
      rate on the same seeds the winner is later certified on is selection on
      the test set, and it inflates every downstream p-value.
    * **The cost is charged.**  A 6-point grid on 2 seeds is 12 extra training
      runs per candidate; leaving them out of the ledger is precisely the
      omission this harness exists to prevent.
    * **Stability beats mean score.**  A learning rate that diverged on *any*
      tuning seed is disqualified while any fully-stable rate exists, even if
      its mean over the surviving seeds is the lowest.  Skipping this picks the
      knife edge: near a stability cliff, the largest rate that happens to
      survive two seeds looks best and then diverges on a third.  The damage is
      not just a bad trial -- a candidate whose scores are bimodal has a huge
      ``sigma_hat``, so stage 2 can never eliminate it and it survives to the
      report on the strength of its own instability.  Measured here: a rate one
      grid point over the cliff kept the random control alive for the full seed
      budget at a mean score four orders of magnitude worse than the baseline's.
    """
    grid = cand.lr_grid or (cand.lr,)
    table, spent = [], 0.0
    for lr in grid:
        probe = replace(cand, lr=lr)
        scores, diverged = [], 0
        for s in seeds:
            res = trial.run(probe, s, cost_model)
            spent += res.flops
            scores.append(res.score)
            diverged += bool(res.diverged)
        table.append({
            "lr": lr,
            "mean_score": sum(scores) / len(scores),
            "scores": scores,
            "n_diverged": diverged,
        })

    def pick(rows):
        finite = [r for r in rows if math.isfinite(r["mean_score"])]
        return min(finite, key=lambda r: r["mean_score"]) if finite else None

    stable = pick([r for r in table if r["n_diverged"] == 0]) or pick(table)
    return (stable["lr"] if stable else cand.lr), spent, table


@dataclass
class QuadraticTrial:
    """A streaming noisy least-squares problem: fast, and the truth is known.

    ``a_i ~ N(0, diag(lam))`` rotated into a random basis, ``b_i = a_i'w* +
    eps``.  The Hessian is ``E[a a']``, so the eigenvalue profile -- and
    therefore whether curvature preconditioning can pay -- is set directly
    rather than hoped for.  Gradient noise is genuine mini-batch noise, not an
    additive fiction, so ``Sigma`` is the thing stage 1 would measure.

    Used to exercise the selector end to end in seconds.  It is not a claim
    about any real workload.
    """

    dim: int = 24
    cond: float = 100.0
    batch: int = 16
    noise: float = 0.1
    flop_budget: float = 4e7
    max_steps: int = 400
    eval_batch: int = 512
    name: str = "quadratic"

    def _problem(self, seed: int):
        gen = torch.Generator().manual_seed(1000 + seed)
        dt = torch.get_default_dtype()
        lam = torch.logspace(0, torch.log10(torch.tensor(self.cond)).item(), self.dim, dtype=dt)
        q, r = torch.linalg.qr(torch.randn(self.dim, self.dim, generator=gen, dtype=dt))
        Q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        root = Q @ torch.diag(lam.sqrt()) @ Q.T
        w_star = torch.randn(self.dim, generator=gen, dtype=dt)
        return gen, root, w_star

    def _batch(self, gen, root, w_star, size):
        a = torch.randn(size, self.dim, generator=gen, dtype=root.dtype) @ root
        b = a @ w_star + self.noise * torch.randn(size, generator=gen, dtype=root.dtype)
        return a, b

    def run(self, cand: Candidate, seed: int, cost_model: CostModel | None = None) -> TrialResult:
        torch.manual_seed(seed)
        gen, root, w_star = self._problem(seed)
        w = torch.zeros(self.dim, dtype=root.dtype, requires_grad=True)
        opt = make_optimizer(cand, [w])
        cm = cost_model or analytic_cost_model(self.dim)
        cost = ProbeCost()

        a_eval, b_eval = self._batch(gen, root, w_star, self.eval_batch)
        with torch.no_grad():
            cap = _DIVERGENCE_CAP * float(0.5 * ((a_eval @ w - b_eval) ** 2).mean())
        steps = 0
        for _ in range(self.max_steps):
            if cm.total(cost) >= self.flop_budget:
                break
            a, b = self._batch(gen, root, w_star, self.batch)

            def loss_fn():
                return 0.5 * ((a @ w - b) ** 2).mean()

            run_step(cand, opt, loss_fn, cost, self.batch)
            steps += 1

        with torch.no_grad():
            raw = float(0.5 * ((a_eval @ w - b_eval) ** 2).mean())
        score, diverged = _cap(raw, cap)
        return TrialResult(
            score=score,
            flops=cm.total(cost),
            steps=steps,
            diverged=diverged,
            detail={"cost": cost.as_dict(), "budget": self.flop_budget,
                    "raw_score": raw, "cap": cap},
        )

    def probe_contexts(
        self,
        cand: Candidate,
        checkpoints: Sequence[int],
        depths: Sequence[int],
        *,
        seed: int = 0,
        curv_batch: int = 512,
        micro: int = 16,
        micro_batch: int = 64,
    ):
        """Probe contexts along the trajectory *this trial actually runs*.

        The gate and the trials must describe the same trajectory or stage 1 is
        screening a different problem from the one stage 2 trains on.  Building
        the contexts from the trial object rather than from a separate script is
        what keeps that true; the alternative -- a measurement script that
        re-creates the setup by hand -- is how the two drift apart.

        The curvature batch and the microbatch pool are drawn fresh from the
        stream and share no sample, so ``Sigma`` is not correlated with ``g``.
        """
        from .probes import build_probe_context

        torch.manual_seed(seed)
        gen, root, w_star = self._problem(seed)
        w = torch.zeros(self.dim, dtype=root.dtype, requires_grad=True)
        opt = make_optimizer(cand, [w])
        cost = ProbeCost()
        want = sorted(set(checkpoints))
        out = []
        for step in range(max(want) + 1):
            if step in want:
                a_c, b_c = self._batch(gen, root, w_star, curv_batch)
                mb = [self._batch(gen, root, w_star, micro_batch) for _ in range(micro)]
                out.append(
                    build_probe_context(
                        lambda: 0.5 * ((a_c @ w - b_c) ** 2).mean(),
                        [w],
                        [(lambda am=am, bm=bm: 0.5 * ((am @ w - bm) ** 2).mean())
                         for am, bm in mb],
                        depths=depths,
                        curv_batch=curv_batch,
                        micro_batch=micro_batch,
                        train_batch=self.batch,
                        step=step,
                        label=self.name,
                    )
                )
            a, b = self._batch(gen, root, w_star, self.batch)
            run_step(cand, opt, lambda: 0.5 * ((a @ w - b) ** 2).mean(), cost, self.batch)
        return out


@dataclass
class MnistTrial:
    """The 784-96-tanh-10 MLP this project already measures, held-out scored.

    Same architecture as ``scripts/measure_rho.py`` and ``scripts/sanity_mnist.py``
    so the trials and the gate are talking about the same trajectory.  Scored on
    a held-out split rather than on training loss: at a fixed FLOP budget a
    method that fits the training batch faster is not thereby better, and the
    design's own metric is validation loss.
    """

    train_size: int = 8192
    eval_size: int = 2048
    batch: int = 256
    flop_budget: float = 2e11
    max_steps: int = 400
    name: str = "mnist-mlp"
    _cache: dict = field(default_factory=dict, repr=False)

    def _data(self):
        if "x" not in self._cache:
            from torchvision import datasets, transforms

            ds = datasets.MNIST("data", train=True, download=True,
                                transform=transforms.ToTensor())
            n = self.train_size + self.eval_size
            x = torch.stack([ds[i][0] for i in range(n)]).to(torch.get_default_dtype())
            y = torch.tensor([ds[i][1] for i in range(n)])
            self._cache["x"], self._cache["y"] = x, y
        return self._cache["x"], self._cache["y"]

    def _model(self):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 96),
            torch.nn.Tanh(),
            torch.nn.Linear(96, 10),
        )

    def run(self, cand: Candidate, seed: int, cost_model: CostModel | None = None) -> TrialResult:
        x, y = self._data()
        x_tr, y_tr = x[: self.train_size], y[: self.train_size]
        x_ev, y_ev = x[self.train_size :], y[self.train_size :]

        torch.manual_seed(seed)
        model = self._model()
        params = list(model.parameters())
        opt = make_optimizer(cand, params)
        ce = torch.nn.functional.cross_entropy
        cm = cost_model or analytic_cost_model(sum(p.numel() for p in params))
        cost = ProbeCost()

        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            cap = _DIVERGENCE_CAP * float(ce(model(x_ev), y_ev))
        steps = 0
        for _ in range(self.max_steps):
            if cm.total(cost) >= self.flop_budget:
                break
            idx = torch.randint(0, self.train_size, (self.batch,), generator=gen)
            xb, yb = x_tr[idx], y_tr[idx]
            run_step(cand, opt, lambda: ce(model(xb), yb), cost, self.batch)
            steps += 1

        with torch.no_grad():
            raw = float(ce(model(x_ev), y_ev))
        score, diverged = _cap(raw, cap)
        return TrialResult(
            score=score,
            flops=cm.total(cost),
            steps=steps,
            diverged=diverged,
            detail={"cost": cost.as_dict(), "budget": self.flop_budget,
                    "raw_score": raw, "cap": cap},
        )

    def probe_contexts(
        self,
        cand: Candidate,
        checkpoints: Sequence[int],
        depths: Sequence[int],
        *,
        seed: int = 0,
        curv_batch: int = 1024,
        micro: int = 16,
        micro_batch: int = 128,
    ):
        """Probe contexts along the trajectory this trial actually runs.

        The curvature pool and the microbatch pool are carved off the *front* of
        the training split and are disjoint from each other.  They overlap the
        training stream, which is correct -- the gate is asking about the
        curvature the optimizer is currently walking through, not about a
        held-out one -- but ``Sigma`` and ``g`` must not share samples with each
        other, and they do not.
        """
        from .probes import build_probe_context

        x, y = self._data()
        x_tr, y_tr = x[: self.train_size], y[: self.train_size]
        need = curv_batch + micro * micro_batch
        if need > self.train_size:
            raise ValueError(
                f"probe pools need {need} samples but train_size is {self.train_size}"
            )
        x_c, y_c = x_tr[:curv_batch], y_tr[:curv_batch]
        x_m, y_m = x_tr[curv_batch:need], y_tr[curv_batch:need]

        torch.manual_seed(seed)
        model = self._model()
        params = list(model.parameters())
        opt = make_optimizer(cand, params)
        ce = torch.nn.functional.cross_entropy
        cost = ProbeCost()
        gen = torch.Generator().manual_seed(seed)

        want = sorted(set(checkpoints))
        out = []
        for step in range(max(want) + 1):
            if step in want:
                out.append(
                    build_probe_context(
                        lambda: ce(model(x_c), y_c),
                        params,
                        [
                            (lambda j=j: ce(
                                model(x_m[j * micro_batch : (j + 1) * micro_batch]),
                                y_m[j * micro_batch : (j + 1) * micro_batch],
                            ))
                            for j in range(micro)
                        ],
                        depths=depths,
                        curv_batch=curv_batch,
                        micro_batch=micro_batch,
                        train_batch=self.batch,
                        step=step,
                        label=self.name,
                    )
                )
            idx = torch.randint(0, self.train_size, (self.batch,), generator=gen)
            xb, yb = x_tr[idx], y_tr[idx]
            run_step(cand, opt, lambda: ce(model(xb), yb), cost, self.batch)
        return out


# ------------------------------------------------------------------- the pool


#: Six points, half-decade spacing. The design document's 16-trial random search
#: is the intended budget; a grid is used here because the harness is being
#: demonstrated on two knobs, not because a grid is better.
DEFAULT_LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def sgd_candidate(lr: float = 3e-3, lr_grid=DEFAULT_LR_GRID) -> Candidate:
    return Candidate(
        name="sgd",
        preconditioner=IdentityPreconditioner(),
        build=lambda p, lr_: PreconditionedSGD(p, lr_),
        lr=lr,
        is_baseline=True,
        lr_grid=tuple(lr_grid),
        notes="P = I. The reference every discovery claim is made against.",
    )


def fixed_random_candidate(dim: int, seed: int = 7, lr: float = 3e-3,
                           log_spread: float = 0.5, lr_grid=DEFAULT_LR_GRID) -> Candidate:
    p = fixed_random_preconditioner(dim, seed=seed, log_spread=log_spread)
    return Candidate(
        name=f"fixed-random(seed={seed},s={log_spread:g})",
        preconditioner=p,
        build=lambda params, lr_: PreconditionedSGD(params, lr_, precond=p),
        lr=lr,
        lr_grid=tuple(lr_grid),
        notes=(
            "Data-independent diagonal. The negative control: it has the shape "
            "of a preconditioner and none of the information."
        ),
    )


def dsn_candidate(lr: float = 1e-2, delta: float = 1e-3, k_max: int = 8,
                  m_recycle: int = 4, tau: float = 0.1,
                  lr_grid=DEFAULT_LR_GRID) -> Candidate:
    from dsn import DSN
    from dsn.subspace import KrylovBuilder

    return Candidate(
        name=f"dsn(delta={delta:g},k_max={k_max})",
        # Screened at the rank it actually runs at, not at full rank. The
        # full-rank form is an upper bound on this one, and screening the bound
        # while training the restriction is how a harness ends up certifying a
        # different optimizer from the one it benchmarks.
        preconditioner=KrylovRestrictedPreconditioner(delta, k_max),
        build=lambda p, lr_: DSN(
            p, lr=lr_, weight_decay=0.0,
            builder=KrylovBuilder(k_max=k_max, m_recycle=m_recycle, tau=tau, damping=delta),
        ),
        lr=lr,
        lr_grid=tuple(lr_grid),
        needs_closure=True,
        notes=(
            "Shipped DSN, unmodified. Screened as P_k = W (|W'HW| + delta)^-1 W' "
            "+ (I - WW'), the rank-k_max operator it actually applies."
        ),
    )


def default_pool(
    dim: int, include_dsn: bool = True, n_controls: int = 1
) -> list[Candidate]:
    """SGD baseline, ``n_controls`` random controls, and DSN.

    No new optimizer is invented.  Extra controls are the same negative control
    at different seeds and spreads: they cost nothing to define, they are known
    to carry no curvature information, and they are what makes the multiplicity
    accounting in stage 2 non-trivial.  A pool of two arms has one comparison and
    nothing interesting to say about family-wise error; a pool of six has five,
    and the naive-versus-controlled gap becomes visible.
    """
    pool = [sgd_candidate()]
    # Spread is the control's only knob and it doubles as a difficulty dial:
    # a wide random diagonal multiplies the problem's condition number, so the
    # later controls are genuinely worse arms rather than merely different ones.
    spreads = (0.5, 0.25, 1.0, 2.0, 3.0)
    for i in range(n_controls):
        pool.append(
            fixed_random_candidate(dim, seed=7 + i, log_spread=spreads[i % len(spreads)])
        )
    if include_dsn:
        pool.append(dsn_candidate())
    return pool
