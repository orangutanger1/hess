"""Stage 2 -- sequential selection under a family-wise error budget.

Given a pool of gate-passing candidates, allocate training seeds adaptively and
say, at every point, what the evidence actually supports.  Three things are
deliberately *not* done, because each is a standard way to bank noise as
progress:

**No fixed N-seeds-per-candidate design.**  Seeds go to arms that are still
plausibly best.  A candidate that is two standard errors behind after three
seeds does not get seventeen more.

**No per-candidate alpha.**  A pool that grows one candidate at a time, each
tested at 0.05, has a family-wise error that grows with it: after 20 candidates
the chance of at least one spurious winner is 64%, not 5%.  The budget here is
*family-wise over the whole pool as it grows*, spent through a convergent
series, so the guarantee survives an unbounded number of future candidates.

**No unadjusted discovery claim.**  Every round reports the false-discovery risk
given how many comparisons have been made *so far*, next to the naive number
that ignores multiplicity.  The gap between them is the point.

Statistical contract, stated so it can be checked
-------------------------------------------------

*Anytime-valid confidence sequences.*  Elimination decisions are made after
every round, so a fixed-sample interval would be invalid -- peeking inflates
error exactly as much as you peek.  Each arm carries

    beta(n, delta) = sigma * sqrt( (2/n) * ln( 2 n (n+1) / delta ) )

which is the sub-Gaussian bound with the union over ``n`` taken through
``delta_n = delta / (n(n+1))``, ``sum_n delta_n = delta``.  It holds
*simultaneously at every n*, so stopping when it looks good is allowed.

*Family-wise budget over a growing pool.*  Candidate ``j``, in order of entry,
is allocated ``alpha_j = alpha * 6 / (pi^2 j^2)``.  That series converges to
``alpha`` over an infinite pool, so no matter how many candidates are added
later, ``P(any arm is wrongly eliminated, ever) <= alpha``.  The cost is
explicit and is reported: later entrants get a smaller budget and therefore need
more seeds to be separated.  ``geometric`` (``alpha * 2^-j``) is stricter still.

This is an *ordering-dependent* allocation, which is a real limitation and not a
subtlety worth burying: a candidate proposed 40th is held to a higher bar than
the same candidate proposed 4th.  Online-FDR procedures (LORD++ and relatives)
exist precisely to re-invest budget as discoveries accrue, and are not
implemented here.

*Discovery claims.*  Beating the incumbent is a separate question from being the
best arm, so it gets a separate, one-sided, anytime-valid p-value on *paired*
differences (all arms share the seed set, so pairing is legitimate and cuts the
variance substantially).  Those p-values then go through Benjamini-Yekutieli,
which is valid under arbitrary dependence -- and the dependence here is
arbitrary: shared seeds, shared data order, adaptive stopping.  Benjamini-
Hochberg would be sharper and is reported alongside, but it is not the number to
quote.

*Where the guarantee is approximate.*  ``sigma`` is estimated, not known.  The
bounds above are exact for a known sub-Gaussian parameter and approximate with a
plug-in estimate; ``sigma_inflate`` exists to buy back some of that, and the
report states the assumption rather than absorbing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

ZETA2 = math.pi**2 / 6.0


# ------------------------------------------------------------------ alpha budget


class AlphaLedger:
    """Family-wise error budget over a pool that may keep growing.

    ``spend`` is idempotent per candidate: asking twice returns the same
    allocation.  Nothing is ever re-allocated, which is what makes the running
    total a bound rather than a wish.
    """

    def __init__(self, alpha: float = 0.05, schedule: str = "zeta2"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if schedule not in ("zeta2", "geometric"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.alpha = alpha
        self.schedule = schedule
        self._order: dict[str, int] = {}

    def entry_index(self, name: str) -> int:
        if name not in self._order:
            self._order[name] = len(self._order) + 1
        return self._order[name]

    def alpha_for(self, name: str) -> float:
        j = self.entry_index(name)
        if self.schedule == "zeta2":
            return self.alpha / (ZETA2 * j * j)
        return self.alpha * 2.0**-j

    @property
    def spent(self) -> float:
        """Total budget allocated so far -- always ``<= alpha``, by construction."""
        return sum(self.alpha_for(n) for n in self._order)

    @property
    def remaining(self) -> float:
        return self.alpha - self.spent

    def as_dict(self) -> dict:
        return {
            "alpha_total": self.alpha,
            "schedule": self.schedule,
            "n_candidates": len(self._order),
            "spent": self.spent,
            "remaining": self.remaining,
            "per_candidate": {n: self.alpha_for(n) for n in self._order},
        }


# ------------------------------------------------------- anytime-valid bounds


def confidence_radius(n: int, sigma: float, delta: float) -> float:
    """Anytime-valid sub-Gaussian radius, simultaneously valid at every ``n``."""
    if n <= 0:
        return float("inf")
    if sigma <= 0:
        return 0.0
    return sigma * math.sqrt((2.0 / n) * math.log(2.0 * n * (n + 1) / delta))


def anytime_p_value(n: int, mean_diff: float, sigma: float) -> float:
    """One-sided anytime-valid p-value for ``H0: E[d] >= 0`` against ``E[d] < 0``.

    ``d`` is the paired difference ``candidate - baseline`` with lower being
    better, so evidence *for* the candidate is ``mean_diff < 0``.  Inverting the
    same union bound the radius uses gives

        p_n = n (n + 1) exp( - n * mean_diff^2 / (2 sigma^2) ),   mean_diff < 0

    clipped to 1.  Because the bound is anytime-valid, the *running minimum* of
    ``p_n`` over rounds is itself a valid p-value -- which is what makes it
    legitimate to look every round.
    """
    if n <= 0 or sigma <= 0 or mean_diff >= 0:
        return 1.0
    exponent = -n * mean_diff * mean_diff / (2.0 * sigma * sigma)
    return min(1.0, n * (n + 1) * math.exp(exponent))


def benjamini_yekutieli(pvalues: Sequence[float], q: float) -> tuple[list[int], float]:
    """BY step-up: indices rejected, and the critical value used.

    Valid under *arbitrary* dependence, at the cost of the ``c(m) = sum 1/k``
    factor.  That is the right procedure here: arms share seeds, share the data
    order, and are stopped adaptively, so no positive-dependence assumption is
    defensible.
    """
    m = len(pvalues)
    if m == 0:
        return [], 0.0
    cm = sum(1.0 / k for k in range(1, m + 1))
    order = sorted(range(m), key=lambda i: pvalues[i])
    crit = 0.0
    k_max = 0
    for rank, i in enumerate(order, start=1):
        c = rank * q / (m * cm)
        if pvalues[i] <= c:
            k_max, crit = rank, c
    return sorted(order[:k_max]), crit


def benjamini_hochberg(pvalues: Sequence[float], q: float) -> tuple[list[int], float]:
    """BH step-up -- sharper than BY, valid only under independence or PRDS.

    Reported for contrast, never as the headline: the dependence structure here
    does not support it.
    """
    m = len(pvalues)
    if m == 0:
        return [], 0.0
    order = sorted(range(m), key=lambda i: pvalues[i])
    crit = 0.0
    k_max = 0
    for rank, i in enumerate(order, start=1):
        c = rank * q / m
        if pvalues[i] <= c:
            k_max, crit = rank, c
    return sorted(order[:k_max]), crit


# ------------------------------------------------------------------------ arms


@dataclass
class TrialResult:
    """One training trial.  ``score`` is lower-is-better.

    ``diverged`` is separate from ``score`` on purpose.  A run that blew up and
    was capped is not "a bad score"; it is evidence that the configuration is
    not viable, and the two must not be pooled into one mean.  Measured on the
    random control: 11 of 12 seeds converged and one hit the cap, which moved
    the mean by four orders of magnitude and inflated ``sigma_hat`` enough that
    the arm could never be eliminated on a mean comparison -- it survived the
    whole budget on the strength of its own instability.
    """

    score: float
    flops: float
    steps: int = 0
    diverged: bool = False
    detail: dict = field(default_factory=dict)


@dataclass
class Arm:
    name: str
    is_baseline: bool = False
    scores: list[float] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    flops: float = 0.0
    n_diverged: int = 0
    eliminated_round: int | None = None
    elimination_kind: str = ""
    elimination_evidence: str = ""

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return sum(self.scores) / self.n if self.n else float("nan")

    def sigma_hat(self, floor: float) -> float:
        if self.n < 2:
            return float("nan")
        m = self.mean
        var = sum((s - m) ** 2 for s in self.scores) / (self.n - 1)
        return max(math.sqrt(var), floor)

    @property
    def active(self) -> bool:
        return self.eliminated_round is None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "is_baseline": self.is_baseline,
            "n_seeds": self.n,
            "mean": self.mean,
            "scores": list(self.scores),
            "flops": self.flops,
            "n_diverged": self.n_diverged,
            "eliminated_round": self.eliminated_round,
            "elimination_kind": self.elimination_kind,
            "elimination_evidence": self.elimination_evidence,
        }


# ------------------------------------------------------------------- reporting


@dataclass
class RiskSnapshot:
    """False-discovery accounting *as of* a given round.

    ``naive_fwer`` is the number this whole module exists to keep off the
    report: the probability of at least one false positive if each of the ``m``
    comparisons made so far had been tested at an unadjusted ``q``.  It is
    printed next to the controlled numbers because "we ran 20 comparisons at
    0.05" and "our error rate is 5%" are not the same sentence.
    """

    round: int
    n_comparisons: int
    fwer_alpha_spent: float
    fwer_alpha_total: float
    naive_fwer: float
    fdr_q: float
    by_critical: float
    bh_critical: float
    by_discoveries: list[str]
    bh_discoveries: list[str]
    naive_discoveries: list[str]
    expected_false_discoveries: float
    pvalues: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "round": self.round,
            "n_comparisons": self.n_comparisons,
            "fwer_alpha_spent": self.fwer_alpha_spent,
            "fwer_alpha_total": self.fwer_alpha_total,
            "naive_fwer_if_unadjusted": self.naive_fwer,
            "fdr_q": self.fdr_q,
            "by_critical": self.by_critical,
            "bh_critical": self.bh_critical,
            "by_discoveries": self.by_discoveries,
            "bh_discoveries": self.bh_discoveries,
            "naive_discoveries": self.naive_discoveries,
            "expected_false_discoveries_bound": self.expected_false_discoveries,
            "pvalues": self.pvalues,
        }


@dataclass
class SelectionReport:
    arms: dict[str, Arm]
    baseline: str
    rounds: int
    winner: str | None
    survivors: list[str]
    eliminations: list[dict]
    risk: list[RiskSnapshot]
    alpha_ledger: dict
    assumptions: list[str]
    total_pulls: int
    stopped_because: str

    @property
    def final_risk(self) -> RiskSnapshot | None:
        return self.risk[-1] if self.risk else None

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "rounds": self.rounds,
            "winner": self.winner,
            "survivors": self.survivors,
            "stopped_because": self.stopped_because,
            "total_pulls": self.total_pulls,
            "arms": {k: v.as_dict() for k, v in self.arms.items()},
            "eliminations": self.eliminations,
            "risk": [r.as_dict() for r in self.risk],
            "alpha_ledger": self.alpha_ledger,
            "assumptions": self.assumptions,
        }


# ------------------------------------------------------------------- selector


class SuccessiveElimination:
    """Fixed-confidence best-arm identification with an adaptive seed budget.

    The pool may grow between rounds (``add``), which is the case this is built
    for: a screening loop proposes candidates continuously and the error budget
    has to hold across all of them, not per batch.
    """

    def __init__(
        self,
        baseline: str,
        alpha: float = 0.05,
        fdr_q: float = 0.10,
        n_pilot: int = 3,
        max_seeds: int = 20,
        max_pulls: int | None = None,
        sigma_inflate: float = 1.25,
        sigma_floor: float = 1e-9,
        schedule: str = "zeta2",
        protect_baseline: bool = True,
        max_divergent_trials: int | None = 0,
    ):
        if n_pilot < 2:
            raise ValueError("need at least 2 pilot seeds to estimate sigma at all")
        self.baseline = baseline
        self.ledger = AlphaLedger(alpha, schedule)
        self.fdr_q = fdr_q
        self.n_pilot = n_pilot
        self.max_seeds = max_seeds
        self.max_pulls = max_pulls
        self.sigma_inflate = sigma_inflate
        self.sigma_floor = sigma_floor
        self.protect_baseline = protect_baseline
        # Stability disqualification is *not* a statistical test and spends no
        # alpha: "this configuration blew up" is a fact about the run, not an
        # inference about a mean. Conflating the two would let a viability
        # judgement consume the error budget that the mean comparisons need.
        self.max_divergent_trials = max_divergent_trials

        self.arms: dict[str, Arm] = {}
        self.add(baseline, is_baseline=True)

    # ------------------------------------------------------------------ pool

    def add(self, name: str, is_baseline: bool = False) -> None:
        """Enter a candidate into the pool and allocate its share of the budget."""
        if name in self.arms:
            return
        self.arms[name] = Arm(name=name, is_baseline=is_baseline)
        self.ledger.entry_index(name)  # fixes its place in the spending series

    # ------------------------------------------------------------- estimates

    def _sigma(self, arm: Arm) -> float:
        s = arm.sigma_hat(self.sigma_floor)
        return float("nan") if math.isnan(s) else s * self.sigma_inflate

    def _paired_sigma(self, arm: Arm) -> tuple[float, float, int]:
        """``(mean, sigma, n)`` of the paired difference ``arm - baseline``.

        Pairing is only valid on seeds both arms actually ran, so the shared
        prefix is used rather than the two means.  An arm eliminated in round 2
        must not be compared against a baseline mean that absorbed 18 more
        seeds; that comparison would look like evidence and would be an artifact
        of the allocation rule.
        """
        base = self.arms[self.baseline]
        shared = [s for s in arm.seeds if s in set(base.seeds)]
        if len(shared) < 2:
            return float("nan"), float("nan"), len(shared)
        ai = {s: v for s, v in zip(arm.seeds, arm.scores)}
        bi = {s: v for s, v in zip(base.seeds, base.scores)}
        d = [ai[s] - bi[s] for s in shared]
        m = sum(d) / len(d)
        var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
        return m, max(math.sqrt(var), self.sigma_floor) * self.sigma_inflate, len(d)

    def pvalues(self) -> dict[str, float]:
        """Anytime-valid one-sided p-values against the baseline, per candidate."""
        out = {}
        for name, arm in self.arms.items():
            if arm.is_baseline:
                continue
            m, s, n = self._paired_sigma(arm)
            out[name] = 1.0 if math.isnan(m) else anytime_p_value(n, m, s)
        return out

    def risk_snapshot(self, rnd: int) -> RiskSnapshot:
        p = self.pvalues()
        names = sorted(p)
        vals = [p[n] for n in names]
        m = len(vals)
        by_idx, by_crit = benjamini_yekutieli(vals, self.fdr_q)
        bh_idx, bh_crit = benjamini_hochberg(vals, self.fdr_q)
        by = [names[i] for i in by_idx]
        return RiskSnapshot(
            round=rnd,
            n_comparisons=m,
            fwer_alpha_spent=self.ledger.spent,
            fwer_alpha_total=self.ledger.alpha,
            naive_fwer=1.0 - (1.0 - self.fdr_q) ** m,
            fdr_q=self.fdr_q,
            by_critical=by_crit,
            bh_critical=bh_crit,
            by_discoveries=by,
            bh_discoveries=[names[i] for i in bh_idx],
            naive_discoveries=[n for n in names if p[n] <= self.fdr_q],
            expected_false_discoveries=self.fdr_q * len(by),
            pvalues=p,
        )

    # ----------------------------------------------------------------- driver

    def run(
        self,
        pull: Callable[[str, int], TrialResult],
        new_candidates: Callable[[int], Iterable[str]] | None = None,
        verbose: bool = False,
    ) -> SelectionReport:
        """Run to a decision.

        ``pull(arm_name, seed) -> TrialResult`` runs one training trial.
        ``new_candidates(round) -> names`` may add candidates as rounds proceed,
        which is what "as the pool grows" means operationally.
        """
        eliminations: list[dict] = []
        risk: list[RiskSnapshot] = []
        total_pulls = 0
        rnd = 0
        stopped = "max_seeds reached"

        while True:
            rnd += 1
            if new_candidates is not None:
                for name in new_candidates(rnd):
                    self.add(name)

            active = [a for a in self.arms.values() if a.active]
            if not active:
                stopped = "every arm eliminated"
                break

            seed = rnd - 1
            for arm in active:
                if arm.n >= self.max_seeds:
                    continue
                if self.max_pulls is not None and total_pulls >= self.max_pulls:
                    break
                res = pull(arm.name, seed)
                arm.scores.append(res.score)
                arm.seeds.append(seed)
                arm.flops += res.flops
                arm.n_diverged += bool(res.diverged)
                total_pulls += 1

            if self.max_divergent_trials is not None:
                for arm in active:
                    if arm.is_baseline and self.protect_baseline:
                        continue
                    if arm.active and arm.n_diverged > self.max_divergent_trials:
                        arm.eliminated_round = rnd
                        arm.elimination_kind = "stability"
                        arm.elimination_evidence = (
                            f"diverged on {arm.n_diverged} of {arm.n} trials "
                            f"(cap {self.max_divergent_trials}); disqualified on "
                            "viability, not on a mean comparison, and no alpha spent"
                        )
                        eliminations.append({
                            "round": rnd, "arm": arm.name, "against": "(stability)",
                            "kind": "stability", "n_seeds": arm.n, "mean": arm.mean,
                            "lcb": float("nan"), "rival_ucb": float("nan"),
                            "alpha_allocated": 0.0,
                            "evidence": arm.elimination_evidence,
                        })
                        if verbose:
                            print(f"  round {rnd}: disqualified {arm.name} -- "
                                  f"{arm.elimination_evidence}")
                active = [a for a in active if a.active]

            # Elimination needs a variance estimate, so nothing is decided
            # before the pilot completes.  This is the design's own noise gate:
            # a single timeboxed run kept because the metric improved banks
            # noise as progress across hundreds of experiments.
            contenders = [a for a in active if a.n >= self.n_pilot]
            if len(contenders) >= 2:
                stats = {}
                for a in contenders:
                    s = self._sigma(a)
                    d = self.ledger.alpha_for(a.name)
                    stats[a.name] = (a.mean, confidence_radius(a.n, s, d), s, d)
                best = min(contenders, key=lambda a: stats[a.name][0])
                bmean, bradius, _, _ = stats[best.name]
                ucb_best = bmean + bradius
                for a in contenders:
                    if a is best or (a.is_baseline and self.protect_baseline):
                        continue
                    mean, radius, sig, dlt = stats[a.name]
                    if mean - radius > ucb_best:
                        a.eliminated_round = rnd
                        a.elimination_kind = "statistical"
                        a.elimination_evidence = (
                            f"LCB {mean - radius:.6g} (mean {mean:.6g}, n={a.n}, "
                            f"sigma_hat {sig:.4g}, delta {dlt:.4g}) exceeds UCB "
                            f"{ucb_best:.6g} of '{best.name}' "
                            f"(mean {bmean:.6g}, n={best.n})"
                        )
                        eliminations.append(
                            {
                                "round": rnd,
                                "arm": a.name,
                                "against": best.name,
                                "kind": "statistical",
                                "n_seeds": a.n,
                                "mean": mean,
                                "lcb": mean - radius,
                                "rival_ucb": ucb_best,
                                "alpha_allocated": dlt,
                                "evidence": a.elimination_evidence,
                            }
                        )
                        if verbose:
                            print(f"  round {rnd}: eliminated {a.name} -- {a.elimination_evidence}")

            risk.append(self.risk_snapshot(rnd))

            remaining = [a for a in self.arms.values() if a.active]
            non_baseline = [a for a in remaining if not a.is_baseline]
            if len(remaining) <= 1 or (self.protect_baseline and not non_baseline):
                stopped = "one arm remains"
                break
            if all(a.n >= self.max_seeds for a in remaining):
                stopped = "seed budget exhausted with several arms alive"
                break
            if self.max_pulls is not None and total_pulls >= self.max_pulls:
                stopped = "trial budget exhausted"
                break

        survivors = [a.name for a in self.arms.values() if a.active]
        candidates = [n for n in survivors if n != self.baseline]
        winner = None
        if candidates:
            winner = min(candidates, key=lambda n: self.arms[n].mean)

        return SelectionReport(
            arms=self.arms,
            baseline=self.baseline,
            rounds=rnd,
            winner=winner,
            survivors=survivors,
            eliminations=eliminations,
            risk=risk,
            alpha_ledger=self.ledger.as_dict(),
            assumptions=[
                "Scores are sub-Gaussian across seeds with the plug-in sigma_hat "
                f"inflated by {self.sigma_inflate}x; the anytime bounds are exact "
                "for a known sigma and approximate for an estimated one.",
                "All arms share the seed set, so candidate-vs-baseline differences "
                "are paired over the seeds both arms actually ran.",
                "Family-wise error is budgeted over the pool in order of entry "
                f"({self.ledger.schedule}); a candidate proposed later is held to a "
                "stricter bar than the same candidate proposed earlier.",
                "Benjamini-Yekutieli is used for discoveries because arms share "
                "seeds, share data order and are stopped adaptively, so no "
                "positive-dependence assumption is available.",
                "Divergent trials are score-capped, which makes an unstable arm's "
                "score distribution bimodal and violates the sub-Gaussian "
                "assumption. Such arms are disqualified on stability instead, "
                "spending no alpha; the mean comparisons are only applied to arms "
                "that stayed in their stable regime.",
            ],
            total_pulls=total_pulls,
            stopped_because=stopped,
        )


# ------------------------------------------------------------ sequential halving


def sequential_halving(
    arms: Sequence[str],
    pull: Callable[[str, int], TrialResult],
    budget_pulls: int,
    seed0: int = 0,
) -> tuple[str, list[dict], dict[str, Arm]]:
    """Fixed-*budget* best-arm identification: halve the pool each round.

    The fixed-confidence complement to ``SuccessiveElimination``.  Use it when
    the constraint is "we have this many trials" rather than "we need this
    confidence"; it spends the whole budget and returns a winner without a
    certificate, so it does **not** by itself support a discovery claim.
    """
    if not arms:
        raise ValueError("sequential halving needs at least one arm")
    state = {a: Arm(name=a) for a in arms}
    live = list(arms)
    n_rounds = max(1, math.ceil(math.log2(len(live)))) if len(live) > 1 else 1
    log: list[dict] = []
    seed = seed0
    for r in range(n_rounds):
        per_arm = max(1, budget_pulls // (len(live) * n_rounds))
        for a in live:
            for _ in range(per_arm):
                res = pull(a, seed)
                state[a].scores.append(res.score)
                state[a].seeds.append(seed)
                state[a].flops += res.flops
                seed += 1
        ranked = sorted(live, key=lambda a: state[a].mean)
        keep = ranked[: max(1, len(ranked) // 2)]
        for a in live:
            if a not in keep:
                state[a].eliminated_round = r + 1
                state[a].elimination_evidence = (
                    f"bottom half at round {r + 1}: mean {state[a].mean:.6g}, n={state[a].n}"
                )
        log.append(
            {
                "round": r + 1,
                "pulls_per_arm": per_arm,
                "kept": list(keep),
                "dropped": [a for a in live if a not in keep],
                "means": {a: state[a].mean for a in live},
            }
        )
        live = keep
        if len(live) == 1:
            break
    return live[0], log, state
