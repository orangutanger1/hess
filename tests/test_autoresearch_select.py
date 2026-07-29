"""Stage-2 selection: budget arithmetic, anytime validity, and error calibration.

The load-bearing tests here are the calibration ones.  A selection procedure
that reports a 5% family-wise error and delivers 40% is worse than no procedure
at all, because the number is what the decision rests on -- so the error rate is
measured by simulation against known-null arms rather than asserted from the
derivation.
"""

import math
import random

import pytest

from autoresearch.select import (
    AlphaLedger,
    SuccessiveElimination,
    TrialResult,
    anytime_p_value,
    benjamini_hochberg,
    benjamini_yekutieli,
    confidence_radius,
    sequential_halving,
)


# ------------------------------------------------------------- alpha ledger


def test_alpha_budget_never_exceeds_alpha_however_large_the_pool_grows():
    """The point of a convergent series: adding candidates cannot overspend."""
    led = AlphaLedger(alpha=0.05)
    for i in range(5000):
        led.entry_index(f"cand{i}")
        assert led.spent <= 0.05 + 1e-12
    assert led.spent == pytest.approx(0.05, rel=0.02)
    assert led.remaining >= 0


def test_alpha_allocation_is_fixed_at_entry_and_never_reissued():
    led = AlphaLedger(alpha=0.05)
    a = led.alpha_for("x")
    led.entry_index("y")
    led.entry_index("z")
    assert led.alpha_for("x") == a


def test_geometric_schedule_overtakes_zeta2_only_after_the_crossover():
    """``2^-j`` decays faster than ``1/(zeta(2) j^2)`` -- but not immediately.

    The ``1/zeta(2) = 0.608`` factor means the quadratic schedule is the
    *stricter* one for the first few entrants and the geometric one takes over
    around the sixth.  Worth pinning: "geometric is stricter" is the natural
    assumption and it is wrong for exactly the pool sizes a small search has.
    """
    z, g = AlphaLedger(0.05, "zeta2"), AlphaLedger(0.05, "geometric")
    names = [f"c{i}" for i in range(10)]
    for n in names:
        z.entry_index(n)
        g.entry_index(n)
    assert g.alpha_for("c3") > z.alpha_for("c3")   # 4th entrant: zeta2 stricter
    assert g.alpha_for("c9") < z.alpha_for("c9")   # 10th entrant: geometric stricter


def test_a_later_entrant_is_held_to_a_stricter_bar():
    """Stated as a limitation in the module docstring; asserted here."""
    led = AlphaLedger(alpha=0.05)
    for i in range(40):
        led.entry_index(f"c{i}")
    assert led.alpha_for("c39") < led.alpha_for("c3") / 50


# ------------------------------------------------------------ anytime bounds


def test_confidence_radius_shrinks_with_seeds_and_widens_with_confidence():
    assert confidence_radius(20, 1.0, 0.05) < confidence_radius(5, 1.0, 0.05)
    assert confidence_radius(5, 1.0, 0.001) > confidence_radius(5, 1.0, 0.05)
    assert confidence_radius(5, 0.0, 0.05) == 0.0


def test_anytime_p_value_is_one_when_the_candidate_is_not_better():
    assert anytime_p_value(10, 0.0, 1.0) == 1.0
    assert anytime_p_value(10, +0.5, 1.0) == 1.0
    # A half-sigma edge at n=10 is not evidence once the union over n is paid
    # for; 1.5 sigma is.
    assert anytime_p_value(10, -0.5, 1.0) == 1.0
    assert anytime_p_value(10, -1.5, 1.0) < 1.0
    assert anytime_p_value(40, -1.5, 1.0) < anytime_p_value(10, -1.5, 1.0)


def test_anytime_p_value_is_conservative_relative_to_a_fixed_sample_z_test():
    """The union-over-n price must be paid, not quietly skipped.

    A fixed-sample one-sided z-test at n=10 with a 1-sigma effect gives
    p ~ 8e-4.  The anytime version must be strictly larger; if it is not, the
    peeking correction is missing and every discovery claim is optimistic.
    """
    n, gap, sigma = 10, -1.0, 1.0
    fixed = 0.5 * math.erfc(abs(gap) * math.sqrt(n) / (sigma * math.sqrt(2)))
    assert anytime_p_value(n, gap, sigma) > fixed


def test_anytime_confidence_sequence_covers_under_continuous_peeking():
    """Simulated coverage: peeking every round must not break the guarantee."""
    rng = random.Random(0)
    delta, sigma, reps, max_n = 0.1, 1.0, 400, 30
    misses = 0
    for _ in range(reps):
        total = 0.0
        for n in range(1, max_n + 1):
            total += rng.gauss(0.0, sigma)
            if abs(total / n) > confidence_radius(n, sigma, delta):
                misses += 1
                break
    assert misses / reps <= delta, misses / reps


# ---------------------------------------------------------------- multiplicity


def test_benjamini_yekutieli_is_never_more_permissive_than_hochberg():
    p = [0.001, 0.01, 0.02, 0.04, 0.2, 0.9]
    by, _ = benjamini_yekutieli(p, 0.1)
    bh, _ = benjamini_hochberg(p, 0.1)
    assert set(by) <= set(bh)


def test_no_discoveries_when_nothing_is_significant():
    p = [0.4, 0.6, 0.8, 0.95]
    assert benjamini_yekutieli(p, 0.1)[0] == []
    assert benjamini_hochberg(p, 0.1)[0] == []


def test_benjamini_yekutieli_controls_fdr_under_the_global_null():
    """All nulls true: the expected false-discovery proportion must be <= q."""
    rng = random.Random(1)
    q, m, reps = 0.1, 12, 600
    any_rejection = 0
    for _ in range(reps):
        p = [rng.random() for _ in range(m)]
        if benjamini_yekutieli(p, q)[0]:
            any_rejection += 1
    # Under the global null FDP is 1 whenever anything is rejected, so
    # E[FDP] = P(any rejection) and it must sit under q.
    assert any_rejection / reps <= q


# ----------------------------------------------------------------- selection


def _bandit(means, sigma=1.0, seed=0):
    rng = random.Random(seed)

    def pull(arm, s):
        return TrialResult(score=rng.gauss(means[arm], sigma), flops=1.0)

    return pull


def test_successive_elimination_finds_a_clearly_better_arm():
    means = {"sgd": 1.0, "good": 0.0, "bad": 2.0, "worse": 3.0}
    sel = SuccessiveElimination("sgd", alpha=0.05, n_pilot=3, max_seeds=60)
    for a in ("good", "bad", "worse"):
        sel.add(a)
    rep = sel.run(_bandit(means, sigma=0.5))
    assert rep.winner == "good", rep.survivors
    assert "worse" not in rep.survivors
    assert all(e["arm"] != "good" for e in rep.eliminations)


def test_eliminations_carry_the_evidence_that_produced_them():
    means = {"sgd": 1.0, "good": 0.0, "bad": 4.0}
    sel = SuccessiveElimination("sgd", n_pilot=3, max_seeds=40)
    sel.add("good")
    sel.add("bad")
    rep = sel.run(_bandit(means, sigma=0.3))
    killed = [e for e in rep.eliminations if e["arm"] == "bad"]
    assert killed, rep.eliminations
    e = killed[0]
    assert e["lcb"] > e["rival_ucb"]
    assert 0 < e["alpha_allocated"] <= 0.05
    assert "exceeds UCB" in e["evidence"]


def test_the_adaptive_budget_spends_fewer_trials_than_a_fixed_design():
    """The reason for an adaptive allocation, measured rather than assumed."""
    means = {"sgd": 1.0, "good": 0.0, "bad": 5.0, "worse": 6.0, "awful": 7.0}
    sel = SuccessiveElimination("sgd", n_pilot=3, max_seeds=40)
    for a in ("good", "bad", "worse", "awful"):
        sel.add(a)
    rep = sel.run(_bandit(means, sigma=0.3))
    fixed_design = len(means) * 40
    assert rep.total_pulls < fixed_design


def test_the_pool_may_grow_mid_run_without_breaking_the_budget():
    means = {"sgd": 1.0, "a": 0.9, "b": 0.5, "late": 0.0}
    sel = SuccessiveElimination("sgd", n_pilot=3, max_seeds=30)
    sel.add("a")
    sel.add("b")
    rep = sel.run(
        _bandit(means, sigma=0.3),
        new_candidates=lambda rnd: ["late"] if rnd == 4 else [],
    )
    assert "late" in sel.arms
    assert rep.alpha_ledger["spent"] <= rep.alpha_ledger["alpha_total"] + 1e-12
    assert rep.alpha_ledger["n_candidates"] == 4


def test_paired_comparison_uses_only_seeds_both_arms_actually_ran():
    """An arm killed in round 2 must not be scored against 30 baseline seeds.

    Otherwise the comparison silently reflects the allocation rule rather than
    the arms, and eliminated candidates acquire evidence they never earned.
    """
    sel = SuccessiveElimination("base", n_pilot=2, max_seeds=10)
    sel.add("short")
    base, short = sel.arms["base"], sel.arms["short"]
    base.scores = [1.0, 1.0, 1.0, 5.0, 5.0]
    base.seeds = [0, 1, 2, 3, 4]
    short.scores = [0.5, 0.5]
    short.seeds = [0, 1]
    mean, sigma, n = sel._paired_sigma(short)
    assert n == 2
    assert mean == pytest.approx(-0.5)


def test_family_wise_error_is_controlled_under_the_global_null():
    """Every arm identical: wrongly eliminating any of them is the error.

    This is the number the whole module exists to deliver, so it is measured.
    """
    reps, alpha = 300, 0.05
    bad = 0
    for r in range(reps):
        means = {n: 0.0 for n in ("base", "a", "b", "c", "d")}
        sel = SuccessiveElimination("base", alpha=alpha, n_pilot=3, max_seeds=25)
        for n in ("a", "b", "c", "d"):
            sel.add(n)
        rep = sel.run(_bandit(means, sigma=1.0, seed=r))
        if rep.eliminations:
            bad += 1
    assert bad / reps <= alpha, f"FWER {bad / reps:.3f} exceeds {alpha}"


def test_the_random_control_is_not_discovered_when_it_is_no_better():
    """The negative control must not produce a discovery claim."""
    means = {"sgd": 0.0, "fixed-random": 0.0}
    sel = SuccessiveElimination("sgd", fdr_q=0.1, n_pilot=3, max_seeds=25)
    sel.add("fixed-random")
    rep = sel.run(_bandit(means, sigma=1.0, seed=3))
    assert rep.final_risk.by_discoveries == []


def test_risk_snapshot_reports_the_naive_rate_alongside_the_controlled_one():
    means = {"sgd": 1.0, "a": 1.0, "b": 1.0, "c": 1.0}
    sel = SuccessiveElimination("sgd", fdr_q=0.05, n_pilot=3, max_seeds=8)
    for n in ("a", "b", "c"):
        sel.add(n)
    rep = sel.run(_bandit(means, sigma=1.0, seed=5))
    risk = rep.final_risk
    assert risk.n_comparisons == 3
    # 1 - 0.95^3
    assert risk.naive_fwer == pytest.approx(1 - 0.95**3)
    assert risk.naive_fwer > risk.fwer_alpha_total
    assert set(risk.by_discoveries) <= set(risk.bh_discoveries) <= set(risk.naive_discoveries)


def test_baseline_is_protected_from_elimination():
    """It is the reference for every discovery claim; killing it ends the run."""
    means = {"sgd": 5.0, "good": 0.0}
    sel = SuccessiveElimination("sgd", n_pilot=3, max_seeds=30)
    sel.add("good")
    rep = sel.run(_bandit(means, sigma=0.2))
    assert sel.arms["sgd"].active
    assert "sgd" in rep.survivors


# ---------------------------------------------------------- sequential halving


def test_sequential_halving_spends_its_budget_and_returns_a_winner():
    means = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}
    winner, log, state = sequential_halving(
        list(means), _bandit(means, sigma=0.2), budget_pulls=64
    )
    assert winner == "d"
    assert len(log) >= 1
    assert sum(s.n for s in state.values()) > 0
    assert all(s.eliminated_round for s in state.values() if s.name != winner)
