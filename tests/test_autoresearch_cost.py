"""Cost accounting and the final ledger.

The ledger's job is to make search cost impossible to omit, so the tests are
mostly about arithmetic that must not silently lose a term.
"""

import math

import pytest
import torch

from autoresearch import report as reportmod
from autoresearch.cost import CostModel, Ledger, analytic_cost_model, calibrate
from autoresearch.probes import ProbeCost
from autoresearch.select import Arm, RiskSnapshot, SelectionReport
from autoresearch.trials import (
    Candidate,
    PreconditionedSGD,
    QuadraticTrial,
    default_pool,
    run_step,
    tune_learning_rate,
)


# ----------------------------------------------------------------- cost model


def test_cost_model_prices_each_operation_by_its_own_multiplier():
    cm = CostModel(fwd_flops_per_sample=100.0, grad_multiplier=3.0, hvp_multiplier=6.0)
    c = ProbeCost()
    c.add_forward(10)
    c.add_grad(10)
    c.add_hvp(10, 2)
    assert cm.total(c) == pytest.approx(100 * 10 * (1 + 3 + 6 * 2))


def test_fractional_shared_costs_survive_into_the_total():
    """A shared cost split four ways must still sum back to the whole."""
    cm = analytic_cost_model(1000)
    c = ProbeCost()
    c.add_hvp(64, 7)
    quarters = sum(cm.total(c.scaled(0.25)) for _ in range(4))
    assert quarters == pytest.approx(cm.total(c))


def test_calibrate_measures_the_model_instead_of_assuming_multipliers():
    """The design's P3 bar is agreement with ``FlopCounterMode``; so measure it."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(784, 96), torch.nn.Tanh(),
        torch.nn.Linear(96, 10),
    )
    params = list(model.parameters())
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))

    def loss_fn():
        return torch.nn.functional.cross_entropy(model(x), y)

    cm = calibrate(model, loss_fn, params, batch=64)
    assert cm.fwd_flops_per_sample > 0
    assert "FlopCounterMode" in cm.provenance
    # A gradient is forward + backward and a backward is about twice a forward;
    # a double-backward curvature product is a few times that. Bounds, not
    # point values -- the assertion is that the measurement is in the physically
    # possible range, not that it equals the analytic guess.
    assert 2.0 <= cm.grad_multiplier <= 4.0, cm.provenance
    assert cm.hvp_multiplier > cm.grad_multiplier, cm.provenance
    assert cm.hvp_multiplier <= 12.0, cm.provenance

    analytic = analytic_cost_model(sum(p.numel() for p in params))
    ratio = cm.fwd_flops_per_sample / analytic.fwd_flops_per_sample
    assert 0.5 < ratio < 2.0, (cm.fwd_flops_per_sample, analytic.fwd_flops_per_sample)


# --------------------------------------------------------------------- ledger


def test_ledger_splits_stage1_stage2_and_never_drops_a_term():
    led = Ledger("x", stage1_exclusive=10, stage1_shared_share=5,
                 stage2_tuning=100, stage2_training=200)
    assert led.stage1 == 15
    assert led.stage2 == 300
    assert led.marginal_total == 315


def _fake_selection(means, survivors, baseline="sgd"):
    arms = {}
    for name, m in means.items():
        a = Arm(name=name, is_baseline=(name == baseline))
        a.scores = [m, m]
        a.seeds = [0, 1]
        if name not in survivors and name != baseline:
            a.eliminated_round = 2
            a.elimination_kind = "statistical"
        arms[name] = a
    risk = RiskSnapshot(
        round=2, n_comparisons=len(means) - 1, fwer_alpha_spent=0.04,
        fwer_alpha_total=0.05, naive_fwer=0.14, fdr_q=0.1, by_critical=0.02,
        bh_critical=0.05, by_discoveries=[], bh_discoveries=[],
        naive_discoveries=[], expected_false_discoveries=0.0, pvalues={},
    )
    return SelectionReport(
        arms=arms, baseline=baseline, rounds=2, winner=(survivors or [None])[0],
        survivors=[baseline] + list(survivors), eliminations=[], risk=[risk],
        alpha_ledger={"schedule": "zeta2", "n_candidates": len(means),
                      "spent": 0.04, "alpha_total": 0.05},
        assumptions=["test"], total_pulls=8, stopped_because="test",
    )


def test_fully_loaded_total_charges_the_survivor_for_the_whole_search():
    """The headline number: search cost cannot be moved off the books."""
    cm = analytic_cost_model(10)
    ledgers = {
        "sgd": Ledger("sgd", stage1_exclusive=1e6, stage2_training=4e6),
        "good": Ledger("good", stage1_exclusive=1e6, stage2_tuning=2e6, stage2_training=4e6),
        "dead": Ledger("dead", stage1_exclusive=1e6, stage2_training=1e6),
    }
    for l in ledgers.values():
        l.n_trials = 4
    sel = _fake_selection({"sgd": 1.0, "good": 0.5, "dead": 9.0}, survivors=["good"])
    rep = reportmod.build(cm, {}, sel, ledgers, max_seeds=4)

    line = next(l for l in rep.lines if l.name == "good")
    assert line.status == "survivor"
    # Everything except its own selection trials is search: 3e6 of stage 1,
    # the other arms' stage 2 (sgd 4e6 + dead 1e6 = 5e6), and its own 2e6 of
    # tuning -- 10e6, on top of its own 4e6 of training.
    assert line.fully_loaded_total == pytest.approx(4e6 + (3e6 + 5e6 + 2e6))
    assert line.search_multiplier == pytest.approx(line.fully_loaded_total / 4e6)
    assert line.fully_loaded_total > line.marginal_total


def test_grand_total_equals_the_sum_of_both_stages():
    cm = analytic_cost_model(10)
    ledgers = {
        "sgd": Ledger("sgd", stage1_exclusive=1e6, stage2_training=4e6, n_trials=2),
        "a": Ledger("a", stage1_exclusive=2e6, stage2_tuning=1e6,
                    stage2_training=3e6, n_trials=2),
    }
    sel = _fake_selection({"sgd": 1.0, "a": 0.5}, survivors=["a"])
    rep = reportmod.build(cm, {}, sel, ledgers, max_seeds=2)
    assert rep.stage1_total == pytest.approx(3e6)
    assert rep.tuning_total == pytest.approx(1e6)
    assert rep.stage2_total == pytest.approx(8e6)
    assert rep.grand_total == pytest.approx(11e6)


def test_a_run_with_no_survivor_says_so_rather_than_dividing_by_zero():
    cm = analytic_cost_model(10)
    ledgers = {"sgd": Ledger("sgd", stage1_exclusive=1e6, stage2_training=1e6, n_trials=2),
               "dead": Ledger("dead", stage1_exclusive=1e6, stage2_training=1e6, n_trials=2)}
    sel = _fake_selection({"sgd": 1.0, "dead": 9.0}, survivors=[])
    rep = reportmod.build(cm, {}, sel, ledgers, max_seeds=2)
    text = reportmod.render(rep)
    assert "No candidate survived" in text
    assert all(l.fully_loaded_total is None for l in rep.lines)


def test_rendered_report_names_both_totals_and_the_naive_risk():
    cm = analytic_cost_model(10)
    ledgers = {"sgd": Ledger("sgd", stage1_exclusive=1e6, stage2_training=4e6, n_trials=4),
               "good": Ledger("good", stage1_exclusive=1e6, stage2_training=4e6, n_trials=4)}
    sel = _fake_selection({"sgd": 1.0, "good": 0.5}, survivors=["good"])
    text = reportmod.render(reportmod.build(cm, {}, sel, ledgers, max_seeds=4))
    assert "stage 1 (gate screen" in text
    assert "stage 2 selection" in text
    assert "search multiplier" in text
    assert "false positive would be" in text


# --------------------------------------------------------------------- trials


def test_preconditioned_sgd_with_no_preconditioner_is_plain_sgd():
    torch.manual_seed(0)
    w = torch.zeros(5, requires_grad=True)
    target = torch.arange(5.0)
    opt = PreconditionedSGD([w], lr=0.1)
    opt.zero_grad()
    ((w - target) ** 2).sum().backward()
    opt.step()
    assert torch.allclose(w.detach(), 0.1 * 2 * target)


def test_trial_respects_its_flop_budget_and_reports_what_it_spent():
    trial = QuadraticTrial(flop_budget=2e6, max_steps=10_000)
    cand = default_pool(trial.dim, include_dsn=False)[0]
    res = trial.run(cand, seed=0)
    assert res.flops <= 2e6 * 1.05
    assert res.steps > 0
    assert math.isfinite(res.score)


def test_a_divergent_trial_is_capped_and_flagged_not_returned_as_nan():
    """Stage 2's sub-Gaussian machinery cannot consume ``inf``."""
    from dataclasses import replace

    trial = QuadraticTrial(flop_budget=2e6)
    cand = replace(default_pool(trial.dim, include_dsn=False)[0], lr=5.0)
    res = trial.run(cand, seed=0)
    assert res.diverged
    assert math.isfinite(res.score)
    assert res.score == pytest.approx(res.detail["cap"])


def test_tuning_refuses_a_learning_rate_that_diverged_on_any_seed():
    """The knife-edge failure: stable on two seeds, catastrophic on a third."""
    from dataclasses import replace

    trial = QuadraticTrial(flop_budget=2e6)
    base = default_pool(trial.dim, include_dsn=False)[0]
    cand = replace(base, lr_grid=(1e-3, 3e-3, 5.0))
    lr, spent, table = tune_learning_rate(cand, trial, seeds=[100, 101])
    assert lr != 5.0
    assert spent > 0
    assert next(r for r in table if r["lr"] == 5.0)["n_diverged"] == 2


def test_tuning_charges_every_trial_it_ran():
    trial = QuadraticTrial(flop_budget=2e6)
    cand = default_pool(trial.dim, include_dsn=False)[0]
    one = trial.run(cand, seed=100).flops
    _, spent, table = tune_learning_rate(cand, trial, seeds=[100, 101])
    assert spent == pytest.approx(one * len(table) * 2, rel=0.35)


def test_dsn_trials_are_charged_for_their_curvature_products():
    """A method that pays curvature must show it in the ledger, not in the loss."""
    from autoresearch.trials import dsn_candidate, make_optimizer

    torch.manual_seed(0)
    cand = dsn_candidate(lr=1e-3)
    w = torch.zeros(8, requires_grad=True)
    A = torch.randn(16, 8)
    b = torch.randn(16)
    opt = make_optimizer(cand, [w])
    cost = ProbeCost()
    run_step(cand, opt, lambda: 0.5 * ((A @ w - b) ** 2).mean(), cost, batch=16)
    assert cost.total_hvps() > 0
    assert cost.grads[16] == 1
    assert cost.forwards[16] == 1  # the same-batch acceptance ratio


def test_probe_contexts_come_from_the_trajectory_the_trial_actually_runs():
    """Stage 1 and stage 2 must describe the same problem, structurally."""
    trial = QuadraticTrial(flop_budget=1e6)
    cand = default_pool(trial.dim, include_dsn=False)[0]
    ctxs = trial.probe_contexts(cand, checkpoints=(0, 5), depths=(4, 8),
                                curv_batch=64, micro=8, micro_batch=32)
    assert [c.step for c in ctxs] == [0, 5]
    assert all(c.n_cross_fit == 4 for c in ctxs)
    assert all(c.noise_scale == pytest.approx(32 / trial.batch) for c in ctxs)
    assert all(c.shared_cost.total_hvps() == 0 for c in ctxs)  # lazy until used
    ctxs[0].rules()
    assert ctxs[0].shared_cost.total_hvps() > 0
