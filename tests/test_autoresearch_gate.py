"""Stage-1 gate: estimator correctness, calibration, and the mutant separation.

The mutant tests are the gate's licence to reject anything.  A screen that has
not been shown to separate a known-good preconditioner from a known-bad one has
no demonstrated power, and running it against real candidates would launder its
own noise into a verdict.
"""

import math

import pytest
import torch

from autoresearch.gate import THRESHOLD, Gate, Verdict, statistics
from autoresearch.preconditioners import (
    IdentityPreconditioner,
    dense_preconditioner,
    dsn_preconditioner,
    frozen_spectral_preconditioner,
    identity_spectral,
    newton_preconditioner,
)
from autoresearch.probes import ProbeContext, ProbeCost, cross_fit, quadrature
from autoresearch.synthetic import (
    blocked_quadratic,
    favorable_pair,
    unfavorable_quadratic,
)

EXACT_DEPTHS = (12,)   # full rank: the quadrature rule is then exact
SWEEP_DEPTHS = (4, 8, 12)


def _gate(depths=SWEEP_DEPTHS, **kw):
    return Gate(depths=depths, n_bootstrap=400, seed=0, **kw)


# ------------------------------------------------------------------- probes


def test_quadrature_reproduces_a_dense_quadratic_form():
    """One Lanczos run must give ``b' f(H) b`` for arbitrary ``f``, exactly."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    b = p.g()
    rules = quadrature(p.matvec, b, (12,))
    H = p.H()
    assert rules[12].apply(lambda lam: lam) == pytest.approx(float(b @ (H @ b)), rel=1e-10)
    Hinv = torch.linalg.inv(H)
    assert rules[12].apply(lambda lam: 1 / lam) == pytest.approx(
        float(b @ (Hinv @ b)), rel=1e-10
    )


def test_quadrature_falls_back_to_the_exhausted_rule_not_to_a_hole():
    """Breakdown at a depth that was never *requested* must still be usable.

    A seed that is an exact eigenvector has a one-dimensional Krylov space, so
    Lanczos breaks down after one step.  Requesting (2, 8) means the depth
    actually reached is not itself a requested depth; both entries must fall
    back to the exact depth-1 rule rather than raising ``KeyError``.
    """
    p = blocked_quadratic(0.1, 10.0, 0.01)
    b = p.exact_cross_fit()[0]  # an eigenvector of H, by construction
    rules = quadrature(p.matvec, b, (2, 8))
    assert set(rules) == {2, 8}
    assert rules[2].depth == 1 and rules[8].depth == 1
    H = p.H()
    assert rules[8].apply(lambda lam: lam) == pytest.approx(float(b @ (H @ b)), rel=1e-10)


def test_cross_fit_requires_an_even_number_of_microbatches():
    v = [torch.randn(3) for _ in range(3)]
    with pytest.raises(ValueError, match="even number"):
        cross_fit(v)


def test_cross_fit_is_unbiased_for_sigma_with_no_estimated_mean():
    """``E[u u'] = Sigma`` with a nonzero common mean present in every sample."""
    torch.manual_seed(0)
    n, h = 6, 20000
    sigma_root = torch.diag(torch.tensor([0.5, 1.0, 2.0, 0.25, 3.0, 1.5]))
    mean = torch.arange(1.0, n + 1)
    samples = [mean + sigma_root @ torch.randn(n) for _ in range(2 * h)]
    u = cross_fit(samples)
    emp = sum(ui.outer(ui) for ui in u) / len(u)
    target = sigma_root @ sigma_root.T

    # Checked against the estimator's own standard error rather than a flat
    # tolerance, which would be tuned to a seed rather than to the statistic.
    # For diagonal Sigma: Var(u_i u_j) = s_i s_j (i != j), Var(u_i^2) = 2 s_i^2.
    s = torch.diagonal(target)
    var = s.outer(s) * (1 + torch.eye(n))
    se = (var / len(u)).sqrt()
    assert bool(((emp - target).abs() < 4 * se).all()), (emp - target) / se


def test_probe_cost_scaling_keeps_fractional_shares():
    c = ProbeCost()
    c.add_hvp(256, 7)
    assert c.scaled(0.5).hvps[256] == pytest.approx(3.5)


# -------------------------------------------------------------- calibration


def test_identity_scores_exactly_one_on_both_statistics_and_both_paths():
    """``P = I`` is the definition of the threshold, so it must hit it exactly.

    Both code paths -- explicit operator and spectral quadrature -- and at a
    *truncated* Lanczos depth, where the underlying quadratic forms are only
    approximate.  The approximations must cancel; if they do not, the threshold
    at 1 drifts with the estimator and every verdict moves with it.
    """
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=(2, 4))
    for precond in (IdentityPreconditioner(), identity_spectral()):
        for depth in (2, 4):
            rho, gain = statistics(precond.forms(ctx, depth))
            assert rho == pytest.approx(1.0, abs=1e-12), precond.name
            assert gain == pytest.approx(1.0, abs=1e-12), precond.name


def test_identity_never_passes_the_gate():
    """A gate that passes ``P = I`` would pass doing nothing."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    report = _gate().evaluate(IdentityPreconditioner(), [p.context(depths=SWEEP_DEPTHS)])
    assert report.verdict is Verdict.FAIL, report.reason


def test_zero_min_effect_leaves_the_identity_at_the_mercy_of_rounding():
    """Why ``min_effect`` defaults above zero, asserted rather than asserted-in-prose.

    ``P = I`` scores exactly 1.0 and its bootstrap is degenerate -- every
    resample returns the same number -- so at ``min_effect=0`` the interval has
    zero width and sits *on* the bar.  The verdict is then decided by
    floating-point dust, and a zero-width interval reports it as certainty.
    """
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ck = Gate(depths=EXACT_DEPTHS, n_bootstrap=200, min_effect=0.0).evaluate_checkpoint(
        IdentityPreconditioner(), p.context(depths=EXACT_DEPTHS)
    )
    assert ck.cells[0].rho.se == 0.0
    assert ck.cells[0].rho.lo == ck.cells[0].rho.hi == pytest.approx(1.0, abs=1e-12)


def test_statistics_are_invariant_to_rescaling_the_preconditioner():
    """The learning rate absorbs scale, so the gate must not reward it."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=EXACT_DEPTHS)
    H = p.H()
    base = torch.linalg.inv(H)
    a = statistics(dense_preconditioner(base, "P").forms(ctx, 12))
    b = statistics(dense_preconditioner(37.0 * base, "37P").forms(ctx, 12))
    assert a[0] == pytest.approx(b[0], rel=1e-9)
    assert a[1] == pytest.approx(b[1], rel=1e-9)


def test_spectral_and_operator_paths_agree_on_the_same_preconditioner():
    """The cheap path must not be a different measurement from the exact one."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=EXACT_DEPTHS)
    delta = 0.05
    spectral = dsn_preconditioner(delta)
    explicit = dense_preconditioner(
        torch.linalg.inv(p.H() + delta * torch.eye(p.n)), "explicit"
    )
    rs, gs = statistics(spectral.forms(ctx, 12))
    re, ge = statistics(explicit.forms(ctx, 12))
    assert rs == pytest.approx(re, rel=1e-8)
    assert gs == pytest.approx(ge, rel=1e-8)


def test_estimator_matches_the_closed_form():
    """Estimator against truth, not against itself."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=EXACT_DEPTHS)
    pdiag = p.newton_diag()
    rho, gain = statistics(newton_preconditioner(1e-12).forms(ctx, 12))
    assert rho == pytest.approx(p.exact_rho(pdiag), rel=1e-8)
    assert gain == pytest.approx(p.exact_gain(pdiag), rel=1e-8)


def test_spectral_function_must_stay_positive_on_the_measured_spectrum():
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=EXACT_DEPTHS)
    bad = newton_preconditioner(1e-12)
    bad.f = lambda lam: lam - 5.0
    with pytest.raises(ValueError, match="non-positive"):
        bad.forms(ctx, 12)


# ------------------------------------------------------------------ mutants


def test_mutant_pass_exact_optimum_on_a_synthetic_quadratic():
    """(b) ``P* = H^-1`` on a problem built so that preconditioning pays."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    report = _gate().evaluate(
        newton_preconditioner(1e-12), [p.context(depths=SWEEP_DEPTHS)]
    )
    assert report.verdict is Verdict.PASS, report.reason
    ck = report.checkpoints[0]
    assert ck.rho_floor.lo > THRESHOLD
    assert ck.gain_floor.lo > THRESHOLD


def test_mutant_pass_is_actually_the_optimum_not_merely_good():
    """``P* = H^-1`` attains the all-``P`` Cauchy-Schwarz ceiling as noise vanishes.

    ``sup_P R(P) <= (g'H^-1 g)(g'C^-1 g)`` with ``C = gg' + Sigma``.  As
    ``Sigma -> 0`` the bound tightens to ``g'H^-1 g`` and ``P = H^-1`` attains
    it, which is what makes "exact optimum" a checkable claim rather than a
    label.
    """
    p = blocked_quadratic(0.1, 10.0, 0.01, noise_scale=1e-10)
    assert p.exact_R(p.newton_diag()) == pytest.approx(p.ceiling_R(), rel=1e-6)


def test_mutant_pass_beats_every_random_perturbation_of_itself():
    """Numeric local optimality: no symmetric perturbation of ``P*`` raises ``R``."""
    torch.manual_seed(0)
    p = blocked_quadratic(0.1, 10.0, 0.01, noise_scale=1e-10)
    star = p.newton_diag()
    best = p.exact_R(star)
    for _ in range(200):
        pert = star * torch.exp(0.15 * torch.randn(p.n, dtype=star.dtype))
        assert p.exact_R(pert) <= best * (1 + 1e-9)


def test_mutant_fail_constant_preconditioner_claim():
    """(a-i) A constant ``P`` is not a preconditioner and must not pass."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=SWEEP_DEPTHS)
    const = dense_preconditioner(3.0 * torch.eye(p.n), "constant(3I)")
    report = _gate().evaluate(const, [ctx])
    assert report.verdict is not Verdict.PASS
    rho, gain = statistics(const.forms(ctx, 12))
    assert rho == pytest.approx(1.0, abs=1e-12)
    assert gain == pytest.approx(1.0, abs=1e-12)


def test_mutant_fail_frozen_preconditioner_after_curvature_drift():
    """(a-ii) The staleness mutant, and the reason the gate is not ``rho`` alone.

    Three claims, all asserted:

    1. the correctly-refreshed ``P = H_T^-1`` PASSes at ``t=T``;
    2. the frozen ``P = H_0^-1`` FAILs at ``t=T``;
    3. ``rho`` alone would have passed the frozen one -- and its value is
       *numerically identical* to what it was at ``t=0``, because ``H`` does not
       appear in ``rho`` at all.
    """
    t0, tT = favorable_pair()
    ctx0 = t0.context(depths=SWEEP_DEPTHS, step=0)
    ctxT = tT.context(depths=SWEEP_DEPTHS, step=1)
    frozen = dense_preconditioner(torch.linalg.inv(t0.H()), "frozen(H0^-1)")
    fresh = dense_preconditioner(torch.linalg.inv(tT.H()), "fresh(HT^-1)")
    gate = _gate()

    fresh_report = gate.evaluate(fresh, [ctxT])
    frozen_report = gate.evaluate(frozen, [ctxT])
    assert fresh_report.verdict is Verdict.PASS, fresh_report.reason
    assert frozen_report.verdict is Verdict.FAIL, frozen_report.reason

    # (3) rho is blind, and provably so: same number at both checkpoints.
    rho_at_0, _ = statistics(frozen.forms(ctx0, 12))
    rho_at_T, gain_at_T = statistics(frozen.forms(ctxT, 12))
    assert rho_at_T == pytest.approx(rho_at_0, rel=1e-12)
    assert rho_at_T > THRESHOLD
    assert gain_at_T < THRESHOLD
    assert frozen_report.rho_only_verdict is Verdict.PASS


def test_gate_rejects_the_dsn_shaped_unfavorable_problem():
    """Gradient in sharp directions, noise in flat ones: nothing to buy."""
    p = unfavorable_quadratic()
    report = _gate().evaluate(newton_preconditioner(1e-12), [p.context(depths=SWEEP_DEPTHS)])
    assert report.verdict is Verdict.FAIL, report.reason
    assert report.rho_ceiling < THRESHOLD


def test_frozen_low_rank_snapshot_fails_once_it_is_rich_enough_to_be_wrong():
    """The realistic frozen form -- a rank-k Lanczos snapshot, never refreshed.

    Rank matters, and not in the direction one would guess.  Lanczos resolves
    extremal eigenvalues first, so a low-rank snapshot has not yet resolved the
    *flattest* block -- which is the one that drifts here.  Below rank 8 the
    snapshot has therefore committed to nothing about the drifting directions
    and falls back to its tail scaling, which happens to damp them; it is
    protected from staleness by its own ignorance rather than by being fresh.
    Only once the snapshot is rich enough to assign those directions a large
    ``1/lambda`` does the drift become catastrophic.

    Both halves are asserted, because "the gate rejects stale preconditioners"
    would be an overclaim: what it rejects is stale preconditioners that are
    *specific enough to be wrong*.
    """
    t0, tT = favorable_pair()
    ctx0 = t0.context(depths=SWEEP_DEPTHS, step=0)
    ctxT = tT.context(depths=SWEEP_DEPTHS, step=1)
    gate = _gate()

    def frozen(rank):
        return frozen_spectral_preconditioner(
            ctx0, lambda lam: lam.abs() + 1e-8, rank=rank, name=f"frozen-rank{rank}"
        )

    for rank in (8, 10, 12):
        assert gate.evaluate(frozen(rank), [ctxT]).verdict is Verdict.FAIL, rank
    # ...and rho is identical at both checkpoints at every rank, because H does
    # not appear in it.
    for rank in (4, 8, 12):
        p = frozen(rank)
        assert statistics(p.forms(ctx0, 12))[0] == pytest.approx(
            statistics(p.forms(ctxT, 12))[0], rel=1e-12
        )


# ------------------------------------------------------------ verdict logic


def test_ceiling_and_floor_are_evaluated_at_opposite_ends_of_the_sweep():
    """Rejection uses the most favorable depth; acceptance uses the least."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ck = _gate().evaluate_checkpoint(
        newton_preconditioner(1e-12), p.context(depths=SWEEP_DEPTHS)
    )
    assert ck.rho_ceiling.point >= ck.rho_floor.point
    assert ck.gain_ceiling.point >= ck.gain_floor.point


def test_a_failing_gain_overrides_a_passing_rho():
    """Both statistics are necessary conditions; either one can reject."""
    _, tT = favorable_pair()
    frozen_P = torch.linalg.inv(favorable_pair()[0].H())
    ck = _gate().evaluate_checkpoint(
        dense_preconditioner(frozen_P, "frozen"), tT.context(depths=SWEEP_DEPTHS)
    )
    assert ck.rho_verdict is Verdict.PASS
    assert ck.gain_verdict is Verdict.FAIL
    assert ck.verdict is Verdict.FAIL


def test_require_gain_false_reproduces_the_rho_only_screen():
    """The H-blind screen is still available -- and still blind."""
    _, tT = favorable_pair()
    frozen_P = torch.linalg.inv(favorable_pair()[0].H())
    report = Gate(depths=SWEEP_DEPTHS, n_bootstrap=200, require_gain=False).evaluate(
        dense_preconditioner(frozen_P, "frozen"), [tT.context(depths=SWEEP_DEPTHS)]
    )
    assert report.verdict is Verdict.PASS


def test_trajectory_rollup_passes_if_any_checkpoint_passes():
    """`rho` moved 16x within one measured DSN run; a phase-dependent pass counts."""
    good = blocked_quadratic(0.1, 10.0, 0.01)
    bad = unfavorable_quadratic()
    gate = _gate()
    p = newton_preconditioner(1e-12)
    mixed = gate.evaluate(
        p, [bad.context(depths=SWEEP_DEPTHS, step=0), good.context(depths=SWEEP_DEPTHS, step=1)]
    )
    assert mixed.verdict is Verdict.PASS
    only_bad = gate.evaluate(
        p, [bad.context(depths=SWEEP_DEPTHS, step=0), bad.context(depths=SWEEP_DEPTHS, step=1)]
    )
    assert only_bad.verdict is Verdict.FAIL


def test_unbounded_quadratic_model_is_undefined_not_infinitely_good():
    """Indefinite curvature breaks the model; that must never read as a PASS."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    indefinite = p.__class__(
        lam=p.lam * torch.tensor([1.0] * 4 + [-1.0] * 8, dtype=p.lam.dtype),
        c=p.c,
        s=p.s,
        Q=p.Q,
        noise_scale=1e6,
        name="indefinite",
    )
    ctx = indefinite.context(depths=EXACT_DEPTHS)
    _, gain = statistics(IdentityPreconditioner().forms(ctx, 12))
    assert math.isnan(gain)
    ck = _gate(depths=EXACT_DEPTHS).evaluate_checkpoint(IdentityPreconditioner(), ctx)
    assert ck.gain_verdict is Verdict.INDETERMINATE
    assert ck.verdict is not Verdict.PASS
    assert ck.cells[0].unbounded_model


def test_screen_splits_the_shared_probe_cost_across_the_pool():
    """The shared Lanczos cost must appear in the ledger, not vanish."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=EXACT_DEPTHS)
    pool = [newton_preconditioner(1e-12), dsn_preconditioner(0.1), IdentityPreconditioner()]
    reports = gate_reports = __import__(
        "autoresearch.gate", fromlist=["screen"]
    ).screen(_gate(depths=EXACT_DEPTHS), pool, [ctx])
    total_shared = sum(
        r.shared_cost_share.total_hvps() for r in gate_reports.values()
    )
    assert total_shared == pytest.approx(ctx.shared_cost.total_hvps())
    assert reports["identity"].exclusive_cost.total_hvps() > 0
    assert reports["newton(floor=1e-12)"].exclusive_cost.total_hvps() == 0


# ------------------------------------------------- gain-targeted mutants (G1-G3)


def test_gain_converges_in_depth_strictly_more_slowly_than_rho():
    """G3. ``gain``'s integrand is the sharper near-singular one, so it lags.

    ``rho`` integrates ``1/f(lam)``; ``gain`` integrates ``lam/f(lam)^2``, which
    peaks at ``lam = delta`` with height ``1/(4 delta)``.  A Gauss rule fits
    polynomials, so the sharper feature is approximated worse at every depth
    short of convergence.  Exact values come from the true spectrum using the
    *same* probe vectors, so what is measured here is truncation alone.
    """
    from autoresearch.synthetic import flat_tail_quadratic

    prob = flat_tail_quadratic(n=300, lo=1e-4, hi=1e2, seed=0)
    u = prob.sampled_cross_fit(6, seed=1)
    g = prob.g()
    delta = 1e-3
    f = lambda lam: lam.abs() + delta
    rho_x, gain_x = prob.exact_spectral_statistics(f, g, u)
    P = dsn_preconditioner(delta)

    checked = 0
    for depth in (10, 25, 50):
        ctx = ProbeContext(matvec=prob.matvec, g=g, u=u, depths=(depth,),
                           noise_scale=prob.noise_scale)
        rho, gain = statistics(P.forms(ctx, depth))
        assert gain / gain_x < rho / rho_x, (depth, rho / rho_x, gain / gain_x)
        checked += 1
    assert checked == 3


def test_a_statistic_still_moving_at_the_deepest_rung_cannot_produce_a_verdict():
    """Non-convergence gates the verdict in BOTH directions, not just PASS.

    The earlier safeguard only annotated the reason string, so a sweep that was
    entirely below the bar returned a confident FAIL no matter how far from
    converged it was.  On this spectrum ``rho`` runs 9.2 at depth 5 and 1104 at
    depth 200, so a shallow FAIL would have been two orders of magnitude wrong.
    """
    from autoresearch.synthetic import flat_tail_quadratic

    prob = flat_tail_quadratic(n=300, lo=1e-4, hi=1e2, seed=0)
    ctx = ProbeContext(matvec=prob.matvec, g=prob.g(),
                       u=prob.sampled_cross_fit(6, seed=1), depths=(5, 10),
                       noise_scale=prob.noise_scale)
    ck = Gate(depths=(5, 10), n_bootstrap=200).evaluate_checkpoint(
        dsn_preconditioner(1e-3), ctx
    )
    assert ck.rho_verdict is Verdict.INDETERMINATE
    assert ck.gain_verdict is Verdict.INDETERMINATE
    assert "NOT CONVERGED IN DEPTH" in ck.reason
    assert ck.depth_convergence("gain") > ck.depth_convergence("rho")


def test_a_single_depth_sweep_cannot_certify_anything():
    """Convergence is unassessable with one rung, and that is not a pass."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ck = Gate(depths=(12,), n_bootstrap=200).evaluate_checkpoint(
        newton_preconditioner(1e-12), p.context(depths=(12,))
    )
    assert ck.verdict is Verdict.INDETERMINATE
    assert "UNASSESSABLE" in ck.reason


def test_spectral_resolution_reports_when_damping_falls_below_the_resolved_spectrum():
    """The mechanism behind the depth trap, as a number.

    A Gauss rule samples the spectrum only down to its smallest node.  Damping
    at or below that node puts the whole near-singular region of the integrand
    in the unsampled part, where a polynomial is extrapolating.
    """
    from autoresearch.synthetic import flat_tail_quadratic

    prob = flat_tail_quadratic(n=300, lo=1e-4, hi=1e2, seed=0)
    ctx = ProbeContext(matvec=prob.matvec, g=prob.g(),
                       u=prob.sampled_cross_fit(6, seed=1), depths=(10, 25),
                       noise_scale=prob.noise_scale)
    gate = Gate(depths=(10, 25), n_bootstrap=100)
    tight = gate.evaluate_checkpoint(dsn_preconditioner(1e-3), ctx)
    loose = gate.evaluate_checkpoint(dsn_preconditioner(1e-1), ctx)
    assert tight.cells[-1].spectral_resolution < 1.0
    assert loose.cells[-1].spectral_resolution > 1.0
    assert "extrapolated" in tight.reason
    assert "extrapolated" not in loose.reason


def test_trajectory_aggregate_overrides_a_single_favorable_checkpoint():
    """G2. Expected decreases add across steps; per-checkpoint ratios do not.

    A preconditioner tuned to one curvature regime, screened along a trajectory
    that alternates between two.  It tops the per-checkpoint ratio in the regime
    where the available per-step decrease is small, and loses badly in the regime
    where it is large -- so ``max`` over checkpoints and ``sum R(P) / sum R(I)``
    disagree by ~170x, in the unsafe direction.
    """
    from autoresearch.gate import trajectory_gain

    a = blocked_quadratic(0.1, 10.0, 0.01, seed=0, name="A")
    b = blocked_quadratic(1e-3, 2e-3, 5e-4, seed=0, name="B")
    ctxs = [
        p.context(depths=SWEEP_DEPTHS, step=i, label=p.name)
        for i, p in enumerate([a, b, a, b])
    ]
    P = dense_preconditioner(torch.linalg.inv(a.H()), "tuned-to-A")
    gate = _gate()
    report = gate.evaluate(P, ctxs)

    per_ckpt = [statistics(P.forms(c, 12))[1] for c in ctxs]
    assert max(per_ckpt) > 10                      # a checkpoint loves it
    assert report.trajectory_gain < 0.1            # the trajectory does not
    assert report.verdict is not Verdict.PASS, report.reason
    assert "does not survive aggregation" in report.reason
    assert trajectory_gain(report.checkpoints, 12) == pytest.approx(
        report.trajectory_gain
    )


def test_a_genuinely_favorable_candidate_still_passes_the_trajectory_check():
    """The aggregate must not reject everything -- a stationary win survives it."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctxs = [p.context(depths=SWEEP_DEPTHS, step=i) for i in range(3)]
    report = _gate().evaluate(newton_preconditioner(1e-12), ctxs)
    assert report.verdict is Verdict.PASS, report.reason
    assert report.trajectory_gain > 1.0


def test_subspace_freeze_is_rejected_by_gain_but_not_by_rho():
    """G1. Freezing a subspace: ``gain`` holds, ``rho`` is fooled a third time.

    ``rho`` rewards shrinking ``P`` on the noise subspace through
    ``tr(P^-1 Sigma)/tr(P Sigma)``, without reference to what that subspace costs
    the objective.  ``gain`` charges it, because the gradient mass the frozen
    directions carry leaves the numerator.
    """
    from autoresearch.preconditioners import subspace_damping_preconditioner

    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=SWEEP_DEPTHS)
    H = p.H()
    _, V = torch.linalg.eigh(0.5 * (H + H.T))
    frozen = subspace_damping_preconditioner(V[:, -6:], 1e-3, "freeze")
    rho, gain = statistics(frozen.forms(ctx, 12))
    assert rho > THRESHOLD
    assert gain < rho


def test_subspace_damping_rejects_a_singular_preconditioner():
    """``epsilon = 0`` has no inverse, so ``rho`` would be undefined."""
    from autoresearch.preconditioners import subspace_damping_preconditioner

    with pytest.raises(ValueError, match="singular"):
        subspace_damping_preconditioner(torch.eye(4)[:, :2], 0.0, "bad")


# --------------------------------------------------- rank-restricted screening


def _krylov(delta, rank, c=1.0):
    from autoresearch.preconditioners import KrylovRestrictedPreconditioner

    return KrylovRestrictedPreconditioner(delta, rank, c)


def test_rank_restriction_at_full_rank_reproduces_the_unrestricted_form():
    """The restriction must be a restriction, not a different operator.

    At ``rank = n`` the Krylov space is the whole space and the complement is
    empty, so ``P_k`` must equal ``(|H| + delta)^-1`` exactly.  Without this the
    rank sweep would be measuring two unrelated things at its two ends.
    """
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=EXACT_DEPTHS)
    full = statistics(dsn_preconditioner(1e-3).forms(ctx, 12))
    restricted = statistics(_krylov(1e-3, p.n).forms(ctx, 12))
    assert restricted[0] == pytest.approx(full[0], rel=1e-9)
    assert restricted[1] == pytest.approx(full[1], rel=1e-9)


def test_rank_restricted_statistics_carry_no_truncation_error():
    """An explicit operator has no depth trap, and that is the point.

    A full-rank spectral candidate is evaluated through quadrature whose
    integrand is near-singular for small ``delta``.  A restricted candidate is
    applied directly, and the only quadrature left is ``g'Hg`` -- degree one,
    which a Gauss rule integrates exactly at any depth.  So the statistic must
    not move with the sweep at all.
    """
    from autoresearch.synthetic import flat_tail_quadratic

    prob = flat_tail_quadratic(n=300, lo=1e-4, hi=1e2, seed=0)
    ctx = ProbeContext(matvec=prob.matvec, g=prob.g(),
                       u=prob.sampled_cross_fit(6, seed=1), depths=(10, 25, 50),
                       noise_scale=prob.noise_scale)
    gate = Gate(depths=(10, 25, 50), n_bootstrap=100)

    restricted = gate.evaluate_checkpoint(_krylov(1e-3, 8), ctx)
    assert restricted.depth_convergence("rho") == pytest.approx(0.0, abs=1e-12)
    assert restricted.depth_convergence("gain") == pytest.approx(0.0, abs=1e-12)

    # ...while the full-rank form on the same context is nowhere near converged.
    full = gate.evaluate_checkpoint(dsn_preconditioner(1e-3), ctx)
    assert full.depth_convergence("gain") > 0.10


def test_a_rank_restricted_verdict_is_labelled_differently_from_a_full_rank_one():
    """Nothing downstream may mistake an upper bound for a verdict."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctxs = [p.context(depths=SWEEP_DEPTHS, step=i) for i in range(2)]
    gate = _gate()
    full = gate.evaluate(dsn_preconditioner(1e-3), ctxs).as_dict()
    restricted = gate.evaluate(_krylov(1e-3, 4), ctxs).as_dict()
    assert full["rank"] is None
    assert full["scope"] == "full-rank"
    assert restricted["rank"] == 4
    assert restricted["scope"] == "rank-4"


def test_a_singular_complement_is_rejected_rather_than_screened():
    """``c = 0`` has no inverse, so ``rho`` would be undefined -- and DSN steps
    in the complement anyway, so it is not the operator being modelled."""
    with pytest.raises(ValueError, match="singular"):
        _krylov(1e-3, 4, c=0.0)


def test_rank_restriction_changes_the_statistics_it_is_supposed_to_change():
    """A restriction that changed nothing would not be measuring the restriction."""
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=EXACT_DEPTHS)
    vals = [statistics(_krylov(1e-3, r).forms(ctx, 12)) for r in (2, 4, 8, 12)]
    assert len({round(v[1], 6) for v in vals}) == len(vals)


def test_the_complement_scaling_is_swept_because_it_is_a_modelling_choice():
    """DSN puts AdamW in the complement; a scalar ``c`` is an approximation.

    If the verdict moves with ``c``, the report is about the complement rather
    than the subspace -- so ``c`` must actually move the statistics, or sweeping
    it would be theatre.
    """
    p = blocked_quadratic(0.1, 10.0, 0.01)
    ctx = p.context(depths=EXACT_DEPTHS)
    vals = [statistics(_krylov(1e-2, 4, c).forms(ctx, 12))[1] for c in (0.1, 1.0, 10.0)]
    assert len({round(v, 6) for v in vals}) == 3
