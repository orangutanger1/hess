import pytest
import torch

from dsn import DSN
from dsn.subspace import KrylovBuilder


def quadratic_problem(n=12, cond=50.0, seed=0):
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    lam = torch.logspace(0, torch.log10(torch.tensor(cond)), n)
    A = Q @ torch.diag(lam) @ Q.T
    b = torch.randn(n)
    x = torch.zeros(n, requires_grad=True)
    return A, b, x


def test_degenerates_to_adamw_when_subspace_is_empty():
    """k=0 must reproduce AdamW exactly, or every comparison is meaningless."""
    torch.manual_seed(0)
    p_ref = torch.randn(8, requires_grad=True)
    p_dsn = p_ref.detach().clone().requires_grad_(True)
    target = torch.randn(8)

    ref = torch.optim.AdamW([p_ref], lr=1e-2, weight_decay=0.0, eps=1e-8)
    dsn = DSN([p_dsn], lr=1e-2, weight_decay=0.0,
              builder=KrylovBuilder(k_max=0, m_recycle=0))

    for _ in range(20):
        ref.zero_grad()
        (p_ref - target).pow(2).sum().backward()
        ref.step()
        dsn.step(lambda: (p_dsn - target).pow(2).sum())

    torch.testing.assert_close(p_dsn.detach(), p_ref.detach(), rtol=0, atol=1e-11)


def test_full_rank_step_equals_exact_newton_on_a_quadratic():
    """The central correctness invariant of the whole method."""
    n = 10
    A, b, x = quadratic_problem(n)

    dsn = DSN([x], lr=1.0, weight_decay=0.0, trust_radius=1e9,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))
    x0 = x.detach().clone()
    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    newton = -torch.linalg.solve(A, A @ x0 + b)
    torch.testing.assert_close(x.detach() - x0, newton, rtol=1e-6, atol=1e-8)


def test_solves_a_quadratic_in_one_step():
    n = 10
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1.0, weight_decay=0.0, trust_radius=1e9,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))

    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    torch.testing.assert_close(A @ x.detach() + b, torch.zeros(n),
                               atol=1e-6, rtol=0)


def test_complement_step_is_orthogonal_to_the_subspace():
    n = 15
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-2, builder=KrylovBuilder(k_max=4, m_recycle=2, tau=0.1))
    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    W = dsn._last_W
    torch.testing.assert_close(
        W.T @ dsn._last_d_complement, torch.zeros(W.shape[1]), atol=1e-10, rtol=0
    )


def test_trust_radius_clips_a_large_subspace_step():
    n = 10
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-3, trust_radius=1e-3,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))
    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    assert dsn.telemetry.step_norm_subspace <= 1e-3 + 1e-12


def test_rho_accounts_for_trust_region_clipping():
    """rho must be computed against the *clipped* step's predicted reduction,
    or the trust region enters a self-reinforcing shrink spiral: a small clip
    factor `s` produces a small rho (rho ~ 2s - s**2 for an unscaled
    prediction), which shrinks the trust radius further, producing an even
    smaller `s` next step, with no path back out since growth requires
    rho > 0.75.

    On an exact quadratic, a correctly-scaled prediction matches the clipped
    step's true reduction exactly, so rho must be ~1 every step and the trust
    radius must not collapse toward zero. Uses the same configuration as
    ``test_trust_radius_clips_a_large_subspace_step`` (a trust radius far
    below the Newton step norm, so clipping is active from the first step),
    run for multiple steps instead of one so the trust-region *trajectory* is
    observable.

    Every step now carries a valid rho, including the first: the ratio is
    measured on the step's own closure rather than lagged onto the next one
    (Plan 2, F1/F2). 11 steps, not more: step 10 is the first whose Newton step
    fits inside the grown radius, and on an exact quadratic it lands on x*
    exactly, so step 11 sees ||g|| ~ 1.6e-15, reports k=0 and predicted = 0.0,
    and rho is nan there -- for a reason unrelated to prediction scaling. What
    happens from step 11 on is
    ``test_an_exact_optimum_is_a_fixed_point_of_the_iteration``.
    """
    n = 10
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-3, trust_radius=1e-3,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))

    rhos = []
    for _ in range(11):
        dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
        rhos.append(dsn.telemetry.rho)

    for i, rho in enumerate(rhos):
        assert abs(rho - 1.0) < 1e-6, (
            f"step {i}: rho={rho} indicates a mis-scaled prediction"
        )

    # A correctly-scaled prediction must not send the trust radius into a
    # self-reinforcing shrink spiral toward zero.
    assert dsn.trust_radius >= 1e-3


def test_trust_radius_is_clamped_from_above():
    """F7: unbounded geometric growth. Conn/Gould/Toint require a Delta_max.

    Before Plan 2 the radius doubled on every well-predicted step with no upper
    bound -- measured 1.34e8 over 40 steps on this suite's own logistic gate and
    ~1.4e45 over 200 fixed-batch MNIST steps, which already overflows float32.
    On the exact quadratic below, rho is ~1 on every step, so growth is
    continuous and the clamp is the only thing that can stop it.
    """
    n = 10
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-3, trust_radius=1e-3, trust_max=1e-2,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))

    for _ in range(30):
        dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
        assert dsn.trust_radius <= 1e-2

    assert dsn.trust_radius == pytest.approx(1e-2)

    # The default is 1000x the initial radius, so the bound scales with the
    # problem scale the caller set rather than being a fixed constant.
    assert DSN([x], trust_radius=4.0).trust_max == 4e3


def test_an_exact_optimum_is_a_fixed_point_of_the_iteration():
    """F9: a zero gradient used to still produce a momentum-driven step.

    The builder short-circuits to an empty subspace at a vanishing gradient, but
    AdamW's first moment still carries the previous steps' gradients and
    `project_out` against a zero-width basis is the identity, so the full AdamW
    displacement was applied on top of a zero Newton step. Measured on this
    exact configuration before Plan 2: the iterate lands on x* at step 10, then
    step 11 applies ||d_comp|| = 2.455e-3 and step 12 measures ||g|| = 2.6e-2 --
    the optimizer walked back off the optimum it had reached.
    """
    n = 10
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-3, trust_radius=1e-3,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))

    for _ in range(11):
        dsn.step(lambda: 0.5 * x @ A @ x + b @ x)

    at_optimum = x.detach().clone()
    assert float((A @ at_optimum + b).norm()) < 1e-12, "setup: not at x* yet"

    for _ in range(5):
        dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
        assert dsn.telemetry.k == 0
        assert dsn.telemetry.step_norm_complement == 0.0

    torch.testing.assert_close(x.detach(), at_optimum, rtol=0, atol=0)


def test_the_complement_is_clipped_by_the_trust_region_too():
    """F4: the trust region must bound the whole step, not just its subspace part.

    `g` lies in span(W) by construction (the Krylov basis is seeded with the
    deflated gradient), so `d_comp` is orthogonal to `g` and carries exactly
    zero first-order descent. When only `d_sub` was scaled, a collapsing radius
    drove the descent-carrying part toward zero while the zero-descent part
    stayed at essentially unit norm -- so the loss *rose* under collapse rather
    than stalling. Both parts must shrink together.
    """
    n = 12
    A, b, x = quadratic_problem(n)
    # k_max well below n leaves a nonzero complement, and lr large enough that
    # the AdamW step alone would exceed the radius.
    dsn = DSN([x], lr=1e-1, trust_radius=1e-4,
              builder=KrylovBuilder(k_max=3, m_recycle=0, tau=0.0))

    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
    t = dsn.telemetry

    assert t.step_norm_complement > 0.0, "setup: complement must be nonzero"
    total = (t.step_norm_subspace**2 + t.step_norm_complement**2) ** 0.5
    assert total <= 1e-4 + 1e-12, f"total step {total} exceeds the trust radius"


def test_telemetry_is_populated():
    n = 15
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-2, builder=KrylovBuilder(k_max=5, m_recycle=2, tau=0.1))
    dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
    t = dsn.telemetry

    assert 0 < t.k <= 5
    assert t.n_hvp >= 1
    # NaN guard only, not a range check: `newton_residual` returns a sqrt, so
    # non-negativity holds by construction. What this catches is
    # `rel_residual` coming back NaN (NaN fails every comparison), which is
    # how a non-finite curvature product would surface here.
    assert 0.0 <= t.rel_residual
    assert t.n_fallback == 0


def test_non_finite_curvature_falls_back_to_adamw_without_crashing():
    x = torch.tensor([1.0], requires_grad=True)

    class Exploding:
        def build(self, matvec, g):
            raise FloatingPointError("simulated non-finite curvature")

        def reset(self):
            pass

    dsn = DSN([x], lr=1e-2, builder=Exploding())
    dsn.step(lambda: (x**2).sum())

    assert dsn.telemetry.n_fallback == 1
    assert torch.isfinite(x).all()


def test_the_fallback_path_carries_every_cumulative_counter_forward():
    """The fallback rebuilds Telemetry from scratch, so each cumulative counter
    has to be threaded through explicitly or it silently resets to zero. Assert
    on all three at once: a counter added later and forgotten here would make a
    run that spent most of its steps rejecting or shrinking indistinguishable
    from one that never did."""
    x = torch.zeros(4, requires_grad=True)
    dsn = DSN([x], lr=1e-2, builder=KrylovBuilder(k_max=2, m_recycle=0, tau=0.0))
    dsn.telemetry.n_shrink = 7
    dsn.telemetry.n_reject = 5
    dsn.telemetry.n_fallback = 3

    class Exploding:
        def build(self, matvec, g):
            raise FloatingPointError("simulated non-finite curvature")

        def reset(self):
            pass

    dsn.builder = Exploding()
    dsn.step(lambda: (x - 1.0).pow(2).sum())

    assert (dsn.telemetry.n_shrink, dsn.telemetry.n_reject) == (7, 5)
    assert dsn.telemetry.n_fallback == 4


def test_damping_conflicting_with_an_explicit_builder_raises():
    """`damping` configures the *default* builder only.

    Before this check, `DSN(params, damping=1e-6, builder=KrylovBuilder(...))`
    silently used the builder's default 1e-3 -- a 1000x difference in the
    regularizer with no error and no telemetry. Raising is the only way the
    caller finds out.
    """
    x = torch.zeros(4, requires_grad=True)

    with pytest.raises(ValueError, match="set damping on the builder"):
        DSN([x], damping=1e-6, builder=KrylovBuilder(k_max=2, damping=1e-6))

    # Neither half alone is a conflict: a builder with DSN's default damping,
    # and a non-default damping with no builder, must both construct fine.
    DSN([x], builder=KrylovBuilder(k_max=2, damping=1e-6))
    assert DSN([x], damping=1e-6).builder.damping == 1e-6


def test_lr_scheduler_changes_the_lr_the_optimizer_actually_uses():
    """`lr` must be read from the param group, not from a constructor copy.

    `torch.optim.lr_scheduler` writes `param_groups[0]["lr"]`. If DSN kept its
    own `self.lr`, every scheduler would be a no-op on it while appearing to
    work. Asserted on behavior, not on the attribute: k=0 makes DSN a pure
    AdamW step, so the step norm is directly proportional to lr, and a decayed
    lr must produce a proportionally smaller step.
    """
    def step_norm_after(n_scheduler_steps):
        x = torch.zeros(6, requires_grad=True)
        target = torch.ones(6)
        dsn = DSN([x], lr=1e-1, builder=KrylovBuilder(k_max=0, m_recycle=0))
        sched = torch.optim.lr_scheduler.StepLR(dsn, step_size=1, gamma=0.1)
        for _ in range(n_scheduler_steps):
            dsn.step(lambda: (x - target).pow(2).sum())
            sched.step()
        before = x.detach().clone()
        dsn.step(lambda: (x - target).pow(2).sum())
        return float((x.detach() - before).norm()), dsn.param_groups[0]["lr"]

    undecayed, lr0 = step_norm_after(0)
    decayed, lr3 = step_norm_after(3)

    assert (lr0, lr3) == pytest.approx((1e-1, 1e-4))
    # AdamW's normalized step is ~lr in magnitude, so a 1000x lr decay must
    # show up as a ~1000x smaller step. Anything reading a stale self.lr would
    # give a ratio of 1.
    assert decayed < undecayed / 100, f"{decayed=} {undecayed=}"


def test_state_dict_round_trips_and_resumes_identically():
    """The base class serializes none of DSN's state, and it looked like it worked.

    Before Plan 2, `state_dict()` returned `{'state': {}, 'param_groups': [...]}`
    with no AdamW moments, no trust radius and no recycled basis, so a "resumed"
    run silently restarted from zero: measured `adam.t = 0` (was 5) and
    `trust_radius = 1.0` (was 16.0), with the next step diverging from the
    un-resumed run by 0.134 in parameter space. Assert on the resumed
    *trajectory*, not on the attributes, so a field added later that is not
    checkpointed still fails this.
    """
    n = 10

    def fresh():
        A, b, x = quadratic_problem(n)
        dsn = DSN([x], lr=1e-2, weight_decay=0.0, trust_radius=1.0,
                  builder=KrylovBuilder(k_max=15, m_recycle=4, tau=1e-3))
        return A, b, x, dsn

    A, b, x, dsn = fresh()
    for _ in range(5):
        dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
    sd = dsn.state_dict()
    mid = x.detach().clone()

    # Keep running the original for 3 more steps: the reference trajectory.
    for _ in range(3):
        dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
    reference = x.detach().clone()

    # A brand-new optimizer, loaded from the checkpoint, must reproduce it.
    x2 = mid.clone().requires_grad_(True)
    dsn2 = DSN([x2], lr=1e-2, weight_decay=0.0, trust_radius=1.0,
               builder=KrylovBuilder(k_max=15, m_recycle=4, tau=1e-3))
    dsn2.load_state_dict(sd)

    assert dsn2.adam.t == 5
    assert dsn2.trust_radius == sd["dsn"]["trust_radius"]

    for _ in range(3):
        dsn2.step(lambda: 0.5 * x2 @ A @ x2 + b @ x2)

    torch.testing.assert_close(x2.detach(), reference, rtol=0, atol=1e-12)


def test_state_dict_round_trip_preserves_the_recycled_basis():
    n = 25
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-2, trust_radius=1.0,
              builder=KrylovBuilder(k_max=18, m_recycle=4, tau=1e-6))
    for _ in range(3):
        dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
    assert dsn.builder.U is not None, "setup: recycling must have engaged"

    x2 = x.detach().clone().requires_grad_(True)
    dsn2 = DSN([x2], lr=1e-2, trust_radius=1.0,
               builder=KrylovBuilder(k_max=18, m_recycle=4, tau=1e-6))
    dsn2.load_state_dict(dsn.state_dict())

    torch.testing.assert_close(dsn2.builder.U, dsn.builder.U, rtol=0, atol=0)


def test_weight_decay_reaches_the_param_group():
    """`weight_decay` must live in the group, where schedulers and state_dict see it."""
    x = torch.zeros(4, requires_grad=True)
    dsn = DSN([x], lr=1e-2, weight_decay=0.1)

    assert dsn.param_groups[0]["weight_decay"] == 0.1
    assert dsn.state_dict()["param_groups"][0]["weight_decay"] == 0.1

    # And rewriting the group must actually change the behavior, which is the
    # part that silently did nothing when the value was an optimizer attribute.
    p = torch.ones(4, requires_grad=True)
    opt = DSN([p], lr=1e-1, weight_decay=0.0,
              builder=KrylovBuilder(k_max=0, m_recycle=0))
    opt.param_groups[0]["weight_decay"] = 0.5
    opt.step(lambda: (p**2).sum() * 0.0)     # zero gradient: decay only

    torch.testing.assert_close(
        p.detach(), torch.full((4,), 1.0 - 1e-1 * 0.5), rtol=0, atol=1e-12
    )


def test_multiple_param_groups_get_their_own_lr_and_weight_decay():
    """All parameters are optimized as one flat vector, but per-group
    hyperparameters must still be honored. Before Plan 2 every group used
    `param_groups[0]["lr"]` and a single optimizer-level `weight_decay`."""
    a = torch.zeros(5, requires_grad=True)
    c = torch.zeros(5, requires_grad=True)
    target = torch.ones(5)

    dsn = DSN(
        [{"params": [a], "lr": 1e-1}, {"params": [c], "lr": 1e-3}],
        builder=KrylovBuilder(k_max=0, m_recycle=0),   # pure AdamW complement
    )
    dsn.step(lambda: (a - target).pow(2).sum() + (c - target).pow(2).sum())

    # AdamW's normalized step is ~lr in magnitude, so a 100x lr gap must show up
    # as a ~100x step-size gap between the groups.
    step_a, step_c = float(a.detach().norm()), float(c.detach().norm())
    assert step_a > 50 * step_c, f"{step_a=} {step_c=}"

    # Per-group weight decay, on a group that also has its own lr.
    p = torch.ones(3, requires_grad=True)
    q = torch.ones(3, requires_grad=True)
    opt = DSN(
        [{"params": [p], "weight_decay": 0.5}, {"params": [q], "weight_decay": 0.0}],
        lr=1e-1, builder=KrylovBuilder(k_max=0, m_recycle=0),
    )
    opt.step(lambda: ((p**2).sum() + (q**2).sum()) * 0.0)

    torch.testing.assert_close(
        p.detach(), torch.full((3,), 1.0 - 1e-1 * 0.5), rtol=0, atol=1e-12
    )
    torch.testing.assert_close(q.detach(), torch.ones(3), rtol=0, atol=1e-12)


def test_add_param_group_extends_the_complement_state():
    x = torch.zeros(4, requires_grad=True)
    dsn = DSN([x], lr=1e-2, builder=KrylovBuilder(k_max=0, m_recycle=0))
    dsn.step(lambda: (x**2).sum())

    y = torch.zeros(6, requires_grad=True)
    dsn.add_param_group({"params": [y], "lr": 1e-2})

    assert dsn.adam.m.numel() == 10
    assert dsn.builder.U is None
    dsn.step(lambda: (x**2).sum() + (y**2).sum())
    assert torch.isfinite(y).all()


def test_step_without_a_closure_raises_rather_than_mis_stepping():
    """The signature matches `Optimizer.step(closure=None)`, but DSN cannot form
    a curvature-vector product from a pre-populated `.grad`, so the omission is
    reported instead of silently taking a different step."""
    x = torch.zeros(3, requires_grad=True)
    dsn = DSN([x], lr=1e-2)

    (x**2).sum().backward()
    with pytest.raises(ValueError, match="requires a closure"):
        dsn.step()


def test_beats_adamw_on_an_ill_conditioned_quadratic():
    """Second order should win decisively where first order is known to struggle."""
    n = 20
    A, b, _ = quadratic_problem(n, cond=1e4)

    x_a = torch.zeros(n, requires_grad=True)
    x_d = torch.zeros(n, requires_grad=True)
    adam = torch.optim.AdamW([x_a], lr=1e-1, weight_decay=0.0)
    dsn = DSN([x_d], lr=1e-1, weight_decay=0.0, trust_radius=1e3,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-6))

    for _ in range(30):
        adam.zero_grad()
        (0.5 * x_a @ A @ x_a + b @ x_a).backward()
        adam.step()
        dsn.step(lambda: 0.5 * x_d @ A @ x_d + b @ x_d)

    f = lambda z: 0.5 * z @ A @ z + b @ z
    assert f(x_d.detach()) < f(x_a.detach())
