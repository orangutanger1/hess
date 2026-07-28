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


def test_lagged_rho_accounts_for_trust_region_clipping():
    """The lagged rho must be computed against the *clipped* step's predicted
    reduction, or the trust region enters a self-reinforcing shrink spiral: a
    small clip factor `s` produces a small rho (rho ~ 2s - s**2 for an
    unscaled prediction), which shrinks the trust radius further, producing an
    even smaller `s` next step, with no path back out since growth requires
    rho > 0.75.

    On an exact quadratic, a correctly-scaled prediction matches the clipped
    step's true reduction exactly, so rho must be ~1 every step and the trust
    radius must not collapse toward zero. Uses the same configuration as
    ``test_trust_radius_clips_a_large_subspace_step`` (a trust radius far
    below the Newton step norm, so clipping is active from the first step),
    run for multiple steps instead of one so the trust-region *trajectory* is
    observable.
    """
    n = 10
    A, b, x = quadratic_problem(n)
    dsn = DSN([x], lr=1e-3, trust_radius=1e-3,
              builder=KrylovBuilder(k_max=n, m_recycle=0, tau=0.0, damping=0.0))

    rhos = []
    # 12 steps, not more: step 12 is a cliff. Step 10 (0-indexed) is the first
    # step whose Newton step fits inside the grown trust radius, and on an
    # exact quadratic it lands on x* exactly, so step 11 sees ||g|| == 0
    # (measured 1.3e-15). The builder then returns k=0, `predicted` is 0.0, and
    # a 13th iteration would read `predicted > 0` as False and report
    # rho = nan -- failing for a reason unrelated to prediction scaling, the
    # property under test here.
    for _ in range(12):
        dsn.step(lambda: 0.5 * x @ A @ x + b @ x)
        rhos.append(dsn.telemetry.rho)

    # The first step has no lagged prediction yet (rho is nan); every step
    # after that is a clipped Newton step on an exact quadratic, so rho must
    # be ~1, not collapsing toward 0 as an unscaled prediction would produce.
    for rho in rhos[1:]:
        assert abs(rho - 1.0) < 1e-6, f"rho={rho} indicates a mis-scaled prediction"

    # A correctly-scaled prediction must not send the trust radius into a
    # self-reinforcing shrink spiral toward zero.
    assert dsn.trust_radius >= 1e-3


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
