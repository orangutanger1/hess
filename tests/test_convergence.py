import torch

from dsn import DSN
from dsn.subspace import KrylovBuilder


def ill_conditioned_logistic(n_samples=200, d=25, cond=100.0, seed=0):
    torch.manual_seed(seed)
    scale = torch.logspace(0, torch.log10(torch.tensor(cond)), d)
    X = torch.randn(n_samples, d) * scale
    w_true = torch.randn(d)
    y = (X @ w_true > 0).to(torch.get_default_dtype())
    return X, y


def logistic_loss(w, X, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y)


def test_dsn_reaches_lower_loss_than_adamw_on_logistic_regression():
    X, y = ill_conditioned_logistic()
    d = X.shape[1]

    w_a = torch.zeros(d, requires_grad=True)
    w_d = torch.zeros(d, requires_grad=True)

    adam = torch.optim.AdamW([w_a], lr=1e-1, weight_decay=0.0)
    dsn = DSN([w_d], lr=1e-1, weight_decay=0.0, damping=1e-4, trust_radius=10.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    for _ in range(60):
        adam.zero_grad()
        logistic_loss(w_a, X, y).backward()
        adam.step()
        dsn.step(lambda: logistic_loss(w_d, X, y))

    with torch.no_grad():
        assert logistic_loss(w_d, X, y) < logistic_loss(w_a, X, y)


def test_recycling_stays_fresh_rather_than_going_stale():
    """Risk 1 from the spec: reuse high while residual climbs means staleness."""
    X, y = ill_conditioned_logistic()
    w = torch.zeros(X.shape[1], requires_grad=True)
    dsn = DSN([w], lr=1e-1, weight_decay=0.0, damping=1e-4, trust_radius=10.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    residuals, reuse = [], []
    for _ in range(40):
        dsn.step(lambda: logistic_loss(w, X, y))
        residuals.append(dsn.telemetry.rel_residual)
        reuse.append(dsn.telemetry.reuse_frac)

    assert sum(reuse[10:]) / len(reuse[10:]) > 0.2
    assert sum(residuals[-10:]) / 10 <= sum(residuals[:10]) / 10 + 0.5


def test_trust_region_does_not_collapse():
    """Risk 2 from the spec: complement fighting the trust region."""
    X, y = ill_conditioned_logistic()
    w = torch.zeros(X.shape[1], requires_grad=True)
    dsn = DSN([w], lr=1e-1, weight_decay=0.0, damping=1e-4, trust_radius=1.0,
              builder=KrylovBuilder(k_max=10, m_recycle=5, tau=1e-2, damping=1e-4))

    for _ in range(40):
        dsn.step(lambda: logistic_loss(w, X, y))

    assert dsn.telemetry.n_shrink < 30
    assert dsn.telemetry.trust_radius > 1e-6


def test_trains_a_small_mlp_on_synthetic_data():
    """No network access: synthetic classification, not MNIST."""
    torch.manual_seed(0)
    X = torch.randn(128, 10)
    y = (X[:, 0] * X[:, 1] > 0).long()
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)
    )
    dsn = DSN(model.parameters(), lr=1e-2, weight_decay=0.0,
              builder=KrylovBuilder(k_max=6, m_recycle=3, tau=0.1))

    def closure():
        return torch.nn.functional.cross_entropy(model(X), y)

    first = float(closure())
    for _ in range(50):
        dsn.step(closure)
    last = float(closure())

    assert last < first
    assert dsn.telemetry.n_fallback == 0
