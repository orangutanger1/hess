import torch

from dsn.complement import AdamWState, project_out


def test_matches_torch_adamw_over_many_steps():
    """Guards bias correction, epsilon placement and decoupled weight decay."""
    torch.manual_seed(0)
    p_ref = torch.randn(10, requires_grad=True)
    p_ours = p_ref.detach().clone()

    lr, wd = 1e-2, 0.0
    opt = torch.optim.AdamW([p_ref], lr=lr, weight_decay=wd, eps=1e-8)
    state = AdamWState(10)

    for _ in range(25):
        g = torch.randn(10)
        opt.zero_grad()
        p_ref.grad = g.clone()
        opt.step()
        p_ours = p_ours + state.step(g, lr)

    torch.testing.assert_close(p_ours, p_ref.detach(), rtol=0, atol=1e-12)


def test_first_step_has_bias_correction_applied():
    """Without bias correction the first step would be far smaller than lr."""
    state = AdamWState(3)
    g = torch.full((3,), 2.0)

    step = state.step(g, lr=0.1)

    torch.testing.assert_close(step, torch.full((3,), -0.1), rtol=1e-6, atol=1e-9)


def test_project_out_leaves_result_orthogonal_to_basis():
    n = 20
    W, _ = torch.linalg.qr(torch.randn(n, 4))
    d = torch.randn(n)

    out = project_out(d, W)

    torch.testing.assert_close(W.T @ out, torch.zeros(4), atol=1e-12, rtol=0)


def test_project_out_is_identity_for_empty_basis():
    d = torch.randn(7)
    torch.testing.assert_close(project_out(d, d.new_zeros(7, 0)), d)


def test_project_out_removes_only_the_in_span_component():
    n = 15
    W, _ = torch.linalg.qr(torch.randn(n, 3))
    inside = W @ torch.randn(3)
    outside = project_out(torch.randn(n), W)

    torch.testing.assert_close(project_out(inside + outside, W), outside,
                               atol=1e-10, rtol=0)
