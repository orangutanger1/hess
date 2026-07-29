"""Candidate optimizer for the autoresearch loop.

This is the ONLY file a future agent session should edit as part of the
optimizer search driven by scripts/run_experiment.py. Every other piece of
the loop (model, data, eval, timeout/divergence handling, keep/discard
logic, logging) lives outside this file and is not meant to change per
candidate.

Experiment 3 (20260729): Shampoo-lite with Adam grafting.

Experiment 2 tried the same Kronecker-factored whitening below but applied
the raw preconditioned step at the AdamW baseline's lr (1e-3) and diverged
(mean_loss 373 vs baseline 0.035) -- L^-1/4 G R^-1/4 is not scaled the way
Adam's per-element-normalized update is, so the same lr is wildly too large.
This version grafts: keep the Shampoo *direction* (from the accumulated,
noise-aware Kronecker factors -- still the favored direction per the seed
knowledge) but rescale each tensor's step to match the norm the diagonal
Adam accumulator would have taken, so the step *magnitude* is known-good and
only the *direction* is the thing being tested.
"""

import torch


class CandidateOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, precond_update_freq=10):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         precond_update_freq=precond_update_freq)
        super().__init__(params, defaults)

    @staticmethod
    def _inv_root(mat, power, eps):
        dim = mat.shape[0]
        eye = torch.eye(dim, device=mat.device, dtype=mat.dtype)
        evals, evecs = torch.linalg.eigh(mat + eps * eye)
        evals = evals.clamp_min(eps)
        inv_root_evals = evals.pow(-1.0 / power)
        return (evecs * inv_root_evals.unsqueeze(0)) @ evecs.T

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            update_freq = group["precond_update_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CandidateOptimizer does not support sparse gradients")

                state = self.state[p]

                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if p.dim() == 2:
                        out_dim, in_dim = p.shape
                        state["L"] = torch.zeros(out_dim, out_dim, device=p.device, dtype=p.dtype)
                        state["R"] = torch.zeros(in_dim, in_dim, device=p.device, dtype=p.dtype)
                        state["L_inv_root"] = torch.eye(out_dim, device=p.device, dtype=p.dtype)
                        state["R_inv_root"] = torch.eye(in_dim, device=p.device, dtype=p.dtype)
                        state["shampoo_exp_avg"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                adam_update = exp_avg / denom * step_size  # the norm we graft onto

                if p.dim() == 2:
                    state["L"].mul_(beta2).add_(grad @ grad.T, alpha=1 - beta2)
                    state["R"].mul_(beta2).add_(grad.T @ grad, alpha=1 - beta2)

                    if step == 1 or step % update_freq == 0:
                        state["L_inv_root"] = self._inv_root(state["L"], 4.0, eps)
                        state["R_inv_root"] = self._inv_root(state["R"], 4.0, eps)

                    precond_grad = state["L_inv_root"] @ grad @ state["R_inv_root"]

                    shampoo_exp_avg = state["shampoo_exp_avg"]
                    shampoo_exp_avg.mul_(beta1).add_(precond_grad, alpha=1 - beta1)

                    shampoo_norm = shampoo_exp_avg.norm().clamp_min(eps)
                    adam_norm = adam_update.norm()
                    graft_direction = shampoo_exp_avg / shampoo_norm
                    p.add_(graft_direction, alpha=-float(adam_norm))
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
