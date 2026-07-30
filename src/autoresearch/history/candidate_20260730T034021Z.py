"""Candidate optimizer for the autoresearch loop.

This is the ONLY file a future agent session should edit as part of the
optimizer search driven by scripts/run_experiment.py. Every other piece of
the loop (model, data, eval, timeout/divergence handling, keep/discard
logic, logging) lives outside this file and is not meant to change per
candidate.

Experiment 4 (20260729): Shampoo-lite + Adam grafting + bias-corrected L/R.
Experiment 6 (20260729): relative eigenvalue damping instead of absolute.
Experiment 11 (20260729): extend the Kronecker whitening to conv weights.
Correctness fix (20260729, fe414a6): pre-eigh stabilization epsilon.
These are the committed history up to mean_loss 0.0124 (mnist, old metric) /
0.6509 (cifar, old metric) -- see src/autoresearch/history/ snapshots and
research_log.md for the full trail; not repeated here since none of it
carries forward into this experiment's design.

Harness fix (20260730, see research_log.md): the measurement noise floor
was root-caused (wall-clock step-count jitter landing on different
minibatches in a trailing-train-batch-average metric, NOT CPU BLAS
nondeterminism as first hypothesized) and fixed in scripts/run_experiment.py
(eval-based metric, 4 seeds, 5% noise-margin gate). Both chains re-baselined
against byte-identical candidate code: mnist 0.0124 -> 0.1284, cifar
0.6733 -> 1.1730 (the eval metric reads different values than the old
trailing-train-batch metric -- not regressions).

Experiment 12 (20260730): real curvature via Hessian-vector product
(dsn.curvature.hvp, enabled by also giving the harness a real closure --
opt.step(closure) instead of loss.backward()+bare opt.step(), mirroring
sanity_mnist.py's own DSN loop -- kept regardless of this result, since it's
a reusable capability, not tied to any one candidate). Replaced Adam-norm
grafting's step-length heuristic with the Newton-optimal scalar (g.v)/(v.H.v)
along the joint Shampoo direction (all dim-2/4 params sharing one global
step length, trust-capped at 3x the aggregate Adam-implied norm after an
uncapped first attempt diverged past loss 50 -- same lesson DSN's own trust
region already encodes: a raw Newton step isn't trustworthy far from where
the local quadratic model is accurate). Discarded: mean_loss 0.4682 vs
baseline 0.1284, clearly worse and not a step-count artifact (2531-2712
steps vs baseline's ~3030-3071, only ~15-20% fewer). Forcing every whitened
tensor to share one global Newton scalar is likely too coarse across layers
of very different scale (784x96 vs 96x10) -- per-tensor Newton scaling is
the natural follow-up if a future attempt wants to revisit HVP-based scaling,
though its added per-tensor double-backward cost would cut step count
further.

Experiment 13 (20260730): Muon (Newton-Schulz orthogonalization), no
covariance accumulation at all.

Blind spot this targets: the seed knowledge in program.md says accumulated,
noise-aware statistics (Shampoo/Adagrad-style, whitening by gradient
covariance built up over many steps) are the theoretically favored
direction over one-shot Hessian estimation -- and every committed candidate
on this branch (experiments 3/4/6/11/12) has taken that seriously, going
further into accumulation (bias correction, relative damping, conv support,
now even real curvature). None of them have questioned the OTHER premise
those methods share: that per-element or per-layer covariance-based
*scaling* is what a good preconditioner should be doing at all. Muon starts
from a different assumption entirely -- that what matters is
*orthogonalizing* the update (equalizing its singular values via the matrix
sign function) rather than reweighting it by any accumulated second-moment
statistic, noise-aware or not. It carries no L/R covariance state, no eigh,
no bias-corrected EMA of a preconditioner -- just a momentum buffer and a
handful of matmuls (Newton-Schulz quintic iteration) computed fresh every
step. This is a real structural alternative to the whole Shampoo-lite
lineage above, not a variant of it, and per program.md it was "reasoned
about but never implemented as its own candidate" until now.

Implementation follows the public reference (Jordan et al., "Muon: An
optimizer for the hidden layers of neural networks"): a nesterov-style
momentum lookahead (grad + momentum*buf) is orthogonalized via 5 steps of
the quintic Newton-Schulz iteration (X <- aX + (bA + cA^2)X, A = XX^T,
tuned coefficients a=3.4445, b=-4.7750, c=2.0315 for fast convergence to the
semi-orthogonal polar factor), then scaled by max(1, out_dim/in_dim)^0.5 (a
correction so the update's spectral norm doesn't shrink for rectangular
layers) and applied at the harness's fixed lr (1e-2) with no extra
multiplier for this first test -- deliberately the most direct, least-tuned
version, so a loss/win here is attributable to the orthogonalization idea
itself rather than a hand-picked scale. 1D/bias params keep the same
bias-corrected diagonal Adam path every prior candidate used (Muon's own
reference implementation likewise reserves a separate optimizer for
non-matrix params); this candidate isn't claiming anything new for those.
"""

import torch


def _newton_schulz5(G, steps=5, eps=1e-7):
    """Approximate msign(G) -- the nearest semi-orthogonal matrix to G --
    via the quintic Newton-Schulz iteration (Jordan et al.'s Muon). No
    eigendecomposition: a handful of matmuls, cheap relative to the
    Kronecker-whitening path's per-tensor eigh calls above.
    """
    assert G.dim() == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class CandidateOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, muon_momentum=0.95, ns_steps=5):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         muon_momentum=muon_momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError(
                "CandidateOptimizer.step() requires a closure (a zero-arg "
                "callable returning the loss) -- the harness always "
                "supplies one."
            )

        all_params = [p for group in self.param_groups for p in group["params"]]
        loss = closure()
        raw_grads = torch.autograd.grad(loss, all_params, create_graph=False, allow_unused=True)
        grad_map = {id(p): g for p, g in zip(all_params, raw_grads)}

        with torch.no_grad():
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                momentum = group["muon_momentum"]
                ns_steps = group["ns_steps"]

                for p in group["params"]:
                    grad = grad_map.get(id(p))
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("CandidateOptimizer does not support sparse gradients")

                    if weight_decay != 0.0:
                        p.mul_(1 - lr * weight_decay)

                    state = self.state[p]

                    if p.dim() in (2, 4):
                        if "muon_buf" not in state:
                            state["muon_buf"] = torch.zeros_like(p)
                        buf = state["muon_buf"]
                        buf.mul_(momentum).add_(grad)
                        lookahead = grad.add(buf, alpha=momentum)  # nesterov-style

                        out_dim = p.shape[0]
                        g2d = lookahead.reshape(out_dim, -1)
                        in_dim = g2d.shape[1]
                        o2d = _newton_schulz5(g2d, steps=ns_steps)
                        o = o2d.reshape(p.shape)

                        scale = max(1.0, out_dim / in_dim) ** 0.5
                        p.add_(o, alpha=-lr * scale)
                    else:
                        if "step" not in state:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] += 1
                        step = state["step"]

                        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step
                        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                        step_size = lr / bias_correction1
                        p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
