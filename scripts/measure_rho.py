"""Measure rho = (lambda_Sig / lambda_g)^2 along DSN's actual MNIST trajectory.

Measurement only -- this script trains nothing new and changes nothing in the
library. It runs the exact configuration of `scripts/sanity_mnist.py` (seed 0,
batch 256, lr 1e-2, KrylovBuilder(k_max=8, m_recycle=4, tau=0.1)) and pauses at
six checkpoints to log two scalars per checkpoint.

Definitions (taken as given from the P0 excerpt):

    lambda_g   = ||g||^2 / (g' M^-1 g)
    lambda_Sig = sqrt( tr(M Sigma) / tr(M^-1 Sigma) )
    rho        = (lambda_Sig / lambda_g)^2

Which operator M?
-----------------
The excerpt writes H, and assumes H^-1 exists. The exact Hessian of this MLP is
indefinite (measured: a substantial fraction of Ritz values are negative at
every checkpoint), so H^-1 is not the operator anything here would precondition
with, and `g' H^-1 g` can be negative. DSN does not invert H either: it inverts
the saddle-free damped operator

    M = |H| + delta

(see `solve.subspace_newton`, `damping=1e-3` by default). M is what "curvature
preconditioning" actually means for this code, so M is what is measured. The
undamped indefinite variant is reported alongside only as a diagnostic
(`neg_frac`, the g^2-weighted mass on negative curvature).

Estimator
---------
Both quadratic forms come from the same Lanczos run per vector -- no new solver
code, `lanczos_iter` from src/dsn/lanczos.py plus an eigh of its tridiagonal,
exactly as `solve.subspace_newton` already does. For a seed vector b with
q1 = b/||b||, the k-step Gauss quadrature rule is

    b' f(H) b  ~=  ||b||^2 * sum_j V[0,j]^2 * f(lam_j)

with (lam, V) = eigh(T_k). One Lanczos run therefore yields b'Mb and b'M^-1b
together, for any delta, at any truncation k' <= k -- so the delta- and
k-sensitivity tables below are free.

Sigma and cross-fitting
-----------------------
Sigma is the covariance of the mini-batch gradient. It is estimated from 16
independent microbatches (size 128) drawn from a pool disjoint from the batch
that defines H and g, and it is never centered on a mean computed from the same
samples it is then evaluated against: the microbatches are split into halves A
and B and only cross-half differences are used,

    u_i = (g_{A_i} - g_{B_i}) / sqrt(2),      E[u_i u_i'] = Sigma_128 exactly,

so tr(M Sigma) ~= mean_i u_i' M u_i is unbiased with no self-referential mean
(the F8/F10 failure mode: a metric computed from the same object it evaluates).
Sigma's overall scale cancels in lambda_Sig, so the microbatch size does not
matter -- only Sigma's shape does.

Usage:  python scripts/measure_rho.py [--steps 200] [--k 24]
"""

import argparse
import json
import time

import torch
from torchvision import datasets, transforms

from dsn import DSN
from dsn.curvature import hvp
from dsn.lanczos import lanczos_iter
from dsn.subspace import KrylovBuilder
from dsn.subspace.krylov import _tridiagonal
from dsn.utils import flat_grad


def build_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 96),
        torch.nn.Tanh(),
        torch.nn.Linear(96, 10),
    )


def quadrature(matvec, b, ks):
    """Gauss-quadrature rules for b' f(H) b at each Lanczos depth in ``ks``.

    A depth-j rule must come from an eigh of the j-by-j tridiagonal T_j, not
    from a subset of the depth-k rule's nodes: eigh returns nodes in ascending
    eigenvalue order, so slicing them keeps the most-negative curvature and
    discards the sharpest directions -- which is the opposite of a shallower
    Krylov space. Building T_j as the recurrence yields it costs no extra
    curvature products.

    Returns {j: (nodes, weights)} plus ||b||^2, with sum(weights) == 1.
    """
    want = sorted(set(ks))
    rules, seen = {}, 0
    for st in lanczos_iter(matvec, b, max(want)):
        seen = st.n_hvp
        if seen in want:
            T = _tridiagonal(st.alphas, st.betas)
            lam, V = torch.linalg.eigh(0.5 * (T + T.T))
            rules[seen] = (lam.double(), (V[0, :] ** 2).double())
    if not rules:
        raise RuntimeError("Lanczos broke down immediately: ||b|| below tol")
    # Krylov breakdown before the deepest requested rule: the space is
    # exhausted, so the deepest available rule is exact. Reuse it rather than
    # silently reporting a missing depth.
    for j in want:
        if j not in rules:
            rules[j] = rules[seen]
    return rules, float(b.dot(b))


def form(rules, nb2, delta, k, inverse=False):
    """nb2 * sum_j w_j f(node_j) for f = (|lam|+delta)^{+-1} at depth k."""
    nodes, weights = rules[k]
    f = nodes.abs() + delta
    return nb2 * float((weights * (1.0 / f if inverse else f)).sum())


def measure(model, params, x_c, y_c, micro, k, deltas):
    """One checkpoint: returns the scalars for every (delta, k) in the sweep."""
    def closure():
        return torch.nn.functional.cross_entropy(model(x_c), y_c)

    def matvec(v):
        return hvp(closure, params, v)

    ks = [k // 4, k // 2, 3 * k // 4, k]
    g = flat_grad(closure(), params, create_graph=False).detach()
    gq = quadrature(matvec, g, ks)

    # Cross-half paired differences: half A is micro[:h], half B is micro[h:].
    h = len(micro) // 2
    uq = [
        quadrature(matvec, (micro[i] - micro[h + i]) / (2.0 ** 0.5), ks)
        for i in range(h)
    ]

    neg_frac = float((gq[0][k][1][gq[0][k][0] < 0]).sum())
    out = {"gnorm": float(g.norm()), "neg_frac": neg_frac, "rows": []}
    for delta in deltas:
        for kk in ks:
            gHig = form(*gq, delta, kk, inverse=True)
            lam_g = float(g.dot(g)) / gHig
            trMS = sum(form(*q, delta, kk) for q in uq) / h
            trMiS = sum(form(*q, delta, kk, inverse=True) for q in uq) / h
            lam_S = (trMS / trMiS) ** 0.5
            # Split-half stability: same statistic on each half of the pairs.
            halves = []
            for sl in (slice(0, h // 2), slice(h // 2, h)):
                a = sum(form(*q, delta, kk) for q in uq[sl]) / len(uq[sl])
                b = sum(form(*q, delta, kk, inverse=True) for q in uq[sl]) / len(uq[sl])
                halves.append(((a / b) ** 0.5 / lam_g) ** 2)
            out["rows"].append({
                "delta": delta, "k": kk,
                "lam_g": lam_g, "lam_Sig": lam_S,
                "gHig": gHig, "trMSig": trMS, "trMiSig": trMiS,
                "rho": (lam_S / lam_g) ** 2,
                "rho_halves": halves,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=24, help="Lanczos depth per vector")
    ap.add_argument("--curv-batch", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=16, help="microbatches for Sigma")
    ap.add_argument("--micro-batch", type=int, default=128)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    ds = datasets.MNIST("data", train=True, download=True,
                        transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Fixed, disjoint pools. The curvature batch defines H and g; the microbatch
    # pool defines Sigma and shares no sample with it.
    n_micro_pool = args.micro * args.micro_batch
    flat = torch.stack([ds[i][0] for i in range(args.curv_batch + n_micro_pool)])
    labels = torch.tensor([ds[i][1] for i in range(args.curv_batch + n_micro_pool)])
    x_c, y_c = flat[: args.curv_batch], labels[: args.curv_batch]
    x_m, y_m = flat[args.curv_batch:], labels[args.curv_batch:]

    model = build_model()
    params = list(model.parameters())
    opt = DSN(model.parameters(), lr=args.lr, weight_decay=0.0,
              builder=KrylovBuilder(k_max=8, m_recycle=4, tau=0.1))

    checkpoints = [0, 40, 80, 120, 160, args.steps - 1]
    deltas = [1e-3, 1e-2, 1e-1]
    results = []

    t0 = time.time()
    it = iter(loader)
    for step in range(args.steps):
        if step in checkpoints:
            micro = []
            for j in range(args.micro):
                sl = slice(j * args.micro_batch, (j + 1) * args.micro_batch)
                lo = torch.nn.functional.cross_entropy(model(x_m[sl]), y_m[sl])
                micro.append(flat_grad(lo, params).detach())
            r = measure(model, params, x_c, y_c, micro, args.k, deltas)
            with torch.no_grad():
                r["step"] = step
                r["loss_curv"] = float(
                    torch.nn.functional.cross_entropy(model(x_c), y_c)
                )
            results.append(r)
            base = [q for q in r["rows"] if q["delta"] == 1e-3 and q["k"] == args.k][0]
            print(f"[ckpt {step:4d}] loss {r['loss_curv']:.4f} "
                  f"|g| {r['gnorm']:.4f} neg_frac {r['neg_frac']:.3f} "
                  f"lam_g {base['lam_g']:.4g} lam_Sig {base['lam_Sig']:.4g} "
                  f"rho {base['rho']:.4g}  ({time.time() - t0:.0f}s)")

        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        def train_closure():
            return torch.nn.functional.cross_entropy(model(x), y)

        loss = opt.step(train_closure)
        if step % 40 == 0:
            print(f"  step {step:4d} train loss {float(loss.detach()):.4f} "
                  f"k={opt.telemetry.k} tr={opt.telemetry.trust_radius:.3g}")

    print(f"\ndone in {time.time() - t0:.0f}s")
    print(f"{'step':>6} {'delta':>8} {'k':>4} {'lam_g':>12} {'lam_Sig':>12} {'rho':>10}")
    for r in results:
        for q in r["rows"]:
            print(f"{r['step']:6d} {q['delta']:8.0e} {q['k']:4d} "
                  f"{q['lam_g']:12.5g} {q['lam_Sig']:12.5g} {q['rho']:10.4g}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
