"""Measure the P4 (FACA) break-even gate's inputs on DSN's real MNIST trajectory.

Measurement only -- trains nothing new, changes nothing in the library. It runs
DSN on the exact `scripts/sanity_mnist.py` configuration and, at a handful of
consecutive-step pairs (t, t+1), logs every quantity the gate is built from.

Quantities (formulas taken as given from the P4 excerpt)
-------------------------------------------------------
    s          = theta_{t+1} - theta_t                      (free)
    y_t        = g_{t+1} - g_t                              (free)
    w          = 2 * (y_t - H_t s)                          (1 extra HVP)
    delta_hat  = ||w|| / ||s||^2                            (estimate of L_H)
    |Delta_H|  = ||w|| / ||s||   = delta_hat * ||s||        (curvature units)

`|Delta_H|` is what the break-even inequality compares, because its right-hand
side `epsilon_K(k) / ||y_U||` is gradient-units over length-units, i.e.
curvature units. `delta_hat` is reported alongside as the L_H estimate the
excerpt names.

Two secant pairs are formed at every t, differing only in which batch the second
gradient is evaluated on:

    diff   g_{t+1} on batch B_{t+1}   -- what consecutive DSN steps actually see
    same   g_{t+1} on batch B_t       -- the excerpt's proposed fix, one extra
                                         gradient on a common curvature batch

Both are differenced against the same g_t and the same H_t s, so the gap between
them is batch resampling and nothing else.

    sigma_batch = rms_r || H_{B_r} s - mean_r H_{B_r} s || / ||s||

over `--sigma-batches` independently resampled batches at the frozen theta_t.
Same units as |Delta_H|, so the two are directly comparable. `sigma_vs_ref` is
the same spread measured against a large reference batch instead of the
resample mean, so a systematic offset of the training batch size is visible too.

    rho_K   = epsilon_K(k+1) / epsilon_K(k)

from the Lanczos residual history at step t+1 -- the *true* residual
||H (W y) + g|| / ||g|| that `KrylovBuilder` already computes per direction, not
an in-span estimate. Reported at the k where the builder actually stopped. NOT
the P0 regime rho measured by `scripts/measure_rho.py`.

    m_eff   = sum_i || (I - Q Q') u_i ||^2

for U the basis the builder carried into step t+1 and Q an undeflated fresh
Krylov basis for (H_{t+1}, g_{t+1}) of the same depth: the mass of the previous
step's basis lying outside the current fresh Krylov space. U is orthonormal, so
each term is in [0, 1] and m_eff <= m. When the builder carried nothing, m_eff
is 0 by definition and the gate's right-hand side is negative.

    ||y_U||  = norm of the recycled block of the subspace solution at step t+1.

Gate:   |Delta_H|  <  (m_eff - 1) * (1 - rho_K) * epsilon_K(k) / ||y_U||

Configurations
--------------
`--k-max 8` is the shipped MNIST configuration, where no Ritz pair converges and
the builder recycles nothing. `--k-max 40` is run as well so the gate can also
be evaluated on steps where recycling *does* engage -- otherwise "m_eff is zero"
would be a statement about one hyperparameter rather than about the method.

Usage:  python scripts/measure_faca_gate.py [--steps 60] [--k-max 8]
"""

import argparse
import copy
import json
import time

import torch
from torchvision import datasets, transforms

from dsn import DSN
from dsn.curvature import hvp
from dsn.lanczos import lanczos_iter
from dsn.solve import newton_residual, subspace_newton
from dsn.subspace import KrylovBuilder
from dsn.utils import flat_grad


def build_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 96),
        torch.nn.Tanh(),
        torch.nn.Linear(96, 10),
    )


def get_flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params]).clone()


@torch.no_grad()
def set_flat(params, vec):
    i = 0
    for p in params:
        n = p.numel()
        p.copy_(vec[i : i + n].view_as(p))
        i += n


def batch_closure(model, x, y):
    def closure():
        return torch.nn.functional.cross_entropy(model(x), y)

    return closure


def build_with_history(builder, matvec, g, revalidate=True):
    """Replicate ``KrylovBuilder.build`` while recording the residual per depth.

    The builder returns only the depth it stopped at, but rho_K is a ratio of
    consecutive depths, so the history has to be kept. This mirrors
    `krylov.KrylovBuilder.build` -- refresh HU, re-validate convergence, deflate,
    assemble T, solve -- and returns the same subspace plus the epsilon
    sequence. It does not mutate `builder`; pass a copy.

    With ``revalidate=False`` the re-convergence check is skipped and every
    carried direction is deflated. That is not what the shipped builder does --
    it is what FACA's break-even inequality assumes, since the inequality is the
    thing that would decide whether to reuse. Measuring it is the only way to
    put a real number on ``||y_U||`` on a trajectory where the shipped gate
    throws the basis away first.

    Returns (eps_list, W, y, m, n_converged) with eps_list[j] the true relative
    residual after j+1 Lanczos directions and n_converged the number of carried
    directions that still pass the Ritz test against this step's operator.
    """
    from dsn.subspace.krylov import _tridiagonal  # local: measurement-only use

    gnorm = g.norm()
    m = builder.U.shape[1] if builder.U is not None else 0
    HU = None
    n_converged = 0
    if m and builder.k_max > 0:
        HU = torch.stack([matvec(builder.U[:, i]) for i in range(m)], dim=1)
        keep = builder._converged(builder.U, HU)
        n_converged = int(keep.sum())
        if revalidate and not bool(keep.all()):
            if not bool(keep.any()):
                builder.U = HU = None
                m = 0
            else:
                builder.U, HU = builder.U[:, keep], HU[:, keep]
                m = int(keep.sum())

    eps, W, y = [], None, None
    for st in lanczos_iter(matvec, g, builder.k_max, U=builder.U):
        W = st.Q if m == 0 else torch.cat([builder.U, st.Q], dim=1)
        HW = st.HQ if m == 0 else torch.cat([HU, st.HQ], dim=1)
        T = builder._assemble(st, m, HU)
        y, _, _ = subspace_newton(T, W.T @ g, builder.damping, builder.saddle_free)
        rel = float(newton_residual(HW, g, y) / gnorm)
        eps.append(rel)
        if rel <= builder.tau:
            break
    return eps, W, y, m, n_converged


def fresh_krylov_basis(matvec, g, k):
    """Undeflated Krylov basis for (H, g) of depth up to k."""
    Q = None
    for st in lanczos_iter(matvec, g, k):
        Q = st.Q
    return Q


def measure_pair(model, params, opt, batches, t, args, results):
    """Diagnostics for the consecutive-step pair (t, t+1). Assumes params are at
    theta_t and that `batches[t]` is about to be stepped on by the caller."""
    x_t, y_t_lab = batches[t]
    x_n, y_n_lab = batches[t + 1]
    cl_t = batch_closure(model, x_t, y_t_lab)
    cl_n = batch_closure(model, x_n, y_n_lab)

    theta_t = get_flat(params)
    g_t = flat_grad(cl_t(), params).detach()

    # ---- the step itself (unmodified DSN, on its own batch) ----------------
    U_before = opt.builder.U
    loss = float(opt.step(cl_t).detach())
    theta_n = get_flat(params)
    s = theta_n - theta_t
    U_carried = (
        None if opt.builder.U is None else opt.builder.U.detach().clone()
    )

    row = {
        "t": t,
        "loss": loss,
        "s_norm": float(s.norm()),
        "g_norm": float(g_t.norm()),
        "m_carried_in": 0 if U_before is None else int(U_before.shape[1]),
        "m_carried_out": 0 if U_carried is None else int(U_carried.shape[1]),
        "k": opt.telemetry.k,
        "reuse_frac": opt.telemetry.reuse_frac,
        "rejected": opt.telemetry.rho <= opt.trust_accept,
    }

    if row["s_norm"] == 0.0:
        results.append(row)
        return

    # ---- gradients at theta_{t+1}: different batch vs common batch --------
    g_n_diff = flat_grad(cl_n(), params).detach()
    g_n_same = flat_grad(cl_t(), params).detach()

    # ---- H_t s and the batch spread, both at the frozen theta_t -----------
    set_flat(params, theta_t)
    Hs = hvp(cl_t, params, s).detach()
    Hs_ref = hvp(batch_closure(model, args.x_ref, args.y_ref), params, s).detach()
    resampled = []
    for r in range(args.sigma_batches):
        idx = args.sigma_idx[r]
        resampled.append(
            hvp(batch_closure(model, args.x_pool[idx], args.y_pool[idx]),
                params, s).detach()
        )
    R = torch.stack(resampled, dim=1)
    mean_R = R.mean(dim=1, keepdim=True)
    sn = float(s.norm())
    row["sigma_batch"] = float(
        ((R - mean_R).norm(dim=0) ** 2).mean().sqrt() / sn
    )
    row["sigma_vs_ref"] = float(
        ((R - Hs_ref.unsqueeze(1)).norm(dim=0) ** 2).mean().sqrt() / sn
    )
    row["Hs_norm_over_s"] = float(Hs.norm() / sn)

    # Gradient noise across the same resampled batches. This is the term that
    # actually contaminates a cross-batch secant pair: y_t = g_{t+1} - g_t picks
    # up the FULL gradient difference between two batches, which does not shrink
    # with ||s||, whereas `sigma_batch` above is the spread of the curvature
    # ACTION H_B s, which does. Both are reported because the excerpt's
    # "sigma_batch" could name either, and they differ by orders of magnitude.
    # In |Delta_H| units the secant contamination is 2*||g_B - g_B'|| / ||s||^2
    # * ||s|| = 2 * sqrt(2) * sigma_g / ||s||, from w = 2(y - H s).
    G = torch.stack(
        [flat_grad(batch_closure(model, args.x_pool[i], args.y_pool[i])(),
                   params).detach()
         for i in args.sigma_idx],
        dim=1,
    )
    sigma_g = float(((G - G.mean(dim=1, keepdim=True)).norm(dim=0) ** 2).mean().sqrt())
    row["sigma_g"] = sigma_g
    row["sigma_secant"] = 2.0 * (2.0 ** 0.5) * sigma_g / sn
    set_flat(params, theta_n)

    # ---- drift estimates from both secant pairs ---------------------------
    for tag, g_n in (("diff", g_n_diff), ("same", g_n_same)):
        w = 2.0 * ((g_n - g_t) - Hs)
        row[f"delta_hat_{tag}"] = float(w.norm()) / sn**2
        row[f"DeltaH_{tag}"] = float(w.norm()) / sn
        row[f"y_norm_{tag}"] = float((g_n - g_t).norm())

    # ---- Krylov quantities at step t+1 ------------------------------------
    def matvec_n(v):
        return hvp(cl_n, params, v)

    # As shipped: the carried basis is re-validated and usually dropped.
    probe = copy.deepcopy(opt.builder)
    probe.U = U_carried
    eps, W, y, m_used, n_conv = build_with_history(probe, matvec_n, g_n_diff)
    k_stop = len(eps)
    row["m_still_converged"] = n_conv
    # The gate's epsilon_K(k) must be the ABSOLUTE residual ||H d + g||: it is
    # divided by ||y_U||, a length, and the result is compared against a
    # curvature. `build_with_history` returns the relative residual the builder
    # reports, so it is rescaled by ||g|| here.
    row["eps_k_rel"] = eps[-1]
    row["eps_k"] = eps[-1] * float(g_n_diff.norm())
    row["eps_hist"] = eps
    # rho_K is a per-matvec contraction, so it needs two consecutive depths.
    # At k_stop == 1 there is no previous depth and the ratio is undefined.
    row["rho_K"] = eps[-1] / eps[-2] if k_stop >= 2 else float("nan")
    row["k_stop"] = k_stop
    # The excerpt treats rho_K as a per-matvec contraction factor, i.e. a
    # property of the sequence, not of one adjacent pair. A single ratio off a
    # non-monotone sequence is a coin flip, so the sequence-level summaries are
    # recorded too: the geometric mean rate over the whole history, and the
    # fraction of consecutive steps that actually contract.
    if k_stop >= 2:
        ratios = [eps[j + 1] / eps[j] for j in range(k_stop - 1) if eps[j] > 0]
        row["rho_K_geo"] = (eps[-1] / eps[0]) ** (1.0 / (k_stop - 1))
        row["frac_contracting"] = sum(r < 1.0 for r in ratios) / len(ratios)
        row["eps_min_rel"] = min(eps)
    else:
        row["rho_K_geo"] = float("nan")
        row["frac_contracting"] = float("nan")
        row["eps_min_rel"] = eps[-1]
    row["m_used"] = m_used

    # As FACA assumes: deflate the whole carried basis whether or not it still
    # passes the Ritz test, since the break-even inequality is what would decide
    # reuse. This is the configuration ``m_eff``, ``||y_U||`` and the gate are
    # evaluated in -- the most favorable one FACA could ask for.
    m_carried = 0 if U_carried is None else int(U_carried.shape[1])
    if m_carried:
        forced = copy.deepcopy(opt.builder)
        forced.U = U_carried.clone()
        eps_f, _, y_f, m_f, _ = build_with_history(
            forced, matvec_n, g_n_diff, revalidate=False
        )
        row["y_U_norm"] = float(y_f[:m_f].norm())
        row["eps_k_forced"] = eps_f[-1] * float(g_n_diff.norm())
        Q_fresh = fresh_krylov_basis(matvec_n, g_n_diff, probe.k_max)
        resid = U_carried - Q_fresh @ (Q_fresh.T @ U_carried)
        row["m_eff"] = float((resid.norm(dim=0) ** 2).sum())
    else:
        row["y_U_norm"] = 0.0
        row["m_eff"] = 0.0

    # ---- the gate ----------------------------------------------------------
    if m_carried and row["y_U_norm"] > 0 and k_stop >= 2:
        rhs = (
            (row["m_eff"] - 1.0)
            * (1.0 - row["rho_K"])
            * row["eps_k"]
            / row["y_U_norm"]
        )
        row["rhs"] = rhs
        # The linearized inequality is only meaningful when both of its factors
        # are positive: m_eff > 1 (the reused block adds more than the one
        # certification matvec it costs) and rho_K < 1 (the Krylov residual is
        # actually contracting). With both negative their product is positive
        # and the comparison "passes" while describing nothing -- a sign
        # artifact, not an open gate.
        row["gate_valid"] = bool(row["m_eff"] > 1.0 and row["rho_K"] < 1.0)
        row["gate_open_diff"] = bool(row["gate_valid"] and row["DeltaH_diff"] < rhs)
        row["gate_open_same"] = bool(row["gate_valid"] and row["DeltaH_same"] < rhs)
        # Same gate with the sequence-level contraction rate in place of the
        # last-pair ratio.
        rhs_geo = (
            (row["m_eff"] - 1.0) * (1.0 - row["rho_K_geo"])
            * row["eps_k"] / row["y_U_norm"]
        )
        row["rhs_geo"] = rhs_geo
        row["gate_valid_geo"] = bool(
            row["m_eff"] > 1.0 and row["rho_K_geo"] < 1.0
        )
        row["gate_open_geo_diff"] = bool(
            row["gate_valid_geo"] and row["DeltaH_diff"] < rhs_geo
        )
        row["gate_open_geo_same"] = bool(
            row["gate_valid_geo"] and row["DeltaH_same"] < rhs_geo
        )
    else:
        # Nothing was carried at all: m_eff = 0 makes (m_eff - 1) negative and
        # the right-hand side non-positive, so no non-negative |Delta_H| can
        # satisfy it. Recorded as such rather than skipped.
        row["rhs"] = 0.0 if not m_carried else float("nan")
        row["rhs_geo"] = row["rhs"]
        row["gate_valid_geo"] = False
        row["gate_open_geo_diff"] = False
        row["gate_open_geo_same"] = False
        row["gate_valid"] = False
        row["gate_open_diff"] = False
        row["gate_open_same"] = False

    results.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--m-recycle", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--sigma-batches", type=int, default=8)
    ap.add_argument("--ref-batch", type=int, default=4096)
    ap.add_argument("--pairs", type=str, default="",
                    help="comma-separated t values; default is 6 spread over the run")
    ap.add_argument("--fixed-batch", action="store_true",
                    help="train every step on one batch: removes batch noise "
                         "from the secant pair entirely, so the 'diff' and "
                         "'same' pairs coincide and only true operator drift "
                         "L_H||s|| remains")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ds = datasets.MNIST("data", train=True, download=True,
                        transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Prefetch the whole trajectory's batches: the diagnostics need B_{t+1}
    # before the optimizer consumes it.
    batches, it = [], iter(loader)
    for _ in range(args.steps + 1):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader)
            batches.append(next(it))

    if args.fixed_batch:
        batches = [batches[0]] * len(batches)

    # Resampling pool for sigma_batch, and a large reference batch. Drawn from
    # the tail of the training set so a resample is an independent draw of the
    # same size, not a re-draw of the step's own batch.
    pool_n = args.sigma_batches * args.batch_size
    tail = len(ds) - pool_n - args.ref_batch
    x_all = torch.stack([ds[i][0] for i in range(tail, len(ds))])
    y_all = torch.tensor([ds[i][1] for i in range(tail, len(ds))])
    args.x_pool, args.y_pool = x_all[:pool_n], y_all[:pool_n]
    args.x_ref, args.y_ref = x_all[pool_n:], y_all[pool_n:]
    args.sigma_idx = [
        slice(r * args.batch_size, (r + 1) * args.batch_size)
        for r in range(args.sigma_batches)
    ]

    model = build_model()
    params = list(model.parameters())
    opt = DSN(model.parameters(), lr=args.lr, weight_decay=0.0,
              builder=KrylovBuilder(k_max=args.k_max, m_recycle=args.m_recycle,
                                    tau=args.tau))

    if args.pairs:
        pairs = [int(v) for v in args.pairs.split(",")]
    else:
        pairs = [max(1, args.steps * j // 6) for j in range(6)]
    pairs = sorted(set(p for p in pairs if p < args.steps - 1))

    results = []
    t0 = time.time()
    for t in range(args.steps):
        if t in pairs:
            measure_pair(model, params, opt, batches, t, args, results)
            r = results[-1]
            print(
                f"[pair {t:4d}] loss {r['loss']:.4f} |s| {r['s_norm']:.4g} "
                f"m_out {r['m_carried_out']} m_eff {r.get('m_eff', 0):.3g} "
                f"rho_K {r.get('rho_K', float('nan')):.3g} "
                f"eps_k {r.get('eps_k', float('nan')):.3g} "
                f"dH_diff {r.get('DeltaH_diff', float('nan')):.4g} "
                f"dH_same {r.get('DeltaH_same', float('nan')):.4g} "
                f"sig_sec {r.get('sigma_secant', float('nan')):.4g} "
                f"sig_Hs {r.get('sigma_batch', float('nan')):.4g} "
                f"({time.time() - t0:.0f}s)"
            )
        else:
            x, yb = batches[t]
            opt.step(batch_closure(model, x, yb))

    print(f"\ndone in {time.time() - t0:.0f}s   k_max={args.k_max}")
    rows = [r for r in results if "DeltaH_diff" in r]

    print("\nA. drift and batch noise, all in |Delta_H| units (curvature)")
    hdr = ("t", "|s|", "L_H^diff", "L_H^same", "|dH|diff", "|dH|same",
           "sigma_sec", "sigma_Hs", "sig_vs_ref")
    print(("{:>5}" + "{:>11}" * 8).format(*hdr))
    for r in rows:
        print(
            f"{r['t']:5d}{r['s_norm']:11.4g}{r['delta_hat_diff']:11.4g}"
            f"{r['delta_hat_same']:11.4g}{r['DeltaH_diff']:11.4g}"
            f"{r['DeltaH_same']:11.4g}{r['sigma_secant']:11.4g}"
            f"{r['sigma_batch']:11.4g}{r['sigma_vs_ref']:11.4g}"
        )
    print("  sigma_sec = gradient-noise contamination of the cross-batch secant")
    print("  sigma_Hs  = spread of the curvature ACTION H_B s over resampled batches")

    print("\nB. the gate")
    hdr = ("t", "m_out", "m_conv", "m_eff", "rho_K", "rho_geo", "contr%",
           "eps_k", "|y_U|", "rhs", "rhs_geo", "open", "open_geo")
    print(("{:>5}" + "{:>7}" * 2 + "{:>11}" * 8 + "{:>9}" * 2).format(*hdr))
    for r in rows:
        op = ("-" if not r["gate_valid"]
              else str(r["gate_open_diff"])[0] + "/" + str(r["gate_open_same"])[0])
        opg = ("-" if not r["gate_valid_geo"]
               else str(r["gate_open_geo_diff"])[0] + "/"
               + str(r["gate_open_geo_same"])[0])
        print(
            f"{r['t']:5d}{r['m_carried_out']:7d}{r.get('m_still_converged', 0):7d}"
            f"{r['m_eff']:11.4g}{r['rho_K']:11.4g}{r['rho_K_geo']:11.4g}"
            f"{100 * r['frac_contracting']:11.0f}{r['eps_k']:11.4g}"
            f"{r['y_U_norm']:11.4g}{r['rhs']:11.4g}{r['rhs_geo']:11.4g}"
            f"{op:>9}{opg:>9}"
        )
    print("  open reads diff/same. '-' where the inequality is vacuous:")
    print("  m_eff <= 1 or rho_K >= 1, which makes both factors negative and")
    print("  their product spuriously positive.")
    print(f"  tau = {args.tau}; every pair above ran to k_max = {args.k_max}"
          if all(r["k_stop"] == args.k_max for r in rows)
          else f"  tau = {args.tau}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": {k: v for k, v in vars(args).items()
                                if isinstance(v, (int, float, str))},
                       "rows": results}, fh, indent=1)


if __name__ == "__main__":
    main()
