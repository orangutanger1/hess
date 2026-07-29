# P4 (FACA) break-even gate — measured on DSN's real trajectories

Instrument: `scripts/measure_faca_gate.py`. Measurement only; nothing in `src/`
changed and no solver code was written. Formulas are taken as given from the P4
excerpt and are not re-derived here.

Gate under test:

    |Delta_H|  <  (m_eff - 1) * (1 - rho_K) * epsilon_K(k) / ||y_U||

## Verdict

**"Different mini-batches across the secant pair" is not what is killing the
gate.** It is a factor-of-~2 contributor. Three other things each close the gate
on their own, and two of them survive the excerpt's proposed common-curvature-batch
fix untouched:

1. **`rho_K` is not a contraction.** Geometric-mean rate over the residual
   history is 0.978–1.014 across every configuration and seed measured, with
   only 41–57% of consecutive Lanczos directions contracting at all. `(1 - rho_K)`
   is therefore ~0.01 rather than O(1), which crushes the right-hand side by
   about two orders of magnitude. This is the dominant killer.
2. **`m_eff` and `|Delta_H|` are anti-correlated, so no regime satisfies both.**
   Where drift is small the previous basis lies *inside* the current fresh
   Krylov space and there is nothing to reuse (`m_eff` → 1e-5); where `m_eff` is
   healthy (2.5–4.4) the drift is large. See "The trap" below.
3. **`L_H*||s||` alone already exceeds the right-hand side**, with the secant
   pair evaluated on a common batch: `|Delta_H|_same` is 0.38–1.83 against a
   right-hand side of 0.011–0.10. The batch fix buys ~2x; the gap is 4–90x.

Separately, and independent of FACA: **`m_conv` is 0 at 23 of the 24 measured
pairs** (four runs of six; the exception is one direction at `t=50` of the
fixed-batch run) — a direction that passed the Ritz convergence test at step `t` never
still passes it against step `t+1`'s operator. The shipped builder therefore
discards its recycled basis every step regardless of what any break-even gate
decides.

## What was measured

Six consecutive-step pairs `(t, t+1)` per run, on the exact `sanity_mnist.py`
model and data (784-96-tanh-10 MLP, 76k parameters, batch 256, lr 1e-2, MNIST).

| quantity | how |
| --- | --- |
| `s`, `y_t` | free, from the trajectory |
| `w = 2(y_t - H_t s)`, `delta_hat = ||w||/||s||^2` | one extra HVP per pair |
| `|Delta_H| = ||w||/||s||` | same `w`; curvature units, which is what the gate compares |
| `sigma_sec` | 2*sqrt(2)*sigma_g/||s||, gradient spread over 8 independently resampled batches at frozen theta_t |
| `sigma_Hs` | spread of the curvature *action* `H_B s` over the same 8 batches |
| `rho_K`, `rho_geo` | true residual `||H(Wy)+g||/||g||` per Lanczos depth, as `KrylovBuilder` computes it |
| `m_eff` | `sum_i ||(I - QQ')u_i||^2`, U the carried basis, Q an undeflated fresh Krylov basis at `t+1` |
| `||y_U||` | recycled block of the subspace solution, with re-validation forced off so the number exists |

`|Delta_H|` rather than `delta_hat` is compared against the gate because
`epsilon_K(k)/||y_U||` is gradient-units over length-units, i.e. curvature.
`epsilon_K` is the absolute residual for the same reason.

Two secant pairs are formed at every `t`, differing only in the batch the second
gradient is evaluated on: `diff` (`g_{t+1}` on `B_{t+1}`, what consecutive DSN
steps actually see) and `same` (`g_{t+1}` on `B_t`, the excerpt's fix). Both are
differenced against the same `g_t` and the same `H_t s`, so the gap between them
is batch resampling and nothing else.

`||y_U||` and `m_eff` are computed with the builder's Ritz re-validation forced
**off**. As shipped it fires and drops the basis, so these quantities would not
exist on this trajectory at all. Forcing it off is the configuration most
favorable to FACA.

## Numbers

### Batching is a 2x term, not the killer

`k_max=40`, `m_recycle=8`, seed 0, mini-batch (`|Delta_H|` units):

| t | \|s\| | \|dH\| diff | \|dH\| same | sigma_sec | sigma_Hs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.0 | 0.750 | 0.384 | 0.467 | 0.089 |
| 10 | 4.0 | 0.770 | 0.420 | 0.652 | 0.182 |
| 20 | 4.0 | 0.736 | 0.441 | 0.794 | 0.276 |
| 30 | 2.0 | 2.370 | 0.981 | 1.514 | 0.358 |
| 40 | 2.0 | 1.335 | 0.772 | 1.262 | 0.296 |
| 50 | 2.0 | 1.810 | 1.825 | 1.428 | 0.523 |

Moving the secant onto a common batch removes 40–60% of `|Delta_H|`, not orders
of magnitude. What remains — `L_H*||s||` with `L_H` measured at 0.11–0.49, stable
across step lengths of 1.4–4.0, i.e. behaving like a real Lipschitz constant —
is still far above any right-hand side the same run produces.

Note the two candidate readings of the excerpt's `sigma_batch` differ by ~5x:
`sigma_Hs` (spread of `H_B s`, 0.09–0.52) is the curvature-operator noise, but
the term that actually contaminates a cross-batch secant is `sigma_sec`
(0.47–1.53), because `y_t = g_{t+1} - g_t` picks up the full gradient difference
between two batches, which does not shrink with `||s||`. `sigma_sec` is the one
that matters and it is the larger of the two.

### The right-hand side, and why it is small

Same run:

| t | m_eff | rho_K (last pair) | rho_geo | contracting | eps_k | \|y_U\| | rhs (last-pair) | rhs (geo) | open (geo) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 2.52 | 0.886 | 0.993 | 54% | 0.874 | 0.760 | 0.199 | 0.0115 | F/F |
| 10 | 3.15 | 1.784 | 1.007 | 49% | 1.310 | 0.176 | -12.55 | -0.116 | vacuous |
| 20 | 3.88 | 0.555 | 0.986 | 51% | 0.952 | 0.387 | 3.148 | 0.100 | F/F |
| 30 | 4.09 | 0.058 | 0.994 | 46% | 0.785 | 0.165 | 13.87 | 0.088 | F/F |
| 40 | 3.09 | 0.352 | 0.986 | 44% | 0.634 | 0.331 | 2.596 | 0.056 | F/F |
| 50 | 4.37 | 2.423 | 1.014 | 46% | 2.357 | 0.528 | -21.39 | -0.209 | vacuous |

Using a single adjacent ratio for `rho_K` makes the gate look open at 3 of 6
pairs. That is an artifact of reading a contraction rate off a sequence that
does not contract. The residual history is a random walk with O(1) excursions —
e.g. at `t=1`: `1.566, 3.496, 21.390, 1.493, 1.272, 4.050, ... , 1.366, 1.210` —
and `tau=0.1` is never reached, so **every** pair in **every** configuration ran
to `k_max`. Whether `rho_K` lands below 1 depends on which side of an excursion
the last two directions fall on. With the sequence-level rate the gate is closed
or vacuous at every pair, by 4–90x, for both the `diff` and `same` secant pairs.

The non-contraction is structural, not a tuning failure: `subspace_newton` solves
the saddle-free damped system `(|H| + delta) d = -g`, and DSN's Hessian here is
genuinely indefinite, so `||H d + g||` is the residual of a system the solver is
deliberately not solving. It has no reason to decrease in `k`, and it doesn't.

Seed 1 reproduces this: `rho_geo` 0.990–1.059, closed or vacuous at all 6 pairs.
One row (`t=50`, seed 1) is the single case in 12 mini-batch pairs where the
batch fix alone flips a last-pair-`rho_K` verdict from closed to open
(`|dH|_same` 0.777 vs rhs 0.950); under `rho_geo` that same row is vacuous.

### The trap: m_eff and |Delta_H| cannot both be favorable

Fixed-batch run (`--fixed-batch`, `k_max=40`) removes batch noise from the
secant entirely — `diff` and `same` coincide by construction:

| t | \|s\| | \|dH\| | m_eff | eps_k | rhs (geo) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.0 | 3.7e-1 | 1.10 | 8.8e-1 | 2.6e-3 |
| 10 | 0.769 | 8.8e-4 | 5.5e-2 | 2.3e-4 | -7.7e-5 |
| 20 | 0.269 | 3.3e-6 | 1.5e-3 | 5.7e-5 | 1.7e-6 |
| 30 | 0.108 | 1.5e-6 | 2.5e-4 | 3.1e-5 | 4.9e-7 |
| 40 | 0.046 | 2.3e-6 | 5.5e-5 | 2.2e-5 | 1.8e-7 |
| 50 | 0.023 | 4.7e-6 | 1.7e-5 | 1.6e-5 | 5.0e-8 |

Drift falls by five orders of magnitude and the gate is *still* never open. As
the iterate settles, the carried basis comes to lie inside the fresh Krylov
space (`m_eff` → 1.7e-5 of 8 carried directions), so reuse contributes no new
dimensions and `(m_eff - 1)` is negative; `epsilon_K` collapses at the same time,
so the right-hand side falls faster than `|Delta_H|` does. The two sides of the
inequality are driven by the same underlying quantity — how much the operator
and iterate have moved — which is why removing batch noise cannot open the gate:
it removes the numerator and the denominator together.

## Consequences for building anything from P4

- The excerpt's stated failure mode ("different mini-batches ... the real
  killer") is real but secondary here, and its proposed fix — one extra gradient
  on a common curvature batch — is measured to be insufficient by 4–90x on this
  problem. It should not be built on the expectation that it opens the gate.
- The gate's `(1 - rho_K)` factor assumes a contracting Krylov residual. DSN's
  saddle-free damped solve does not produce one. Any FACA-derived quantity read
  off `epsilon_K` on this codebase is being read off a non-monotone sequence, and
  a single-pair `rho_K` will report open/closed roughly at random (3 of 6 vs 0 of
  6 on identical data, seed 0).
- Before any FACA implementation work, the prerequisite question is whether a
  contracting residual and a simultaneously-large `m_eff` can coexist on *any*
  regime of this problem. The fixed-batch run says the two are anti-correlated.

## Reproduce

    python scripts/measure_faca_gate.py --steps 60 --k-max 8                     # shipped config
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8       # recycling engages
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --fixed-batch
    python scripts/measure_faca_gate.py --steps 60 --k-max 40 --m-recycle 8 --seed 1

At `--k-max 8` (the shipped MNIST configuration) the gate is additionally dead
for a trivial reason: at most 1 direction is ever carried, `m_eff` maxes at 0.40,
so `(m_eff - 1) < 0` at every pair. That is a statement about one hyperparameter,
which is why the `k_max=40` runs above are the ones the verdict rests on.
