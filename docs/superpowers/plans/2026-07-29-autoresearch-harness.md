# Autoresearch harness — two-stage candidate screening

**Date:** 2026-07-29
**Status:** Infrastructure. Built and validated against DSN and two trivial
baselines. **No new optimizer is implemented.**

The design document's P8 sketched an autoresearch loop conditional on the kill
criterion firing. This is the screening half of it, built to the constraints the
negative result argued for rather than to Karpathy's loop as-is.

Two stages, and one ledger that neither of them can hide in.

| | question | cost | code |
| --- | --- | --- | --- |
| Stage 1 | is preconditioner `P` favorable *at this point on this trajectory*? | a handful of Lanczos solves | `src/autoresearch/gate.py` |
| Stage 2 | of the survivors, which actually wins, and how much do we believe it? | training trials | `src/autoresearch/select.py` |

---

## 1. Stage 1 — the gate

### 1.1 The predicate is generalized; the training loop is not touched

`scripts/measure_rho.py` fixes `M = |H| + delta` and reports

    lambda_g   = ||g||^2 / (g' M^-1 g)
    lambda_Sig = sqrt( tr(M Sigma) / tr(M^-1 Sigma) )
    rho        = (lambda_Sig / lambda_g)^2

Written in terms of the preconditioner `P = M^-1`:

    rho(P) = [ tr(P^-1 Sigma) / tr(P Sigma) ] * (g'Pg)^2 / ||g||^4

which is defined for any SPD `P` and reduces to the shipped formula at
`P = (|H| + delta)^-1`. Two properties the verdict rule leans on:

- `rho` is invariant under `P -> cP`, so the gate cannot be gamed by rescaling.
- **`rho(I) = 1` exactly.** The threshold at 1 is not a tuned constant — it is
  literally "does this beat plain SGD by this criterion".

**Verified against the original to machine precision.** At the step-0 checkpoint
of DSN's MNIST trajectory, on the same batches and the same probe vectors, the
generalized `rho` reproduces `measure_rho.py` at every `(delta, k)` cell:
0.176 / 0.193 / 0.064 / 0.045 at `delta=1e-3` and depths 25/50/75/100, matching
the published step-0 row. Relative difference `< 1e-15` throughout. The
generalization is the same measurement, not a new one.

Adding a new preconditioner form means adding a `Preconditioner` subclass. It
does not mean touching `gate.py`, and it never means touching a training loop.

### 1.2 `rho` is H-blind, and this is not a small thing

`H` does not appear in the `rho` expression above. It enters only through the
*definition* of `P`. So `rho` measures whether `P`'s shape matches the
`(g, Sigma)` geometry, and it cannot tell whether `P` still inverts the curvature
it claims to invert.

Two independent demonstrations, both logged in
[`autoresearch/logs/gate_mutants.md`](../../../autoresearch/logs/gate_mutants.md):

**Staleness.** On a synthetic quadratic where `g` and `Sigma` are held fixed
while `H` drifts, `rho` of a *frozen* `P = H_0^-1` is numerically **identical**
at `t=0` and `t=T` — 11.08 both times, to machine precision. It is not that the
gate is underpowered against staleness; it is structurally incapable of
detecting it. Meanwhile the correct `P = H_T^-1` scores `rho = 1.0e6`.

**Uninformative spread.** A random diagonal `P`, drawn from a seed and never
shown `H`, `g` or `Sigma`, gets *better* on `rho` as its spread grows:

| log spread | cond(P) | rho median | rho > 1 | gain median | gain > 1 | composite PASS | rho-only PASS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.65 | 1.029 | 67% | 1.035 | 58% | 8% | 21% |
| 0.50 | 2.72 | 1.097 | 75% | 1.075 | 58% | 12% | 33% |
| 1.00 | 7.39 | 1.41 | 88% | 1.148 | 54% | 8% | 46% |
| 2.00 | 54.6 | 3.642 | 100% | 0.979 | 46% | **0%** | **79%** |

At the widest spread the `rho`-only screen passes a preconditioner that knows
nothing 79% of the time. The mechanism is arithmetic, not sampling: `rho`
contains `(g'Pg)^2 / ||g||^4`, which grows quadratically in `P`'s action on the
gradient's support, while the compensating trace ratio only recovers part of it.

Shipping a generalized `rho` as the sole screen would have rebuilt, at the
harness level, exactly the failure this project already paid for: a
maximally-stale frozen basis scoring *better* than the correct implementation on
both gates written to catch it (negative-result doc, §8.2).

### 1.3 The second statistic

So the gate also evaluates the expected one-step loss decrease of preconditioned
SGD under a noisy quadratic model. For `x <- x - eta P (g + noise)`,

    E[dL] = -eta g'Pg + (eta^2/2) [ g'PHPg + tr(P H P Sigma) ]

minimized over `eta` at

    R(P) = (g'Pg)^2 / ( g'PHPg + tr(P H P Sigma) )
    gain(P) = R(P) / R(I)

Also scale-invariant in `P`, also exactly 1 at `P = I`, and — unlike `rho` — it
contains `H`. It is also *not* invariant to the scale of `Sigma`, which `rho` is;
`ProbeContext.noise_scale` converts the measured microbatch covariance to the
covariance of the gradient the screened optimizer actually sees, and getting that
wrong silently rescales half the gate.

Both are necessary conditions: PASS requires both, FAIL needs only one. Every
report also logs what the verdict *would* have been on `rho` alone, so §1.2 is a
measured claim in every run rather than an assertion in a doc.

### 1.4 Cost: screening the spectral family is free

For `P = f(H)^-1`, all four quadratic forms the gate needs —
`v'Pv`, `v'P^-1v`, `v'PHPv`, `v'Hv` — are values of *some* function of `H` at the
same probe vector, so all four come out of one Gauss quadrature rule per vector.
**Screening the entire spectral family costs the same curvature products as
screening one member.** Explicit (diagonal, dense, frozen, random) preconditioners
need `H` applied to the rotated vector `Pv`, so they pay one extra curvature
product per probe vector each.

This shows up in the run report: on the quadratic problem DSN's stage-1 cost is
*lower* than the random control's, because DSN is spectral and the control is not.

### 1.5 Ceiling, floor, and the depth trap

Following the project's own idiom ("`rho` maxes at 0.717 against a threshold of
1"), each statistic is swept over Lanczos depth and judged at both ends:

- **FAIL** if the *upper* CI bound at the *most favorable* depth is below the bar.
- **PASS** if the *lower* CI bound at the *least favorable* depth is above it.
- **INDETERMINATE** otherwise, which is a real answer and is counted as one.

Maximizing over a sweep and then testing at the maximum would inflate the
false-pass rate by exactly the width of the sweep.

**The sweep is only protective if it spans the converged region.** Found the hard
way: at depths (4, 8, 12) the MNIST checkpoint reports `rho ~ 9.8`; at (25, 50,
75, 100) it reports 0.045–0.19. `lambda_g` is a harmonic mean, dominated by the
smallest eigenvalues, which Lanczos resolves last — a shallow sweep measures
truncation, not `rho`. The ceiling/floor rule returned INDETERMINATE rather than
a false PASS, which is the intended behavior, but the reason was invisible until
`CheckpointResult.depth_spread` was added to name it. **Read the spread before
the verdict.**

### 1.6 `min_effect` is not cosmetic

`P = I` and every scalar multiple of it score exactly 1.0, *and* have a
degenerate bootstrap — every resample returns the same number. At a zero margin
the verdict is decided by whether rounding lands at `1 + 1e-16` or `1 - 1e-16`,
and the zero-width interval reports that dust as a **certified** pass. The
default 5% margin exists for that reason as much as for practical significance.

---

## 2. Stage 2 — selection

### 2.1 Family-wise, over a pool that keeps growing

Candidate `j`, in order of entry, is allocated `alpha_j = alpha * 6/(pi^2 j^2)`.
The series converges to `alpha` over an *infinite* pool, so no matter how many
candidates are proposed later, `P(any arm ever wrongly eliminated) <= alpha`.

Measured under the global null (five identical arms, 300 replications): the
family-wise error rate lands under the nominal 0.05. The anytime confidence
sequence's coverage under continuous peeking is measured too, not assumed.

The cost is stated rather than buried: **this allocation is
ordering-dependent.** A candidate proposed 40th is held to a bar 50x stricter
than the same candidate proposed 4th. Online-FDR procedures (LORD++ and
relatives) exist to re-invest budget as discoveries accrue and are not
implemented here.

### 2.2 Elimination is not the same question as discovery

Elimination uses an anytime-valid sub-Gaussian confidence sequence, so peeking
every round is legitimate. Discovery — "does this beat the incumbent" — is a
separate one-sided anytime p-value on **paired** differences over the seeds both
arms actually ran. Pairing matters twice: it cuts variance, and it stops an arm
killed in round 2 from being scored against a baseline mean that later absorbed
18 more seeds.

Those p-values go through **Benjamini–Yekutieli**, valid under arbitrary
dependence — which is what we have: shared seeds, shared data order, adaptive
stopping. Benjamini–Hochberg is reported alongside for contrast and is explicitly
not the number to quote.

### 2.3 Stability is a viability judgement, not a mean comparison

Found by running it. The random control at a knife-edge learning rate converged
on 11 of 12 seeds and blew up on one. The capped outlier moved its mean by four
orders of magnitude and inflated `sigma_hat` enough that **it could never be
eliminated** — it survived the entire seed budget on the strength of its own
instability.

Two fixes, both kept:

1. Tuning disqualifies any learning rate that diverged on *any* tuning seed while
   a fully-stable rate exists.
2. An arm that diverges in selection is disqualified on **stability**, recorded
   separately from statistical eliminations and spending **no alpha**. "This blew
   up" is a fact about the run, not an inference about a mean, and pooling the
   two would let a viability judgement consume the budget the comparisons need.

The general lesson is worth carrying: score-capping makes an unstable arm's
distribution bimodal, which violates the sub-Gaussian assumption the whole
procedure rests on. The procedure's conservatism then protects the *wrong* arm.

---

## 3. Cost accounting

Two totals per survivor, and the fully-loaded one is the headline:

- **marginal** — its own stage-1 share plus its own trials.
- **fully loaded** — every candidate's gate screen, every tuning run, and every
  *other* arm's trials, plus its own training.

A measured run (quadratic, 5 arms, 12 seeds max): grand total 2.47e8 FLOPs,
of which stage 1 was 1.16e8, tuning 8.3e7, and selection trials 4.8e7. The
survivors' **search multiplier was 8.1x** — the fully-loaded cost was eight times
the number a training-only ledger would have printed.

That multiplier is the number to quote against any published
FLOPs-to-target-loss figure. Omitting it is what makes a comparison unfairly
favorable to whichever candidate the search happened to keep.

Attribution conventions, all stated because all are choices: shared stage-1 probe
cost is split evenly across the candidates that used it (the marginal cost of the
second spectral candidate is genuinely zero, so any split is arbitrary — what is
not arbitrary is that the total appears); the baseline's trials are search cost;
tuning is search cost even though it is spent on training runs; fully-loaded cost
is divided equally when more than one candidate survives.

---

## 4. What this does not establish

- **No new optimizer was designed or tested.** The harness was built against DSN
  as shipped, plain SGD, and a fixed-random-diagonal control. Whether it finds
  anything is a question for whatever gets plugged into it.
- **Stage 1 screened a preconditioner *form*, not an implementation.**
  **Superseded** — see
  [`2026-07-29-gain-mutants-and-rank.md`](2026-07-29-gain-mutants-and-rank.md).
  `KrylovRestrictedPreconditioner` now screens DSN's shipped rank-`k_max`
  operator, and doing so **flips the verdict from FAIL to PASS**: the full-rank
  form is a different operator, not an upper bound. The `gain` figure of "4–79"
  reported above was also wrong — the 79 end was a Lanczos-depth artifact, and
  the gate now gates verdicts on per-statistic depth convergence in both
  directions. The residual `rho`/`gain` disagreement is real but smaller
  (0.17–0.57 against 0.98–9.7) and remains unresolved.
- **The stage-2 guarantees are exact for a known sub-Gaussian `sigma` and
  approximate for the plug-in estimate** used in practice. `sigma_inflate` buys
  some of that back; the report states the assumption rather than absorbing it.
- **Gate intervals cover sampling error in `Sigma` only.** `g` is a single sample
  and its error is not quantified; Lanczos truncation is handled by the depth
  sweep, not by the interval.
- **The demonstration problems are a 24-dimensional quadratic and a 76k-parameter
  MNIST MLP on CPU.** Nothing here has been run at the GPU tiers.

---

## 5. Reproduce

    python -m pytest tests/test_autoresearch_gate.py \
                     tests/test_autoresearch_select.py \
                     tests/test_autoresearch_cost.py

    python scripts/validate_gate.py               # mutants; exit 1 if not separated
    python scripts/validate_gate.py --mnist       # plus the real-trajectory cross-check

    python scripts/run_autoresearch.py --no-dsn --controls 4
    python scripts/run_autoresearch.py --controls 1
    python scripts/run_autoresearch.py --problem mnist

`run_autoresearch.py` refuses to start unless `validate_gate.py` has recorded a
successful separation. Outputs land in `autoresearch/logs/`.
