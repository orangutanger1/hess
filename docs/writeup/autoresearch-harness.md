# The autoresearch harness

**Date:** 2026-07-29
**Status:** Infrastructure, built and validated. Implements **no new optimizer**.
**Code:** `src/autoresearch/` — `gate.py`, `select.py`, `cost.py`,
`preconditioners.py`, `probes.py`, `trials.py`, `report.py`.
**Scale:** CPU. Demonstrated on a 24-dimensional streaming quadratic and a
76k-parameter MNIST MLP.

A two-stage screen for candidate preconditioners, built so a mechanism gets
instrumented *before* it gets a training budget rather than after. It exists
because the project that produced it spent an implementation cycle on a
mechanism that a few Lanczos runs at a handful of checkpoints could have flagged
in minutes.

| | question | cost | code |
| --- | --- | --- | --- |
| **Stage 1** | is preconditioner `P` favorable *at this point on this trajectory*? | a handful of Lanczos solves, no training | `gate.py` |
| **Stage 2** | of the survivors, which actually wins, and how much do we believe it? | training trials on an adaptive seed budget | `select.py` |
| **Ledger** | what did the whole search cost, including the parts that found nothing? | — | `cost.py` |

Adding a candidate means adding a `Preconditioner` subclass and a `Candidate`
that pairs it with the optimizer that applies it. It never means touching
`gate.py`, and it never means touching a training loop.

---

## 1. Stage 1 — the gate

### 1.1 Two statistics, both scale-free, both exactly 1 at `P = I`

    rho(P)  = [ tr(P^-1 Sigma) / tr(P Sigma) ] * (g'Pg)^2 / ||g||^4

    R(P)    = (g'Pg)^2 / ( g'PHPg + tr(P H P Sigma) )
    gain(P) = R(P) / R(I)

`rho` is the payoff gate from the project's own trajectory measurement,
generalized off the fixed `M = |H| + delta` to any SPD `P` — verified to
reproduce the original at `P = (|H|+delta)^-1` to machine precision at every
`(delta, depth)` cell (relative difference `< 1e-15`). `R` is the expected
one-step loss decrease of preconditioned SGD under a noisy quadratic model,
minimized over the step size.

Both are invariant under `P -> cP`, so the gate cannot be gamed by rescaling,
and both are **exactly 1 at `P = I`**. The threshold is not a tuned constant —
it is literally "does this beat plain SGD by this criterion".

**PASS requires both. FAIL needs only one.** Every report also logs what the
verdict *would* have been on `rho` alone, so the reason the second statistic
exists stays a measured quantity in every run rather than a claim in a document.
That reason is that **`H` does not appear in `rho` at all** — it enters only
through the definition of `P` — so `rho` cannot tell whether `P` still inverts
the curvature it claims to invert. `gain` contains `H`. The full argument, with
four independent demonstrations, is in
[`self-referential-diagnostics.md`](self-referential-diagnostics.md).

One consequence worth stating for anyone reusing this: `gain` is **not**
invariant to the scale of `Sigma`, which `rho` is. `ProbeContext.noise_scale`
converts the measured microbatch covariance into the covariance of the gradient
the screened optimizer actually sees, and getting that conversion wrong silently
rescales half the gate.

### 1.2 Verdicts: ceiling, floor, and depth convergence

Each statistic is swept over Lanczos depth and judged at both ends:

- **FAIL** if the *upper* CI bound at the *most favorable* depth is below the bar.
- **PASS** if the *lower* CI bound at the *least favorable* depth is above it.
- **INDETERMINATE** otherwise, which is a real answer and is counted as one.

Maximizing over a sweep and then testing at the maximum would inflate the
false-pass rate by exactly the width of the sweep.

**The sweep is only protective if it spans the converged region**, which was
found the hard way: at depths (4, 8, 12) an MNIST checkpoint reports `rho ≈ 9.8`;
at (25, 50, 75, 100) it reports 0.045–0.19. `lambda_g` is a harmonic mean,
dominated by the smallest eigenvalues, which Lanczos resolves last — so a shallow
sweep measures truncation, not `rho`. `CheckpointResult.depth_spread` and
`Cell.spectral_resolution` (`= delta / min|node|`) exist to name that condition.
**Read the spread before the verdict.**

Depth convergence is measured **per statistic** as the relative change between
the two deepest rungs and **gates the verdict in both directions**. A statistic
still moving at the deepest rung cannot support a PASS *or* a FAIL. The earlier
version only appended a warning to the reason string, so a sweep sitting entirely
below the bar still returned a confident FAIL — and on a flat-tailed spectrum the
same statistic runs 9.2 at depth 5 and 1104 at depth 200, so such a FAIL can be
two orders of magnitude wrong.

`GateReport.trajectory_gain` computes `sum R(P) / sum R(I)` over the trajectory
and a per-checkpoint PASS must survive it. `R` is a decrease *per step*;
decreases add across steps, ratios do not. Skipping this is worth 171x in the
unsafe direction on a constructed counterexample.

`min_effect` (default 5%) is not cosmetic. `P = I` and every scalar multiple
score exactly 1.0 *and* have a degenerate bootstrap — every resample returns the
same number. At a zero margin the verdict is decided by whether rounding lands at
`1 + 1e-16` or `1 - 1e-16`, and a zero-width interval reports that dust as a
*certified* pass.

### 1.3 Cost: screening the whole spectral family costs one screen

For `P = f(H)^-1`, all four quadratic forms the gate needs — `v'Pv`, `v'P^-1 v`,
`v'PHPv`, `v'Hv` — are values of *some* function of `H` at the same probe vector,
so all four come out of **one Gauss quadrature rule per vector**. Screening the
entire spectral family costs the same curvature products as screening one member.

Explicit preconditioners (diagonal, dense, frozen, random, rank-restricted) need
`H` applied to the rotated vector `Pv`, so they pay one extra curvature product
per probe vector each. This shows up in the ledger: on the quadratic problem a
spectral candidate's stage-1 cost is *lower* than the random control's.

A useful side effect: an explicit `P` has **no truncation error** in the three
`P`-dependent forms, because they are computed by applying it. The only
quadrature left is `g'Hg`, a degree-one polynomial a Gauss rule integrates
exactly at any depth. Rank-restricted rows measure depth convergence 0.000.

### 1.4 Scope is a first-class field

Every gate report carries `scope` and `rank`. A verdict about one operator must
not read as a verdict about another — and this is not hypothetical: an earlier
report screened an idealized full-rank `P = (|H| + delta)^-1` while the optimizer
shipped a rank-`k` Krylov restriction with a first-order complement,

    P_k = W (|W'HW| + delta)^-1 W'  +  c (I - WW')

The full-rank form was framed as "an upper bound" on the restriction. It is not.
`P_k` with `c > 0` is itself a full-rank invertible matrix — a *member* of the
class the full-rank screen draws from, not a restriction of it. An upper bound
would require maximizing the statistic over the class, which no gate here does,
and which is vacuous anyway since `P = I` is always feasible so the supremum is
always at least 1. Screening the shipped operator flipped its verdict from FAIL
to PASS. `KrylovRestrictedPreconditioner` now screens what ships, and
`Candidate` bolts the screened `P` and the trained optimizer into one object so
the two stages cannot silently describe different things.

---

## 2. Stage 2 — selection

### 2.1 Family-wise error over a pool that keeps growing

Candidate `j`, in order of entry, is allocated `alpha_j = alpha * 6/(pi^2 j^2)`.
The series converges to `alpha` over an **infinite** pool, so no matter how many
candidates are proposed later,
`P(any arm ever wrongly eliminated) <= alpha`. Measured under the global null
(five identical arms, 300 replications) the family-wise error rate lands under
the nominal 0.05, and the anytime confidence sequence's coverage under continuous
peeking is measured rather than assumed.

The cost is stated rather than buried: **this allocation is ordering-dependent.**
A candidate proposed 40th is held to a bar 50x stricter than the same candidate
proposed 4th. Online-FDR procedures (LORD++ and relatives) exist to re-invest
budget as discoveries accrue and are not implemented here.

### 2.2 Elimination and discovery are different questions

Elimination uses an anytime-valid sub-Gaussian confidence sequence, so peeking
every round is legitimate. Discovery — "does this beat the incumbent" — is a
separate one-sided anytime p-value on **paired** differences over the seeds both
arms actually ran. Pairing matters twice: it cuts variance, and it stops an arm
killed in round 2 from being scored against a baseline mean that later absorbed
18 more seeds.

Those p-values go through **Benjamini–Yekutieli**, valid under arbitrary
dependence — which is what this is: shared seeds, shared data order, adaptive
stopping. Benjamini–Hochberg is computed alongside for contrast and is explicitly
not the number to quote. The report also prints the naive unadjusted rate beside
the controlled one, so the gap the machinery buys is visible per round.

### 2.3 Instability is a viability judgement, not a mean comparison

Found by running it. A random control at a knife-edge learning rate converged on
11 of 12 seeds and blew up on one. The capped outlier moved its mean by four
orders of magnitude and inflated `sigma_hat` enough that **it could never be
eliminated** — it survived the entire seed budget on the strength of its own
instability.

Two fixes, both kept:

1. Tuning disqualifies any learning rate that diverged on *any* tuning seed while
   a fully-stable rate exists.
2. An arm that diverges in selection is disqualified on **stability**, recorded
   separately from statistical eliminations and spending **no alpha**.

The general lesson: score-capping makes an unstable arm's distribution bimodal,
which violates the sub-Gaussian assumption the whole procedure rests on. The
procedure's conservatism then protects the *wrong* arm.

Two further conventions that are easy to get wrong and are enforced here:
learning-rate tuning runs on seeds `10_000+`, **disjoint** from the selection
seeds (tuning on the seeds a discovery is later certified on is selection on the
test set); and trials are budgeted in **FLOPs, not steps**, because a fixed step
count structurally favours whichever method does the most work per step — exactly
the class under test.

---

## 3. The ledger

Two totals per survivor, and the fully-loaded one is the headline:

- **marginal** — its own stage-1 share plus its own trials.
- **fully loaded** — every candidate's gate screen, every tuning run, and every
  *other* arm's trials, plus its own training.

From the logged demonstration run (quadratic, 3 arms, 12-seed cap): grand total
**1.486e8** FLOPs, of which stage 1 was 1.117e8, tuning 2.212e7, and selection
trials 1.475e7. The surviving candidate's own training was 7.37e6 FLOPs, so its
**search multiplier was 20.15x** — the fully-loaded cost was twenty times the
number a training-only ledger would have printed. An earlier 5-arm run measured
8.1x. The multiplier is problem- and pool-dependent; the point is that it is
never 1 and it is never reported by default anywhere else.

**That multiplier is the number to quote against any published
FLOPs-to-target-loss figure.** Omitting it is what makes a comparison unfairly
favorable to whichever candidate the search happened to keep.

Attribution conventions, all stated because all are choices: shared stage-1 probe
cost is split evenly across the candidates that used it (the marginal cost of the
second spectral candidate is genuinely zero, so any split is arbitrary — what is
not arbitrary is that the total appears); the baseline's trials are search cost;
tuning is search cost even though it is spent on training runs; fully-loaded cost
is divided equally when more than one candidate survives; and stage-1 cost
includes candidates the gate rejected before any training FLOP was spent, since
hiding those would overstate the saving.

FLOPs come from `analytic_cost_model(n_params)` and `calibrate()` cross-checks it
against `torch.utils.flop_counter.FlopCounterMode` rather than assuming the
multipliers.

---

## 4. Mutant validation, and what it caught

**`run_autoresearch.py` refuses to start** unless `validate_gate.py` has recorded
a successful mutant separation. A gate that has not been shown to separate a
known-good preconditioner from a known-bad one has no demonstrated power, and
using it to allocate a training budget spends that budget on its noise. `--force`
exists and marks every downstream verdict uncertified.

Separation is judged on **synthetic mutants only**, where the right answer is
known by construction. Real-trajectory rows are logged as a cross-check and
explicitly do not count: their right answer is the thing under study, so they
cannot also be the thing that certifies the instrument.

Current status: **9 of 9 mutants land on their expected verdict.**

| mutant | expected | verdict | `rho` ceil | `gain` ceil | `rho`-only would say |
| --- | --- | --- | ---: | ---: | --- |
| exact optimum `P* = H^-1`, favorable quadratic | PASS | PASS | 11.08 | 10.84 | PASS |
| constant `P = 3I` | FAIL | FAIL | 1 | 1 | FAIL |
| identity `P = I` (SGD control) | FAIL | FAIL | 1 | 1 | FAIL |
| **frozen `P = H_0^-1`, evaluated at `t=T`** | FAIL | FAIL | 11.08 | 0.0112 | **PASS** |
| refreshed `P = H_T^-1` at `t=T` | PASS | PASS | 1.004e6 | 3.943e4 | PASS |
| frozen rank-4 Krylov snapshot at `t=T` | PASS | PASS | 4428 | 585.6 | PASS |
| **frozen rank-8 Krylov snapshot at `t=T`** | FAIL | FAIL | 61.48 | 0.0458 | **PASS** |
| **frozen rank-12 Krylov snapshot at `t=T`** | FAIL | FAIL | 11.08 | 0.0112 | **PASS** |
| `P* = H^-1` on an unfavorable quadratic | FAIL | FAIL | 3.2e-4 | 5.5e-3 | FAIL |

The PASS mutant is verified to be the *optimum*, not merely good: in the noiseless
limit `sup_P R(P)` tightens to `g'H^-1 g`, and `P = H^-1` attains it —
`R(P*) = 10.5279` against a ceiling of 10.5279, with closed-form and estimated
`rho`/`gain` agreeing to four decimals.

### What the discipline actually caught

Each of these was a live defect in the shipped harness, found by a mutant rather
than by review:

| mutant | what it caught |
| --- | --- |
| frozen-`P` staleness | `rho` is provably constant before and after arbitrary curvature drift (11.0837 both times, to machine precision) — structurally incapable of detecting staleness. Produced the requirement for a second, `H`-aware statistic. |
| random diagonal swept over spread | `rho`-only passes a preconditioner that knows nothing **79%** of the time at wide spread, and the two statistics move in *opposite* directions as spread grows. |
| **G1** subspace freeze | `rho` scored **33.5** on a configuration measured to train **2,264x worse than SGD**; fooled by 5 of 9, `gain` by 0. First `rho` failure with trained ground truth attached. Also identified `rho` as untrustworthy on *rank-restricted* operators, which is the shape a real candidate has. |
| **G2** two-phase trajectory | The roll-up rule was max-over-checkpoints; the correct aggregate `sum R(P)/sum R(I)` disagreed by **171x** in the unsafe direction. Produced `GateReport.trajectory_gain`. |
| **G3** flat-tail spectrum | `gain` converges in Lanczos depth strictly more slowly than `rho` — further from converged in **13 of 13** unconverged cells — and had shipped with no depth safeguard, because it was added *after* the safeguard was found necessary for `rho`. A previously reported 80x `rho`/`gain` gap on MNIST collapsed to 7–20x once depth was controlled. Produced per-statistic two-directional depth gating and `Cell.spectral_resolution`. |
| rank-restriction check | The full-rank verdict was being reported as if it bounded the shipped rank-`k` operator. It does not, and screening the shipped operator flips FAIL → PASS. Produced `KrylovRestrictedPreconditioner` and the `scope`/`rank` fields. |
| stability sweep in stage 2 | An arm that diverges on one seed of twelve can become **uneliminable** through its own inflated `sigma_hat`. Produced stability disqualification with no alpha spend. |

Three of the seven changed a previously published number. That is the argument
for the discipline: the harness was reviewed carefully before each of them, and
review did not find any of them.

---

## 5. Calibration — the open limitation

The only test that settles a statistic is whether it orders candidates the way
training does. A one-parameter family `P = (|H| + 0.01)^-p` for `p` in
`[0, 1.5]`, scored by both statistics and then **actually trained** at a fixed
FLOP budget with a per-arm tuned learning rate:

| p | `rho` | `gain` | trained loss |
| ---: | ---: | ---: | ---: |
| 0 (SGD) | 1 | 1 | **0.00624** |
| 0.25 | 1.397 | 1.458 | 0.01067 |
| 0.5 | 1.064 | 1.324 | 0.00956 |
| 0.75 | 0.806 | 1.172 | 0.19477 |
| 1 | 0.660 | 0.946 | 1.5212 |
| 1.25 | 0.563 | 0.574 | 4.0293 |
| 1.5 | 0.508 | 0.256 | 11.222 |

Spearman against trained loss (a correct statistic is *negatively* correlated):
`rho` **-0.857**, `gain` **-0.750**.

**Both statistics correlate directionally with real training. Both mis-rank the
top candidate on this trajectory** — each ranks `p = 0.25` best while training
says plain SGD wins. This is stated as an open limitation of the current gate
statistics, not as a footnote:

- The screen is trustworthy for **rejecting** clearly bad regions of a
  preconditioner family. Above `p = 0.5` both statistics fall as the trained loss
  explodes by three orders of magnitude, in the right order.
- The screen is **not** trustworthy for picking a winner near the top. On this
  family it puts a candidate 1.7x worse than the baseline ahead of the baseline.
  The trajectory-aggregate fix downgraded `p = 0.25` from PASS to INDETERMINATE,
  so the gate no longer *claims* it beats SGD — but the ranking error remains.

A related open item, stated because it is unresolved rather than resolved: on the
MNIST trajectory `rho` says FAIL (0.17–0.56) and `gain` says PASS (3.3–9.7),
**both converged**, at every checkpoint where either can be trusted. Depth
explains the *magnitude* of the disagreement (80x collapses to 7–20x) and
explains nothing about the disagreement itself. None of the three constructed
mutants accounts for it. The composite gate rejects on it, which is the
conservative direction, and that is all that can currently be justified.

The honest summary: **stage 1 is a screen with measured power against known
failure modes and measured directional correlation with training, and it is not
an oracle.** Its correct use is to spend fewer training FLOPs on obviously
unfavorable candidates, not to certify a winner. Stage 2 exists precisely because
stage 1 cannot do stage 2's job.

---

## 6. What the demonstration run actually produced

Worth reporting, because it is the honest picture of an end-to-end run rather
than a showcase. On the 24-dimensional quadratic with 3 arms (SGD baseline, one
fixed-random control, DSN at its shipped rank):

- DSN was **gate-rejected** at stage 1 and never entered stage 2 in that run, so
  no training FLOPs were spent on it — which is the point of having a stage 1.
- The random control passed the gate and reached mean 0.006978 against the SGD
  baseline's 0.006427, i.e. it did not beat the baseline.
- **No arm was eliminated and no discovery was certified.** The run stopped with
  the seed budget exhausted and several arms alive. Family-wise budget spent:
  0.038 of 0.05. BY discoveries: none. Naive-if-unadjusted false-positive risk at
  q=0.1 across the comparisons made: 10.0%.

A screening harness whose demonstration run finds nothing is behaving correctly
when there is nothing to find. The pool contained one real candidate, one
deliberate negative control and a baseline; the correct output is "no discovery",
and it printed that alongside what an unadjusted analysis would have claimed.

A separate direct measurement, at an equal 4e7-FLOP budget on the same trial with
both arms at their separately tuned learning rate, 3 seeds: SGD reaches 0.00656
in 17,362 steps; DSN reaches 0.01244 in 1,432 steps — 12.1x fewer steps for the
same FLOPs and 1.90x worse. This does not go through the gate at all, which is
why it is quoted in the DSN writeup as gate-independent evidence.

---

## 7. What this does not establish

- **No new optimizer was designed or tested.** The harness was built against DSN
  as shipped, plain SGD, and a fixed-random-diagonal control. Whether it finds
  anything is a question for whatever gets plugged into it.
- **Both gate statistics mis-rank the top of a preconditioner family** against
  trained ground truth (§5).
- **The converged `rho`/`gain` disagreement on MNIST is unexplained** (§5).
- **One-step locality is unpatched.** The trajectory aggregate fixes the *roll-up*
  arithmetic, not the fact that each checkpoint's statistic is a one-step model
  evaluated at a sparse, equally-weighted sample of the trajectory. Freezing a
  subspace is harmful only when the gradient *later* acquires mass in it, and no
  statistic here is given a second point in time.
- **Stage-2 guarantees are exact for a known sub-Gaussian `sigma`** and
  approximate for the plug-in estimate used in practice. `sigma_inflate` (1.25x)
  buys some of that back; the report states the assumption rather than absorbing
  it.
- **Gate intervals cover sampling error in `Sigma` only.** `g` is a single sample
  and its error is not quantified; Lanczos truncation is handled by the depth
  sweep, not by the interval.
- **The alpha allocation is ordering-dependent** (§2.1), and online-FDR
  alternatives are not implemented.
- **`c`, the complement scaling in the rank-restricted preconditioner, is a
  scalar stand-in for AdamW.** Swept, not modelled.
- **Everything is CPU-tier**: a 24-dimensional quadratic and a 76k-parameter
  MNIST MLP. Nothing here has been run at a GPU tier.

---

## 8. Reproduce

    python -m pytest tests/test_autoresearch_gate.py \
                     tests/test_autoresearch_select.py \
                     tests/test_autoresearch_cost.py

    python scripts/validate_gate.py               # 9 mutants; exit 1 if not separated
    python scripts/validate_gate.py --mnist       # plus the real-trajectory cross-check
    python scripts/validate_gain.py               # G1-G3 + the training calibration
    python scripts/diagnose_gain_depth.py         # the MNIST depth ladder
    python scripts/measure_rank_restricted.py     # a candidate at its shipped rank

    python scripts/run_autoresearch.py --no-dsn --controls 4
    python scripts/run_autoresearch.py --controls 1
    python scripts/run_autoresearch.py --problem mnist

Outputs land in `autoresearch/logs/` as paired `.json` and `.md`:
`gate_mutants`, `gain_mutants`, `gain_depth_diagnosis`, `rank_restricted`,
`run_report`.

Design notes as originally written:
`docs/superpowers/plans/2026-07-29-autoresearch-harness.md` and
`docs/superpowers/plans/2026-07-29-gain-mutants-and-rank.md`.
