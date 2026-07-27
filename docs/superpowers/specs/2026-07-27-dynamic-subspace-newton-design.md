# Dynamic Subspace Newton (DSN) — Design

**Date:** 2026-07-27
**Status:** Approved, pre-implementation

## Problem

Full second-order optimization is impractical at scale: constructing and inverting the Hessian
costs `O(n²)` memory and `O(n³)` time for `n` parameters. Existing methods approximate curvature
in low-rank, block-structured (K-FAC, Shampoo), or otherwise predefined subspaces. None directly
optimizes for the *smallest dynamically changing subspace that most accurately predicts the true
second-order update*.

This project asks: can second-order optimization be made cheaper by dynamically identifying such
a subspace from gradient, update, and implicit curvature information, estimating curvature only
inside it, and running a first-order method in the orthogonal complement?

## Key insight

The true Newton step `d* = −H⁻¹g` is unknown, so "accuracy of the predicted second-order update"
looks unmeasurable. But the **Newton residual** is measurable:

```
r(d) = H d + g          ‖r(d*)‖ = 0
```

evaluating it costs one Hessian-vector product. Minimizing `‖Hd + g‖` over `d ∈ S` is exactly what
Krylov methods (MINRES / conjugate residual) do by construction. So:

- The subspace-quality objective in the problem statement becomes an **online, computable** criterion.
- The Krylov space `K_k(H, g) = span{g, Hg, …, H^{k−1}g}` is the *optimal* `k`-dimensional subspace
  reachable from `g` via `H`-products under that criterion.
- "Smallest such subspace" becomes a concrete stopping rule: expand until relative residual crosses
  a tolerance.

One consequence worth stating because most low-rank curvature work gets it backwards: the Newton
step weights eigendirections by `1/λ`, so **small** eigenvalues dominate the update. Methods that
approximate the **top** eigenspace optimize the wrong end of the spectrum. A Krylov space seeded
from `g` naturally weights by where the gradient actually has mass. Backend C below exists to
demonstrate this empirically.

## Novelty

Adaptive Krylov curvature alone is Hessian-free / truncated Newton (Martens 2010), which is not new.
The contributions here are:

1. **Cross-step subspace recycling.** Standard Hessian-free discards its Krylov basis every step and
   pays 20–100 HVPs. Curvature eigenvectors are strongly correlated between adjacent steps. Retain
   the top Ritz vectors, deflate against them, and build only a few *new* directions per step.
   Recycled Krylov (GCRO-DR, Parks et al. 2006) is standard in numerical linear algebra and
   essentially unused in neural network training. This is the "dynamically changing subspace."
2. **First-order complement.** Hessian-free returns `d ∈ K_k` and does nothing outside it. DSN takes
   a Newton step inside `S` and an Adam step in `S⊥`, so no coordinate is left unoptimized.
3. **Adaptive `k` by marginal value.** Grow the subspace while the residual reduction per HVP exceeds
   a threshold — a cost-aware realization of "smallest subspace," not a fixed rank.

## Algorithm

Persistent state: `U ∈ ℝ^{n×m}` recycled orthonormal basis, `HU` its curvature images, Adam moments
`(m₁, m₂)`, trust radius `Δ`.

Step *t*:

1. `g = ∇L(θ; B)`.
2. Curvature operator `H·v`: Gauss-Newton-vector product (forward-over-reverse) by default; raw
   Hessian-vector product optional. Evaluated on curvature batch `B_c`, which may differ from `B`.
3. Deflate: `g⊥ = g − U Uᵀ g`.
4. **Adaptive Lanczos** on `P H P` with `P = I − UUᵀ`, seeded `q₁ = g⊥/‖g⊥‖`. Full
   reorthogonalization against `[U, Q]` each iteration — `O(nk)`, negligible beside an HVP.
   Stop when any of:
   - relative Newton residual `‖Hd + g‖ / ‖g‖ < τ`
   - marginal residual drop per HVP `< ε`
   - `j = k_max`

   The residual comes free from the Lanczos recurrence (`|β_{j+1} eⱼᵀ y|`); no extra HVP.
5. `W = [U, Q]`. Assemble `T = WᵀHW` from the Lanczos tridiagonal plus stored `HU` blocks.
   Eigendecompose `T = V Λ Vᵀ` — size `≤ m + k_max`, cost negligible.
6. **Saddle-free damped inverse:**
   `y = −V diag(1/(|λᵢ| + λ_d)) Vᵀ Wᵀ g`, `d_S = W y`.
   Negative curvature is handled by absolute value (Dauphin et al. 2014), not clipped away.
   Having `Λ` explicitly is a side benefit of Lanczos that a CG-based solver does not get.
7. **Complement:** `a = AdamStep(g)`, `d⊥ = a − W Wᵀ a`.
8. `d = clip_Δ(d_S) + d⊥`. Adapt `Δ` on the actual/predicted reduction ratio `ρ`.
9. **Recycle:** retain top-`m` Ritz vectors ranked by step contribution `|yᵢ|`. Ranking rule is a
   config knob (`|y|` / smallest-`|λ|` / largest-`|λ|`) and an ablation. Store `U`, `HU`.

### Correctness invariants (enforced as tests)

| Condition | Required behavior |
|---|---|
| `m = k = 0` | update is bit-identical to AdamW |
| `k = n`, no damping, no trust region | `d` equals the exact Newton step to solver tolerance |
| after reorthogonalization | `Q ⊥ U` to machine precision |
| complement projection | `d⊥ ⊥ W` |
| Lanczos residual estimate | matches explicitly computed `‖Hd + g‖` |

These are the difference between a working optimizer and one that silently degrades to a badly-tuned
first-order method — the most likely failure mode, and the hardest to notice from loss curves alone.

## Structure

```
src/dsn/
  optimizer.py      DSN(torch.optim.Optimizer)
  curvature.py      HVP / GGN-vp, damping, saddle-free inverse
  subspace/
    base.py         SubspaceBuilder protocol -> (W, T)
    krylov.py       A: recycled adaptive Lanczos          [main method]
    history.py      B: gradient/update secant pairs, zero HVP   [ablation]
    sketch.py       C: randomized block power iteration         [ablation]
  complement.py     Adam state + orthogonal projection
  telemetry.py      residual, k_t, FLOP counter, memory probe
benchmarks/
  models/ data/ baselines/ run.py sweep.py
analysis/           plots.py tables.py
configs/            mlp_mnist.yaml resnet_cifar.yaml gpt_wikitext.yaml
remote/             vast_up.sh vast_sync.sh vast_run.sh
autoresearch/       harness.py     immutable: model, data, budget, metric
                    candidate.py   the single agent-mutable file
                    program.md results.tsv journal.md              [phase 8]
tests/
```

All three subspace backends satisfy one `SubspaceBuilder` interface returning `(W, T)`, so ablations
require no rewrite of the optimizer.

## Measurement

The headline metric is **loss versus FLOPs**, not loss versus steps. A method that halves step count
at 4× per-step cost is a regression, and step-count plots hide exactly that. FLOPs are counted
analytically and cross-checked against `torch.utils.flop_counter.FlopCounterMode`.

Reported per the problem statement:

- time to target loss (wall-clock, same GPU)
- total FLOPs to target loss
- peak memory (`torch.cuda.max_memory_allocated`)
- convergence rate (loss vs step *and* vs FLOP)
- final accuracy

Method telemetry: `k_t` trajectory, Newton residual, recycling reuse fraction, Ritz spectrum.

Baselines: SGD+momentum, AdamW, K-FAC, Sophia, and **fixed-`k` Hessian-free**. The last is the
critical ablation — it isolates "adaptive + recycled" from "Krylov at all."

### Hardware

Development and unit tests run on local CPU (12 cores, 7 GB RAM, no usable GPU — installed torch is
built for CUDA 13.0 against driver 12030). Sweeps run on a rented vast.ai 3090.

| Tier | Params | Peak VRAM (k=16, bf16 basis) |
|---|---|---|
| MLP / MNIST | ~0.1 M | CPU |
| ResNet-18 / CIFAR-10 | 11 M | ~1.6 GB |
| 6-layer GPT / WikiText-103 | 30 M | ~4.5 GB |

VRAM is not the binding constraint; GPU-hours are. Estimated ~35 GPU-hr ≈ $8–12 for the full study.

## Failure handling

| Condition | Response |
|---|---|
| Lanczos breakdown (`β ≈ 0`) | subspace exhausted; stop early, use what exists |
| loss of orthogonality | full reorthogonalization always (`k` is small, so affordable) |
| non-finite curvature | fall back to pure Adam step for that iteration, log it |
| trust-region rejection | shrink `Δ`, take Adam-only step; do not burn HVPs on a retry |
| curvature-batch OOM | halve `B_c`, log |

Every fallback increments a counter in telemetry. A run where DSN silently spent 90% of its steps in
the Adam fallback path must be visibly distinguishable from one where it did not.

## Phases

| Phase | Content | Verification |
|---|---|---|
| P0 | repo, curvature ops | HVP matches finite differences and dense autograd Hessian on `n < 50` |
| P1 | DSN optimizer, Krylov backend | exact-Newton and Adam-degeneracy invariants pass |
| P2 | local sanity (CPU) | quadratic bowl converges in `≤ rank(H)` steps; logistic regression beats SGD per-iteration; MNIST MLP trains |
| P3 | harness, baselines, metrics | FLOP counter agrees with `FlopCounterMode` within 5% |
| P4 | vast.ai plumbing, CIFAR-10 sweep | 3 seeds × 6 optimizers complete, results reproducible from jsonl |
| P5 | LM tier sweep | same |
| P6 | ablations: B, C, fixed-`k`, no-recycle, ranking rule | each isolates one design decision |
| P7 | analysis, writeup | plots and tables regenerate from results with one command |
| P8 | autoresearch loop (conditional — see below) | agent completes ≥50 screened experiments unattended |

## Pre-registered success and kill criteria

Declared before any data is collected, so the bar cannot move afterward.

Definitions, fixed now:

- **Target loss** = the validation loss AdamW reaches at the end of its standard budget for that
  tier (CIFAR-10: 100 epochs; WikiText: 50k steps). Every optimizer is then scored on the FLOPs it
  needed to first reach that value.
- **Equal tuning budget** = 16 trials of random search per optimizer, over learning rate plus that
  optimizer's one main knob (DSN: `τ`; AdamW: weight decay; K-FAC: damping; Sophia: `ρ`;
  Hessian-free: fixed `k`). Best trial per optimizer advances to the 3-seed comparison.

**DSN succeeds** if, under that tuning budget, it reaches target loss in fewer total FLOPs than
AdamW on at least one of the two GPU tiers, with 3 seeds and non-overlapping error bars.

**DSN fails** if it does not, on either tier, under that same standard.

A negative result is a real result and gets written up as one. If the mechanism fails, the most
valuable output is *why* — which of the two named risks fired:

- recycled eigenvectors go stale faster than cross-step curvature correlation predicts (measurable
  directly via the reuse-fraction telemetry), or
- Adam's per-coordinate scaling in the complement conflicts with the trust region inside `S`
  (measurable via the trust-region rejection rate).

P2 is designed to expose both on CPU, before any money is spent.

## Phase 8 — autoresearch loop

Triggered if the kill criterion fires. Modeled on Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch): an agent edits one file, runs a timeboxed
experiment, keeps or reverts by git, and repeats unattended.

Adopted from it:

- **Immutable harness owns the metric.** `autoresearch/harness.py` fixes model, data, budget, and
  evaluation; neither human nor agent edits it, so every experiment is measured on one yardstick.
- **One mutable file.** The agent edits only the subspace-selection and update rule. Everything in
  that file is fair game.
- **Human directives in `program.md`.** Research direction, things already ruled out, things not to
  try.
- **`results.tsv` audit trail** — commit hash, score, memory, pass/fail, description.
- **Git ratchet** — keep the commit or reset it.

Changed, because copying it directly would produce garbage for an optimizer study:

1. **Noise gate.** Karpathy's loop takes a single timeboxed run and keeps it if the metric improved.
   Across hundreds of experiments against a noisy metric, that banks noise as progress. Here:
   estimate baseline `σ` from repeated seeds up front, and promote a candidate only if it beats the
   incumbent by `> 2σ` on 3 seeds.
2. **Incumbent pool, not a single trunk.** A strict ratchet cannot express "worse before better."
   Keep the top 3 incumbents; the agent may branch from any of them.
3. **FLOP budget, not wall-clock timebox.** A fixed 5-minute box structurally penalizes methods that
   pay setup cost early and win later — precisely the class under test. The metric is loss at a
   fixed FLOP budget, and promoted candidates get a longer confirmation run.
4. **Two-tier screening.** Cheap MNIST-MLP screen on local CPU (~1 min/run, free), promotion to the
   GPU CIFAR tier only on survivors. Keeps vast.ai spend near zero during search.
5. **`journal.md`** — accumulated findings, so the agent does not rediscover the same dead ends.

The loop's own success criterion is the same pre-registered bar: a discovered variant must beat
AdamW on FLOPs-to-target-loss on a GPU tier with non-overlapping error bars over 3 seeds.

## Open item

vast.ai CLI (`vastai` 1.5.0) is installed but unauthenticated. `vastai set api-key <KEY>` is required
before P4. Phases P0–P3 and the P8 screening tier need no GPU.

## References

- Martens 2010, *Deep learning via Hessian-free optimization*
- Parks et al. 2006, *Recycling Krylov subspaces for sequences of linear systems* (GCRO-DR)
- Dauphin et al. 2014, *Identifying and attacking the saddle point problem*
- Karpathy 2026, [autoresearch](https://github.com/karpathy/autoresearch)
