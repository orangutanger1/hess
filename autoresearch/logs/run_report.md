# Autoresearch harness -- run report

## Cost ledger

FLOP model: `analytic: 2 * 24 params/sample, 3x grad, 6x curvature`

| candidate | gate | scope | rho ceil | gain ceil | stage-1 FLOPs | tuning FLOPs | stage-2 FLOPs | trials | lr | mean score | status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed-random(seed=7,s=0.5) | PASS | full-rank | 1.597 | 1.198 | 3.723e+07 | 1.106e+07 | 7.373e+06 | 8 | 0.01 | 0.006978 | survivor |
| dsn(delta=0.001,k_max=8) | FAIL | rank-8 | 3.646 | 0.2895 | 3.723e+07 | 0 | 0 | 0 | - | - | gate-rejected |
| sgd | FAIL | full-rank | 1 | 1 | 3.723e+07 | 1.106e+07 | 7.373e+06 | 8 | 0.01 | 0.006427 | baseline |

- stage 1 (gate screen, all candidates): **1.117e+08** FLOPs
- stage 2 tuning (learning-rate search, all arms): **2.212e+07** FLOPs
- stage 2 selection (training trials, all arms): **1.475e+07** FLOPs
- grand total: **1.486e+08** FLOPs
- a fixed N-seeds-per-candidate design at the same seed cap would have cost **1.486e+08** FLOPs (1.00x this run)

## Total cost per surviving candidate

| survivor | own training FLOPs | search + training (fully loaded) | search multiplier |
| --- | ---: | ---: | ---: |
| fixed-random(seed=7,s=0.5) | 7.373e+06 | 1.486e+08 | 20.15x |

The multiplier is the number to quote against any published training-FLOPs-to-target figure. Omitting it makes the comparison unfairly favorable to whichever candidate the search happened to keep.

## Stage 2 -- eliminations and the evidence for them

Stopped after 8 rounds and 16 trials: seed budget exhausted with several arms alive.

No arm was eliminated: no pair separated at its allocated alpha.

## False-discovery risk after 1 comparisons

- family-wise budget: **0.038** spent of **0.05** allocated (`zeta2` schedule over 2 candidates)
- FDR target q = 0.1; Benjamini-Yekutieli critical value 0
- **discoveries (BY, valid under arbitrary dependence): none**
- discoveries (BH, needs independence/PRDS -- not defensible here): none
- discoveries if multiplicity were ignored: none
- expected false discoveries among the BY set: <= 0

**Had each of these 1 comparisons been tested at an unadjusted 0.1, the chance of at least one false positive would be 10.0%** -- against the 0.05 family-wise budget actually enforced. That gap is why the pool-level budget exists, and it widens with every candidate added.

### Running risk by round

| round | comparisons | alpha spent | naive FWER if unadjusted | BY discoveries |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 0.038 | 10.0% | none |
| 2 | 1 | 0.038 | 10.0% | none |
| 3 | 1 | 0.038 | 10.0% | none |
| 4 | 1 | 0.038 | 10.0% | none |
| 5 | 1 | 0.038 | 10.0% | none |
| 6 | 1 | 0.038 | 10.0% | none |
| 7 | 1 | 0.038 | 10.0% | none |
| 8 | 1 | 0.038 | 10.0% | none |

### Assumptions this rests on

- Scores are sub-Gaussian across seeds with the plug-in sigma_hat inflated by 1.25x; the anytime bounds are exact for a known sigma and approximate for an estimated one.
- All arms share the seed set, so candidate-vs-baseline differences are paired over the seeds both arms actually ran.
- Family-wise error is budgeted over the pool in order of entry (zeta2); a candidate proposed later is held to a stricter bar than the same candidate proposed earlier.
- Benjamini-Yekutieli is used for discoveries because arms share seeds, share data order and are stopped adaptively, so no positive-dependence assumption is available.
- Divergent trials are score-capped, which makes an unstable arm's score distribution bimodal and violates the sub-Gaussian assumption. Such arms are disqualified on stability instead, spending no alpha; the mean comparisons are only applied to arms that stayed in their stable regime.

### Ledger conventions

- The `scope` column says which operator each gate verdict is about. A full-rank verdict does not transfer to a rank-restricted implementation of the same form, in either direction: a restricted P with a positive complement is itself full-rank and invertible, so it is a different operator rather than a weaker one. Screen the restriction directly with KrylovRestrictedPreconditioner.
- Stage-1 cost includes every candidate screened, including the ones the gate rejected before any training FLOP was spent -- that is the point of having a stage 1, and hiding it would overstate the saving.
- Fully-loaded totals charge the survivor for the whole search. Marginal totals charge it for itself. Both are shown because they answer different questions, and quoting only the second is the failure mode this ledger exists to prevent.
- Gate mutant validation: ACHIEVED (autoresearch/logs/gate_mutants.json).
- Wall clock: stage 1 6.2s, stage 2 4.3s. FLOPs, not seconds, are the comparison metric; timings are reported only so a run can be reproduced at a known scale.
