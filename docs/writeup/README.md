# Writeup

Four results from this project. **Each is readable on its own** and none depends
on the others being read first. They are deliberately kept separate because
three of them stand independently of whether DSN worked.

| document | what it is | depends on DSN? |
| --- | --- | --- |
| [`dsn-negative-result.md`](dsn-negative-result.md) | DSN as designed does not beat AdamW: four independent instruments, the full chain of evidence, and an appendix of the F1–F10 defect history that establishes the implementation was correct | this **is** the DSN result |
| [`self-referential-diagnostics.md`](self-referential-diagnostics.md) | A metric computed from the same approximation it evaluates cannot detect that approximation's failure — and gets *better* as the approximation degrades. Six measured instances, and the countermeasures that worked | no |
| [`krylov-recycling-in-trust-region-newton.md`](krylov-recycling-in-trust-region-newton.md) | Deflation recycling is break-even by an exact accounting; augmented (FACA) recycling does not open under a saddle-free damped solve, and the reason is not batching noise | no |
| [`autoresearch-harness.md`](autoresearch-harness.md) | `src/autoresearch/` as reusable infrastructure: cheap gate screen, FWER-controlled sequential selection, mutant-validation discipline, and the calibration limitation the gate statistics still have | no |

The two standalone numerical findings (2 and 3) and the infrastructure (4) were
produced *while* investigating DSN, not *because* DSN failed. Their evidence is
arithmetic and controlled measurement, not a benchmark outcome.

## Scope, for all four

Everything measured here is CPU-tier and float64: a 784-96-tanh-10 MLP (76k
parameters) on MNIST, synthetic quadratics of dimension 24–400, ill-conditioned
logistic problems, and synthetic mini-batch streams. No GPU tier was run.

## Provenance

These documents supersede the contemporaneous records in
`docs/superpowers/plans/`, which are kept unedited for provenance and are cited
from each writeup. Where a writeup corrects a number or a claim in those
records, it says so explicitly.
