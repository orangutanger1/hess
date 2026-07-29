# Autoresearch log

Fixed-budget optimizer search over src/autoresearch/candidate_optimizer.py,
driven by scripts/run_experiment.py. Each entry is one experiment attempt:
2 seeds x 90s wall-clock each, against scripts/sanity_mnist.py's model/data.

`status: committed` entries are candidates that beat the prior committed
mean and whose optimizer was snapshotted to src/autoresearch/history/.
Everything else is `status: discarded`; `reason` distinguishes
"tried and lost" (worse_loss) from "tried and broke" (any of DIVERGED,
EXCEPTION, TIMEOUT, possibly combined).

## 20260729T121730Z — committed (baseline)

```json
{
  "timestamp": "20260729T121730Z",
  "status": "committed",
  "decision": "KEEP",
  "reason": "baseline",
  "seed_results": [
    {
      "seed": 0,
      "result": "OK",
      "loss": 0.03919068006798625,
      "n_steps": 3514
    },
    {
      "seed": 1,
      "result": "OK",
      "loss": 0.03059880044311285,
      "n_steps": 3442
    }
  ],
  "mean_loss": 0.03489474025554955,
  "baseline_mean": null,
  "snapshot": "src/autoresearch/history/candidate_20260729T121730Z.py"
}
```

## 20260729T122558Z — discarded (worse_loss)

```json
{
  "timestamp": "20260729T122558Z",
  "status": "discarded",
  "decision": "DISCARD",
  "reason": "worse_loss",
  "seed_results": [
    {
      "seed": 0,
      "result": "OK",
      "loss": 0.7265118300914765,
      "n_steps": 2719
    },
    {
      "seed": 1,
      "result": "OK",
      "loss": 0.5839226901531219,
      "n_steps": 2614
    }
  ],
  "mean_loss": 0.6552172601222992,
  "baseline_mean": 0.03489474025554955,
  "snapshot": null
}
```

