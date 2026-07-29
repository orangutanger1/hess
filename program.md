# autoresearch — optimizer research, DSN successor

*Modeled directly on karpathy/autoresearch. Same shape: one file the agent
edits, one fixed harness, one metric, a fixed time budget, keep-or-discard,
repeat overnight. The difference from that repo is only the domain (a
PyTorch optimizer, not a GPT training loop) and the seed knowledge below,
which this project already paid for and shouldn't have to re-learn.*

## Fixed budget

Each experiment trains for a **fixed wall-clock budget of 90 seconds**
(excluding setup), on the existing MNIST sanity harness. Fixed budget, not
fixed steps -- this keeps experiments comparable even when the agent's
change alters per-step cost (e.g. adding curvature products), the same
reason the nanochat version fixes wall-clock time rather than step count.

## Metric

**Loss at the end of the fixed budget. Lower is better.** One number. Do not
report FLOPs, memory, and accuracy as a bundle for keep/discard decisions --
that's a comparison-validity question for a final writeup, not a per-
experiment gate. Keep it to the one number that decides keep-or-discard.

## What you edit

`src/autoresearch/candidate_optimizer.py` -- one file. Implement the
`torch.optim.Optimizer` interface already used by DSN/AdamW in this repo.
Everything inside this file is fair game: preconditioner design, damping,
batching strategy, whether it uses curvature at all. Swap it out entirely
between experiments if you want -- there is no fixed architecture to mutate,
only a fixed interface to satisfy.

## What's fixed -- do not modify

- `scripts/sanity_mnist.py`'s data loading, model, and eval code
- `scripts/measure_rho.py` and the Lanczos machinery in `src/dsn/lanczos.py`,
  if you choose to reuse them from inside your candidate
- Anything under `src/dsn/` -- that project is closed; read from it, don't
  edit it

## Keep or discard

After each run, compare the candidate's loss to the current best-committed
optimizer. Strictly better -> commit it as the new best and note why in
`research_log.md`. Not better -> discard, revert `candidate_optimizer.py` to
the last committed version, try something else.

Append every experiment to `research_log.md`: what changed, the loss, the
decision, one sentence of why you think it moved (or didn't). This is a
paper trail for the morning review, not a gate -- nothing here blocks the
loop, but an un-logged run doesn't count as having happened.

## Seed knowledge -- read once before starting, and re-read after every few
experiments so it doesn't get lost as the file grows

These are measured facts from this repo's own history, not hunches. Don't
spend a night rediscovering them.

- **Gradient-seeded, single-batch curvature is measured disfavored here.**
  A Krylov subspace seeded by the current mini-batch's gradient, inverting
  that batch's curvature alone (no noise-awareness), scored rho < 1 at every
  checkpoint measured on this trajectory (max 0.72). If your candidate is a
  variant of this, say explicitly what's different and why it might escape
  that result -- don't just re-run it under a new name.
- **Accumulated, noise-aware statistics are the theoretically favored
  direction.** Whitening by gradient covariance (Shampoo/Adagrad-style,
  accumulated over many steps rather than estimated fresh each step) targets
  the actually-optimal direction this project's own ceiling analysis found.
  Bias exploration this way over one-shot Hessian estimation.
- **Any new self-scoring diagnostic must be mutant-tested before you trust
  it.** If you add a statistic used to decide whether some internal
  mechanism (reuse, recycling, adaptive damping) is helping, check that the
  statistic doesn't score BETTER as the underlying approximation gets WORSE
  -- construct one deliberately-broken variant (frozen/stale state, or a
  variant that ignores the thing it's supposed to measure) and confirm the
  statistic correctly flags it before you let it influence a keep/discard
  decision. This project shipped a metric that got 85x more optimistic as
  the real error got 3x worse. Don't repeat it. This is the one rule in this
  file that isn't optional.
- **Trust-region ratios computed across two different mini-batches are
  invalid.** Any accept/reject decision needs a same-batch or common-batch
  comparison, not one spanning the step boundary.

## Running the agent

Point Claude Code at this repo, open `program.md`, disable the permissions
you're comfortable disabling, and say something like:

> Have a look at program.md and let's kick off a new experiment.

## The experiment loop

Work on a dedicated branch (e.g. `autoresearch/jul29` or
`autoresearch/jul29-gpu0`).

`scripts/run_experiment.py` is the fixed harness -- it already does most of
the bookkeeping a generic loop would need you to do by hand: it runs 2 seeds
x 90s wall-clock each against `scripts/sanity_mnist.py`'s model/data, kills a
seed that exceeds 180s (2x budget) and records it as `TIMEOUT`, compares the
mean loss to the last committed baseline, appends a json entry to
`research_log.md`, snapshots the candidate to `src/autoresearch/history/` on
KEEP, and **reverts `src/autoresearch/candidate_optimizer.py` to the last
committed content on its own** if the result is DISCARD/DIVERGED/EXCEPTION/
TIMEOUT. You do not need to hand-manage a timeout or revert the file
yourself -- the script already did it by the time it exits.

LOOP FOREVER:

1. Check git state: current branch, `git status`, `git log -1`.
2. Edit `src/autoresearch/candidate_optimizer.py` with one experimental
   idea. Re-read the seed knowledge section above every few iterations.
3. Run the experiment: `uv run scripts/run_experiment.py > run.log 2>&1`
   (redirect everything -- do NOT use `tee` or let output flood your
   context).
4. Read the result: `tail -n 5 run.log`. The last line is one of
   `BASELINE COMMITTED: ...`, `KEEP: ...`, `DISCARD (worse_loss): ...`, or
   `DISCARD (<CRASH_REASON>); reverted`.
   - If the log is empty or the process didn't reach that line, the run
     crashed before the harness could log it -- `tail -n 50 run.log` for
     the Python traceback. If it's something dumb (typo, missing import,
     shape mismatch), fix it and re-run. If the idea itself is broken,
     skip it -- the harness already reverted the file for you.
5. On `BASELINE COMMITTED` or `KEEP`: the candidate is now the new best,
   already snapshotted under `src/autoresearch/history/`. `git add -A &&
   git commit` (message: what changed + the new `mean_loss` from the
   json entry `research_log.md` just appended). This advances the branch.
6. On any `DISCARD`: `src/autoresearch/candidate_optimizer.py` is already
   back to the last-committed content (the harness did this). Confirm
   with `git status` -- it should show no diff on that file (fallback:
   `git checkout -- src/autoresearch/candidate_optimizer.py` if it
   somehow doesn't). Move on to the next idea. Do not commit a discarded
   attempt.
7. Append the attempt to `results.csv` (untracked, see below) regardless
   of outcome, then go to 1.

If you feel like you're getting stuck, you can rewind further than one
commit, but do this very, very sparingly.

**Timeout:** already enforced inside `run_experiment.py` (180s per seed,
360s worst case for both seeds). You shouldn't need to kill anything by
hand; if a call to `uv run scripts/run_experiment.py` itself hangs past
~7 minutes, something is wrong with the harness, not the candidate --
investigate rather than discard-and-continue.

**Crashes:** `DIVERGED` (NaN/Inf loss), `EXCEPTION` (Python error), and
`TIMEOUT` are all logged as discard reasons in `research_log.md` and in
`results.csv`. Same judgment call as any other discard: fix the dumb stuff,
skip the fundamentally broken ideas.

**NEVER STOP:** once the loop has begun, do not pause to ask whether to
continue. Do not ask "should I keep going?". The human may be away from
the computer and expects you to keep iterating until manually interrupted.
If you run out of ideas: re-read the seed knowledge section, read
`src/dsn/` and `src/autoresearch/preconditioners.py` for approaches not
yet tried, revisit near-misses (candidates that lost narrowly) with a
tweak, or try a more radical rewrite of `candidate_optimizer.py`. At ~90s
of training + a few seconds of eval/subprocess overhead per attempt (x2
seeds), expect on the order of 15-20 attempts/hour -- the loop runs until
the human interrupts it, period.

### results.csv

Untracked (do not `git add` it -- it's a local run log, not part of the
repo history; `research_log.md` is the durable, committed record). Columns:

```
timestamp,branch,commit,status,reason,mean_loss,baseline_mean,seed0_result,seed0_loss,seed1_result,seed1_loss,snapshot,idea
```

Append one row per `run_experiment.py` invocation (`status` is
`committed`/`discarded`; `reason` is `baseline`/`improved`/`worse_loss`/
`DIVERGED`/`EXCEPTION`/`TIMEOUT`; `idea` is a short free-text note on what
changed that attempt). Leave `mean_loss`/seed fields blank when the run
never produced a loss (e.g. pure crash before any step).
