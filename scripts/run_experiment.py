"""Fixed-budget runner for the autoresearch loop.

Trains src/autoresearch/candidate_optimizer.py's CandidateOptimizer on a
benchmark's model/data/eval setup, for a fixed wall-clock time per seed
(default 90s, override with --budget-seconds), across 4 seeds (bumped from 2
as part of the harness noise-floor fix, see research_log.md). The reported
per-seed loss is a deterministic full-test-set eval pass at the end of
training, not a trailing-train-batch average (the latter was the harness's
own measurement-noise bug: the wall-clock budget makes step count jitter
run-to-run even for identical code/seed, and averaging a handful of recent
train minibatches turned that jitter into large loss swings). Compares the
mean of the seed losses against the last-committed best *for that
benchmark* in research_log.md, requiring more than NOISE_MARGIN_REL relative
improvement (not a bare `<`) before keeping (snapshotting
candidate_optimizer.py under src/autoresearch/history/); otherwise discards
(reverting candidate_optimizer.py to the last committed snapshot for that
benchmark). Every attempt is logged, tagged with which benchmark it ran.

Benchmarks (--benchmark, default mnist):
    mnist  scripts/sanity_mnist.py's model, MNIST digits (Linear-only).
    cifar  scripts/sanity_cifar.py's model, CIFAR-10 (has conv layers --
           the thing MNIST's benchmark can never exercise). Each benchmark
           keeps its own independent committed-baseline chain (entries are
           tagged "benchmark" in research_log.md; untagged legacy entries
           are treated as "mnist" for backward compatibility) so a mnist
           improvement and a cifar improvement never get compared against
           each other's loss scale.

This file is the ONLY thing invoked by the loop. It is not the file future
agent sessions edit -- that's src/autoresearch/candidate_optimizer.py.

Usage:
    python scripts/run_experiment.py                        # one mnist run
    python scripts/run_experiment.py --benchmark cifar \\
        --budget-seconds 300                                # one cifar run
    python scripts/run_experiment.py --worker ...    # internal: single-seed
                                                      # subprocess, not for
                                                      # direct use.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchvision import datasets, transforms

from _device import resolve_device
from sanity_mnist import build_model as build_mnist_model
from sanity_cifar import build_model as build_cifar_model
from autoresearch.candidate_optimizer import CandidateOptimizer

SEEDS = (0, 1, 2, 3)
BUDGET_SECONDS = 90
# Measured empirically (research_log.md, harness noise-floor fix): 3 repeats
# of the unchanged committed mnist candidate at the real 4-seed/90s budget,
# under the eval-based metric above, gave 4-seed means 0.13352/0.13083/0.12842
# -- a ~4% relative spread from timing jitter alone (n_steps varies run to
# run even for identical code/seed; see run_worker's eval-metric comment).
# A bare `mean_loss < baseline_mean` would flag noise as "improved" a
# meaningful fraction of the time. Require improvement to clear this margin.
NOISE_MARGIN_REL = 0.05
TIMEOUT_SECONDS = 2 * BUDGET_SECONDS
DEFAULT_BENCHMARK = "mnist"

BENCHMARKS = {
    "mnist": {
        "build_model": build_mnist_model,
        "dataset_cls": datasets.MNIST,
        "dataset_kwargs": {"transform": transforms.ToTensor()},
    },
    "cifar": {
        "build_model": build_cifar_model,
        "dataset_cls": datasets.CIFAR10,
        "dataset_kwargs": {"transform": transforms.ToTensor()},
    },
}

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
CANDIDATE_PATH = REPO_ROOT / "src" / "autoresearch" / "candidate_optimizer.py"
HISTORY_DIR = REPO_ROOT / "src" / "autoresearch" / "history"
LOG_PATH = REPO_ROOT / "research_log.md"

LOG_HEADER = """# Autoresearch log

Fixed-budget optimizer search over src/autoresearch/candidate_optimizer.py,
driven by scripts/run_experiment.py. Each entry is one experiment attempt:
4 seeds x 90s wall-clock each, against scripts/sanity_mnist.py's model/data.
Per-seed loss is a deterministic full-test-set eval pass at budget end.

`status: committed` entries are candidates that beat the prior committed
mean and whose optimizer was snapshotted to src/autoresearch/history/.
Everything else is `status: discarded`; `reason` distinguishes
"tried and lost" (worse_loss) from "tried and broke" (any of DIVERGED,
EXCEPTION, TIMEOUT, possibly combined).

"""


# --------------------------------------------------------------------------
# Worker: trains one seed in its own process so the parent can enforce a
# hard wall-clock timeout by killing it.
# --------------------------------------------------------------------------

def _write_result(result_path, result):
    with open(result_path, "w") as f:
        json.dump(result, f)


def run_worker(seed, budget_seconds, lr, batch_size, device_name, result_path,
                benchmark=DEFAULT_BENCHMARK):
    result = {"seed": seed}
    try:
        bench = BENCHMARKS[benchmark]
        device = resolve_device(device_name)
        torch.manual_seed(seed)
        ds = bench["dataset_cls"](
            "data", train=True, download=True, **bench["dataset_kwargs"]
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True
        )
        model = bench["build_model"]().to(device)
        opt = CandidateOptimizer(model.parameters(), lr=lr, weight_decay=0.0)

        losses = []
        it = iter(loader)
        step = 0
        t0 = time.time()
        while time.time() - t0 < budget_seconds:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

            loss_value = float(loss.detach())
            if math.isnan(loss_value) or math.isinf(loss_value):
                result["result"] = "DIVERGED"
                result["step"] = step
                _write_result(result_path, result)
                return
            losses.append(loss_value)
            step += 1

        if not losses:
            result["result"] = "EXCEPTION"
            result["traceback"] = (
                f"budget of {budget_seconds}s elapsed before a single step completed"
            )
            _write_result(result_path, result)
            return

        # Root-cause fix for the harness noise floor (see research_log.md):
        # the wall-clock budget makes n_steps jitter run-to-run even for an
        # unchanged candidate + seed (confirmed: 343/364/380 steps at a 10s
        # budget on this machine -- pure timing jitter, not FP nondeterminism;
        # fixed-step reruns of the same code are bit-identical, see the
        # 20260729T noise-floor investigation). Averaging the last 10 TRAIN
        # minibatches made the reported loss a tiny, sampling-noisy statistic
        # that lands on whichever few random minibatches happen to fall in
        # the tail -- amplifying that step-count jitter into large loss
        # swings. A deterministic eval pass over the full (fixed, unshuffled)
        # test split has zero minibatch-sampling noise: the only remaining
        # variance is genuine (different training trajectories), not an
        # artifact of when the wall clock happened to cut off training.
        model.eval()
        eval_ds = bench["dataset_cls"](
            "data", train=False, download=True, **bench["dataset_kwargs"]
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_ds, batch_size=batch_size, shuffle=False
        )
        eval_losses = []
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                eval_losses.append(float(torch.nn.functional.cross_entropy(model(x), y)))

        result["result"] = "OK"
        result["n_steps"] = step
        result["loss"] = sum(eval_losses) / len(eval_losses)
        _write_result(result_path, result)
    except Exception:
        result["result"] = "EXCEPTION"
        result["traceback"] = traceback.format_exc()
        _write_result(result_path, result)


def run_seed_subprocess(seed, budget_seconds, timeout_seconds, lr, batch_size, device,
                         benchmark=DEFAULT_BENCHMARK):
    fd, result_path = tempfile.mkstemp(prefix=f"autoresearch_seed{seed}_", suffix=".json")
    os.close(fd)
    os.remove(result_path)  # worker creates it fresh; absence means it never got there
    cmd = [
        sys.executable, str(THIS_FILE),
        "--worker",
        "--seed", str(seed),
        "--budget-seconds", str(budget_seconds),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--device", device,
        "--benchmark", benchmark,
        "--result-file", result_path,
    ]
    proc = subprocess.Popen(cmd)
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        if os.path.exists(result_path):
            os.remove(result_path)
        return {"seed": seed, "result": "TIMEOUT"}

    if os.path.exists(result_path):
        with open(result_path) as f:
            data = json.load(f)
        os.remove(result_path)
        return data
    return {
        "seed": seed,
        "result": "EXCEPTION",
        "traceback": f"worker exited (code {proc.returncode}) without writing a result",
    }


# --------------------------------------------------------------------------
# Log: append-only, ```json``` blocks inside markdown headers so both
# humans (skim the headers) and this script (parse the json) read it.
# --------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def get_last_committed(benchmark=DEFAULT_BENCHMARK):
    if not LOG_PATH.exists():
        return None
    text = LOG_PATH.read_text()
    committed = []
    for match in _JSON_BLOCK_RE.finditer(text):
        try:
            entry = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        # Legacy entries predate --benchmark and are all mnist runs.
        if entry.get("status") == "committed" and entry.get("benchmark", "mnist") == benchmark:
            committed.append(entry)
    return committed[-1] if committed else None


def clean_seed_result(r):
    out = {"seed": r["seed"], "result": r["result"]}
    if r["result"] == "OK":
        out["loss"] = r["loss"]
        out["n_steps"] = r.get("n_steps")
    elif r["result"] == "DIVERGED":
        out["step"] = r.get("step")
    elif r["result"] == "EXCEPTION":
        out["traceback"] = r.get("traceback", "")[-4000:]
    return out


def append_entry(entry):
    if not LOG_PATH.exists():
        LOG_PATH.write_text(LOG_HEADER)
    header = f"## {entry['timestamp']} — {entry['status']} ({entry['reason']})\n\n"
    body = "```json\n" + json.dumps(entry, indent=2) + "\n```\n\n"
    with open(LOG_PATH, "a") as f:
        f.write(header + body)


def save_snapshot(timestamp):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    dest = HISTORY_DIR / f"candidate_{timestamp}.py"
    shutil.copy2(CANDIDATE_PATH, dest)
    return str(dest.relative_to(REPO_ROOT))


def revert_to_snapshot(relative_snapshot_path):
    src = REPO_ROOT / relative_snapshot_path
    shutil.copy2(src, CANDIDATE_PATH)


# --------------------------------------------------------------------------
# Main: run both seeds, decide keep/discard, log, exit.
# --------------------------------------------------------------------------

def main_experiment(lr, batch_size, device, budget_seconds=BUDGET_SECONDS,
                     benchmark=DEFAULT_BENCHMARK):
    timeout_seconds = 2 * budget_seconds
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    seed_results = [
        run_seed_subprocess(seed, budget_seconds, timeout_seconds, lr, batch_size, device,
                             benchmark)
        for seed in SEEDS
    ]
    for r in seed_results:
        print(f"seed {r['seed']}: {r['result']}"
              + (f" loss={r['loss']:.4f}" if r["result"] == "OK" else ""))

    statuses = [r["result"] for r in seed_results]
    all_ok = all(s == "OK" for s in statuses)
    cleaned = [clean_seed_result(r) for r in seed_results]

    last = get_last_committed(benchmark)

    if last is None:
        if all_ok:
            mean_loss = sum(r["loss"] for r in seed_results) / len(seed_results)
            snapshot = save_snapshot(timestamp)
            entry = {
                "timestamp": timestamp,
                "benchmark": benchmark,
                "status": "committed",
                "decision": "KEEP",
                "reason": "baseline",
                "seed_results": cleaned,
                "mean_loss": mean_loss,
                "baseline_mean": None,
                "snapshot": snapshot,
            }
            append_entry(entry)
            print(f"BASELINE COMMITTED: mean_loss={mean_loss:.4f} snapshot={snapshot}")
            return 0
        else:
            reason = ",".join(sorted({s for s in statuses if s != "OK"}))
            entry = {
                "timestamp": timestamp,
                "benchmark": benchmark,
                "status": "discarded",
                "decision": "DISCARD",
                "reason": reason,
                "seed_results": cleaned,
                "mean_loss": None,
                "baseline_mean": None,
                "snapshot": None,
                "note": "baseline establishment failed; fix candidate_optimizer.py and rerun",
            }
            append_entry(entry)
            print(f"BASELINE ESTABLISHMENT FAILED: {reason}")
            return 1

    baseline_mean = last["mean_loss"]

    if all_ok:
        mean_loss = sum(r["loss"] for r in seed_results) / len(seed_results)
        threshold = baseline_mean * (1 - NOISE_MARGIN_REL)
        if mean_loss < threshold:
            snapshot = save_snapshot(timestamp)
            entry = {
                "timestamp": timestamp,
                "benchmark": benchmark,
                "status": "committed",
                "decision": "KEEP",
                "reason": "improved",
                "seed_results": cleaned,
                "mean_loss": mean_loss,
                "baseline_mean": baseline_mean,
                "noise_margin_rel": NOISE_MARGIN_REL,
                "snapshot": snapshot,
            }
            append_entry(entry)
            print(f"KEEP: mean_loss={mean_loss:.4f} < threshold {threshold:.4f} "
                  f"(baseline {baseline_mean:.4f} - {NOISE_MARGIN_REL:.0%} margin); snapshot={snapshot}")
            return 0
        else:
            # Distinguish "looked numerically better but within measured
            # noise" from "actually worse" -- both discard, but conflating
            # them would hide how often near-misses are noise, not signal.
            reason = "within_noise_band" if mean_loss < baseline_mean else "worse_loss"
            revert_to_snapshot(last["snapshot"])
            entry = {
                "timestamp": timestamp,
                "benchmark": benchmark,
                "status": "discarded",
                "decision": "DISCARD",
                "reason": reason,
                "seed_results": cleaned,
                "mean_loss": mean_loss,
                "baseline_mean": baseline_mean,
                "noise_margin_rel": NOISE_MARGIN_REL,
                "snapshot": None,
            }
            append_entry(entry)
            print(f"DISCARD ({reason}): mean_loss={mean_loss:.4f} >= threshold {threshold:.4f} "
                  f"(baseline {baseline_mean:.4f} - {NOISE_MARGIN_REL:.0%} margin); reverted")
            return 0
    else:
        reason = ",".join(sorted({s for s in statuses if s != "OK"}))
        revert_to_snapshot(last["snapshot"])
        entry = {
            "timestamp": timestamp,
            "benchmark": benchmark,
            "status": "discarded",
            "decision": "DISCARD",
            "reason": reason,
            "seed_results": cleaned,
            "mean_loss": None,
            "baseline_mean": baseline_mean,
            "snapshot": None,
        }
        append_entry(entry)
        print(f"DISCARD ({reason}); reverted")
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--result-file", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--budget-seconds", type=float, default=BUDGET_SECONDS,
                     help="wall-clock seconds per seed (default 90)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK,
                     choices=sorted(BENCHMARKS), help="which benchmark to train/eval on")
    args = ap.parse_args()

    if args.worker:
        assert args.seed is not None and args.result_file is not None
        run_worker(args.seed, args.budget_seconds, args.lr, args.batch_size,
                   args.device, args.result_file, args.benchmark)
        return 0

    return main_experiment(args.lr, args.batch_size, args.device, args.budget_seconds,
                            args.benchmark)


if __name__ == "__main__":
    sys.exit(main())
