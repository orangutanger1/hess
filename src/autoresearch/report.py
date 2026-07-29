"""The final ledger: stage-1 cost, stage-2 cost, and the total per survivor.

This project's own finding is that a comparison which omits search cost flatters
whichever candidate got lucky early -- the eliminated arms' trials were spent
finding the survivor, and a ledger that lists only the survivor's training FLOPs
has quietly moved that spend off the books.  So two totals are reported for
every survivor and the *fully-loaded* one is the headline:

``marginal``      what this candidate cost on its own: its share of stage-1 plus
                  its own training trials.  The right number for "what would one
                  more seed of this cost".
``fully_loaded``  what the search that produced it cost: every candidate's
                  stage-1 screen, plus every *other* arm's training trials,
                  divided among the survivors, plus its own training.  The right
                  number for "what did this result cost".

The ratio between them is printed as one number, because that ratio is the whole
argument.  A search that screened forty candidates to promote one has a large
ratio and a comparison that ignores it is not a comparison.

Attribution conventions -- stated, because every one of them is a choice:

* Shared stage-1 probe cost (one set of Lanczos runs serving every spectral
  candidate at a checkpoint) is split evenly across the candidates that used it.
  The *marginal* cost of the second spectral candidate is genuinely zero, so any
  split is arbitrary; what is not arbitrary is that the total appears.
* The baseline arm's trials are search cost, not the survivor's cost.  They
  exist to support a comparison, and they would have been run whatever the
  survivor turned out to be.
* Fully-loaded cost is divided equally among survivors when more than one
  survives.  With a single survivor it carries the entire search.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .cost import CostModel, Ledger
from .gate import GateReport
from .select import SelectionReport


@dataclass
class CandidateLine:
    name: str
    gate_verdict: str
    #: "rank-k" or "full-rank" -- which operator the verdict is about.  A
    #: full-rank verdict does not transfer to a rank-restricted implementation
    #: of the same form, in either direction.
    gate_scope: str
    gate_reason: str
    rho_ceiling: float
    gain_ceiling: float
    stage1_flops: float
    stage2_tuning_flops: float
    stage2_flops: float
    n_trials: int
    chosen_lr: float | None
    mean_score: float
    status: str  # "survivor" | "eliminated" | "gate-rejected" | "baseline"
    marginal_total: float
    fully_loaded_total: float | None
    search_multiplier: float | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class FinalReport:
    lines: list[CandidateLine]
    cost_model: CostModel
    stage1_total: float
    tuning_total: float
    stage2_total: float
    survivors: list[str]
    baseline: str
    selection: SelectionReport | None
    gate_reports: Mapping[str, GateReport] = field(default_factory=dict)
    counterfactual_fixed_n: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def grand_total(self) -> float:
        return self.stage1_total + self.stage2_total

    def as_dict(self) -> dict:
        return {
            "cost_model": {
                "fwd_flops_per_sample": self.cost_model.fwd_flops_per_sample,
                "grad_multiplier": self.cost_model.grad_multiplier,
                "hvp_multiplier": self.cost_model.hvp_multiplier,
                "provenance": self.cost_model.provenance,
            },
            "stage1_total_flops": self.stage1_total,
            "stage2_tuning_total_flops": self.tuning_total,
            "stage2_total_flops": self.stage2_total,
            "grand_total_flops": self.grand_total,
            "counterfactual_fixed_n_flops": self.counterfactual_fixed_n,
            "baseline": self.baseline,
            "survivors": self.survivors,
            "candidates": [line.as_dict() for line in self.lines],
            "selection": self.selection.as_dict() if self.selection else None,
            "gate": {k: v.as_dict() for k, v in self.gate_reports.items()},
            "notes": self.notes,
        }


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.4g}"


def build(
    cost_model: CostModel,
    gate_reports: Mapping[str, GateReport],
    selection: SelectionReport | None,
    ledgers: Mapping[str, Ledger],
    gate_rejected: Sequence[str] = (),
    max_seeds: int | None = None,
) -> FinalReport:
    """Assemble the two stages into one ledger.

    ``ledgers`` carries per-candidate stage-1 and stage-2 FLOPs; it is built by
    the driver, which is the only place that knows both what was screened and
    what was trained.
    """
    stage1_total = sum(l.stage1 for l in ledgers.values())
    tuning_total = sum(l.stage2_tuning for l in ledgers.values())
    stage2_total = sum(l.stage2 for l in ledgers.values())

    survivors: list[str] = []
    baseline = ""
    if selection is not None:
        baseline = selection.baseline
        survivors = [n for n in selection.survivors if n != selection.baseline]

    lines: list[CandidateLine] = []
    for name, led in ledgers.items():
        gr = gate_reports.get(name)
        arm = selection.arms.get(name) if selection else None
        if name in gate_rejected:
            status = "gate-rejected"
        elif selection and name == selection.baseline:
            status = "baseline"
        elif arm is not None and not arm.active:
            status = "eliminated"
        elif name in survivors:
            status = "survivor"
        else:
            status = "screened"

        fully_loaded = None
        multiplier = None
        if status == "survivor" and survivors:
            # Everything except this candidate's own selection trials is search:
            # every gate screen, every tuning run (including this candidate's own
            # -- tuning buys a hyperparameter, not the result), and every other
            # arm's trials.
            others = sum(
                l.stage2 for n2, l in ledgers.items() if n2 != name
            )
            search = stage1_total + others + led.stage2_tuning
            fully_loaded = led.stage2_training + search / len(survivors)
            if led.stage2_training > 0:
                multiplier = fully_loaded / led.stage2_training

        lines.append(
            CandidateLine(
                name=name,
                gate_verdict=gr.verdict.value if gr else "not-screened",
                gate_scope=(
                    ("rank-%d" % gr.rank if gr.rank else "full-rank") if gr else "-"
                ),
                gate_reason=gr.reason if gr else "",
                rho_ceiling=gr.rho_ceiling if gr else float("nan"),
                gain_ceiling=gr.gain_ceiling if gr else float("nan"),
                stage1_flops=led.stage1,
                stage2_tuning_flops=led.stage2_tuning,
                stage2_flops=led.stage2_training,
                n_trials=led.n_trials,
                chosen_lr=led.chosen_lr,
                mean_score=arm.mean if arm else float("nan"),
                status=status,
                marginal_total=led.marginal_total,
                fully_loaded_total=fully_loaded,
                search_multiplier=multiplier,
            )
        )

    counterfactual = None
    if selection is not None and max_seeds:
        trained = [l for l in ledgers.values() if l.n_trials > 0]
        if trained:
            per_trial = sum(l.stage2_training for l in trained) / sum(
                l.n_trials for l in trained
            )
            counterfactual = (
                stage1_total + tuning_total + per_trial * max_seeds * len(trained)
            )

    notes = [
        "The `scope` column says which operator each gate verdict is about. A "
        "full-rank verdict does not transfer to a rank-restricted implementation "
        "of the same form, in either direction: a restricted P with a positive "
        "complement is itself full-rank and invertible, so it is a different "
        "operator rather than a weaker one. Screen the restriction directly with "
        "KrylovRestrictedPreconditioner.",
        "Stage-1 cost includes every candidate screened, including the ones the "
        "gate rejected before any training FLOP was spent -- that is the point of "
        "having a stage 1, and hiding it would overstate the saving.",
        "Fully-loaded totals charge the survivor for the whole search. Marginal "
        "totals charge it for itself. Both are shown because they answer "
        "different questions, and quoting only the second is the failure mode "
        "this ledger exists to prevent.",
    ]
    return FinalReport(
        lines=sorted(lines, key=lambda l: (l.status != "survivor", l.name)),
        cost_model=cost_model,
        stage1_total=stage1_total,
        tuning_total=tuning_total,
        stage2_total=stage2_total,
        survivors=survivors,
        baseline=baseline,
        selection=selection,
        gate_reports=dict(gate_reports),
        counterfactual_fixed_n=counterfactual,
        notes=notes,
    )


def render(rep: FinalReport) -> str:
    out = [
        "# Autoresearch harness -- run report",
        "",
        "## Cost ledger",
        "",
        f"FLOP model: `{rep.cost_model.provenance}`",
        "",
        "| candidate | gate | scope | rho ceil | gain ceil | stage-1 FLOPs | "
        "tuning FLOPs | stage-2 FLOPs | trials | lr | mean score | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for l in rep.lines:
        out.append(
            f"| {l.name} | {l.gate_verdict} | {l.gate_scope} | {_fmt(l.rho_ceiling)} | "
            f"{_fmt(l.gain_ceiling)} | {_fmt(l.stage1_flops)} | "
            f"{_fmt(l.stage2_tuning_flops)} | {_fmt(l.stage2_flops)} | {l.n_trials} | "
            f"{_fmt(l.chosen_lr)} | {_fmt(l.mean_score)} | {l.status} |"
        )

    out += [
        "",
        f"- stage 1 (gate screen, all candidates): **{rep.stage1_total:.4g}** FLOPs",
        f"- stage 2 tuning (learning-rate search, all arms): "
        f"**{rep.tuning_total:.4g}** FLOPs",
        f"- stage 2 selection (training trials, all arms): "
        f"**{rep.stage2_total - rep.tuning_total:.4g}** FLOPs",
        f"- grand total: **{rep.grand_total:.4g}** FLOPs",
    ]
    if rep.counterfactual_fixed_n:
        saving = rep.counterfactual_fixed_n / rep.grand_total if rep.grand_total else float("nan")
        out.append(
            f"- a fixed N-seeds-per-candidate design at the same seed cap would "
            f"have cost **{rep.counterfactual_fixed_n:.4g}** FLOPs "
            f"({saving:.2f}x this run)"
        )

    out += ["", "## Total cost per surviving candidate", ""]
    if not rep.survivors:
        out.append(
            "No candidate survived. The whole run's cost is search cost, and there "
            "is nothing to amortize it against -- which is a result, not a failure "
            "of accounting."
        )
    else:
        out += [
            "| survivor | own training FLOPs | search + training (fully loaded) | "
            "search multiplier |",
            "| --- | ---: | ---: | ---: |",
        ]
        for l in rep.lines:
            if l.status != "survivor":
                continue
            out.append(
                f"| {l.name} | {_fmt(l.stage2_flops)} | {_fmt(l.fully_loaded_total)} | "
                f"{_fmt(l.search_multiplier)}x |"
            )
        out.append("")
        out.append(
            "The multiplier is the number to quote against any published "
            "training-FLOPs-to-target figure. Omitting it makes the comparison "
            "unfairly favorable to whichever candidate the search happened to keep."
        )

    if rep.selection is not None:
        sel = rep.selection
        out += [
            "",
            "## Stage 2 -- eliminations and the evidence for them",
            "",
            f"Stopped after {sel.rounds} rounds and {sel.total_pulls} trials: "
            f"{sel.stopped_because}.",
            "",
        ]
        if sel.eliminations:
            out += [
                "| round | eliminated | against | seeds | mean | LCB | rival UCB | alpha |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for e in sel.eliminations:
                out.append(
                    f"| {e['round']} | {e['arm']} | {e['against']} | {e['n_seeds']} | "
                    f"{e['mean']:.4g} | {e['lcb']:.4g} | {e['rival_ucb']:.4g} | "
                    f"{e['alpha_allocated']:.4g} |"
                )
        else:
            out.append("No arm was eliminated: no pair separated at its allocated alpha.")

        risk = sel.final_risk
        if risk is not None:
            out += [
                "",
                "## False-discovery risk after "
                f"{risk.n_comparisons} comparisons",
                "",
                f"- family-wise budget: **{risk.fwer_alpha_spent:.4g}** spent of "
                f"**{risk.fwer_alpha_total:.4g}** allocated "
                f"(`{sel.alpha_ledger['schedule']}` schedule over "
                f"{sel.alpha_ledger['n_candidates']} candidates)",
                f"- FDR target q = {risk.fdr_q:.4g}; Benjamini-Yekutieli critical "
                f"value {risk.by_critical:.4g}",
                f"- **discoveries (BY, valid under arbitrary dependence): "
                f"{risk.by_discoveries or 'none'}**",
                f"- discoveries (BH, needs independence/PRDS -- not defensible here): "
                f"{risk.bh_discoveries or 'none'}",
                f"- discoveries if multiplicity were ignored: "
                f"{risk.naive_discoveries or 'none'}",
                f"- expected false discoveries among the BY set: "
                f"<= {risk.expected_false_discoveries:.4g}",
                "",
                f"**Had each of these {risk.n_comparisons} comparisons been tested at "
                f"an unadjusted {risk.fdr_q:.2g}, the chance of at least one false "
                f"positive would be {risk.naive_fwer:.1%}** -- against the "
                f"{risk.fwer_alpha_total:.2g} family-wise budget actually enforced. "
                "That gap is why the pool-level budget exists, and it widens with "
                "every candidate added.",
                "",
                "### Running risk by round",
                "",
                "| round | comparisons | alpha spent | naive FWER if unadjusted | "
                "BY discoveries |",
                "| ---: | ---: | ---: | ---: | --- |",
            ]
            for r in sel.risk:
                out.append(
                    f"| {r.round} | {r.n_comparisons} | {r.fwer_alpha_spent:.4g} | "
                    f"{r.naive_fwer:.1%} | {', '.join(r.by_discoveries) or 'none'} |"
                )

        out += ["", "### Assumptions this rests on", ""]
        out += [f"- {a}" for a in sel.assumptions]

    out += ["", "### Ledger conventions", ""]
    out += [f"- {n}" for n in rep.notes]
    out.append("")
    return "\n".join(out)


def write(rep: FinalReport, directory) -> tuple[str, str]:
    import pathlib

    d = pathlib.Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    j = d / "run_report.json"
    m = d / "run_report.md"
    j.write_text(json.dumps(rep.as_dict(), indent=1, default=str))
    m.write_text(render(rep))
    return str(j), str(m)
