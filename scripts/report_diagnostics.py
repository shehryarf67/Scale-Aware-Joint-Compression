r"""Mechanism diagnostics for the confirmatory run, computed entirely from committed records.

WHY THIS EXISTS
---------------
[F-37](../docs/findings_log.md#f-37) reports *whether* joint beats sequential. It does not say why
the effect is 4-bit-specific, nor why it does not grow with scale. These five diagnostics are the
evidence for the mechanism discussion, and every one comes out of run records that already exist --
**nothing here re-runs or re-tunes the frozen grid.**

WHAT IT REPORTS, AND WHAT EACH ONE CAN AND CANNOT SUPPORT
---------------------------------------------------------
1. **Mask disagreement, joint vs sequential, by scale.** `joint_trace[].mask_divergence` is the
   fraction of mask positions the joint refinement moved away from the mask a sequential pass would
   have chosen. It is recorded on the joint arm only, which is the correct place: it *is* the
   joint-versus-sequential difference. If the joint mechanism is inert, this is ~0.

2. **Layerwise joint advantage, by scale.** Within the joint arm, `loss_before` -> `loss_proposed`
   per accepted iteration. This is the *local* objective gain the joint step buys, before any
   interaction with the rest of the network. A positive model-level gain with a near-zero layer gain
   would mean the effect is not coming from where the method says it does.

3. **Solver efficiency, by scale and arm.** `relative_improvement` = (naive - final) / naive per
   layer: how much of the reconstruction gain the one-pass sweep captures relative to naive masking.
   **This is not slack against the provable optimum** -- that needs the exact solve of
   [F-21](../docs/findings_log.md#f-21), which ran at 160M only and cannot be recomputed from
   records. Read it as a comparability check across arms, not as an efficiency bound.

4. **Additive NLL advantage.** Retention is a ratio and compresses differences at high retention;
   per-token NLL is additive and does not. If the two disagree on ordering, the ratio metric is
   doing the work rather than the method. Sensitivity check for F-37's headline.

5. **Budget realisation.** Measured sparsity against target, effective bits per weight, and the
   targeted-parameter fraction, per scale. §3.11 requires the arms matched on all three; this is
   where that is checkable after the fact rather than asserted.

    python scripts/report_diagnostics.py
    python scripts/report_diagnostics.py --output results/evidence/diagnostics_report.txt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

SCALES = ("pythia-160m", "pythia-410m", "pythia-1b")
BUDGETS = ("moderate", "aggressive")
SEQUENTIAL_ARMS = ("sequential", "sequential_qp")
ARMS = ("pruning", "quantisation", *SEQUENTIAL_ARMS, "joint")


def load_test_records(metrics: Path) -> list[dict[str, Any]]:
    """Every successful test-split record."""
    records = []
    for path in sorted(metrics.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        data = (record.get("config") or {}).get("data") or {}
        if data.get("eval_split") != "test" or record.get("status") != "success":
            continue
        records.append(record)
    return records


def statistics_of(record: dict[str, Any]) -> dict[str, Any]:
    """The compression statistics block."""
    return ((record.get("compression") or {}).get("statistics")) or {}


def layers_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-layer diagnostics, empty when the arm records none."""
    return (statistics_of(record).get("layerwise") or {}).get("layers") or []


def _fmt(value: Any, spec: str = "") -> str:
    """Format a value for the table, or a dash when the record carries none."""
    if value is None:
        return "--"
    return format(value, spec)


def per_token_nll(record: dict[str, Any]) -> float | None:
    """Mean per-token NLL over the evaluation window."""
    quality = (record.get("quality") or {}).get("perplexity") or {}
    total, tokens = quality.get("total_nll"), quality.get("total_tokens")
    if total is None or not tokens:
        return None
    return float(total) / float(tokens)


def report(records: list[dict[str, Any]], stream: TextIO) -> None:
    """Write all five diagnostics."""
    say = lambda text="": print(text, file=stream)  # noqa: E731

    by_cell: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cell[
            (record["model_name"], record.get("budget_label"), record["compression_method"])
        ].append(record)

    say("=" * 79)
    say("MECHANISM DIAGNOSTICS -- confirmatory (test-split) records only")
    say("=" * 79)
    say(f"{len(records)} records. Nothing here re-runs or re-tunes the frozen grid.")

    # ---------------------------------------------------------------- 1. mask disagreement
    say("")
    say("=" * 79)
    say("1. MASK DISAGREEMENT, JOINT vs SEQUENTIAL, BY SCALE")
    say("=" * 79)
    say("")
    say("  Fraction of mask positions the joint refinement moved away from the sequential choice,")
    say("  averaged over layers then over replicates. Recorded on the joint arm, where the")
    say("  difference lives. An inert mechanism gives ~0.")
    say("")
    say(
        f"  {'scale':<13} {'budget':<11} {'bits':>5} {'mean divergence':>16} {'max layer':>11} {'R':>3}"
    )
    for model in SCALES:
        for budget in BUDGETS:
            joint = by_cell.get((model, budget, "joint")) or []
            if not joint:
                continue
            per_replicate, maxima = [], []
            for record in joint:
                values = [
                    step["mask_divergence"]
                    for layer in layers_of(record)
                    for step in (layer.get("joint_trace") or [])
                    if "mask_divergence" in step
                ]
                if values:
                    per_replicate.append(statistics.mean(values))
                    maxima.append(max(values))
            if not per_replicate:
                continue
            bits = statistics_of(joint[0]).get("target_bits")
            say(
                f"  {model:<13} {budget:<11} {bits:>5} {statistics.mean(per_replicate):>15.4%} "
                f"{max(maxima):>10.4%} {len(per_replicate):>3}"
            )
    say("")
    say("  F-05 measured 8.86% divergence at W4 and 0.46% at W8 on six real 160M layers.")

    # ---------------------------------------------------------------- 2. layerwise advantage
    say("")
    say("=" * 79)
    say("2. LAYERWISE JOINT ADVANTAGE, BY SCALE")
    say("=" * 79)
    say("")
    say("  Local objective gain from the joint step, (loss_before - loss_proposed) / loss_before,")
    say("  over ACCEPTED iterations only. The incumbent guard means a rejected proposal leaves the")
    say("  layer at its sequential value, so rejections are counted separately.")
    say("")
    say(
        f"  {'scale':<13} {'budget':<11} {'bits':>5} {'mean layer gain':>16} {'accepted':>9} {'rejected':>9}"
    )
    for model in SCALES:
        for budget in BUDGETS:
            joint = by_cell.get((model, budget, "joint")) or []
            if not joint:
                continue
            gains, accepted, rejected = [], 0, 0
            for record in joint:
                for layer in layers_of(record):
                    for step in layer.get("joint_trace") or []:
                        before, proposed = step.get("loss_before"), step.get("loss_proposed")
                        if step.get("accepted"):
                            accepted += 1
                            if before:
                                gains.append((before - proposed) / before)
                        else:
                            rejected += 1
            if not gains:
                continue
            bits = statistics_of(joint[0]).get("target_bits")
            say(
                f"  {model:<13} {budget:<11} {bits:>5} {statistics.mean(gains):>15.4%} "
                f"{accepted:>9} {rejected:>9}"
            )

    # ---------------------------------------------------------------- 3. layer objective
    say("")
    say("=" * 79)
    say("3. FINAL LAYER OBJECTIVE, JOINT vs SEQUENTIAL, MATCHED PER LAYER")
    say("=" * 79)
    say("")
    say("  ** `relative_improvement` is NOT comparable across arms and is deliberately not used")
    say("  here.** Each arm measures it against its own naive baseline, and the arms do not share")
    say("  one. Q->P quantises first, so at pythia-1b/moderate its naive_loss is 9,548 against the")
    say("  joint arm's 1,635,000 -- the same layer, a reference three orders of magnitude apart.")
    say("  Dividing by it produced an apparent -6921% 'efficiency' for sequential, which is an")
    say("  artefact of the denominator and not a property of the solver.")
    say("")
    say("  `final_loss` IS comparable: it is the same quantity at the end of the same layer. Below")
    say("  is the mean per-layer relative difference, (seq - joint) / seq, matched layer by layer")
    say("  and replicate by replicate. Positive => joint ends at a lower objective.")
    say("")
    say(
        f"  {'scale':<13} {'budget':<11} {'bits':>5} {'joint better by':>16} "
        f"{'layers joint wins':>18} {'R':>3}"
    )
    for model in SCALES:
        for budget in BUDGETS:
            joint_by_replicate = {
                ((r.get("config") or {}).get("data") or {}).get("calibration_replicate"): r
                for r in by_cell.get((model, budget, "joint")) or []
            }
            sequential_by_replicate: dict[Any, dict[str, Any]] = {}
            for arm in SEQUENTIAL_ARMS:
                for record in by_cell.get((model, budget, arm)) or []:
                    key = ((record.get("config") or {}).get("data") or {}).get(
                        "calibration_replicate"
                    )
                    sequential_by_replicate[key] = record

            replicate_means, wins, total_layers = [], 0, 0
            for key, joint_record in joint_by_replicate.items():
                sequential_record = sequential_by_replicate.get(key)
                if sequential_record is None:
                    continue
                joint_layers = {layer["name"]: layer for layer in layers_of(joint_record)}
                differences = []
                for layer in layers_of(sequential_record):
                    counterpart = joint_layers.get(layer["name"])
                    if counterpart is None:
                        continue
                    sequential_loss = layer.get("final_loss")
                    joint_loss = counterpart.get("final_loss")
                    if not sequential_loss:
                        continue
                    differences.append((sequential_loss - joint_loss) / sequential_loss)
                    total_layers += 1
                    if joint_loss < sequential_loss:
                        wins += 1
                if differences:
                    replicate_means.append(statistics.mean(differences))
            if not replicate_means:
                continue
            bits = statistics_of(joint_by_replicate[next(iter(joint_by_replicate))]).get(
                "target_bits"
            )
            say(
                f"  {model:<13} {budget:<11} {bits:>5} {statistics.mean(replicate_means):>15.4%} "
                f"{wins:>10}/{total_layers:<7} {len(replicate_means):>3}"
            )
    say("")
    say("  This is the LOCAL objective. It is not the model-level result and need not agree with")
    say("  it: layers are optimised independently, and a lower sum of layer objectives does not")
    say("  guarantee lower end-to-end perplexity.")

    # ---------------------------------------------------------------- 4. additive NLL
    say("")
    say("=" * 79)
    say("4. ADDITIVE NLL ADVANTAGE -- sensitivity check on the retention metric")
    say("=" * 79)
    say("")
    say("  Retention is a ratio and compresses differences where retention is high; per-token NLL")
    say("  is additive. If the two disagree on SIGN, the metric is doing the work, not the method.")
    say("")
    say(
        f"  {'scale':<13} {'budget':<11} {'mean NLL adv':>14} {'positive':>9} {'retention pp':>13} {'agree':>6}"
    )
    for model in SCALES:
        for budget in BUDGETS:
            joint = {
                ((r.get("config") or {}).get("data") or {}).get("calibration_replicate"): r
                for r in by_cell.get((model, budget, "joint")) or []
            }
            sequential: dict[Any, dict[str, Any]] = {}
            for arm in SEQUENTIAL_ARMS:
                for record in by_cell.get((model, budget, arm)) or []:
                    key = ((record.get("config") or {}).get("data") or {}).get(
                        "calibration_replicate"
                    )
                    sequential[key] = record
            advantages, retention_gains = [], []
            for key, joint_record in joint.items():
                sequential_record = sequential.get(key)
                if sequential_record is None:
                    continue
                joint_nll, sequential_nll = (
                    per_token_nll(joint_record),
                    per_token_nll(sequential_record),
                )
                if joint_nll is None or sequential_nll is None:
                    continue
                # Lower NLL is better, so the joint advantage is sequential - joint.
                advantages.append(sequential_nll - joint_nll)
                joint_ret = ((joint_record.get("quality") or {}).get("retention") or {}).get(
                    "perplexity_retention"
                )
                sequential_ret = (
                    (sequential_record.get("quality") or {}).get("retention") or {}
                ).get("perplexity_retention")
                if joint_ret is not None and sequential_ret is not None:
                    retention_gains.append(joint_ret - sequential_ret)
            if not advantages:
                continue
            mean_advantage = statistics.mean(advantages)
            mean_retention = statistics.mean(retention_gains) if retention_gains else float("nan")
            positive = sum(1 for value in advantages if value > 0)
            agree = (mean_advantage > 0) == (mean_retention > 0)
            say(
                f"  {model:<13} {budget:<11} {mean_advantage:>+14.6f} "
                f"{positive:>4}/{len(advantages):<4} {mean_retention:>+12.4f} "
                f"{'yes' if agree else 'NO':>6}"
            )
    say("")
    say("  Units are nats/token. 'agree' compares the SIGN of the two metrics, nothing more.")

    # ---------------------------------------------------------------- 5. budget realisation
    say("")
    say("=" * 79)
    say("5. BUDGET REALISATION -- measured vs target, across scales")
    say("=" * 79)
    say("")
    say("  §3.11 requires the arms matched on sparsity, bit width and module coverage. This is")
    say("  where that is checkable after the fact. mask_sparsity is the pruning budget;")
    say("  realised_sparsity additionally counts survivors that quantisation rounded to zero.")
    say("")
    say(
        f"  {'scale':<13} {'budget':<11} {'arm':<14} {'target':>7} {'mask':>8} {'realised':>9} "
        f"{'eff.bits':>9} {'modules':>8} {'targeted params':>16}"
    )
    for model in SCALES:
        for budget in BUDGETS:
            for arm in ARMS:
                found = by_cell.get((model, budget, arm)) or []
                if not found:
                    continue
                statistic = statistics_of(found[0])
                conversion = statistic.get("conversion") or {}
                target = statistic.get("target_sparsity")
                mask = statistic.get("measured_sparsity")
                realised = statistic.get("numeric_zero_fraction")
                bits = conversion.get("effective_bits_per_weight")
                modules = statistic.get("num_target_modules")
                targeted = statistic.get("targeted_parameters")
                say(
                    f"  {model:<13} {budget:<11} {arm:<14} "
                    f"{_fmt(target, '.2f'):>7} "
                    f"{_fmt(mask, '.4f'):>8} "
                    f"{_fmt(realised, '.4f'):>9} "
                    f"{_fmt(bits, '.4f'):>9} "
                    f"{_fmt(modules):>8} "
                    f"{_fmt(targeted, ','):>16}"
                )


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=REPOSITORY_ROOT / "outputs" / "metrics")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    records = load_test_records(arguments.metrics)
    if not records:
        print(f"no successful test-split records under {arguments.metrics}", file=sys.stderr)
        return 1

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
            report(records, handle)
        print(f"wrote {arguments.output}")
    else:
        report(records, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
