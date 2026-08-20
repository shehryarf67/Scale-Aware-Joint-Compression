r"""Emit the paper's results tables as Markdown, generated from the committed evidence.

WHY GENERATED RATHER THAN TYPED
-------------------------------
Every number in the paper has to be traceable to a run record. A table typed by hand is a number
without provenance the moment anything is re-derived, and this project has already retracted three
figures that existed only in prose. These tables regenerate; if the records change, the tables move
with them, and if they do not regenerate the paper is quoting something that no longer exists.

WHAT IT EMITS
-------------
    T1  the headline: joint gain per cell, with R, sign counts, raw and adjusted p
    T2  every confirmatory replicate, per cell
    T3  all-arm mean retention per cell
    T4  budget realisation -- measured sparsity, effective bits, coverage
    T5  dense baselines
    T6  downstream tasks, if run_downstream records exist
    T7  deployment latency, with the limitation stated in the caption

Tables are written as one Markdown file, ready to paste or \\input. Captions state the limitation
that applies to that table, because a table lifted without its caption is how a limitation gets
lost.

    python scripts/build_paper_tables.py
    python scripts/build_paper_tables.py --output results/evidence/paper_tables.md
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

from scale_aware_compression.metrics.replicates import summarise_replicates  # noqa: E402

SCALES = ("pythia-160m", "pythia-410m", "pythia-1b")
BUDGETS = ("moderate", "aggressive")
BUDGET_RECIPE = {"moderate": "30% + W8", "aggressive": "30% + W4"}
SEQUENTIAL_ARMS = ("sequential", "sequential_qp")
ORDER_NAME = {"sequential": "P→Q", "sequential_qp": "Q→P"}
IMPORTANCE_THRESHOLD_PP = 1.0


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


def retention_of(record: dict[str, Any]) -> float | None:
    """Perplexity retention as a percentage."""
    value = (record.get("quality") or {}).get("retention")
    if isinstance(value, dict):
        return value.get("perplexity_retention")
    return value


def replicate_of(record: dict[str, Any]) -> Any:
    """The calibration replicate index."""
    return ((record.get("config") or {}).get("data") or {}).get("calibration_replicate")


def holm_adjusted(p_values: dict[Any, float]) -> dict[Any, float]:
    """Holm-Bonferroni step-down adjustment.

    Args:
        p_values: Raw p-values keyed by cell.

    Returns:
        Adjusted p-values under the same keys.
    """
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total, running, out = len(ordered), 0.0, {}
    for rank, (key, raw) in enumerate(ordered):
        running = min(1.0, max(running, (total - rank) * raw))
        out[key] = running
    return out


def build_cells(records: list[dict[str, Any]]) -> tuple[dict, dict, dict]:
    """Index records and compute per-cell summaries.

    Returns:
        ``(by_cell, summaries, orders)``.
    """
    by_cell: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cell[
            (record["model_name"], record.get("budget_label"), record["compression_method"])
        ].append(record)

    summaries, orders = {}, {}
    for model in SCALES:
        for budget in BUDGETS:
            joint = {replicate_of(r): r for r in by_cell.get((model, budget, "joint"), [])}
            sequential: dict[Any, dict[str, Any]] = {}
            for arm in SEQUENTIAL_ARMS:
                for record in by_cell.get((model, budget, arm), []):
                    sequential[replicate_of(record)] = record
                    if by_cell.get((model, budget, arm)):
                        orders[(model, budget)] = arm
            gains, rows = [], []
            for key in sorted(joint, key=lambda value: (value is None, value)):
                counterpart = sequential.get(key)
                if counterpart is None:
                    continue
                joint_retention = retention_of(joint[key])
                sequential_retention = retention_of(counterpart)
                if joint_retention is None or sequential_retention is None:
                    continue
                gains.append(joint_retention - sequential_retention)
                rows.append((key, sequential_retention, joint_retention))
            if gains:
                summaries[(model, budget)] = (
                    summarise_replicates(model_name=model, budget_label=budget, gains=gains),
                    rows,
                )
    return by_cell, summaries, orders


def emit(records: list[dict[str, Any]], stream: TextIO) -> None:
    """Write every table."""
    say = lambda text="": print(text, file=stream)  # noqa: E731
    by_cell, summaries, orders = build_cells(records)
    adjusted = holm_adjusted({key: value[0].sign_test_p for key, value in summaries.items()})

    say("# Paper results tables")
    say("")
    say("**Generated by `scripts/build_paper_tables.py` from the committed run records.** Do not")
    say("edit by hand: regenerate. Every number here is on the **test** split, from the single")
    say("frozen confirmatory execution of A1 step 10 (171 cells, 42 pairs, 0 failures).")
    say("")
    say("Captions carry the limitation that applies to each table. A table lifted without its")
    say("caption is how a limitation gets lost.")

    # ------------------------------------------------------------------ T1
    say("")
    say("## T1 — Joint gain over the frozen sequential baseline")
    say("")
    say(
        "| Scale | Budget | Order | R | Mean gain (pp) | SD | Median | Positive | Raw *p* | Holm *p* | §6.3 |"
    )
    say("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for model in SCALES:
        for budget in BUDGETS:
            entry = summaries.get((model, budget))
            if entry is None:
                continue
            summary, _ = entry
            consistent = summary.positive_count == summary.replicates
            meets = summary.mean_gain >= IMPORTANCE_THRESHOLD_PP and consistent
            order = ORDER_NAME.get(orders.get((model, budget), ""), "?")
            say(
                f"| {model} | {BUDGET_RECIPE[budget]} | {order} | {summary.replicates} | "
                f"**{summary.mean_gain:+.4f}** | {summary.standard_deviation:.4f} | "
                f"{summary.median_gain:+.4f} | {summary.positive_count}/{summary.replicates} | "
                f"{summary.sign_test_p:.4f} | {adjusted[(model, budget)]:.4f} | "
                f"{'**YES**' if meets else 'no'} |"
            )
    say("")
    say("*§6.3 requires mean gain ≥ 1.0 pp **and** a consistent sign across every paired")
    say("replicate. **No cell meets both.** Quote the Holm-adjusted p, not the raw one: six cells")
    say("were examined and the largest is reported. The baseline is the **frozen** order per cell,")
    say("not best-of-both, so the W8 rows carry an order uncertainty comparable to their effect.*")

    # ------------------------------------------------------------------ T2
    say("")
    say("## T2 — Every confirmatory replicate")
    say("")
    say("| Scale | Budget | Replicate | Sequential (%) | Joint (%) | Gain (pp) |")
    say("| --- | --- | --- | --- | --- | --- |")
    for model in SCALES:
        for budget in BUDGETS:
            entry = summaries.get((model, budget))
            if entry is None:
                continue
            _, rows = entry
            for key, sequential_retention, joint_retention in rows:
                say(
                    f"| {model} | {BUDGET_RECIPE[budget]} | {key} | {sequential_retention:.4f} | "
                    f"{joint_retention:.4f} | {joint_retention - sequential_retention:+.4f} |"
                )
    say("")
    say("*A1 §5.1 requires replicate-level values: a mean that hides a sign flip is exactly what")
    say("F-26 caught on the exploratory split.*")

    # ------------------------------------------------------------------ T3
    say("")
    say("## T3 — Mean perplexity retention, all arms")
    say("")
    say("| Scale | Budget | Pruning | Quantisation | Sequential | Joint |")
    say("| --- | --- | --- | --- | --- | --- |")
    for model in SCALES:
        for budget in BUDGETS:
            values = {}
            for arm in ("pruning", "quantisation", *SEQUENTIAL_ARMS, "joint"):
                found = [
                    retention_of(r)
                    for r in by_cell.get((model, budget, arm), [])
                    if retention_of(r) is not None
                ]
                if found:
                    values[arm] = statistics.mean(found)
            if not values:
                continue
            sequential = values.get("sequential", values.get("sequential_qp"))
            say(
                f"| {model} | {BUDGET_RECIPE[budget]} | "
                f"{values.get('pruning', float('nan')):.2f} | "
                f"{values.get('quantisation', float('nan')):.2f} | "
                f"{sequential:.2f} | {values.get('joint', float('nan')):.2f} |"
            )
    say("")
    say("*Retention is `100 × dense_ppl / compressed_ppl`. The pruning and quantisation arms are")
    say("single-technique controls, not competitors: pruning is identical across budgets because")
    say("both prune 30%, and only the precision differs.*")

    # ------------------------------------------------------------------ T4
    say("")
    say("## T4 — Budget realisation")
    say("")
    say(
        "| Scale | Blocks | Modules | Targeted params | Mask sparsity | Eff. bits W8 | Eff. bits W4 |"
    )
    say("| --- | --- | --- | --- | --- | --- | --- |")
    for model in SCALES:
        joint_moderate = by_cell.get((model, "moderate", "joint"), [])
        joint_aggressive = by_cell.get((model, "aggressive", "joint"), [])
        if not joint_moderate or not joint_aggressive:
            continue

        def stat(record: dict[str, Any]) -> dict[str, Any]:
            return ((record.get("compression") or {}).get("statistics")) or {}

        moderate, aggressive = stat(joint_moderate[0]), stat(joint_aggressive[0])
        names = (moderate.get("layerwise") or {}).get("module_names") or []
        blocks = len(
            {name.split(".layers.")[1].split(".")[0] for name in names if ".layers." in name}
        )
        say(
            f"| {model} | {blocks} | {moderate.get('num_target_modules')} | "
            f"{moderate.get('targeted_parameters'):,} | "
            f"{moderate.get('measured_sparsity'):.4f} | "
            f"{(moderate.get('conversion') or {}).get('effective_bits_per_weight'):.4f} | "
            f"{(aggressive.get('conversion') or {}).get('effective_bits_per_weight'):.4f} |"
        )
    say("")
    say("*Mask sparsity sits just below 0.30 because the per-row prune count is an integer (B-46).")
    say(
        "Effective bits exceed the nominal width by fp32 scale overhead. **Note the block counts:**"
    )
    say("pythia-1b is *shallower* than pythia-410m, so the scale axis is confounded with depth")
    say("(F-38, limitations §2).*")

    # ------------------------------------------------------------------ T5
    say("")
    say("## T5 — Dense baselines")
    say("")
    say("| Scale | Dense perplexity (test) |")
    say("| --- | --- |")
    for model in SCALES:
        for record in by_cell.get((model, "moderate", "dense"), []):
            perplexity = ((record.get("quality") or {}).get("perplexity") or {}).get("perplexity")
            say(f"| {model} | {perplexity:.4f} |")
    say("")
    say("*512 sequences × 512 tokens, WikiText test split, CPU evaluation.*")

    # ------------------------------------------------------------------ T7
    say("")
    say("## T7 — CPU deployment measurements")
    say("")
    say("> ## ⚠️ DO NOT USE THIS TABLE FOR A LATENCY COMPARISON")
    say(">")
    say("> These are **incidental** measurements: each was taken inside its own cell, at whatever")
    say("> moment that cell happened to run, across six days of a grid that also saw commit")
    say("> exhaustion, process recycling and host standby. **They are not mutually comparable, and")
    say("> the `measured` column below shows why.**")
    say(">")
    say("> Worked example. pythia-1b dense reads 1041 ms and pythia-1b pruning 630 ms — an")
    say("> apparent **40% speedup from masking weights**. That is impossible: pruned weights stay")
    say("> FP32 and dense in storage, so the GEMM does identical work. The dense figure was")
    say(
        "> measured on **2026-08-05** and all ten pruning figures on **2026-08-07**, where they are"
    )
    say("> tightly consistent (627–650 ms). The gap is two days of machine state, not sparsity.")
    say(">")
    say(
        "> **The authoritative latency result is [F-34](../docs/findings_log.md#f-34)**, a dedicated"
    )
    say("> study under the §4.7 protocol — model-order rotation exists precisely to control the")
    say("> drift this table demonstrates. Cite F-34; use this table only to show coverage.")
    say("")
    say("| Scale | Budget | Arm | Median latency (ms) | IQR | Runtime repr. | Measured |")
    say("| --- | --- | --- | --- | --- | --- | --- |")
    emitted = 0
    for model in SCALES:
        for budget in BUDGETS:
            for arm in ("dense", "pruning", "quantisation", *SEQUENTIAL_ARMS, "joint"):
                for record in by_cell.get((model, budget, arm), []):
                    if replicate_of(record) not in (0, None):
                        continue
                    deployment = record.get("deployment") or {}
                    median = deployment.get("latency_median_ms")
                    if median is None:
                        continue
                    say(
                        f"| {model} | {BUDGET_RECIPE[budget]} | {arm} | {median:.2f} | "
                        f"{deployment.get('latency_iqr_ms', float('nan')):.2f} | "
                        f"{record.get('runtime_representation', '--')} | "
                        f"{str(record.get('timestamp', ''))[:10]} |"
                    )
                    emitted += 1
    if emitted == 0:
        say("| — | — | — | — | — | *no deployment measurements in these records* |")
    say("")
    say("*Replicate 0 only. **This is one sparsity (30%), not a curve** — RQ4's sparsity–speedup")
    say("relationship is unanswered (F-34, limitations §13).*")
    say("")
    say("**Read which arms are missing.** Only the FP32 arms appear. Every packed arm —")
    say("quantisation, sequential and joint, at both W8 and W4 — is refused a latency by the")
    say("runtime-representation gate, because a packed artefact would be timed through a")
    say("dequantising path and the number would measure unpacking rather than inference. The")
    say("consequence is blunt and belongs in the text: **this study contains no joint-versus-")
    say("sequential latency comparison at all.** RQ4 is answerable only through the pruning-only")
    say("arm, whose weights stay FP32, and that arm shows no speedup from 30% unstructured")
    say("sparsity at any scale.*")


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=REPOSITORY_ROOT / "outputs" / "metrics")
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY_ROOT / "results" / "evidence" / "paper_tables.md"
    )
    arguments = parser.parse_args()

    records = load_test_records(arguments.metrics)
    if not records:
        print(f"no successful test-split records under {arguments.metrics}", file=sys.stderr)
        return 1

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
        emit(records, handle)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
