r"""The confirmatory result: every joint gain on the TEST split, with the pre-registered verdict.

WHAT THIS IS
------------
A1 step 10 runs the frozen grid once on the held-out test split and forbids tuning afterwards. This
script reads the resulting run records and reports, per cell:

    * every per-replicate retention and gain, individually -- A1 §5.1 requires the replicate-level
      values, because a mean that hides a sign flip is exactly what F-26 caught
    * R, the mean, sd, median, and the sign counts
    * the exact two-sided sign-test p over NON-TIED replicates (B-40)
    * a paired block-bootstrap interval on the per-token NLL advantage, resampling whole evaluation
      windows with one index draw applied to both arms
    * the §6.3 verdict: practically important iff mean >= 1.0 pp AND the sign is consistent across
      every paired replicate

WHY IT READS RECORDS AND NOT results/evidence/joint_gains.csv
------------------------------------------------------------
That table carries no `eval_split` column, so it mixes the exploratory validation rows with these.
The split is the whole distinction between screening and confirmation.

WHAT THE BASELINE IS
--------------------
The **frozen** sequential order, not best-of-both. A1 §3 selects the order on validation and freezes
it per cell before test, so only that order was run here. At pythia-1b/moderate the frozen order is
Q->P (`sequential_qp`); everywhere else it is P->Q (`sequential`). Reporting this as "best-of" would
overstate what was measured.

    python scripts/report_confirmatory.py
    python scripts/report_confirmatory.py --output results/evidence/confirmatory_report.txt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, TextIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scale_aware_compression.metrics.replicates import (  # noqa: E402
    paired_block_bootstrap,
    summarise_replicates,
)

SEQUENTIAL_ARMS = ("sequential", "sequential_qp")
ORDER_NAME = {"sequential": "P->Q", "sequential_qp": "Q->P"}
SCALES = ("pythia-160m", "pythia-410m", "pythia-1b")
BUDGETS = ("moderate", "aggressive")
IMPORTANCE_THRESHOLD_PP = 1.0


def load_test_records(metrics: Path) -> list[dict[str, Any]]:
    """Every successful test-split record.

    Args:
        metrics: Directory of JSON run records.

    Returns:
        The records, unordered.
    """
    records = []
    for path in sorted(metrics.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        data = (record.get("config") or {}).get("data") or {}
        if data.get("eval_split") != "test" or record.get("status") != "success":
            continue
        records.append(record)
    return records


def retention_of(record: dict[str, Any]) -> float | None:
    """Perplexity retention as a percentage, or None when the record carries none."""
    value = (record.get("quality") or {}).get("retention")
    if isinstance(value, dict):
        return value.get("perplexity_retention")
    return value


def replicate_of(record: dict[str, Any]) -> int | None:
    """The calibration replicate index the record was produced at."""
    return ((record.get("config") or {}).get("data") or {}).get("calibration_replicate")


def bootstrap_for(sequential: dict[str, Any], joint: dict[str, Any]) -> Any:
    """Paired block bootstrap over evaluation windows, or None if a record lacks per-window NLL."""
    seq_quality = (sequential.get("quality") or {}).get("perplexity") or {}
    joint_quality = (joint.get("quality") or {}).get("perplexity") or {}
    seq_nll, joint_nll = seq_quality.get("window_nll"), joint_quality.get("window_nll")
    tokens = seq_quality.get("window_tokens")
    if not seq_nll or not joint_nll or not tokens:
        return None
    if not (len(seq_nll) == len(joint_nll) == len(tokens)):
        return None
    return paired_block_bootstrap(
        sequential_window_nll=list(seq_nll),
        joint_window_nll=list(joint_nll),
        window_tokens=list(tokens),
    )


def report(records: list[dict[str, Any]], stream: TextIO) -> None:
    """Write the whole confirmatory report.

    Args:
        records: Successful test-split records.
        stream: Where to write.
    """
    say = lambda text="": print(text, file=stream)  # noqa: E731

    index: dict[tuple, dict[str, dict]] = {}
    for record in records:
        method = record["compression_method"]
        if method not in SEQUENTIAL_ARMS and method != "joint":
            continue
        key = (record["model_name"], record.get("budget_label"), replicate_of(record))
        index.setdefault(key, {})[method] = record

    say("=" * 79)
    say("CONFIRMATORY RESULT -- test split, run once, no tuning after the A1 step-9 freeze")
    say("=" * 79)
    say(f"{len(records)} successful test-split records")
    say("")
    say("Baseline is the FROZEN sequential order per cell, not best-of-both: A1 §3 selects the")
    say("order on validation and freezes it before test, so only that order was run.")
    say("§6.3: practically important iff mean gain >= 1.0 pp AND sign-consistent across all R.")

    summaries: dict[tuple, Any] = {}
    for model in SCALES:
        for budget in BUDGETS:
            rows, gains, orders = [], [], set()
            pairs = [
                (key, arms) for key, arms in index.items() if key[0] == model and key[1] == budget
            ]
            for key, arms in sorted(pairs, key=lambda item: item[0][2] or 0):
                seq_arm = next((a for a in SEQUENTIAL_ARMS if a in arms), None)
                if seq_arm is None or "joint" not in arms:
                    continue
                sequential, joint = arms[seq_arm], arms["joint"]
                seq_ret, joint_ret = retention_of(sequential), retention_of(joint)
                if seq_ret is None or joint_ret is None:
                    continue
                orders.add(seq_arm)
                gains.append(joint_ret - seq_ret)
                rows.append((key[2], seq_ret, joint_ret, joint_ret - seq_ret, sequential, joint))
            if not gains:
                continue

            summary = summarise_replicates(model_name=model, budget_label=budget, gains=gains)
            summaries[(model, budget)] = summary
            order_text = ", ".join(sorted(ORDER_NAME.get(o, o) for o in orders))

            say("")
            say("-" * 79)
            say(f"{model} / {budget}    sequential order: {order_text}")
            say("-" * 79)
            say(
                f"  {'rep':>3} {'sequential %':>13} {'joint %':>11} {'gain pp':>9}   95% CI on NLL advantage"
            )
            for rep, seq_ret, joint_ret, gain, sequential, joint in rows:
                interval = bootstrap_for(sequential, joint)
                if interval is None:
                    tail = "  (no per-window NLL recorded)"
                else:
                    excl = "excludes 0" if interval.excludes_zero else "includes 0"
                    tail = f"  [{interval.lower:+.5f}, {interval.upper:+.5f}] nats/token, {excl}"
                say(f"  {rep:>3} {seq_ret:>12.4f} {joint_ret:>10.4f} {gain:>+9.4f}{tail}")

            say("")
            say(
                f"  R = {summary.replicates}    mean {summary.mean_gain:+.4f} pp    "
                f"sd {summary.standard_deviation:.4f}    median {summary.median_gain:+.4f} pp    "
                f"range [{summary.minimum:+.4f}, {summary.maximum:+.4f}]"
            )
            say(
                f"  positive {summary.positive_count}    negative {summary.negative_count}    "
                f"ties {summary.tie_count}    sign-test n = {summary.sign_test_n}    "
                f"exact two-sided p = {summary.sign_test_p:.4f}"
            )
            consistent = summary.positive_count == summary.replicates
            meets_size = summary.mean_gain >= IMPORTANCE_THRESHOLD_PP
            say(
                f"  >= 1.0 pp: {'YES' if meets_size else 'NO':<3}    "
                f"sign-consistent: {'YES' if consistent else 'NO':<3}    "
                f"=> §6.3 PRACTICALLY IMPORTANT: {'YES' if (meets_size and consistent) else 'NO'}"
            )

    say("")
    say("=" * 79)
    say("SCALE TREND -- mean joint gain by scale")
    say("=" * 79)
    for budget in BUDGETS:
        say(f"\n  {budget}:")
        for model in SCALES:
            summary = summaries.get((model, budget))
            if summary is None:
                continue
            say(
                f"    {model:<13} {summary.mean_gain:+.4f} pp   R={summary.replicates}   "
                f"{summary.positive_count}/{summary.replicates} positive   "
                f"p={summary.sign_test_p:.4f}"
            )

    say("")
    say("=" * 79)
    say("MULTIPLE COMPARISONS -- Holm-Bonferroni over every cell examined")
    say("=" * 79)
    say("")
    say("  validity_threats.md warned before any result existed that examining several joint-gain")
    say("  values and reporting the largest inflates the false-positive rate. It applies: the")
    say("  headline significance comes from one cell out of the number below.")
    say("")
    ordered = sorted(summaries.items(), key=lambda item: item[1].sign_test_p)
    total, running = len(ordered), 0.0
    say(f"  {'cell':<26} {'raw p':>9} {'adjusted':>10}")
    for rank, ((model, budget), summary) in enumerate(ordered):
        adjusted = min(1.0, max(running, (total - rank) * summary.sign_test_p))
        running = adjusted
        mark = "  <- significant" if adjusted < 0.05 else ""
        say(f"  {model + '/' + budget:<26} {summary.sign_test_p:>9.4f} {adjusted:>10.4f}{mark}")
    survivors = sum(
        1
        for rank, (_, summary) in enumerate(ordered)
        if min(1.0, (total - rank) * summary.sign_test_p) < 0.05
    )
    say("")
    say(f"  {survivors} of {total} cells significant after correction.")
    say("  Report the ADJUSTED value. Quoting the raw p while reporting the largest of several")
    say("  cells is the inflation this correction exists to remove.")

    say("")
    say("=" * 79)
    say("DENSE BASELINES (test split)")
    say("=" * 79)
    for record in sorted(records, key=lambda r: r["model_name"]):
        if record["compression_method"] != "dense":
            continue
        perplexity = ((record.get("quality") or {}).get("perplexity") or {}).get("perplexity")
        say(f"  {record['model_name']:<13} {perplexity:.4f}")

    say("")
    say("=" * 79)
    say("ALL ARMS -- mean retention per cell")
    say("=" * 79)
    by_cell: dict[tuple, list[float]] = {}
    for record in records:
        value = retention_of(record)
        if value is None or record["compression_method"] == "dense":
            continue
        key = (record["model_name"], record.get("budget_label"), record["compression_method"])
        by_cell.setdefault(key, []).append(value)
    for model in SCALES:
        for budget in BUDGETS:
            parts = []
            for arm in ("pruning", "quantisation", *SEQUENTIAL_ARMS, "joint"):
                values = by_cell.get((model, budget, arm))
                if values:
                    parts.append(f"{arm}={statistics.mean(values):.2f}%")
            if parts:
                say(f"  {model:<13} {budget:<11} " + "  ".join(parts))


def main() -> int:
    """Entry point.

    Returns:
        0 always -- this reports, it does not gate. `audit_confirmatory_run.py` is the gate.
    """
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
        # newline="\n" so regeneration is idempotent against the committed file: .gitattributes
        # pins LF, and the default translation would rewrite every line on Windows.
        with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
            report(records, handle)
        print(f"wrote {arguments.output}")
    else:
        report(records, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
