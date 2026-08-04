"""Summarise a budget-screening grid against the research plan's §5.3 selection rule.

Reads the run records a screening sweep produced and prints one row per budget: quality retention for
each arm, the joint gain between them, and whether the budget satisfies each clause of the selection
rule. The output is the *evidence* §5.3 asks to be recorded alongside the frozen choice, so it is
written to disk as well as printed.

Deliberately does not choose the budgets. The rule has a judgement clause ("enough separation to test
whether joint gain changes with scale") that no single draw can settle, and screening runs one draw by
design. This tool lays out the evidence; a human freezes the choice in protocol_freeze.md.

Everything it emits is **exploratory** under Amendment A1 §4: validation split, one calibration draw,
no uncertainty estimate. The output says so, because a table that does not say so gets quoted as
though it were confirmatory.

Usage::

    python scripts/summarise_screening.py
    python scripts/summarise_screening.py --metrics outputs/metrics --model pythia-160m
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# §5.3: "measurable but non-catastrophic". Neither bound is in the plan as a number, so both are
# stated here and reported alongside the verdict rather than hidden inside it.
MEASURABLE_BELOW = 99.0
"""Retention at or above this is not measurably degraded -- the budget is too mild to compare arms."""

CATASTROPHIC_BELOW = 50.0
"""Retention below this is catastrophic: the model is broken, not compressed."""


@dataclass(slots=True)
class Cell:
    """One arm at one budget."""

    budget: str
    method: str
    sparsity: float
    bits: int
    perplexity: float | None
    retention: float | None
    status: str
    experiment_id: str
    replicate: int | None = None
    """Which calibration draw. `None` for records predating Amendment A1's replicate axis.

    Carried so this tool can *refuse* rather than silently pick one draw out of several. Reporting
    a single draw as a point estimate is the fault B-31 retracted a headline for."""
    eval_sequences: int | None = None
    eval_sequence_length: int | None = None
    local_steps: int | None = None
    """Solver budget this arm actually consumed. Two arms with different totals are not a fair
    comparison however good the numbers look (§3.11)."""

    @property
    def window(self) -> tuple[int | None, int | None]:
        """The evaluation window this cell was measured on.

        Two cells measured on different windows are not comparable, and mixing them in one table is
        the kind of thing §3.11 exists to prevent.
        """
        return (self.eval_sequences, self.eval_sequence_length)


def load_cells(metrics_dir: Path, model: str | None) -> list[Cell]:
    """Read every record in ``metrics_dir`` into a flat list of cells."""
    cells: list[Cell] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"  ! skipping {path.name}: {error}", file=sys.stderr)
            continue
        if model and record.get("model_name") != model:
            continue
        quality = record.get("quality", {})
        perplexity = (quality.get("perplexity") or {}).get("perplexity")
        cells.append(
            Cell(
                budget=record.get("budget_label", "?"),
                method=record.get("compression_method", "?"),
                sparsity=float(record.get("sparsity") or 0.0),
                bits=int(record.get("quantisation_bits") or 32),
                perplexity=float(perplexity) if perplexity is not None else None,
                retention=_retention(quality),
                status=record.get("status", "?"),
                experiment_id=record.get("experiment_id", path.stem),
                replicate=((record.get("config") or {}).get("data") or {}).get(
                    "calibration_replicate"
                ),
                eval_sequences=_window(quality, "num_sequences"),
                eval_sequence_length=_window(quality, "sequence_length"),
                local_steps=((record.get("compression") or {}).get("statistics") or {}).get(
                    "total_local_steps"
                ),
            )
        )
    return cells


def _window(quality: dict[str, Any], key: str) -> int | None:
    """Read one evaluation-window field from a record's perplexity payload."""
    payload = quality.get("perplexity") or {}
    value = payload.get(key)
    return int(value) if value is not None else None


def _retention(quality: dict[str, Any]) -> float | None:
    """Pull perplexity retention as a percentage from a record's quality payload.

    The record nests it as ``quality.retention.perplexity_retention``. Read from the record rather
    than recomputed from perplexities, so this reports the same number the run itself recorded
    against its own dense reference.
    """
    nested = quality.get("retention")
    if isinstance(nested, dict):
        value = nested.get("perplexity_retention")
        if value is not None:
            return float(value)
    return None


def verdict(sequential: Cell | None, joint: Cell | None) -> tuple[str, str]:
    """Apply the §5.3 selection rule to one budget.

    Returns:
        A short verdict and the reason, so a rejected budget records *why* it was rejected.
    """
    if sequential is None or joint is None:
        return "INCOMPLETE", "missing an arm, so no joint gain is defined"
    if sequential.status != "success" or joint.status != "success":
        return "UNSTABLE", f"status {sequential.status}/{joint.status}"
    if sequential.retention is None or joint.retention is None:
        return "NO REFERENCE", "no dense baseline recorded for this model and seed"

    # Eligibility first. A budget that breaks the model is rejected whether or not the arms were
    # matched, so an unmatched-budget note must not hide a catastrophic verdict.
    best = max(sequential.retention, joint.retention)
    if best >= MEASURABLE_BELOW:
        label, reason = "TOO MILD", f"best retention {best:.1f}% -- not measurably degraded"
    elif best < CATASTROPHIC_BELOW:
        label, reason = "CATASTROPHIC", f"best retention {best:.1f}% -- the model is broken"
    else:
        label, reason = "ELIGIBLE", f"best retention {best:.1f}%"

    # §3.11: a score obtained with more optimisation cannot be attributed to the method. This
    # actually happened -- the joint arm ran on twice the solver budget through a whole grid -- so the
    # check belongs in the output rather than in someone's memory. It qualifies the *gain*, not the
    # budget's eligibility, which is why it is appended rather than substituted.
    if (
        sequential.local_steps is not None
        and joint.local_steps is not None
        and sequential.local_steps != joint.local_steps
    ):
        reason += (
            f" · **gain NOT usable**: solver budgets differ (sequential "
            f"{sequential.local_steps}, joint {joint.local_steps}), §3.11"
        )
    return label, reason


def main(argv: list[str] | None = None) -> int:
    """Print and save the screening summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/metrics", help="Directory of run records")
    parser.add_argument("--model", default=None, help="Restrict to one model short name")
    parser.add_argument(
        "--budgets",
        default=None,
        help=(
            "Comma-separated budget labels to include. Without it, every record in the "
            "directory is included -- which will mix exploratory runs into the screening table."
        ),
    )
    parser.add_argument(
        "--out",
        default="outputs/tables/screening_summary.md",
        help="Where to write the evidence table",
    )
    arguments = parser.parse_args(argv)

    metrics_dir = Path(arguments.metrics)
    if not metrics_dir.is_dir():
        print(f"No metrics directory at {metrics_dir}", file=sys.stderr)
        return 1

    cells = load_cells(metrics_dir, arguments.model)
    if arguments.budgets:
        wanted = {label.strip() for label in arguments.budgets.split(",")}
        cells = [cell for cell in cells if cell.budget in wanted or cell.method == "dense"]
    if not cells:
        print("No records found.", file=sys.stderr)
        return 1

    # A retention figure is only comparable against a dense baseline measured the same way, and only
    # comparable across budgets if every cell used the same window. Refuse to print a table that
    # silently mixes them.
    windows = {cell.window for cell in cells if cell.perplexity is not None}
    if len(windows) > 1:
        print(
            "REFUSING to summarise: the selected records span more than one evaluation window "
            f"{sorted(windows)}. Retention is not comparable across windows, so this table would "
            "be misleading. Narrow the selection with --budgets, or re-run the odd cells.",
            file=sys.stderr,
        )
        for cell in sorted(cells, key=lambda c: (c.window, c.experiment_id)):
            print(f"  {cell.window}  {cell.experiment_id}", file=sys.stderr)
        return 2

    dense = [cell for cell in cells if cell.method == "dense"]
    budgets = sorted({cell.budget for cell in cells if cell.method != "dense"})

    # REFUSE on multiple calibration draws, the same way this tool already refuses on mixed
    # evaluation windows. The row builder below takes the FIRST matching cell per (budget, method),
    # so with replicates in the directory it would silently report one arbitrary draw as though it
    # were the number -- which is exactly the fault B-31 retracted a headline for, except automated
    # and invisible. A multi-draw record set belongs in `metrics/replicates.py`, which reports a
    # mean, an sd and n; this table has no column for uncertainty and should not pretend otherwise.
    draws_per_cell: dict[tuple[str, str], set[int]] = {}
    for cell in cells:
        if cell.replicate is not None:
            draws_per_cell.setdefault((cell.budget, cell.method), set()).add(cell.replicate)
    replicated = {key: draws for key, draws in draws_per_cell.items() if len(draws) > 1}
    if replicated:
        print(
            "REFUSING to summarise: these cells span more than one calibration draw, and this "
            "table reports a single value per cell. Aggregate with "
            "`metrics.replicates.summarise_replicates`, which reports mean, sd and n, or filter "
            "the record set to one draw and say which.",
            file=sys.stderr,
        )
        for (budget, method), draws in sorted(replicated.items()):
            print(f"  {budget} / {method}: draws {sorted(draws)}", file=sys.stderr)
        return 2

    lines: list[str] = []
    lines.append("# Budget screening evidence (research plan §5.3)")
    lines.append("")
    window = next(iter(windows), (None, None))
    lines.append(
        f"Evaluation window: **{window[0]} sequences x {window[1]} tokens** on the "
        "**validation** split. One calibration draw -- the refusal above guarantees it, rather "
        "than this line asserting it."
    )
    lines.append("")
    if dense:
        best_dense = dense[0]
        lines.append(
            f"Dense reference: **{best_dense.perplexity:.2f}** perplexity "
            f"(`{best_dense.experiment_id}`)"
        )
        lines.append("")
    lines.append(
        f"Selection rule thresholds used: measurable below **{MEASURABLE_BELOW:.0f}%** retention, "
        f"catastrophic below **{CATASTROPHIC_BELOW:.0f}%**."
    )
    lines.append("")
    lines.append(
        "| Budget | Sparsity | Bits | Sequential ppl | Joint ppl | Seq ret. | Joint ret. | "
        "Joint gain (pp) | Verdict | Reason |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for budget in budgets:
        at_budget = [cell for cell in cells if cell.budget == budget]
        sequential = next((c for c in at_budget if c.method == "sequential"), None)
        joint = next((c for c in at_budget if c.method == "joint"), None)
        reference = sequential or joint
        gain = (
            joint.retention - sequential.retention
            if sequential
            and joint
            and sequential.retention is not None
            and joint.retention is not None
            else None
        )
        label, reason = verdict(sequential, joint)
        lines.append(
            "| `{budget}` | {sparsity:.0%} | W{bits} | {sp} | {jp} | {sr} | {jr} | {gain} | "
            "**{label}** | {reason} |".format(
                budget=budget,
                sparsity=reference.sparsity if reference else 0.0,
                bits=reference.bits if reference else 32,
                sp=f"{sequential.perplexity:.2f}" if sequential and sequential.perplexity else "—",
                jp=f"{joint.perplexity:.2f}" if joint and joint.perplexity else "—",
                sr=f"{sequential.retention:.1f}%" if sequential and sequential.retention else "—",
                jr=f"{joint.retention:.1f}%" if joint and joint.retention else "—",
                gain=f"{gain:+.2f}" if gain is not None else "—",
                label=label,
                reason=reason,
            )
        )

    lines.append("")
    lines.append(
        "**EXPLORATORY. Not confirmatory evidence, and no sign in this table is interpretable as "
        "evidence for or against joint compression.** Three reasons, per Protocol Amendment A1 §4: "
        "this is the validation split, which is a selection surface because the budgets were chosen "
        "by looking at it; it is a single calibration draw, so there is no uncertainty estimate; and "
        "the confirmatory comparison uses paired calibration replicates on the **test** split. "
        "A1 §5.1 withdrew the old 'must exceed the seed spread' rule, which was vacuous here -- run "
        "seeds are inert under this method (F-15), so the spread was exactly zero and any nonzero "
        "gain passed it trivially."
    )

    report = "\n".join(lines)
    print(report)

    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report + "\n", encoding="utf-8")
    print(f"\nWrote {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
