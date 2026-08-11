"""Result tables in Markdown, LaTeX, and CSV.

:func:`rows_to_markdown` and :func:`rows_to_csv` are implemented, since they are pure text
formatting and are what the protocol documents' placeholder tables get filled from. The
study-specific table builders are placeholders.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)

MISSING = "--"
"""Rendered for absent values, so a missing measurement is visibly different from a zero."""


def format_value(value: Any, *, precision: int = 3) -> str:
    """Render one cell.

    Args:
        value: The value to render.
        precision: Decimal places for floats.

    Returns:
        The rendered string. ``None`` becomes :data:`MISSING`.
    """
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return MISSING
        return f"{value:.{precision}f}"
    return str(value)


def rows_to_markdown(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    precision: int = 3,
    headers: Mapping[str, str] | None = None,
) -> str:
    """Render rows as a GitHub-flavoured Markdown table.

    Args:
        rows: The rows. Keys are column names.
        columns: Column order. Defaults to the first row's key order.
        precision: Decimal places for floats.
        headers: Optional display names, keyed by column name.

    Returns:
        The Markdown table, without a trailing newline.

    Raises:
        ValueError: If ``rows`` is empty and no ``columns`` were given.
    """
    if not rows and columns is None:
        raise ValueError("rows_to_markdown needs either rows or an explicit column list")
    resolved = list(columns) if columns is not None else list(rows[0].keys())
    display = [(headers or {}).get(column, column) for column in resolved]

    lines = [
        f"| {' | '.join(display)} |",
        f"| {' | '.join('---' for _ in resolved)} |",
    ]
    for row in rows:
        cells = [format_value(row.get(column), precision=precision) for column in resolved]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def rows_to_latex(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    precision: int = 3,
    caption: str = "",
    label: str = "",
    headers: Mapping[str, str] | None = None,
) -> str:
    """Render rows as a LaTeX ``tabular`` inside a ``table`` environment.

    Args:
        rows: The rows.
        columns: Column order. Defaults to the first row's key order.
        precision: Decimal places for floats.
        caption: Table caption.
        label: LaTeX label for cross-referencing.
        headers: Optional display names, keyed by column name.

    Returns:
        The LaTeX source.

    Raises:
        ValueError: If ``rows`` is empty and no ``columns`` were given.
    """
    if not rows and columns is None:
        raise ValueError("rows_to_latex needs either rows or an explicit column list")
    resolved = list(columns) if columns is not None else list(rows[0].keys())
    display = [(headers or {}).get(column, column) for column in resolved]

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\begin{{tabular}}{{{'l' * len(resolved)}}}",
        r"    \toprule",
        f"    {' & '.join(_escape_latex(item) for item in display)} \\\\",
        r"    \midrule",
    ]
    for row in rows:
        cells = [
            _escape_latex(format_value(row.get(column), precision=precision)) for column in resolved
        ]
        lines.append(f"    {' & '.join(cells)} \\\\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}"])
    if caption:
        lines.append(f"  \\caption{{{_escape_latex(caption)}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _escape_latex(text: str) -> str:
    """Escape the LaTeX special characters that appear in model names and metrics."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for character, replacement in replacements.items():
        text = text.replace(character, replacement)
    return text


def rows_to_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    columns: Sequence[str] | None = None,
) -> Path:
    """Write rows to a CSV file.

    Args:
        rows: The rows.
        path: Destination file. Parent directories are created.
        columns: Column order. Defaults to the union of all keys, first-seen order.

    Returns:
        The path written.

    Raises:
        ValueError: If ``rows`` is empty and no ``columns`` were given.
    """
    if not rows and columns is None:
        raise ValueError("rows_to_csv needs either rows or an explicit column list")
    resolved = list(columns) if columns is not None else _union_of_keys(rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in resolved})
    LOGGER.info("Wrote %d rows to %s", len(rows), destination)
    return destination


def _union_of_keys(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return every key across the rows, in first-seen order."""
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def _collect(records: Sequence[Mapping[str, Any]], getter: Any) -> list[float]:
    """Pull one numeric field out of every record that carries it.

    Taken as an argument rather than closed over, so the extractor cannot capture a loop variable
    and read the wrong group -- a late-binding bug that would silently mix arms.

    Args:
        records: Records to read.
        getter: Callable returning the field, or None when absent.

    Returns:
        The values present, as floats.
    """
    values: list[float] = []
    for record in records:
        value = getter(record)
        if value is not None:
            values.append(float(value))
    return values


def _retention_of(record: Mapping[str, Any]) -> float | None:
    """Perplexity retention as a percentage, or None."""
    value = (record.get("quality") or {}).get("retention")
    if isinstance(value, dict):
        return value.get("perplexity_retention")
    return value


def _method_order() -> tuple[Any, ...]:
    """Fixed arm ordering, with the Q->P sequential variant beside its P->Q sibling."""
    from scale_aware_compression.constants import CompressionMethod

    return (
        CompressionMethod.DENSE,
        CompressionMethod.PRUNING,
        CompressionMethod.QUANTISATION,
        CompressionMethod.SEQUENTIAL,
        CompressionMethod.SEQUENTIAL_QP,
        CompressionMethod.JOINT,
    )


def build_main_results_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the main results table: every arm at every model scale.

    Args:
        records: Loaded run records.

    Returns:
        Table rows, ordered by parameter count then by
        :data:`~scale_aware_compression.visualisation.plots.METHOD_ORDER`.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    import statistics
    from collections import defaultdict

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "success":
            continue
        grouped[
            (
                str(record.get("model_name")),
                str(record.get("compression_method")),
                str(record.get("budget_label")),
            )
        ].append(record)

    def mean_std(values: list[float]) -> str:
        """Format a mean with its spread, or a bare value when there is only one."""
        if not values:
            return "--"
        if len(values) == 1:
            return f"{values[0]:.4g}"
        return f"{statistics.mean(values):.4g} ± {statistics.stdev(values):.2g}"

    rows: list[dict[str, Any]] = []
    for key, members in grouped.items():
        model_name, method, budget = key
        first = members[0]
        statistics_block = ((first.get("compression") or {}).get("statistics")) or {}

        perplexities = _collect(
            members, lambda r: ((r.get("quality") or {}).get("perplexity") or {}).get("perplexity")
        )
        retentions = _collect(members, _retention_of)
        latencies = _collect(
            members, lambda r: (r.get("deployment") or {}).get("latency_median_ms")
        )
        p95 = _collect(members, lambda r: (r.get("deployment") or {}).get("latency_p95_ms"))
        throughput = _collect(
            members, lambda r: (r.get("deployment") or {}).get("throughput_tokens_per_s")
        )
        peak_memory = _collect(members, lambda r: (r.get("deployment") or {}).get("peak_memory_mb"))
        sizes = _collect(members, lambda r: (r.get("checkpoint") or {}).get("checkpoint_total_mb"))

        rows.append(
            {
                "model": model_name,
                "targeted_parameters": statistics_block.get("targeted_parameters")
                or first.get("parameter_count"),
                "method": method,
                "budget": budget,
                # R is reported per row because §5.1 requires it, and because a row averaged over
                # one replicate must never be read as though it were averaged over eight.
                "replicates": len(members),
                "sparsity": first.get("sparsity"),
                "bits": first.get("quantisation_bits"),
                "perplexity": mean_std(perplexities),
                "retention_pct": mean_std(retentions),
                "checkpoint_mb": mean_std(sizes),
                "latency_median_ms": mean_std(latencies),
                "latency_p95_ms": mean_std(p95),
                "throughput_tok_s": mean_std(throughput),
                "peak_memory_mb": mean_std(peak_memory),
            }
        )

    order = {method.value: index for index, method in enumerate(_method_order())}
    rows.sort(
        key=lambda row: (
            row["targeted_parameters"] or 0,
            order.get(str(row["method"]), len(order)),
            str(row["budget"]),
        )
    )
    return rows


def build_joint_gain_table(trend: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the joint-gain-versus-scale table that accompanies Figure 1.

    Args:
        trend: Rows from :func:`~scale_aware_compression.experiments.scale_sweep.scale_trend`.

    Returns:
        Table rows, ordered by parameter count.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    import statistics
    from collections import defaultdict

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trend:
        if row.get("joint_gain_retention_pp") is None:
            continue
        grouped[(str(row.get("model_name")), str(row.get("budget_label")))].append(row)

    rows: list[dict[str, Any]] = []
    for (model_name, budget), members in grouped.items():
        gains = [float(member["joint_gain_retention_pp"]) for member in members]
        sequential = [
            float(member["sequential_retention"])
            for member in members
            if member.get("sequential_retention") is not None
        ]
        joint = [
            float(member["joint_retention"])
            for member in members
            if member.get("joint_retention") is not None
        ]
        advantage = [
            float(member["joint_advantage_nll"])
            for member in members
            if member.get("joint_advantage_nll") is not None
        ]
        positive = sum(1 for gain in gains if gain > 0.0)
        # A row built from any incomparable pair is flagged rather than dropped: dropping it would
        # shrink the denominator silently, which is how B-50 hid five pairs.
        incomparable = [
            member.get("joint_experiment_id")
            for member in members
            if not member.get("comparable", True)
        ]
        rows.append(
            {
                "model": model_name,
                "targeted_parameters": members[0].get("targeted_parameters"),
                "budget": budget,
                "replicates": len(gains),
                "sequential_retention_pct": f"{statistics.mean(sequential):.4f}"
                if sequential
                else "--",
                "joint_retention_pct": f"{statistics.mean(joint):.4f}" if joint else "--",
                "joint_gain_pp": f"{statistics.mean(gains):+.4f}",
                "gain_sd": f"{statistics.stdev(gains):.4f}" if len(gains) > 1 else "--",
                "positive_replicates": f"{positive}/{len(gains)}",
                "sign_consistent": positive == len(gains),
                "nll_advantage": f"{statistics.mean(advantage):+.6f}" if advantage else "--",
                "incomparable_pairs": len(incomparable),
            }
        )

    rows.sort(key=lambda row: (row["targeted_parameters"] or 0, str(row["budget"])))
    return rows
