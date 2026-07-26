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
    # TODO(tables): one row per (model, method, budget), averaged over seeds with the standard
    # deviation reported as "mean +/- std". Columns: model, size, method, budget, sparsity, bits,
    # perplexity, retention, checkpoint size, median latency, p95 latency, throughput, peak
    # memory. Include the seed count per row: a row averaged over one seed must not be read as
    # if it were averaged over three.
    raise NotImplementedError(
        "build_main_results_table is not implemented yet; see the TODO in visualisation/tables.py"
    )


def build_joint_gain_table(trend: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the joint-gain-versus-scale table that accompanies Figure 1.

    Args:
        trend: Rows from :func:`~scale_aware_compression.experiments.scale_sweep.scale_trend`.

    Returns:
        Table rows, ordered by parameter count.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(tables): columns: model, size, budget, sequential score, joint score, absolute gain,
    # relative gain, seed spread, and the joint arm's training-cost overhead. Flag rows where
    # match_sequential_budget was false, since their gain is confounded with extra training.
    raise NotImplementedError(
        "build_joint_gain_table is not implemented yet; see the TODO in visualisation/tables.py"
    )
