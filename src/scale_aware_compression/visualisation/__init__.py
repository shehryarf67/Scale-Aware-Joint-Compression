"""Figures and tables. matplotlib is imported lazily, never at module import time."""

from __future__ import annotations

from scale_aware_compression.visualisation.plots import (
    METHOD_LABELS,
    METHOD_ORDER,
    apply_style,
    plot_joint_gain_vs_scale,
    plot_latency_vs_sparsity,
    plot_quality_vs_size,
    plot_training_cost,
    save_figure,
)
from scale_aware_compression.visualisation.tables import (
    build_joint_gain_table,
    build_main_results_table,
    format_value,
    rows_to_csv,
    rows_to_latex,
    rows_to_markdown,
)

__all__ = [
    "METHOD_LABELS",
    "METHOD_ORDER",
    "apply_style",
    "build_joint_gain_table",
    "build_main_results_table",
    "format_value",
    "plot_joint_gain_vs_scale",
    "plot_latency_vs_sparsity",
    "plot_quality_vs_size",
    "plot_training_cost",
    "rows_to_csv",
    "rows_to_latex",
    "rows_to_markdown",
    "save_figure",
]
