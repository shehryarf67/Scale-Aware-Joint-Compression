"""Figures for the write-up.

matplotlib is imported lazily inside each function, so importing this module does not pull in a
plotting backend or touch a display.

Figure 1 is :func:`plot_joint_gain_vs_scale`: it is the answer to the research question, and
everything else is supporting evidence.

Status: placeholder; :func:`save_figure` and the shared style helpers are implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from scale_aware_compression.constants import CompressionMethod
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

LOGGER = get_logger(__name__)

DEFAULT_DPI = 200
DEFAULT_FORMATS: tuple[str, ...] = ("png", "pdf")
"""PNG for quick viewing, PDF for the paper. Both written from the same figure object."""

METHOD_ORDER: tuple[CompressionMethod, ...] = (
    CompressionMethod.DENSE,
    CompressionMethod.PRUNING,
    CompressionMethod.QUANTISATION,
    CompressionMethod.SEQUENTIAL,
    CompressionMethod.JOINT,
)
"""Fixed legend order across every figure, so a reader learns it once."""

METHOD_LABELS: dict[CompressionMethod, str] = {
    CompressionMethod.DENSE: "Dense FP32",
    CompressionMethod.PRUNING: "Pruning only",
    CompressionMethod.QUANTISATION: "Quantisation only",
    CompressionMethod.SEQUENTIAL: "Sequential (prune → quantise)",
    CompressionMethod.JOINT: "Joint (pruning-aware quantisation)",
}


def save_figure(
    figure: Figure,
    output_dir: str | Path,
    name: str,
    *,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Write a figure to every configured format.

    Args:
        figure: The matplotlib figure.
        output_dir: Destination directory. Created if absent.
        name: Base filename, without an extension.
        formats: Extensions to write.
        dpi: Raster resolution.

    Returns:
        The paths written.

    Raises:
        ValueError: If ``formats`` is empty.
    """
    if not formats:
        raise ValueError("save_figure needs at least one output format")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        path = destination / f"{name}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    LOGGER.info("Wrote figure %s (%s)", name, ", ".join(formats))
    return written


def apply_style() -> None:
    """Apply the shared figure style.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "figure.autolayout": True,
            "font.size": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def plot_joint_gain_vs_scale(
    trend: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str = "joint_gain_vs_scale",
) -> list[Path]:
    """Figure 1: joint gain against model scale, one line per compression budget.

    Args:
        trend: Rows from :func:`~scale_aware_compression.experiments.scale_sweep.scale_trend`.
        output_dir: Destination directory.
        name: Base filename.

    Returns:
        The paths written.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(plots): x = parameter_count on a log scale, ticked at the actual model sizes and
    # labelled with size_label; y = joint gain. One line per budget, error bars from the
    # seed-to-seed spread. Draw a horizontal line at y=0: the reader's question is whether the
    # gain is above it, and by more than the error bars.
    # Mark the Qwen validation point with a distinct marker and exclude it from any fitted line,
    # since it is not part of the controlled sweep.
    raise NotImplementedError(
        "plot_joint_gain_vs_scale is not implemented yet; see the TODO in visualisation/plots.py"
    )


def plot_quality_vs_size(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str = "quality_vs_size",
) -> list[Path]:
    """Quality retention against checkpoint size, one series per arm.

    The Pareto view: whether the joint arm sits above and to the left of the sequential arm.

    Args:
        records: Loaded run records.
        output_dir: Destination directory.
        name: Base filename.

    Returns:
        The paths written.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(plots): x = checkpoint_size_mb, y = quality_retention, one subplot per model size so
    # the axes stay readable. Series coloured by method in METHOD_ORDER. Connect points from the
    # same method across budgets to make each arm's trade-off curve visible.
    raise NotImplementedError(
        "plot_quality_vs_size is not implemented yet; see the TODO in visualisation/plots.py"
    )


def plot_latency_vs_sparsity(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str = "latency_vs_sparsity",
) -> list[Path]:
    """Measured CPU latency against sparsity, with the theoretical bound drawn alongside.

    Answers the secondary question of whether theoretical sparsity produces real CPU speedups.
    The gap between the measured points and the dashed bound is the finding.

    Args:
        records: Loaded run records.
        output_dir: Destination directory.
        name: Base filename.

    Returns:
        The paths written.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(plots): x = sparsity, y = median latency normalised to each model's dense baseline.
    # Overlay 1/(1-sparsity) from metrics.efficiency.theoretical_speedup_from_sparsity as a
    # dashed line. Annotate the thread count in the axis label: the curve's shape depends on it,
    # and a reader comparing against another paper needs to know.
    raise NotImplementedError(
        "plot_latency_vs_sparsity is not implemented yet; see the TODO in visualisation/plots.py"
    )


def plot_training_cost(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str = "training_cost",
) -> list[Path]:
    """Joint gain against the extra training cost the joint arm required.

    Args:
        records: Loaded run records.
        output_dir: Destination directory.
        name: Base filename.

    Returns:
        The paths written.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(plots): x = training_cost_overhead (joint steps / sequential steps), y = joint gain,
    # one point per (model, budget), sized by parameter count. Points at x=1 are the matched
    # budgets and are the only ones that support a clean claim; mark the others explicitly.
    raise NotImplementedError(
        "plot_training_cost is not implemented yet; see the TODO in visualisation/plots.py"
    )
