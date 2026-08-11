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

GAIN_KEY = "joint_gain_retention_pp"
"""The trend column every figure plots. Named once so a rename cannot half-apply."""

IMPORTANCE_THRESHOLD_PP = 1.0
"""§6.3's practical-importance bar, drawn so a reader sees the criterion, not just the sign."""

BUDGET_LABELS: dict[str, str] = {
    "moderate": "30% + W8 (control)",
    "aggressive": "30% + W4 (headline)",
}
"""Budget labels say the recipe. "moderate" alone tells a reader nothing about precision."""

SEQUENTIAL_METHOD_VALUES = ("sequential", "sequential_qp")
"""Both frozen orders count as the sequential arm (B-42, B-50)."""


def _retention(record: dict[str, Any]) -> float | None:
    """Perplexity retention as a percentage, or None when the record carries none."""
    value = (record.get("quality") or {}).get("retention")
    if isinstance(value, dict):
        return value.get("perplexity_retention")
    return value


def _checkpoint_mb(record: dict[str, Any]) -> float | None:
    """Checkpoint size in MB, or None."""
    checkpoint = record.get("checkpoint") or {}
    for key in ("checkpoint_total_mb", "checkpoint_size_mb"):
        if checkpoint.get(key) is not None:
            return float(checkpoint[key])
    return None


def _method_label(method: str) -> str:
    """Human-readable arm name, tolerating the two sequential orders."""
    for enum_member in METHOD_ORDER:
        if enum_member.value == method:
            return METHOD_LABELS[enum_member]
    if method == "sequential_qp":
        return "Sequential (quantise → prune)"
    return method


METHOD_COLOURS: dict[str, str] = {
    "dense": "0.35",
    "pruning": "C1",
    "quantisation": "C2",
    "sequential": "C3",
    "sequential_qp": "C5",
    "joint": "C0",
}
"""Colour per ARM, fixed across every figure and every subplot.

Assigning colours by position within a subplot means the same colour denotes different arms in
different panels -- 1B carries ``sequential_qp`` where the other scales carry ``sequential`` -- and
a reader comparing panels is then silently misled. Joint takes C0 because it is the arm under test.
"""


def _method_colour(method: str) -> str:
    """Stable colour for an arm."""
    return METHOD_COLOURS.get(method, "0.6")


def _ordered_methods(by_method: dict[str, Any]) -> list[str]:
    """Arms in the fixed legend order, with anything unrecognised appended."""
    order = [member.value for member in METHOD_ORDER]
    order.insert(order.index("sequential") + 1, "sequential_qp")
    known = [value for value in order if value in by_method]
    return known + sorted(set(by_method) - set(known))


def trend_rows_from(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Comparable trend rows for these records.

    Kept here so a figure never re-implements pairing. B-50 was exactly that: a second copy of the
    pairing rule that had not learned about the Q->P order and silently dropped five pairs.

    Args:
        records: Loaded run records.

    Returns:
        Rows from :func:`~scale_aware_compression.experiments.scale_sweep.scale_trend` that are
        marked comparable and carry a gain.
    """
    from scale_aware_compression.experiments.scale_sweep import scale_trend

    return [
        row
        for row in scale_trend(records)
        if row.get("comparable") and row.get(GAIN_KEY) is not None
    ]


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
    import statistics
    from collections import defaultdict

    import matplotlib.pyplot as plt

    apply_style()
    usable = [row for row in trend if row.get("comparable") and row.get(GAIN_KEY) is not None]
    if not usable:
        raise ValueError("plot_joint_gain_vs_scale needs at least one comparable trend row")

    # (budget, targeted_parameters) -> every replicate gain. The replicates are plotted
    # individually, not just their mean: F-26 exists because a mean concealed a sign flip, and a
    # figure that shows only means reproduces exactly that failure.
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    labels: dict[int, str] = {}
    for row in usable:
        size = int(row.get("targeted_parameters") or 0)
        grouped[(str(row.get("budget_label")), size)].append(float(row[GAIN_KEY]))
        labels[size] = str(row.get("model_name"))

    figure, axes = plt.subplots()
    axes.axhline(0.0, color="0.4", linewidth=1.0, zorder=1)
    # The pre-registered practical-importance bar. Without it a reader cannot see that the
    # question was never "is it positive" but "does it clear 1.0 pp with a consistent sign".
    axes.axhline(
        IMPORTANCE_THRESHOLD_PP,
        color="0.4",
        linewidth=1.0,
        linestyle=":",
        zorder=1,
    )
    axes.text(
        0.995,
        IMPORTANCE_THRESHOLD_PP,
        " §6.3 practical-importance bar",
        transform=axes.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    for index, budget in enumerate(sorted({key[0] for key in grouped})):
        sizes = sorted(size for label, size in grouped if label == budget)
        means, errors = [], []
        for size in sizes:
            gains = grouped[(budget, size)]
            means.append(statistics.mean(gains))
            # Standard error of the mean: the figure is about where the mean lies, and the SD
            # would overstate the uncertainty on that.
            errors.append(statistics.stdev(gains) / (len(gains) ** 0.5) if len(gains) > 1 else 0.0)
            axes.scatter(
                [size] * len(gains),
                gains,
                s=14,
                alpha=0.35,
                color=f"C{index}",
                zorder=2,
                linewidths=0,
            )
        axes.errorbar(
            sizes,
            means,
            yerr=errors,
            marker="o",
            capsize=3,
            color=f"C{index}",
            label=BUDGET_LABELS.get(budget, budget),
            zorder=3,
        )

    axes.set_xscale("log")
    ticks = sorted(labels)
    axes.set_xticks(ticks)
    axes.set_xticklabels([labels[size] for size in ticks])
    axes.minorticks_off()
    axes.set_xlabel("Model scale (targeted non-embedding parameters, log)")
    axes.set_ylabel("Joint gain (pp of perplexity retention)")
    axes.set_title("Joint gain against model scale")
    axes.legend(loc="best")
    written = save_figure(figure, output_dir, name)
    plt.close(figure)
    return written


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
    from collections import defaultdict

    import matplotlib.pyplot as plt

    apply_style()
    points: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        size = _checkpoint_mb(record)
        retention = _retention(record)
        if size is None or retention is None:
            continue
        points[str(record.get("model_name"))][str(record.get("compression_method"))].append(
            (size, retention)
        )
    if not points:
        raise ValueError("plot_quality_vs_size found no record with both a size and a retention")

    models = sorted(
        points, key=lambda name: min(s for arm in points[name].values() for s, _ in arm)
    )
    figure, axes_list = plt.subplots(
        1, len(models), figsize=(4.2 * len(models), 3.8), squeeze=False, sharey=True
    )
    for axes, model in zip(axes_list[0], models, strict=False):
        for method in _ordered_methods(points[model]):
            values = sorted(points[model][method])
            axes.plot(
                [size for size, _ in values],
                [retention for _, retention in values],
                marker="o",
                markersize=5,
                linewidth=1.2,
                # Colour keyed to the ARM, never to its index within this subplot. Indexing per
                # subplot gives the same colour to different arms in different panels, because 1B
                # carries `sequential_qp` where the others carry `sequential`.
                color=_method_colour(method),
                label=_method_label(method),
            )
        axes.set_title(model)
        axes.set_xlabel("Checkpoint size (MB)")
    axes_list[0][0].set_ylabel("Perplexity retention (%)")
    # One legend for the row, gathered across EVERY subplot: the first panel alone omits any arm
    # that only appears at another scale, which would leave the 1B Q->P series unlabelled.
    merged: dict[str, Any] = {}
    for axes in axes_list[0]:
        for handle, legend_label in zip(*axes.get_legend_handles_labels(), strict=False):
            merged.setdefault(legend_label, handle)
    figure.legend(
        list(merged.values()),
        list(merged),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.10),
    )
    written = save_figure(figure, output_dir, name)
    plt.close(figure)
    return written


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
    from collections import defaultdict

    import matplotlib.pyplot as plt

    from scale_aware_compression.metrics.efficiency import theoretical_speedup_from_sparsity

    apply_style()

    # Only FP32 arms carry a usable timing. A packed artefact is timed through a dequantising
    # path, so its number measures unpacking, and the runner refuses it -- which is why there is
    # no joint-versus-sequential latency comparison anywhere in this study.
    timed: dict[str, dict[float, list[tuple[float, str]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        deployment = record.get("deployment") or {}
        median = deployment.get("latency_median_ms")
        if median is None:
            continue
        sparsity = float(record.get("sparsity") or 0.0)
        timed[str(record.get("model_name"))][sparsity].append(
            (float(median), str(record.get("timestamp", ""))[:10])
        )
    if not timed:
        raise ValueError("plot_latency_vs_sparsity found no record carrying a latency")

    figure, axes = plt.subplots()
    sparsities = sorted({value for model in timed.values() for value in model})
    contaminated: list[str] = []
    for index, model in enumerate(sorted(timed)):
        baseline = timed[model].get(0.0)
        if not baseline:
            LOGGER.warning("No dense baseline for %s; its latencies cannot be normalised", model)
            continue
        reference, reference_day = min(baseline)
        xs, ys, cross_session = [], [], False
        for sparsity in sorted(timed[model]):
            fastest, day = min(timed[model][sparsity])
            xs.append(sparsity)
            ys.append(reference / fastest)
            if day != reference_day:
                cross_session = True

        # A speedup above 1/(1-s) is not a fast kernel, it is a broken comparison: the bound
        # assumes every zero is skipped for free, which no dense FP32 GEMM approaches. B-49 is the
        # cause -- the dense baseline and the sparse measurement were taken days apart, on a host
        # whose state moved between them. Such a series is drawn hollow and dashed and named in
        # the caption, never as a plain result.
        violates = any(
            value > theoretical_speedup_from_sparsity(sparsity) + 1e-9
            for sparsity, value in zip(xs, ys, strict=False)
            if sparsity > 0.0
        )
        suspect = violates or cross_session
        if violates:
            contaminated.append(model)
            LOGGER.error(
                "%s exceeds the theoretical speedup bound; its dense baseline (%s) and sparse "
                "measurement were taken in different sessions. See B-49.",
                model,
                reference_day,
            )
        axes.plot(
            xs,
            ys,
            marker="o",
            color=f"C{index}",
            linestyle="--" if suspect else "-",
            markerfacecolor="none" if suspect else f"C{index}",
            label=f"{model} — cross-session, not comparable" if suspect else model,
        )

    # The optimistic bound, drawn only across the sparsities actually measured. Extending it
    # further would invite reading this as a curve, which it is not.
    if sparsities:
        bound_x = list(sparsities)
        axes.plot(
            bound_x,
            [theoretical_speedup_from_sparsity(value) for value in bound_x],
            linestyle="--",
            color="0.4",
            label="Theoretical bound 1/(1−s)",
        )

    axes.axhline(1.0, color="0.7", linewidth=1.0, zorder=1)
    axes.set_xlabel("Unstructured sparsity")
    axes.set_ylabel("Speedup over dense (higher is faster)")
    axes.set_title(
        "Measured CPU speedup at a single sparsity\n"
        "NOT a sparsity curve: one non-zero sparsity was run"
    )
    if contaminated:
        axes.text(
            0.5,
            -0.28,
            "Dashed/hollow: dense baseline and sparse measurement taken in different sessions,\n"
            f"so the ratio is not interpretable ({', '.join(sorted(contaminated))} exceeds the\n"
            "theoretical bound, which is impossible). See B-49; cite F-34 for latency.",
            transform=axes.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="0.25",
        )
    axes.legend(loc="best")
    written = save_figure(figure, output_dir, name)
    plt.close(figure)
    return written


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
    import statistics
    from collections import defaultdict

    import matplotlib.pyplot as plt

    apply_style()

    # Solver budget per arm. §3.11 makes `local_steps` the unit of fairness and
    # `assert_matched_plans` refuses to run a mismatched grid, so every ratio here is expected to
    # be exactly 1.0. That is the point of the figure: it is a fairness AUDIT, not a trade-off
    # curve, and any point off 1.0 means the comparison it belongs to is void.
    steps: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    gains: dict[tuple[str, str], list[float]] = defaultdict(list)
    sizes: dict[tuple[str, str], int] = {}
    for record in records:
        key = (str(record.get("model_name")), str(record.get("budget_label")))
        statistics_block = ((record.get("compression") or {}).get("statistics")) or {}
        total = statistics_block.get("total_local_steps")
        method = str(record.get("compression_method"))
        if total is not None:
            steps[key][method] = int(total)
        sizes[key] = int(statistics_block.get("targeted_parameters") or 0)

    for row in trend_rows_from(records):
        gains[(str(row["model_name"]), str(row["budget_label"]))].append(float(row[GAIN_KEY]))

    figure, axes = plt.subplots()
    plotted = 0
    for key, arms in sorted(steps.items()):
        joint = arms.get("joint")
        sequential = arms.get("sequential") or arms.get("sequential_qp")
        if not joint or not sequential or key not in gains or not gains[key]:
            continue
        ratio = joint / sequential
        mean_gain = statistics.mean(gains[key])
        axes.scatter(
            [ratio],
            [mean_gain],
            s=30 + 90 * (sizes.get(key, 0) / max(sizes.values() or [1])),
            alpha=0.85,
            zorder=3,
        )
        # Every point lands on x=1.0 by design, so labels stack. Alternate the side and nudge
        # vertically, otherwise the two near-zero W8 cells overprint each other illegibly.
        offset_x = 10 if plotted % 2 == 0 else -10
        axes.annotate(
            f"{key[0]}\n{BUDGET_LABELS.get(key[1], key[1])}",
            (ratio, mean_gain),
            textcoords="offset points",
            xytext=(offset_x, 6 if plotted % 4 < 2 else -14),
            ha="left" if offset_x > 0 else "right",
            fontsize=7,
        )
        plotted += 1
    if not plotted:
        raise ValueError("plot_training_cost found no cell with both arms' step counts and a gain")

    axes.axvline(1.0, color="0.4", linewidth=1.0, linestyle="--", zorder=1)
    axes.axhline(0.0, color="0.7", linewidth=1.0, zorder=1)
    axes.set_xlim(0.5, 1.5)
    axes.set_xlabel("Solver budget ratio, joint / sequential (1.0 = matched)")
    axes.set_ylabel("Mean joint gain (pp)")
    axes.set_title(
        "Fairness audit: gain against solver budget\n"
        "Every point must sit on the dashed line; marker size is model scale"
    )
    written = save_figure(figure, output_dir, name)
    plt.close(figure)
    return written
