#!/usr/bin/env python
"""Generate figures and tables from recorded results.

Reads the JSON run records written by the sweep and produces the four figures and two tables the
write-up uses. Runs nothing and loads no model, so it is cheap to re-run after every sweep.

Figures:
    1. joint gain vs model scale       -- the primary result
    2. measured latency vs sparsity    -- with the theoretical bound overlaid
    3. quality retention vs size       -- the Pareto view
    4. joint gain vs training cost     -- what the gain cost to obtain

Examples:
    python scripts/generate_plots.py --results outputs/metrics --output outputs/figures
    python scripts/generate_plots.py --results outputs/metrics --tables-only
    python scripts/generate_plots.py --results outputs/metrics --figures 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scale_aware_compression.logging_utils import configure_logging, get_logger, log_key_values

LOGGER = get_logger(__name__)

FIGURE_NUMBERS = ("1", "2", "3", "4")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default="outputs/metrics",
        help="Directory of JSON run records (default: outputs/metrics)",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures",
        help="Destination for figures (default: outputs/figures)",
    )
    parser.add_argument(
        "--tables-output",
        default="outputs/tables",
        help="Destination for tables (default: outputs/tables)",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=FIGURE_NUMBERS,
        help="Generate only these figures (default: all)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tables-only", action="store_true", help="Generate tables, no figures")
    group.add_argument("--figures-only", action="store_true", help="Generate figures, no tables")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats to write (default: png pdf)",
    )
    parser.add_argument(
        "--metric",
        default="quality_retention",
        help="Quality metric to compute joint gain on (default: quality_retention)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what was found in the records without writing any figure or table",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def summarise_records(records: list[dict[str, object]]) -> dict[str, object]:
    """Summarise a record set, so a missing arm is visible before plotting.

    Args:
        records: Loaded run records.

    Returns:
        Counts by model and by method, plus the distinct hardware and thread counts. More than one
        of either means the records are not comparable and must not share a figure.
    """
    models: dict[str, int] = {}
    methods: dict[str, int] = {}
    hardware: set[str] = set()
    threads: set[int] = set()

    for record in records:
        model = str(record.get("model_name", "unknown"))
        method = str(record.get("compression_method", "unknown"))
        models[model] = models.get(model, 0) + 1
        methods[method] = methods.get(method, 0) + 1

        machine = record.get("hardware")
        if isinstance(machine, dict) and machine.get("cpu_model"):
            hardware.add(str(machine["cpu_model"]))
        deployment = record.get("deployment")
        if isinstance(deployment, dict) and deployment.get("num_threads") is not None:
            threads.add(int(deployment["num_threads"]))

    return {
        "num_records": len(records),
        "by_model": models,
        "by_method": methods,
        "distinct_cpu_models": sorted(hardware),
        "distinct_thread_counts": sorted(threads),
    }


def main(argv: list[str] | None = None) -> int:
    """Generate the figures and tables.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if no records were found, 3 when the plotting code is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)

    results_dir = Path(arguments.results)
    if not results_dir.is_dir():
        LOGGER.error("Results directory does not exist: %s", results_dir)
        return 1

    from scale_aware_compression.experiments.runner import ExperimentTracker

    records = ExperimentTracker(results_dir).load_all()
    if not records:
        LOGGER.error("No JSON records found in %s; run some experiments first.", results_dir)
        return 1

    summary = summarise_records(records)
    log_key_values(LOGGER, f"Loaded {len(records)} record(s) from {results_dir}", summary)

    if len(summary["distinct_cpu_models"]) > 1:  # type: ignore[arg-type]
        LOGGER.warning(
            "Records span %d different CPUs. Latencies from different machines are not comparable "
            "and must not appear in the same figure.",
            len(summary["distinct_cpu_models"]),  # type: ignore[arg-type]
        )
    if len(summary["distinct_thread_counts"]) > 1:  # type: ignore[arg-type]
        LOGGER.warning(
            "Records span thread counts %s. Plot each as its own series; never average across "
            "them.",
            summary["distinct_thread_counts"],
        )

    if arguments.dry_run:
        LOGGER.info("Dry run: no figures or tables were written")
        return 0

    from scale_aware_compression.experiments.scale_sweep import scale_trend
    from scale_aware_compression.visualisation import plots, tables

    selected = set(arguments.figures or FIGURE_NUMBERS)
    figure_directory = Path(arguments.output)
    table_directory = Path(arguments.tables_output)
    formats = tuple(arguments.formats)

    try:
        trend = scale_trend(records, metric=arguments.metric)

        if not arguments.tables_only:
            plots.apply_style()
            if "1" in selected:
                plots.plot_joint_gain_vs_scale(trend, figure_directory)
            if "2" in selected:
                plots.plot_latency_vs_sparsity(records, figure_directory)
            if "3" in selected:
                plots.plot_quality_vs_size(records, figure_directory)
            if "4" in selected:
                plots.plot_training_cost(records, figure_directory)

        if not arguments.figures_only:
            main_table = tables.build_main_results_table(records)
            gain_table = tables.build_joint_gain_table(trend)
            table_directory.mkdir(parents=True, exist_ok=True)
            (table_directory / "main_results.md").write_text(
                tables.rows_to_markdown(main_table), encoding="utf-8"
            )
            (table_directory / "joint_gain.md").write_text(
                tables.rows_to_markdown(gain_table), encoding="utf-8"
            )
            tables.rows_to_csv(main_table, table_directory / "main_results.csv")
            tables.rows_to_csv(gain_table, table_directory / "joint_gain.csv")
    except NotImplementedError as error:
        LOGGER.error("Not implemented yet: %s", error)
        return 3

    LOGGER.info(
        "Wrote figures to %s (%s) and tables to %s",
        figure_directory,
        ", ".join(formats),
        table_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
