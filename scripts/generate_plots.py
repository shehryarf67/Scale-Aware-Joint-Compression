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

from scale_aware_compression.hardware import host_key
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
        "--eval-split",
        default="test",
        help=(
            "Only plot records evaluated on this split (default: test, the confirmatory one). "
            "Pass 'all' to plot everything, which mixes the exploratory validation records with "
            "the confirmatory ones and is almost never what a figure should show"
        ),
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
        Counts by model and by method, the distinct hosts overall, and -- separately -- the
        distinct hosts among records that actually carry a deployment measurement. Only the
        latter set breaks comparability, which is why the two are counted apart.
    """
    models: dict[str, int] = {}
    methods: dict[str, int] = {}
    hosts: set[str] = set()
    deployment_hosts: set[str] = set()
    threads: set[int] = set()

    for record in records:
        model = str(record.get("model_name", "unknown"))
        method = str(record.get("compression_method", "unknown"))
        models[model] = models.get(model, 0) + 1
        methods[method] = methods.get(method, 0) + 1

        machine = record.get("hardware")
        host = host_key(machine) if isinstance(machine, dict) else "unknown"
        if host != "unknown":
            hosts.add(host)

        # A deployment measurement is a property of the machine as much as of the model. A
        # compression-only record is not: it carries perplexity and sizes, which differ across
        # hosts by floating-point reduction order alone. Counting the two together made this
        # warning fire on the benign case as soon as a second machine ran any compression, and a
        # warning that fires when nothing is wrong stops being read when something is.
        deployment = record.get("deployment")
        if isinstance(deployment, dict) and deployment:
            if host != "unknown":
                deployment_hosts.add(host)
            if deployment.get("num_threads") is not None:
                threads.add(int(deployment["num_threads"]))

    return {
        "num_records": len(records),
        "by_model": models,
        "by_method": methods,
        "distinct_hosts": sorted(hosts),
        "distinct_deployment_hosts": sorted(deployment_hosts),
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

    # Filter by evaluation split BEFORE anything is plotted. `outputs/metrics` holds the exploratory
    # validation records alongside the confirmatory test ones, and a figure that silently averages
    # the two is the B-45 failure in visual form: the shapes match, so nothing complains. The
    # default is the confirmatory split, because that is what a published figure should show.
    if arguments.eval_split != "all":
        wanted = arguments.eval_split
        kept = [
            record
            for record in records
            if (((record.get("config") or {}).get("data") or {}).get("eval_split")) == wanted
        ]
        LOGGER.info(
            "Filtered to eval_split=%r: %d of %d record(s) kept", wanted, len(kept), len(records)
        )
        if not kept:
            LOGGER.error(
                "No records evaluated on split %r. Use --eval-split all to plot everything.",
                wanted,
            )
            return 1
        records = kept

    summary = summarise_records(records)
    log_key_values(LOGGER, f"Loaded {len(records)} record(s) from {results_dir}", summary)

    deployment_hosts: list[str] = summary["distinct_deployment_hosts"]  # type: ignore[assignment]
    if len(deployment_hosts) > 1:
        # Not a warning. Latency, throughput and peak memory from two machines cannot be averaged
        # or plotted together under any correction, so a figure built from this record set would
        # be wrong rather than imprecise. benchmarking_protocol.md: one machine per results table.
        LOGGER.error(
            "Deployment measurements span %d machines: %s. CPU latencies from different hosts "
            "are not comparable and must never share a table. Re-run the deployment benchmarks on "
            "one machine, or filter the record set before plotting.",
            len(deployment_hosts),
            deployment_hosts,
        )
        return 1
    if len(summary["distinct_hosts"]) > 1:  # type: ignore[arg-type]
        # Benign and expected once two people share the compression work. Quality and size are
        # portable; only the comparison must not span hosts, and that is checked per record by
        # `ExperimentTracker.exists_valid` rather than here.
        LOGGER.info(
            "Records come from %d machines. That is fine for compression and quality; only "
            "deployment measurements are host-bound, and none of those are mixed.",
            len(summary["distinct_hosts"]),  # type: ignore[arg-type]
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
