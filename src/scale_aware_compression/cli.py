"""Console entry point: ``sajc <subcommand>``.

Mirrors the scripts in ``scripts/``, so the same operation is available either way. The CLI is a
thin layer: it parses arguments, configures logging, seeds, and delegates. No pipeline logic
lives here.

Only ``sajc info`` does real work in the current scaffold; every other subcommand validates its
configuration and then reports which module still needs implementing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from scale_aware_compression import __version__
from scale_aware_compression.config import ConfigError, ExperimentConfig, load_config
from scale_aware_compression.constants import CompressionMethod
from scale_aware_compression.logging_utils import configure_logging, get_logger, log_key_values

LOGGER = get_logger(__name__)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_ERROR = 1

_RUN_SUBCOMMANDS: dict[str, CompressionMethod] = {
    "baseline": CompressionMethod.DENSE,
    "prune": CompressionMethod.PRUNING,
    "quantise": CompressionMethod.QUANTISATION,
    "sequential": CompressionMethod.SEQUENTIAL,
    "joint": CompressionMethod.JOINT,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="sajc",
        description=(
            "Scale-aware joint compression: does model scale change whether joint "
            "pruning-aware quantisation beats sequential pruning then quantisation?"
        ),
        epilog=(
            "All deployment measurements (latency, throughput, memory, checkpoint size) are "
            "CPU-only. GPUs may be used for fine-tuning, recovery, calibration, and joint "
            "training. See docs/benchmarking_protocol.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    info = subparsers.add_parser(
        "info",
        help="Print the model registry and this machine's hardware/software details",
        description=(
            "Reports the registered models and the environment. Downloads nothing and runs no "
            "compute; use it to check an environment before launching a sweep."
        ),
    )
    info.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table"
    )

    for name, method in _RUN_SUBCOMMANDS.items():
        subparser = subparsers.add_parser(
            name,
            help=f"Run the {method.value} arm for one model",
            description=f"Runs the {method.value} arm described by a YAML configuration.",
        )
        _add_config_arguments(subparser)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate quality for an existing checkpoint (CPU)",
        description="Computes perplexity, agreement, and generation diagnostics on CPU.",
    )
    _add_config_arguments(evaluate)
    evaluate.add_argument(
        "--checkpoint", help="Checkpoint directory to evaluate instead of the configured model"
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run the CPU deployment benchmark",
        description=(
            "Measures latency, throughput, and peak memory on CPU with a pinned thread count."
        ),
    )
    _add_config_arguments(benchmark)
    benchmark.add_argument(
        "--threads", type=int, help="Override benchmark.num_threads for this run"
    )

    sweep = subparsers.add_parser(
        "sweep",
        help="Run the full scale sweep",
        description="Expands models x methods x budgets x seeds and runs every cell.",
    )
    _add_config_arguments(sweep)
    sweep.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the expanded run plan and exit without running anything",
    )

    plots = subparsers.add_parser(
        "plots",
        help="Generate figures and tables from recorded results",
        description="Reads run records and writes figures and tables.",
    )
    plots.add_argument("--results", required=True, help="Directory of JSON run records")
    plots.add_argument("--output", default="outputs/figures", help="Destination for figures")
    plots.add_argument("--log-level", default="INFO", help="Logging level")

    return parser


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments every config-driven subcommand shares."""
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config value, e.g. --override runtime.seed=7. Repeatable. Overrides are "
            "recorded in the run record."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the plan without running anything",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")


def _load(arguments: argparse.Namespace) -> ExperimentConfig:
    """Load and validate the configuration named on the command line."""
    config = load_config(arguments.config, arguments.override)
    if arguments.log_level:
        config.runtime.log_level = arguments.log_level.upper()
    return config


def run_arm(
    method: CompressionMethod,
    *,
    config_path: str,
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    log_level: str | None = None,
) -> int:
    """Run one compression arm from a configuration path.

    Shared by the ``sajc`` subcommands and by the per-arm scripts in ``scripts/``, so both entry
    points behave identically.

    Args:
        method: Which arm to run. Overrides the method declared in the config, with a warning.
        config_path: Path to a YAML experiment configuration.
        overrides: ``dotted.key=value`` overrides applied after includes.
        dry_run: Validate and print the plan without executing anything.
        log_level: Override ``runtime.log_level``.

    Returns:
        A process exit code.

    Raises:
        ConfigError: If the configuration is invalid.
        NotImplementedError: When the requested pipeline is still a placeholder.
    """
    arguments = argparse.Namespace(
        config=config_path,
        override=list(overrides),
        dry_run=dry_run,
        log_level=log_level,
    )
    return _run_arm(arguments, method)


def _run_arm(arguments: argparse.Namespace, method: CompressionMethod) -> int:
    """Handle the five arm subcommands."""
    from scale_aware_compression.experiments.runner import ExperimentRunner
    from scale_aware_compression.seed import set_global_seed

    # The solver's OpenMP deadlock mitigation is scoped to compress_layer, not applied here:
    # a process-wide inter-op pin cannot be undone and would override benchmark.interop_threads.
    config = _load(arguments)
    if config.compression.method is not method:
        LOGGER.warning(
            "Config declares method=%s but the '%s' subcommand was used; forcing %s.",
            config.compression.method.value,
            method.value,
            method.value,
        )
        config.compression.method = method

    configure_logging(config.runtime.log_level)
    LOGGER.info("%s", config.describe())

    runner = ExperimentRunner(config)
    if arguments.dry_run:
        log_key_values(LOGGER, "Dry run -- nothing was executed:", runner.dry_run())
        return EXIT_OK

    set_global_seed(config.runtime.seed, deterministic=config.runtime.deterministic)
    runner.run()
    return EXIT_OK


def _run_info(arguments: argparse.Namespace) -> int:
    """Handle ``sajc info``."""
    import json

    from scale_aware_compression.hardware import describe_environment
    from scale_aware_compression.models.registry import registry_table
    from scale_aware_compression.visualisation.tables import rows_to_markdown

    configure_logging("INFO")
    environment = describe_environment()
    models = registry_table()

    if arguments.json:
        sys.stdout.write(
            json.dumps({"models": models, **environment}, indent=2, default=str) + "\n"
        )
        return EXIT_OK

    sys.stdout.write("Registered models\n-----------------\n")
    sys.stdout.write(rows_to_markdown(models) + "\n\n")
    log_key_values(LOGGER, "Hardware", environment["hardware"])
    log_key_values(LOGGER, "Software", environment["software"])
    return EXIT_OK


def _run_evaluate(arguments: argparse.Namespace) -> int:
    """Handle ``sajc evaluate``."""
    from scale_aware_compression.evaluation.quality import check_evaluation_device

    config = _load(arguments)
    configure_logging(config.runtime.log_level)
    check_evaluation_device(config.evaluation)

    if arguments.dry_run:
        log_key_values(
            LOGGER,
            "Dry run -- nothing was executed:",
            {
                "model": config.model.name,
                "checkpoint": arguments.checkpoint or "(from config)",
                "metrics": config.evaluation.metrics,
                "device": config.evaluation.device.value,
                "max_samples": config.evaluation.max_samples,
            },
        )
        return EXIT_OK

    raise NotImplementedError(
        "Quality evaluation is not implemented yet; see the TODO in evaluation/quality.py"
    )


def _run_benchmark(arguments: argparse.Namespace) -> int:
    """Handle ``sajc benchmark``."""
    config = _load(arguments)
    if arguments.threads:
        config.benchmark.num_threads = arguments.threads
    configure_logging(config.runtime.log_level)

    if arguments.dry_run:
        log_key_values(
            LOGGER,
            "Dry run -- nothing was executed:",
            {
                "model": config.model.name,
                "device": config.benchmark.device.value,
                "num_threads": config.benchmark.num_threads,
                "batch_size": config.benchmark.batch_size,
                "sequence_length": config.benchmark.sequence_length,
                "warmup_runs": config.benchmark.warmup_runs,
                "measured_runs": config.benchmark.measured_runs,
            },
        )
        return EXIT_OK

    raise NotImplementedError(
        "Model benchmarking needs build_forward_callable; see the TODO in benchmarking/cpu.py"
    )


def _run_sweep(arguments: argparse.Namespace) -> int:
    """Handle ``sajc sweep``."""
    from scale_aware_compression.experiments.scale_sweep import (
        build_sweep_plan,
        executable_cells,
        find_comparison_pairs,
        run_sweep,
    )

    config = _load(arguments)
    configure_logging(config.runtime.log_level)
    plan = build_sweep_plan(config)

    if arguments.plan_only or arguments.dry_run:
        log_key_values(LOGGER, "Sweep plan", plan.summary())
        executable = executable_cells(plan)
        LOGGER.info(
            "%d logical grid slot(s), %d executable cell(s)", len(plan.cells), len(executable)
        )
        for cell in executable:
            LOGGER.info("  %s", cell.experiment_id)
        pairs = find_comparison_pairs(plan)
        LOGGER.info("%d joint/sequential pair(s) can yield a joint gain", len(pairs))
        return EXIT_OK

    run_sweep(config, plan)
    return EXIT_OK


def _run_plots(arguments: argparse.Namespace) -> int:
    """Handle ``sajc plots``."""
    from scale_aware_compression.experiments.runner import ExperimentTracker

    configure_logging(arguments.log_level)
    tracker = ExperimentTracker(arguments.results)
    records = tracker.load_all()
    LOGGER.info("Loaded %d record(s) from %s", len(records), arguments.results)
    if not records:
        LOGGER.error("No records found in %s; run some experiments first.", arguments.results)
        return EXIT_ERROR

    raise NotImplementedError(
        "Figure generation is not implemented yet; see the TODOs in visualisation/plots.py"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code: 0 on success, 1 on error, 2 on usage error, 3 when the requested
        operation is not implemented yet.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if not arguments.command:
        parser.print_help()
        return EXIT_USAGE

    handlers: dict[str, Any] = {
        "info": _run_info,
        "evaluate": _run_evaluate,
        "benchmark": _run_benchmark,
        "sweep": _run_sweep,
        "plots": _run_plots,
    }

    try:
        if arguments.command in _RUN_SUBCOMMANDS:
            return int(_run_arm(arguments, _RUN_SUBCOMMANDS[arguments.command]))
        return int(handlers[arguments.command](arguments))
    except ConfigError as error:
        LOGGER.error("Invalid configuration: %s", error)
        return EXIT_USAGE
    except NotImplementedError as error:
        LOGGER.error("Not implemented yet: %s", error)
        return EXIT_NOT_IMPLEMENTED
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return EXIT_ERROR
    except Exception as error:  # noqa: BLE001 - top-level boundary: report, do not traceback
        LOGGER.error("%s: %s", type(error).__name__, error)
        LOGGER.debug("Traceback:", exc_info=True)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
