#!/usr/bin/env python
r"""Run the full scale sweep: models x arms x budgets x seeds.

This is the experiment that answers the research question. It is also long: the shipped grid is 120
runs. Use ``--plan-only`` first to see the expansion, and run the pilot config end to end before
committing compute.

The sweep is resumable. With ``sweep.skip_existing: true`` an interrupted run re-executes only the
cells that have no record yet.

Examples:
    python scripts/run_scale_sweep.py --config configs/experiments/main_scale_sweep.yaml --plan-only
    python scripts/run_scale_sweep.py --config configs/experiments/main_scale_sweep.yaml
    python scripts/run_scale_sweep.py --config configs/experiments/main_scale_sweep.yaml \\
        --override sweep.seeds=[1234] --override sweep.budgets=[moderate]
    python scripts/run_scale_sweep.py --config configs/experiments/qwen_validation.yaml
"""

from __future__ import annotations

import argparse

from scale_aware_compression.config import ConfigError, load_config
from scale_aware_compression.logging_utils import configure_logging, get_logger, log_key_values

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to a YAML sweep configuration")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --override sweep.seeds=[1234]. Repeatable.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the expanded run plan and the joint/sequential pairs, then exit",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="NAME", help="Restrict the sweep to these models"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["dense", "pruning", "quantisation", "sequential", "joint"],
        help="Restrict the sweep to these arms",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Alias for --plan-only; runs nothing"
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the sweep.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)

    overrides = list(arguments.override)
    if arguments.models:
        overrides.append(f"sweep.models=[{','.join(arguments.models)}]")
    if arguments.methods:
        overrides.append(f"sweep.methods=[{','.join(arguments.methods)}]")

    try:
        config = load_config(arguments.config, overrides)
    except ConfigError as error:
        configure_logging(arguments.log_level or "INFO")
        LOGGER.error("Invalid configuration: %s", error)
        return 2

    configure_logging(arguments.log_level or config.runtime.log_level)

    from scale_aware_compression.experiments.scale_sweep import (
        build_sweep_plan,
        find_comparison_pairs,
        run_sweep,
    )

    try:
        plan = build_sweep_plan(config)
    except (ConfigError, KeyError) as error:
        LOGGER.error("Could not build the sweep plan: %s", error)
        return 2

    log_key_values(LOGGER, "Sweep plan", plan.summary())
    pairs = find_comparison_pairs(plan)
    LOGGER.info("%d joint/sequential pair(s) can yield a joint gain", len(pairs))

    if arguments.plan_only or arguments.dry_run:
        for cell in plan.cells:
            LOGGER.info("  %s", cell.experiment_id)
        LOGGER.info("Plan only: nothing was executed")
        return 0

    if not pairs:
        LOGGER.warning(
            "The plan contains no matched joint/sequential pair, so it cannot produce a joint "
            "gain. Include both 'sequential' and 'joint' in sweep.methods."
        )

    try:
        records = run_sweep(config, plan)
    except NotImplementedError as error:
        LOGGER.error("Not implemented yet: %s", error)
        return 3

    LOGGER.info("Completed %d run(s)", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
