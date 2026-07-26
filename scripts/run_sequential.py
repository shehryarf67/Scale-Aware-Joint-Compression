#!/usr/bin/env python
r"""Run the sequential arm: prune, recover, then quantise.

Pipeline, explicitly:

    dense model -> pruning -> recovery -> quantisation -> conversion

This is the baseline pipeline the joint arm is measured against, and standard practice in the
literature. The quantisation observers see the already-pruned weight distribution, so the grid is
fitted to the sparse model; what the ordering cannot do is let the pruning decision see the grid.

Recovery may run on GPU. Deployment measurements are CPU-only.

**Its ``compression.recovery.max_steps`` is the budget the joint arm must match.** If the two differ,
any joint gain computed from the pair is confounded with extra training.

Examples:
    python scripts/run_sequential.py --config configs/experiments/pilot.yaml --dry-run
    python scripts/run_sequential.py --config configs/experiments/pilot.yaml
    python scripts/run_sequential.py --config configs/experiments/main_scale_sweep.yaml \\
        --override compression.pruning.sparsity=0.7 --override compression.quantisation.bits=4
"""

from __future__ import annotations

import argparse

from scale_aware_compression.cli import run_arm
from scale_aware_compression.config import ConfigError
from scale_aware_compression.constants import CompressionMethod
from scale_aware_compression.logging_utils import configure_logging, get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config value, e.g. --override compression.recovery.max_steps=1000. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the stage plan without running anything",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the sequential arm.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)
    try:
        return run_arm(
            CompressionMethod.SEQUENTIAL,
            config_path=arguments.config,
            overrides=arguments.override,
            dry_run=arguments.dry_run,
            log_level=arguments.log_level,
        )
    except ConfigError as error:
        configure_logging(arguments.log_level or "INFO")
        LOGGER.error("Invalid configuration: %s", error)
        return 2
    except NotImplementedError as error:
        configure_logging(arguments.log_level or "INFO")
        LOGGER.error("Not implemented yet: %s", error)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
