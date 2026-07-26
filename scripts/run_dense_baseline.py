#!/usr/bin/env python
r"""Run the dense FP32 baseline for one model.

The reference point every other arm at that model scale is measured against. No compression is
applied; the model is loaded, evaluated on CPU, and benchmarked on CPU.

**Run this before any other arm for a given model.** Quality retention is defined relative to this
run's perplexity, so a compressed run without a matching dense record has no primary score and
cannot contribute to a joint-gain comparison.

Examples:
    python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml --dry-run
    python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml
    python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml \\
        --override model.name=pythia-410m
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
        help="Override a config value, e.g. --override model.name=pythia-1b. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the plan without loading or measuring anything",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dense baseline.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)
    try:
        return run_arm(
            CompressionMethod.DENSE,
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
