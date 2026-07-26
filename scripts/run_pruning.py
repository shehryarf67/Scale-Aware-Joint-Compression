#!/usr/bin/env python
r"""Run the pruning-only arm for one model.

Pipeline: dense -> prune -> recovery fine-tune -> convert -> evaluate (CPU) -> benchmark (CPU).

Isolates the effect of sparsity, as a reference for the sequential and joint arms. Recovery may run
on GPU; the deployment measurements are CPU-only.

Check two things in the resulting record: that measured sparsity matches the target, and whether the
sparsity produced any latency improvement. For unstructured sparsity it usually does not, because
dense CPU GEMM kernels do not skip scattered zeros — try ``--override
compression.pruning.granularity=2:4`` for the pattern that has kernel support.

Examples:
    python scripts/run_pruning.py --config configs/experiments/pilot.yaml --dry-run
    python scripts/run_pruning.py --config configs/experiments/pilot.yaml
    python scripts/run_pruning.py --config configs/experiments/pilot.yaml \\
        --override compression.pruning.sparsity=0.7
    python scripts/run_pruning.py --config configs/experiments/pilot.yaml \\
        --override compression.pruning.granularity=2:4
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
            "Override a config value, e.g. --override compression.pruning.sparsity=0.7. Repeatable."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the plan without pruning anything",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pruning-only arm.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)
    try:
        return run_arm(
            CompressionMethod.PRUNING,
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
