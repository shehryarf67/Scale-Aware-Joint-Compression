#!/usr/bin/env python
r"""Run the quantisation-only arm for one model.

Pipeline: dense -> observers -> calibrate -> quantise -> convert -> evaluate (CPU) -> benchmark (CPU).

Post-training, weight-only quantisation with no recovery, which is how it is normally deployed.
Isolates the effect of reduced precision. Calibration may run on GPU; the deployment measurements
are CPU-only.

Before believing the resulting record, check ``is_converted`` and ``storage_efficiency``. A model
that was fake-quantised but never converted is numerically quantised and still FP32 on disk, which
produces correct-looking quality with a meaningless size and latency.

Examples:
    python scripts/run_quantisation.py --config configs/experiments/pilot.yaml --dry-run
    python scripts/run_quantisation.py --config configs/experiments/pilot.yaml
    python scripts/run_quantisation.py --config configs/experiments/pilot.yaml \\
        --override compression.quantisation.bits=4
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
            "Override a config value, e.g. --override compression.quantisation.bits=4. Repeatable."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the plan without quantising anything",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the quantisation-only arm.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)
    try:
        return run_arm(
            CompressionMethod.QUANTISATION,
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
