#!/usr/bin/env python
r"""Run the joint arm: pruning-aware quantisation.

Pipeline, explicitly:

    dense model
        -> fake quantisation preparation
        -> gradual pruning during optimisation
        -> recovery / joint fine-tuning
        -> final conversion

Fake quantisation is inserted before any weight is pruned, so the pruning criterion ranks weights as
the quantisation grid will actually represent them, and one optimisation run compensates for both
perturbations together. This is the arm under test.

Joint training may run on GPU. Deployment measurements are CPU-only.

**Budget matching is a correctness requirement, not a nicety.** ``compression.joint.joint_max_steps``
must equal the sequential arm's ``compression.recovery.max_steps``. If the joint run is longer, the
measured joint gain includes extra training and the comparison says nothing about pipeline design.
The resulting record carries ``match_sequential_budget`` and the exact optimiser-step count so this
is checkable afterwards.

Examples:
    python scripts/run_joint.py --config configs/experiments/pilot.yaml --dry-run
    python scripts/run_joint.py --config configs/experiments/pilot.yaml
    python scripts/run_joint.py --config configs/experiments/main_scale_sweep.yaml \\
        --override compression.joint.joint_max_steps=500
"""

from __future__ import annotations

import argparse

from scale_aware_compression.cli import run_arm
from scale_aware_compression.config import ConfigError, load_config
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
            "Override a config value, e.g. --override compression.joint.joint_max_steps=500. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--sequential-steps",
        type=int,
        help=(
            "Optimiser steps the sequential arm used, for a budget-match check. Warns loudly if "
            "the joint budget differs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the stage plan without running anything",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def check_budget_match(
    config_path: str, overrides: list[str], sequential_steps: int | None
) -> None:
    """Warn when the joint budget does not match the sequential one.

    Args:
        config_path: Path to the configuration being run.
        overrides: Overrides applied to it.
        sequential_steps: The sequential arm's step count, if supplied.
    """
    config = load_config(config_path, overrides)
    joint = config.compression.joint
    joint_steps = joint.joint_max_steps
    reference = (
        sequential_steps if sequential_steps is not None else config.compression.recovery.max_steps
    )

    if joint_steps is None or reference is None:
        LOGGER.warning(
            "Could not verify budget matching: joint_max_steps=%s, sequential reference=%s. Set "
            "both explicitly before collecting results.",
            joint_steps,
            reference,
        )
        return
    if joint_steps != reference:
        LOGGER.warning(
            "BUDGET MISMATCH: joint arm has %d optimiser steps against the sequential arm's %d "
            "(%.2fx). Any joint gain from this pair is confounded with extra training.",
            joint_steps,
            reference,
            joint_steps / reference,
        )
    else:
        LOGGER.info("Optimisation budgets match at %d optimiser steps", joint_steps)


def main(argv: list[str] | None = None) -> int:
    """Run the joint arm.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)
    try:
        configure_logging(arguments.log_level or "INFO")
        check_budget_match(arguments.config, arguments.override, arguments.sequential_steps)
        return run_arm(
            CompressionMethod.JOINT,
            config_path=arguments.config,
            overrides=arguments.override,
            dry_run=arguments.dry_run,
            log_level=arguments.log_level,
        )
    except ConfigError as error:
        LOGGER.error("Invalid configuration: %s", error)
        return 2
    except NotImplementedError as error:
        LOGGER.error("Not implemented yet: %s", error)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
