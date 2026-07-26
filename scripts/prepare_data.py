#!/usr/bin/env python
"""Build and cache the evaluation and calibration splits.

Materialising both once means every arm and every seed provably reads the same bytes, rather than
relying on the selection being reproducible. The printed fingerprints are what a later comparison
checks against.

Examples:
    python scripts/prepare_data.py --config configs/experiments/pilot.yaml --dry-run
    python scripts/prepare_data.py --config configs/experiments/main_scale_sweep.yaml
    python scripts/prepare_data.py --config configs/experiments/pilot.yaml --calibration-only
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
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --override data.sequence_length=1024. Repeatable.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--evaluation-only", action="store_true", help="Prepare only the evaluation split"
    )
    group.add_argument(
        "--calibration-only", action="store_true", help="Prepare only the calibration set"
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even when a matching cache already exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved data plan without downloading or tokenising anything",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare the configured datasets.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the operation is not implemented.
    """
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config, arguments.override)
    except ConfigError as error:
        configure_logging(arguments.log_level or "INFO")
        LOGGER.error("Invalid configuration: %s", error)
        return 2

    configure_logging(arguments.log_level or config.runtime.log_level)
    data = config.data

    plan = {
        "dataset": data.dataset,
        "subset": data.subset,
        "train_split": data.train_split,
        "eval_split": data.eval_split,
        "text_column": data.text_column,
        "sequence_length": data.sequence_length,
        "max_eval_samples": data.max_eval_samples,
        "calibration_split": data.calibration_split,
        "calibration_samples": data.calibration_samples,
        "calibration_seed": data.calibration_seed,
        "tokenizer": config.model.name,
        "cache_dir": str(data.cache_dir) if data.cache_dir else "(default)",
    }
    log_key_values(LOGGER, "Data plan", plan)

    if data.calibration_split == data.eval_split:
        LOGGER.warning(
            "data.calibration_split == data.eval_split (%s). Calibration must not overlap the "
            "evaluation set, or quantisation scales are fitted on the test data.",
            data.eval_split,
        )

    if arguments.dry_run:
        LOGGER.info("Dry run: nothing was downloaded or tokenised")
        return 0

    from scale_aware_compression.data.calibration import cache_calibration_set
    from scale_aware_compression.data.preprocessing import prepare_dataset
    from scale_aware_compression.models.loader import load_tokenizer

    tokenizer = load_tokenizer(config.model)

    try:
        if not arguments.calibration_only:
            summary = prepare_dataset(data, tokenizer, data.eval_split)
            LOGGER.info("Prepared evaluation split: %s", summary)
        if not arguments.evaluation_only:
            calibration = cache_calibration_set(data, tokenizer)
            LOGGER.info("Prepared calibration set: %s", calibration.to_dict())
    except NotImplementedError as error:
        LOGGER.error("Not implemented yet: %s", error)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
