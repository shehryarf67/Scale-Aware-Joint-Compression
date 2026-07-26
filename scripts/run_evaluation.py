#!/usr/bin/env python
r"""Evaluate quality for a model or a saved checkpoint.

Computes perplexity, dense-versus-compressed prediction agreement, and generation diagnostics.
Final reported numbers must come from CPU; evaluating on GPU during development is fine and only
warns.

Quality retention needs the dense baseline's perplexity. Pass ``--dense-record`` pointing at that
model's dense run, so the reference is *loaded* rather than recomputed — recomputing it risks a
different evaluation window and an incomparable retention figure.

Examples:
    python scripts/run_evaluation.py --config configs/experiments/pilot.yaml --dry-run
    python scripts/run_evaluation.py --config configs/experiments/pilot.yaml
    python scripts/run_evaluation.py --config configs/experiments/pilot.yaml \\
        --checkpoint outputs/pilot/checkpoint \\
        --dense-record outputs/metrics/pythia-160m_dense_pilot_s00_b32_seed1234.json
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
        help="Override a config value, e.g. --override evaluation.max_samples=1024. Repeatable.",
    )
    parser.add_argument(
        "--checkpoint", help="Evaluate a saved checkpoint directory instead of the base model"
    )
    parser.add_argument(
        "--dense-record",
        help="JSON record of this model's dense baseline, used as the retention reference",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["perplexity", "agreement", "generation"],
        help="Override evaluation.metrics",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved evaluation plan without loading a model",
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate quality.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config, arguments.override)
    except ConfigError as error:
        configure_logging(arguments.log_level or "INFO")
        LOGGER.error("Invalid configuration: %s", error)
        return 2

    if arguments.metrics:
        config.evaluation.metrics = list(arguments.metrics)

    configure_logging(arguments.log_level or config.runtime.log_level)

    from scale_aware_compression.evaluation.quality import check_evaluation_device

    check_evaluation_device(config.evaluation)

    log_key_values(
        LOGGER,
        "Evaluation plan",
        {
            "model": config.model.name,
            "checkpoint": arguments.checkpoint or "(base model from config)",
            "method": config.compression.method.value,
            "device": config.evaluation.device.value,
            "metrics": config.evaluation.metrics,
            "batch_size": config.evaluation.batch_size,
            "sequence_length": config.evaluation.sequence_length,
            "stride": config.evaluation.stride or "(non-overlapping)",
            "max_samples": config.evaluation.max_samples,
            "dense_record": arguments.dense_record or "(none -- no retention will be computed)",
        },
    )

    if not arguments.dense_record and config.compression.method.value != "dense":
        LOGGER.warning(
            "No --dense-record given for a compressed model. Without the dense reference this run "
            "has no quality retention and cannot contribute to a joint-gain comparison."
        )

    if arguments.dry_run:
        LOGGER.info("Dry run: no model was loaded and nothing was evaluated")
        return 0

    from scale_aware_compression.evaluation.quality import evaluate_model
    from scale_aware_compression.models.loader import load_model_and_tokenizer

    try:
        loaded = load_model_and_tokenizer(config.model)
        report = evaluate_model(loaded.model, loaded.tokenizer, config)
    except NotImplementedError as error:
        LOGGER.error("Not implemented yet: %s", error)
        return 3

    LOGGER.info("%s", report.summary_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
