#!/usr/bin/env python
"""Prefetch model checkpoints into the local Hugging Face cache.

Run this before a sweep. Downloading part-way through a long run wastes the time already spent
if the network fails, and a sweep that stalls overnight on a download is a wasted night.

This is the only script that touches the network, and only when invoked explicitly.

Examples:
    python scripts/download_models.py --list
    python scripts/download_models.py --models pythia-160m pythia-410m
    python scripts/download_models.py --sweep
    python scripts/download_models.py --all --dry-run
"""

from __future__ import annotations

import argparse
import sys

from scale_aware_compression.config import ModelConfig
from scale_aware_compression.logging_utils import configure_logging, get_logger
from scale_aware_compression.models.registry import (
    UnknownModelError,
    get_model_spec,
    list_models,
    registry_table,
    scale_sweep_models,
    validation_models,
)
from scale_aware_compression.visualisation.tables import rows_to_markdown

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="Registry short names to fetch, e.g. pythia-160m pythia-1b",
    )
    selection.add_argument(
        "--sweep",
        action="store_true",
        help="Fetch the Pythia scale sweep (add --include-optional for pythia-1.4b)",
    )
    selection.add_argument(
        "--validation", action="store_true", help="Fetch the external validation model only"
    )
    selection.add_argument("--all", action="store_true", help="Fetch every registered model")
    selection.add_argument(
        "--list", action="store_true", help="Print the registry and exit without downloading"
    )

    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="With --sweep, also fetch the optional pythia-1.4b",
    )
    parser.add_argument(
        "--revision",
        help="Pin a Hub revision for every requested model. Normally set per model in configs/models/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without downloading anything",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def resolve_selection(arguments: argparse.Namespace) -> list[str]:
    """Turn the mutually exclusive selection flags into a list of model names.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        Registry short names, ordered by size.

    Raises:
        UnknownModelError: If an explicitly named model is not registered.
    """
    if arguments.models:
        return [get_model_spec(name).short_name for name in arguments.models]
    if arguments.sweep:
        return scale_sweep_models(include_optional=arguments.include_optional)
    if arguments.validation:
        return validation_models()
    if arguments.all:
        return list_models()
    # No selection flag: the sweep is the sensible default, since it is what most runs need.
    LOGGER.info("No selection given; defaulting to the Pythia scale sweep")
    return scale_sweep_models(include_optional=arguments.include_optional)


def main(argv: list[str] | None = None) -> int:
    """Fetch the requested checkpoints.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if any download failed, 2 on a usage error.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)

    if arguments.list:
        sys.stdout.write(rows_to_markdown(registry_table()) + "\n")
        return 0

    try:
        names = resolve_selection(arguments)
    except UnknownModelError as error:
        LOGGER.error("%s", error)
        return 2

    LOGGER.info("Selected %d model(s): %s", len(names), ", ".join(names))
    if arguments.dry_run:
        for name in names:
            spec = get_model_spec(name)
            LOGGER.info(
                "  would fetch %-14s -> %-26s (%s, %s)",
                spec.short_name,
                spec.hf_id,
                spec.size_label,
                spec.role,
            )
        LOGGER.info("Dry run: nothing was downloaded")
        return 0

    from scale_aware_compression.models.loader import ModelLoadError, prefetch

    failures: list[str] = []
    for name in names:
        config = ModelConfig(name=name, revision=arguments.revision)
        try:
            spec = prefetch(config)
        except ModelLoadError as error:
            LOGGER.error("Failed to fetch %s: %s", name, error)
            failures.append(name)
        else:
            LOGGER.info("Fetched %s (%s)", spec.short_name, spec.hf_id)

    if failures:
        LOGGER.error(
            "%d of %d downloads failed: %s", len(failures), len(names), ", ".join(failures)
        )
        return 1
    LOGGER.info("All %d model(s) present in the local cache", len(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
