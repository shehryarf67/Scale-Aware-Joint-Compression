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
import dataclasses

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
    parser.add_argument(
        "--isolate-cells",
        action="store_true",
        help=(
            "Run every cell in its own child process, so memory is released at the cell boundary "
            "by construction. Fixes B-48: the runner accumulates ~4 GiB of commit per 1B "
            "compression cell and never returns it, which exhausts the commit limit on a long grid"
        ),
    )
    parser.add_argument(
        "--only-cell",
        metavar="EXPERIMENT_ID",
        help=(
            "Run exactly this one cell and exit. Used by --isolate-cells to drive the children; "
            "also useful for re-running a single failed cell by hand"
        ),
    )
    parser.add_argument("--log-level", default=None, help="Override runtime.log_level")
    return parser


def _run_isolated(arguments: argparse.Namespace, executable: list) -> int:
    """Run each cell in its own child process.

    Memory is released at the cell boundary because the process exits -- which is the only reliable
    fix for B-48 short of finding every retention inside the runner. The parent keeps no model
    state, so its own footprint stays flat across an arbitrarily long grid.

    A child that fails is reported and the sweep continues, matching ``sweep.continue_on_error``
    semantics at the process level. Note the same caveat applies as in-process: a failed cell
    silently removes a comparison, so check pairs against the records afterwards, not the count.

    Args:
        arguments: Parsed command line, reused verbatim for each child.
        executable: The cells to run.

    Returns:
        0 when every child succeeded, 1 otherwise.
    """
    import subprocess
    import sys

    base = [sys.executable, "-u", __file__, "--config", arguments.config]
    for override in arguments.override:
        base += ["--override", override]
    if arguments.models:
        base += ["--models", *arguments.models]
    if arguments.methods:
        base += ["--methods", *arguments.methods]
    if arguments.log_level:
        base += ["--log-level", arguments.log_level]

    failed: list[str] = []
    for index, cell in enumerate(executable, start=1):
        LOGGER.info("[%d/%d] isolated child for %s", index, len(executable), cell.experiment_id)
        completed = subprocess.run([*base, "--only-cell", cell.experiment_id], check=False)
        if completed.returncode != 0:
            LOGGER.error(
                "Child for %s exited %d, continuing", cell.experiment_id, completed.returncode
            )
            failed.append(cell.experiment_id)

    if failed:
        LOGGER.warning("%d isolated cell(s) failed: %s", len(failed), ", ".join(failed))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the sweep.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a configuration error, 3 when the pipeline is not implemented yet.
    """
    # The OpenMP deadlock mitigation lives in compress_layer, scoped to the solve, rather than
    # process-wide here: pinning inter-op threads at an entry point is irreversible for the rest of
    # the process and would silently override benchmark.interop_threads.
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
        executable_cells,
        find_comparison_pairs,
        run_sweep,
    )

    try:
        plan = build_sweep_plan(config)
    except (ConfigError, KeyError) as error:
        LOGGER.error("Could not build the sweep plan: %s", error)
        return 2

    log_key_values(LOGGER, "Sweep plan", plan.summary())
    executable = executable_cells(plan)
    LOGGER.info("%d logical grid slot(s), %d executable cell(s)", len(plan.cells), len(executable))
    pairs = find_comparison_pairs(plan)
    LOGGER.info("%d joint/sequential pair(s) can yield a joint gain", len(pairs))

    if arguments.plan_only or arguments.dry_run:
        for cell in executable:
            LOGGER.info("  %s", cell.experiment_id)
        LOGGER.info("Plan only: nothing was executed")
        return 0

    if arguments.isolate_cells and arguments.only_cell:
        LOGGER.error("--isolate-cells and --only-cell are mutually exclusive")
        return 2

    if not pairs:
        LOGGER.warning(
            "The plan contains no matched joint/sequential pair, so it cannot produce a joint "
            "gain. Include both 'sequential' and 'joint' in sweep.methods."
        )

    if arguments.isolate_cells:
        return _run_isolated(arguments, executable)

    if arguments.only_cell:
        wanted = [cell for cell in executable if cell.experiment_id == arguments.only_cell]
        if not wanted:
            LOGGER.error(
                "--only-cell %s is not in this plan. Run --plan-only to list the cell ids.",
                arguments.only_cell,
            )
            return 2
        # Narrow the plan rather than build a new one, so the grid metadata (models, methods,
        # budgets, seeds) survives -- `_assert_grid_is_fair` reads it, and a child that dropped it
        # would skip the fairness check the parent passed.
        plan = dataclasses.replace(plan, cells=wanted)
        LOGGER.info("Running one cell only: %s", arguments.only_cell)
        # A one-cell plan has no counterpart by construction, so run_sweep's incomplete-pair
        # warning will fire and mean nothing here. Say so, rather than let it read as a defect.
        LOGGER.info(
            "Single-cell mode: the 'no joint/sequential counterpart' warning below is expected "
            "and does not indicate a missing record. Check pairs against the full plan instead."
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
