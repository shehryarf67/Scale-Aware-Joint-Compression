r"""Resolve and validate the confirmatory configuration into one auditable manifest (A1 step 9).

Step 10 is a multi-day run, runs **once**, and forbids methodological tuning afterwards. Everything it
depends on therefore has to be pinned, resolved and checked *before* it starts -- not described in
prose across six documents and reassembled from memory later.

This produces `results/evidence/confirmatory_manifest.json`: every commit, revision, config, cell,
replicate, order, device and exclusion rule, fully resolved, with the checks that had to pass.

WHY IT REFUSES ON A DIRTY TREE
------------------------------
A freeze recorded at a ``-dirty`` commit is not a freeze: the code that produced it cannot be
recovered. This project has exactly one unusable set of numbers, and its cause was a working tree
22 commits behind `main` recorded as `aec5099-dirty`. So this refuses rather than warns.

WHAT IT VALIDATES, EACH BECAUSE SOMETHING WENT WRONG THERE BEFORE
-----------------------------------------------------------------
* **clean tree at a real commit** -- the `aec5099-dirty` lesson;
* **test split**, not validation -- A1 §5.2; validation is a declared selection surface;
* **CPU evaluation and CPU benchmark** -- §4.6, and the reported numbers must come from CPU;
* **frozen sequential order resolved per cell** -- §6.1 requires best-of, and one frozen cell is
  Q→P (B-42). A cell whose order is unfrozen aborts the manifest;
* **R per model**, with whether significance is reachable at that R -- A1 §5.1, and R=5 can never
  reach p < 0.05 whatever the effect size;
* **every model revision pinned to a 40-character SHA** -- §2.7, and B-13 was a sweep inheriting one
  model's revision for every cell;
* **budgets match the frozen pair** -- §6.3 forbids revisiting them once results exist.

    python scripts/build_confirmatory_manifest.py
    python scripts/build_confirmatory_manifest.py --allow-dirty   # inspection only, never a freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scale_aware_compression.logging_utils import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/experiments/main_scale_sweep.yaml")
DEFAULT_OUTPUT = Path("results/evidence/confirmatory_manifest.json")

FROZEN_BUDGETS = {
    "moderate": {"sparsity": 0.3, "bits": 8},
    "aggressive": {"sparsity": 0.3, "bits": 4},
}
"""The pair frozen 2026-07-29 and confirmed at all three scales (F-23, F-25, F-32).

Duplicated here on purpose: the config could be edited, and this is the independent statement the
manifest checks it against. A silent budget change is the failure §6.3 exists to prevent.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Build from a dirty tree for INSPECTION only. The manifest is marked invalid and must "
        "never be used to freeze.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve, validate and write the confirmatory manifest.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 when every check passed, 1 otherwise. A non-zero exit means **do not freeze**.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)

    from scale_aware_compression.config import load_config
    from scale_aware_compression.constants import (
        CALIBRATION_REPLICATE_SEEDS,
        METHOD_VERSION,
        RESULT_SCHEMA_VERSION,
        CompressionMethod,
    )
    from scale_aware_compression.experiments.runner import get_git_commit
    from scale_aware_compression.experiments.scale_sweep import (
        build_sweep_plan,
        executable_cells,
    )
    from scale_aware_compression.hardware import get_hardware_info, get_software_versions, host_key
    from scale_aware_compression.metrics.replicates import MIN_R_FOR_SIGNIFICANCE
    from scale_aware_compression.protocol import (
        FROZEN_ORDER_EVIDENCE,
        FROZEN_SEQUENTIAL_ORDER,
    )

    failures: list[str] = []
    warnings: list[str] = []

    commit = get_git_commit()
    dirty = commit is not None and commit.endswith("-dirty")
    if commit is None:
        failures.append("not a git checkout: the confirmatory run must be traceable to a commit")
    elif dirty and not arguments.allow_dirty:
        failures.append(
            f"working tree is dirty ({commit}). A freeze recorded at a -dirty commit cannot be "
            "reproduced; this project's one unusable result set came from exactly that. Commit "
            "first, or pass --allow-dirty for inspection only."
        )

    config = load_config(arguments.config)
    plan = build_sweep_plan(config)

    # --- the checks -------------------------------------------------------------------------
    if config.data.eval_split != "test":
        failures.append(
            f"data.eval_split is {config.data.eval_split!r}, not 'test'. A1 §5.2 reserves test for "
            "confirmation; validation is a declared selection surface and the budgets were chosen "
            "on it."
        )
    if config.evaluation.device.value != "cpu":
        failures.append(
            f"evaluation.device is {config.evaluation.device.value!r}. Reported quality must come "
            "from CPU."
        )
    if config.benchmark.device.value != "cpu":
        failures.append("benchmark.device is not cpu; deployment measurements are CPU-only (§4.6)")
    if not config.sweep.use_frozen_order:
        failures.append(
            "sweep.use_frozen_order is off, so a `sequential` cell means P→Q everywhere -- "
            "including pythia-1b/moderate, where Q→P is frozen and P→Q is the weaker baseline "
            "(B-42)"
        )
    if not config.sweep.continue_on_error:
        failures.append(
            "sweep.continue_on_error is false. Amendment A2 requires a multi-day run to continue "
            "after a failed cell and rely on the strict final audit"
        )

    for label, expected in FROZEN_BUDGETS.items():
        override = (config.sweep.budget_overrides.get(label) or {}).get("compression") or {}
        sparsity = (override.get("pruning") or {}).get("sparsity")
        bits = (override.get("quantisation") or {}).get("bits")
        if sparsity != expected["sparsity"] or bits != expected["bits"]:
            failures.append(
                f"budget {label!r} is {sparsity}/{bits}, not the frozen "
                f"{expected['sparsity']}/{expected['bits']} (§6.3 forbids revisiting these)"
            )
    unexpected = set(config.sweep.budgets) - set(FROZEN_BUDGETS)
    if unexpected:
        failures.append(f"sweep names budgets outside the frozen pair: {sorted(unexpected)}")

    # Replicate counts, and whether significance is even reachable at each.
    replicates: dict[str, dict] = {}
    for model in plan.models:
        count = config.sweep.replicates_by_model.get(model, config.sweep.replicates)
        reachable = count >= MIN_R_FOR_SIGNIFICANCE
        replicates[model] = {
            "R": count,
            "significance_reachable": reachable,
            "seeds": list(CALIBRATION_REPLICATE_SEEDS[:count]),
        }
        if count == 0:
            failures.append(f"{model} has no replicates; A1 §5.1 requires the paired draw axis")
        if not reachable:
            # Not a failure: A1 §6 sets R=5 at 1B deliberately. But it must be stated, because a
            # null there says nothing about nature -- only about the replicate count.
            warnings.append(
                f"{model} runs R={count}, below {MIN_R_FOR_SIGNIFICANCE}: no significance claim is "
                "reachable at any effect size. Report effect size and sign consistency only."
            )

    # Every cell, fully resolved. This is the artefact's point: no cell is left to interpretation.
    logical_grid_cell_count = len(plan.cells)
    executable = executable_cells(plan)
    cells = []
    for cell in executable:
        entry = {
            "experiment_id": cell.experiment_id,
            "model": cell.model_name,
            "method": cell.method.value,
            "budget": cell.budget_label,
            "replicate": cell.replicate,
            "sparsity": cell.sparsity,
            "bits": cell.bits,
        }
        if cell.method in {CompressionMethod.SEQUENTIAL, CompressionMethod.SEQUENTIAL_QP}:
            entry["frozen_order_evidence"] = FROZEN_ORDER_EVIDENCE.get(
                (cell.model_name, cell.budget_label), "NOT FROZEN"
            )
        cells.append(entry)

    revisions: dict[str, str | None] = {}
    for model in plan.models:
        from scale_aware_compression.experiments.scale_sweep import _revision_for

        revision = _revision_for(model, config.to_dict())
        revisions[model] = revision
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            failures.append(
                f"{model} revision {revision!r} is not a 40-character commit SHA (§2.7)"
            )

    config_bytes = arguments.config.read_bytes()

    manifest = {
        "schema": "confirmatory_manifest/2",
        "valid_for_freeze": not failures and not dirty,
        "purpose": (
            "The fully resolved confirmatory configuration for A1 step 10, as operationally "
            "amended by A2. Step 10 runs once and forbids methodological tuning afterwards, so "
            "everything it depends on is "
            "pinned and checked here rather than reassembled later."
        ),
        "git_commit": commit,
        "tree_clean": not dirty,
        "config_path": arguments.config.as_posix(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "method_version": METHOD_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation": {
            "split": config.data.eval_split,
            "device": config.evaluation.device.value,
            "max_samples": config.evaluation.max_samples,
            "sequence_length": config.data.sequence_length,
        },
        "benchmark": {
            "device": config.benchmark.device.value,
            "num_threads": config.benchmark.num_threads,
            "warmup_runs": config.benchmark.warmup_runs,
            "measured_runs": config.benchmark.measured_runs,
        },
        "models": plan.models,
        "model_revisions": revisions,
        "budgets": FROZEN_BUDGETS,
        "replicates": replicates,
        "frozen_sequential_order": {
            f"{model}/{budget}": method.value
            for (model, budget), method in sorted(FROZEN_SEQUENTIAL_ORDER.items())
        },
        # `cell_count` remains as a compatibility alias, but now means executable records. The
        # explicit fields below prevent logical-grid scope from being confused with actual work.
        "cell_count": len(cells),
        "logical_grid_cell_count": logical_grid_cell_count,
        "executable_cell_count": len(cells),
        "deduplicated_dense_slots": logical_grid_cell_count - len(cells),
        "dense_policy": (
            "one dense evaluation per model; dense is independent of budget and calibration draw"
        ),
        "cells": cells,
        "practical_importance_rule": {
            "form": "amended (A1 §5.1, §6.3)",
            "criteria": [
                "perplexity retention improves by >= 1.0 percentage point",
                "sign consistent across every paired calibration replicate, ties excluded (B-40)",
                "R reported per cell, with whether significance was reachable at that R",
            ],
            "withdrawn_clause": (
                "the original 'mean exceeds the seed spread' clause is withdrawn: run seeds are "
                "inert (F-15) so the spread is exactly zero and the gate excluded nothing. "
                "Replacing an unmeasurable criterion with a weaker measurable one is a REDUCTION "
                "in pre-registered strength and the write-up must say so."
            ),
        },
        "exclusion_rules": {
            "latency": (
                "only FP32 runtime representations are timed -- dense and pruning-only. A packed "
                "W4/W8 layer dequantises on every forward, so a timing would measure the unpacking "
                "kernel (decision D1). The absence is recorded per record, not omitted."
            ),
            "downstream": (
                "descriptive secondary endpoint, one calibration draw, no formal joint-superiority "
                "claim. The seed-era downstream importance rule is withdrawn, not amended."
            ),
            "1b_significance": (
                "R=5 at 1B cannot reach p < 0.05 at any effect size; 1B carries effect-size "
                "evidence only."
            ),
            "extended_models": (
                "pythia-1.4b and qwen2.5-0.5b are excluded from this manifest and from the scale "
                "trend. Neither has a frozen sequential order, and §8.2 forbids them consuming the "
                "primary sweep's time."
            ),
        },
        "environment": {
            "host": host_key(),
            "hardware": get_hardware_info(),
            "software": get_software_versions(),
        },
        "checks_failed": failures,
        "warnings": warnings,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\n  manifest: {arguments.output}")
    print(f"  commit           {commit}")
    print(f"  logical slots    {logical_grid_cell_count}")
    print(f"  executable cells {len(cells)}")
    print(f"  split / device   {config.data.eval_split} / {config.evaluation.device.value}")
    summary = ", ".join(f"{model}=R{value['R']}" for model, value in replicates.items())
    print(f"  R per model      {summary}")
    print(f"  frozen orders    {len(FROZEN_SEQUENTIAL_ORDER)} cells")
    for warning in warnings:
        print(f"  [warn] {warning}")
    if failures:
        print(f"\n  [FAIL] {len(failures)} check(s) failed -- DO NOT FREEZE")
        for failure in failures:
            print(f"      - {failure}")
        return 1
    if dirty:
        print("\n  [FAIL] dirty tree -- inspection only, not valid for freeze")
        return 1
    print("\n  [OK] all checks passed; valid for freeze")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
