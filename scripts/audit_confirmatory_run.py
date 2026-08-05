#!/usr/bin/env python
"""Fail closed unless the frozen confirmatory execution is complete and internally consistent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scale_aware_compression.config import load_config  # noqa: E402
from scale_aware_compression.constants import CompressionMethod  # noqa: E402
from scale_aware_compression.experiments.runner import (  # noqa: E402
    ExperimentError,
    ExperimentTracker,
)
from scale_aware_compression.experiments.scale_sweep import (  # noqa: E402
    build_cell_config,
    build_sweep_plan,
    executable_cells,
    find_comparison_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/main_scale_sweep.yaml")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/evidence/confirmatory_manifest.json")
    )
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics"))
    return parser


def _manifest_entry(cell: Any) -> dict[str, Any]:
    """Return the cell fields whose exact values are frozen in the manifest."""
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
        from scale_aware_compression.protocol import FROZEN_ORDER_EVIDENCE

        entry["frozen_order_evidence"] = FROZEN_ORDER_EVIDENCE.get(
            (cell.model_name, cell.budget_label), "NOT FROZEN"
        )
    return entry


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object, raising a useful audit error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExperimentError(f"{path} does not contain a JSON object")
    return value


def _artifact_sha256(checkpoint: Path) -> str:
    """Compute the same deterministic directory digest recorded by the runner."""
    digest = hashlib.sha256()
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        relative = path.relative_to(checkpoint).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def audit(config_path: Path, manifest_path: Path, metrics_path: Path) -> list[str]:
    """Return every completeness failure; an empty list means analysis may begin."""
    problems: list[str] = []
    try:
        manifest = _read_json(manifest_path)
    except ExperimentError as error:
        return [str(error)]

    config = load_config(config_path)
    plan = build_sweep_plan(config)
    cells = executable_cells(plan)
    expected_ids = {cell.experiment_id for cell in cells}

    if manifest.get("valid_for_freeze") is not True or manifest.get("checks_failed"):
        problems.append("manifest is not valid_for_freeze or records failed freeze checks")
    if manifest.get("schema") != "confirmatory_manifest/2":
        problems.append(f"manifest schema is {manifest.get('schema')!r}, expected version 2")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if manifest.get("config_sha256") != digest:
        problems.append("config SHA-256 differs from the frozen manifest")
    if config.data.eval_split != "test":
        problems.append(f"evaluation split is {config.data.eval_split!r}, expected 'test'")
    if config.evaluation.device.value != "cpu" or config.benchmark.device.value != "cpu":
        problems.append("quality evaluation and deployment benchmark must both use CPU")
    if not config.sweep.continue_on_error:
        problems.append("continue_on_error is false, contrary to Amendment A2")

    expected_manifest_cells = [_manifest_entry(cell) for cell in cells]
    if manifest.get("cells") != expected_manifest_cells:
        problems.append("manifest cells do not exactly match executable_cells(plan)")
    logical_count = len(plan.cells)
    if manifest.get("logical_grid_cell_count") != logical_count:
        problems.append("manifest logical_grid_cell_count does not match the plan")
    if manifest.get("executable_cell_count") != len(cells):
        problems.append("manifest executable_cell_count does not match the runner")
    if manifest.get("deduplicated_dense_slots") != logical_count - len(cells):
        problems.append("manifest dense-deduplication count is inconsistent")

    tracker = ExperimentTracker(metrics_path)
    seen_internal_ids: Counter[str] = Counter()
    for path in sorted(metrics_path.glob("*.json")) if metrics_path.is_dir() else []:
        try:
            record = _read_json(path)
        except ExperimentError as error:
            # Unrelated historical files are still unsafe in the confirmatory metrics directory.
            problems.append(str(error))
            continue
        internal_id = record.get("experiment_id")
        if isinstance(internal_id, str):
            seen_internal_ids[internal_id] += 1
        if path.stem in expected_ids and internal_id != path.stem:
            problems.append(f"{path} contains experiment_id {internal_id!r}")
    duplicates = sorted(key for key, count in seen_internal_ids.items() if count > 1)
    if duplicates:
        problems.append(f"duplicate JSON records for experiment IDs: {duplicates}")

    if tracker.csv_path.is_file():
        try:
            with tracker.csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_ids = [row.get("experiment_id", "") for row in csv.DictReader(handle)]
            duplicate_csv = sorted(
                key for key, count in Counter(csv_ids).items() if key and count > 1
            )
            if duplicate_csv:
                problems.append(f"duplicate CSV rows for experiment IDs: {duplicate_csv}")
        except (OSError, csv.Error) as error:
            problems.append(f"cannot audit {tracker.csv_path}: {error}")

    valid_ids: set[str] = set()
    for cell in cells:
        cell_config = build_cell_config(config, cell)
        path = tracker.record_path(cell.experiment_id)
        if not path.is_file():
            problems.append(f"missing record: {cell.experiment_id}")
            continue
        try:
            record = _read_json(path)
        except ExperimentError as error:
            problems.append(str(error))
            continue
        status = record.get("status")
        if status != "success":
            problems.append(f"{cell.experiment_id}: status is {status!r}, not 'success'")
            continue
        if not tracker.exists_valid(cell.experiment_id, cell_config):
            problems.append(f"{cell.experiment_id}: record is stale or mismatched")
            continue
        if record.get("config") != cell_config.to_dict():
            problems.append(f"{cell.experiment_id}: resolved config differs from the frozen cell")
            continue
        if cell.method is not CompressionMethod.DENSE:
            checkpoint = record.get("checkpoint") or {}
            checkpoint_path = record.get("checkpoint_path")
            if checkpoint.get("reload_verified") is not True:
                problems.append(f"{cell.experiment_id}: checkpoint reload was not verified")
                continue
            if checkpoint.get("artifact_retained") is not True or not checkpoint_path:
                problems.append(
                    f"{cell.experiment_id}: checkpoint was not retained for final audit"
                )
                continue
            artifact = Path(checkpoint_path)
            if not artifact.is_dir():
                problems.append(f"{cell.experiment_id}: checkpoint path does not exist: {artifact}")
                continue
            recorded_hash = checkpoint.get("artifact_sha256")
            if not recorded_hash or _artifact_sha256(artifact) != recorded_hash:
                problems.append(f"{cell.experiment_id}: checkpoint SHA-256 does not match")
                continue
        valid_ids.add(cell.experiment_id)

    pairs = find_comparison_pairs(plan)
    complete_pairs = sum(
        left.experiment_id in valid_ids and right.experiment_id in valid_ids
        for left, right in pairs
    )
    if complete_pairs != len(pairs):
        problems.append(f"only {complete_pairs}/{len(pairs)} sequential-joint pairs are complete")

    if len(cells) != 171 or logical_count != 210 or len(pairs) != 42:
        problems.append(
            f"unexpected frozen scope: {logical_count} logical, {len(cells)} executable, "
            f"{len(pairs)} pairs (expected 210/171/42)"
        )

    if not problems:
        print(f"{len(valid_ids)} / {len(cells)} executable cells valid")
        print(f"{complete_pairs} / {len(pairs)} sequential-joint pairs complete")
        print("0 failures\n0 stale records\n0 missing records")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the audit and return nonzero unless the run is complete."""
    arguments = build_parser().parse_args(argv)
    problems = audit(arguments.config, arguments.manifest, arguments.metrics)
    if problems:
        print(f"AUDIT FAILED: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("AUDIT PASSED: confirmatory analysis may begin")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
