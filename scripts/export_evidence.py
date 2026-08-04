r"""Export the committed evidence set: enough to recompute every table, and nothing more.

THE PROBLEM THIS SOLVES
-----------------------
`outputs/` is git-ignored, correctly -- it holds checkpoints and transient artefacts. But it also
holds the only copy of every number in `docs/findings_log.md`, so a reviewer with a fresh clone
cannot recompute a single table. The findings log becomes a set of assertions rather than a
derivation.

This writes a small, plain-text, **committed** set that closes that gap:

    results/evidence/cells.csv        one normalised row per completed run record
    results/evidence/joint_gains.csv  per-replicate joint gains against best-of-sequential
    results/evidence/windows.csv      per-evaluation-window NLL and token counts
    results/evidence/MANIFEST.json    sha256 of every source record, plus what produced this

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Checkpoints, datasets, model weights, packed artefacts, logs. Those are large, derivable, or
licence-encumbered. `MANIFEST.json` carries a **hash per source record** instead, so an excluded
artefact is still identifiable: a reviewer can check that the record they have is the record these
tables came from.

WHY WINDOWS.CSV IS WORTH ITS SIZE
---------------------------------
The paired block bootstrap (A1 §5.1) resamples **whole evaluation windows** using the same indices
for both arms. Without per-window NLL there is no way to reproduce an interval -- only to take the
reported one on trust. That is the single most expensive thing here and the least substitutable.

    python scripts/export_evidence.py
    python scripts/export_evidence.py --check    # verify the committed set is current
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scale_aware_compression.logging_utils import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

EVIDENCE_DIR = Path("results/evidence")

CELL_COLUMNS = (
    # identity and provenance -- everything needed to say WHICH run this was
    "experiment_id",
    "model_name",
    "compression_method",
    "budget_label",
    "calibration_replicate",
    "seed",
    "sparsity",
    "quantisation_bits",
    "status",
    "git_commit",
    "model_revision",
    "method_version",
    "host",
    # the measurement
    "perplexity",
    "perplexity_retention",
    "total_nll",
    "total_tokens",
    "num_sequences",
    "sequence_length",
    "evaluation_device",
    "eval_split",
    "dataset_fingerprint",
    "calibration_fingerprint",
    "measured_sparsity",
    "targeted_parameters",
)


def _get(mapping: dict, *path, default=None):
    """Read a nested key path, returning ``default`` at the first miss."""
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _cell_row(record: dict) -> dict:
    """Flatten one run record into the normalised evidence row."""
    quality = record.get("quality") or {}
    perplexity = quality.get("perplexity") or {}
    retention = quality.get("retention") or {}
    compression = record.get("compression") or {}
    statistics = compression.get("statistics") or {}
    hardware = record.get("hardware") or {}

    from scale_aware_compression.hardware import host_key

    return {
        "experiment_id": record.get("experiment_id"),
        "model_name": record.get("model_name"),
        "compression_method": record.get("compression_method"),
        "budget_label": record.get("budget_label"),
        "calibration_replicate": _get(record, "config", "data", "calibration_replicate"),
        "seed": record.get("seed"),
        "sparsity": record.get("sparsity"),
        "quantisation_bits": record.get("quantisation_bits"),
        "status": record.get("status"),
        "git_commit": record.get("git_commit"),
        "model_revision": _get(record, "config", "model", "revision"),
        "method_version": record.get("method_version"),
        "host": host_key(hardware) if hardware else "unknown",
        "perplexity": perplexity.get("perplexity"),
        "perplexity_retention": retention.get("perplexity_retention"),
        "total_nll": perplexity.get("total_nll"),
        "total_tokens": perplexity.get("total_tokens"),
        "num_sequences": perplexity.get("num_sequences"),
        "sequence_length": perplexity.get("sequence_length"),
        "evaluation_device": perplexity.get("evaluation_device"),
        "eval_split": _get(record, "config", "data", "eval_split"),
        "dataset_fingerprint": perplexity.get("dataset_fingerprint"),
        "calibration_fingerprint": compression.get("calibration_fingerprint"),
        "measured_sparsity": statistics.get("realised_sparsity"),
        "targeted_parameters": statistics.get("targeted_parameters"),
    }


def _joint_gain_rows(rows: list[dict]) -> list[dict]:
    """Compute per-replicate joint gains against **best-of-sequential**, as §6.1 requires.

    Emitted as its own table rather than left to the reader, because the best-of step is exactly
    where B-30 went wrong: measuring against P→Q alone flattered joint. Making the chosen order an
    explicit column means a reader can see *which* baseline each gain used.
    """
    by_cell: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row["status"] != "success" or row["perplexity_retention"] is None:
            continue
        method = row["compression_method"]
        if method not in {"sequential", "sequential_qp", "joint"}:
            continue
        key = (row["model_name"], row["budget_label"], row["calibration_replicate"])
        by_cell.setdefault(key, {})[method] = row

    gains: list[dict] = []
    for (model, budget, replicate), arms in sorted(by_cell.items(), key=repr):
        if "joint" not in arms:
            continue
        available = {k: v["perplexity_retention"] for k, v in arms.items() if k != "joint"}
        if not available:
            continue
        best_order = max(available, key=lambda k: available[k])
        best = available[best_order]
        joint = arms["joint"]["perplexity_retention"]
        gains.append(
            {
                "model_name": model,
                "budget_label": budget,
                "calibration_replicate": replicate,
                "sequential_pq_retention": available.get("sequential"),
                "sequential_qp_retention": available.get("sequential_qp"),
                "best_of_order": best_order,
                "best_of_retention": best,
                "joint_retention": joint,
                "joint_gain_pp": joint - best,
                "orders_available": len(available),
            }
        )
    return gains


def _window_rows(records: list[tuple[str, dict]]) -> list[dict]:
    """Per-window NLL and token counts, long-form.

    Long form rather than one row per record: a 493-column CSV is unreadable and brittle, and the
    long form lets a reader filter to the cells a given bootstrap used.
    """
    out: list[dict] = []
    for name, record in records:
        perplexity = (record.get("quality") or {}).get("perplexity") or {}
        nll = perplexity.get("window_nll")
        tokens = perplexity.get("window_tokens")
        if not nll or not tokens or len(nll) != len(tokens):
            continue
        for index, (value, count) in enumerate(zip(nll, tokens, strict=True)):
            out.append(
                {
                    "experiment_id": record.get("experiment_id", name),
                    "window_index": index,
                    "window_nll": value,
                    "window_tokens": count,
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...] | list[str]) -> None:
    """Write rows deterministically, so re-running produces a byte-identical file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" and \n: the repository pins LF, and csv would otherwise emit \r\n on Windows and
    # produce a diff on every export from a different platform.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metrics", default="outputs/metrics", help="Directory of JSON run records"
    )
    parser.add_argument("--output", type=Path, default=EVIDENCE_DIR, help="Where to write")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed set matches the records without rewriting it. Exit 1 on drift.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Export or verify the committed evidence set.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if no records were found or ``--check`` found drift.
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(arguments.log_level)

    metrics = Path(arguments.metrics)
    if not metrics.is_dir():
        LOGGER.error("No metrics directory at %s", metrics)
        return 1

    records: list[tuple[str, dict]] = []
    digests: dict[str, str] = {}
    for path in sorted(metrics.glob("*.json")):
        raw = path.read_bytes()
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            LOGGER.warning("Skipping %s: %s", path.name, error)
            continue
        records.append((path.name, record))
        digests[path.name] = hashlib.sha256(raw).hexdigest()

    if not records:
        LOGGER.error("No run records found in %s", metrics)
        return 1

    cells = [_cell_row(record) for _, record in records]
    cells.sort(key=lambda row: str(row["experiment_id"]))
    gains = _joint_gain_rows(cells)
    windows = _window_rows(records)

    from scale_aware_compression.experiments.runner import get_git_commit

    manifest = {
        "schema": "evidence/1",
        "purpose": (
            "Enough committed, plain-text evidence to recompute every table in "
            "docs/findings_log.md from a fresh clone. outputs/ is git-ignored and holds the only "
            "copy of the source records, so without this the findings log is assertions rather "
            "than a derivation."
        ),
        "generated_by": "scripts/export_evidence.py",
        "generated_at_commit": get_git_commit(),
        "source_records": len(records),
        "cells": len(cells),
        "joint_gain_rows": len(gains),
        "window_rows": len(windows),
        "excluded_and_why": {
            "checkpoints, packed artefacts, model weights": "large and derivable from the config",
            "datasets": "licence and size; the dataset fingerprint in cells.csv identifies them",
            "logs": "transient, and their content is in the findings log",
        },
        "record_sha256": digests,
    }

    targets = {
        arguments.output / "cells.csv": (cells, CELL_COLUMNS),
        arguments.output / "joint_gains.csv": (gains, list(gains[0]) if gains else []),
        arguments.output / "windows.csv": (
            windows,
            ["experiment_id", "window_index", "window_nll", "window_tokens"],
        ),
    }

    if arguments.check:
        import io

        drift = []
        for path, (rows, columns) in targets.items():
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            if not path.exists() or path.read_text(encoding="utf-8") != buffer.getvalue():
                drift.append(path)
        if drift:
            LOGGER.error(
                "The committed evidence set is stale: %s. Re-run without --check.",
                ", ".join(str(p) for p in drift),
            )
            return 1
        LOGGER.info("Committed evidence set is current (%d records).", len(records))
        return 0

    for path, (rows, columns) in targets.items():
        _write_csv(path, rows, columns)
        LOGGER.info("Wrote %d row(s) to %s", len(rows), path)

    manifest_path = arguments.output / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote %s", manifest_path)

    total = sum(path.stat().st_size for path in [*targets, manifest_path])
    print(f"\n  evidence set: {total / 1024:.0f} KiB across {len(targets) + 1} files")
    print(f"  {len(cells)} cells · {len(gains)} joint-gain rows · {len(windows)} window rows")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
