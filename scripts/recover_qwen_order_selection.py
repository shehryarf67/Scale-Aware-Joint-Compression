r"""Recover the Qwen order-selection evidence that B-51 destroyed, from git history.

WHY THIS EXISTS
---------------
[B-51](../docs/findings_log.md#4-bugs-found-that-would-have-invalidated-results): a record id
encodes the cell but not the evaluation split, so the Qwen test grid overwrote the **validation**
records of the same cells. Four were destroyed on disk -- `sequential` (P->Q) and `joint` at both
budgets -- and only the two `sequential_qp` records and a smoke copy of the dense baseline survive
under `outputs/metrics`.

Those four are the evidence for [F-40](../docs/findings_log.md#f-40), the sequential-order freeze.
Without them the freeze rests on a claim no reader can recompute.

They are not lost. The evidence set committed at **2832914** -- the F-40 commit, made *before* the
test grid ran -- contains every one of them, with a sha256 per source record. This script extracts
them into a dedicated, committed artefact so the order-selection evidence stands on its own rather
than depending on a reader knowing which commit to look in.

WHAT IT DOES NOT DO
-------------------
It does not re-run order selection. The test results are known now, and re-running a selection step
after seeing the outcome it feeds is the definition of a post-hoc choice -- it would invalidate the
freeze it is meant to document. Recovery from history is the only legitimate route.

    python scripts/recover_qwen_order_selection.py
    python scripts/recover_qwen_order_selection.py --check   # verify the committed artefact
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "2832914"
"""The F-40 commit. Made before the test grid ran, so its evidence set predates the overwrite."""

MODEL = "qwen2.5-0.5b"
DESTINATION = REPOSITORY_ROOT / "results" / "evidence" / "qwen_order_selection.csv"
PROVENANCE = REPOSITORY_ROOT / "results" / "evidence" / "qwen_order_selection_provenance.json"


def _git_show(path: str, commit: str = SOURCE_COMMIT) -> str:
    """Read a file as it existed at a commit.

    Args:
        path: Repository-relative path.
        commit: The commit to read from.

    Returns:
        File contents.

    Raises:
        RuntimeError: If git cannot produce the file.
    """
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read {path} at {commit}: {completed.stderr.strip()}")
    return completed.stdout


def recover() -> tuple[list[dict[str, str]], dict[str, str]]:
    """Extract the Qwen validation rows and their recorded hashes from the source commit.

    Returns:
        ``(rows, hashes)``.
    """
    cells = _git_show("results/evidence/cells.csv")
    rows = [
        row
        for row in csv.DictReader(io.StringIO(cells))
        if row.get("model_name") == MODEL and row.get("eval_split") == "validation"
    ]
    manifest = json.loads(_git_show("results/evidence/MANIFEST.json"))
    hashes = {
        name: digest
        for name, digest in (manifest.get("record_sha256") or {}).items()
        if MODEL in name or "qwen" in name.lower()
    }
    return rows, hashes


def main() -> int:
    """Write or verify the recovered artefact.

    Returns:
        0 on success, 1 when ``--check`` finds drift.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed artefact matches what history holds, without rewriting it",
    )
    arguments = parser.parse_args()

    rows, hashes = recover()
    if not rows:
        print(f"no {MODEL} validation rows found at {SOURCE_COMMIT}", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: (row["budget_label"], row["compression_method"]))
    columns = list(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue()

    surviving = sorted(
        path.name
        for path in (REPOSITORY_ROOT / "outputs" / "metrics").glob("*qwen*.json")
        if json.loads(path.read_text(encoding="utf-8"))
        .get("config", {})
        .get("data", {})
        .get("eval_split")
        == "validation"
    )
    provenance = {
        "purpose": (
            "Order-selection evidence for F-40, recovered from git history after B-51 overwrote "
            "the validation records with their test-split counterparts. Not a re-run: re-running "
            "a selection step after seeing the results it feeds would be post-hoc."
        ),
        "source_commit": SOURCE_COMMIT,
        "source_files": ["results/evidence/cells.csv", "results/evidence/MANIFEST.json"],
        "rows": len(rows),
        "csv_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "record_sha256_at_source": hashes,
        "still_on_disk": surviving,
        "destroyed_by_b51": sorted(
            set(hashes) - set(surviving) - {"qwen2.5-0.5b_dense_moderate_s00_b32_rep0.json"}
        ),
        "note": (
            "qwen2.5-0.5b_dense_moderate_s00_b32_rep0.json was also overwritten, but an identical "
            "run survives under the smoke experiment id "
            "qwen_smoke_dense__qwen2.5-0.5b_dense_moderate_s00_b32_seed1234.json (ppl 17.7758)."
        ),
    }
    provenance_payload = json.dumps(provenance, indent=2) + "\n"

    if arguments.check:
        if not DESTINATION.exists() or not PROVENANCE.exists():
            print("recovered artefact is missing; run without --check", file=sys.stderr)
            return 1
        if DESTINATION.read_text(encoding="utf-8") != payload:
            print(f"{DESTINATION} differs from what history holds", file=sys.stderr)
            return 1
        recorded = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        if recorded.get("csv_sha256") != provenance["csv_sha256"]:
            print("provenance hash does not match the recovered CSV", file=sys.stderr)
            return 1
        print(f"recovered artefact is current ({len(rows)} rows from {SOURCE_COMMIT})")
        return 0

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(payload, encoding="utf-8", newline="")
    PROVENANCE.write_text(provenance_payload, encoding="utf-8", newline="")
    print(f"wrote {DESTINATION} ({len(rows)} rows from {SOURCE_COMMIT})")
    print(f"wrote {PROVENANCE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
