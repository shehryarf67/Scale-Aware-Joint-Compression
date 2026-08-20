r"""Check whether any deployment measurement was taken across a host suspend (B-47).

THE PROBLEM THIS SOLVES
-----------------------
The benchmark host enters **Modern Standby** -- S0 low-power idle -- and a suspended process keeps
its wall clock running. A latency or throughput benchmark that spans one therefore records the
suspended interval inside its own timing.

Quality numbers are immune: suspension is not arithmetic, and a perplexity computed across a
standby is the same perplexity. §4.6 deployment figures are not immune. A median survives one
inflated sample; a p95 or an IQR need not.

Modern Standby is easy to miss. It logs Kernel-Power **506/507**, *not* the classic **42/107** that
a sleep check greps for, so a host can suspend all day while the sleep history reads clean. Both
pairs are checked here.

WHAT IT COMPARES
----------------
Each record's cell window against every suspend window. The cell window is a deliberate
**superset** of the benchmark window -- the benchmark is one stage late in the cell -- so a cell
that clears this check cannot contain a suspend. A cell that does *not* clear it needs its stage
timings read out of the run log before any verdict: an overlap with the cell is not an overlap
with the benchmark, and in the one real case it was the quality stage that absorbed the suspend.

    python scripts/audit_suspend_windows.py
    python scripts/audit_suspend_windows.py --metrics outputs/metrics

Windows-only: it reads the System event log through PowerShell.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SUSPEND_ENTRY = {506, 42}
SUSPEND_EXIT = {507, 107}

_QUERY = (
    "Get-WinEvent -FilterHashtable @{LogName='System'; "
    "ProviderName='Microsoft-Windows-Kernel-Power'; Id=506,507,42,107} "
    "-ErrorAction SilentlyContinue"
    " | Select-Object @{n='utc';e={$_.TimeCreated.ToUniversalTime()"
    ".ToString('yyyy-MM-ddTHH:mm:ss')}},Id | ConvertTo-Json -Depth 3"
)


def read_suspend_windows() -> list[tuple[datetime.datetime, datetime.datetime]]:
    """Read paired suspend windows from the System event log, in UTC.

    Returns:
        ``(entered, exited)`` pairs, ascending. An entry with no matching exit is dropped: it
        means the log was truncated or the host is suspended right now, and neither can bound a
        window.

    Raises:
        RuntimeError: If the event log cannot be read.
    """
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", _QUERY],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read the System event log: {completed.stderr.strip()}")

    payload = completed.stdout.strip()
    if not payload:
        return []
    events = json.loads(payload)
    if isinstance(events, dict):
        events = [events]

    windows: list[tuple[datetime.datetime, datetime.datetime]] = []
    opened: datetime.datetime | None = None
    for stamp, identifier in sorted(
        (datetime.datetime.fromisoformat(event["utc"]), event["Id"]) for event in events
    ):
        if identifier in SUSPEND_ENTRY:
            opened = stamp
        elif identifier in SUSPEND_EXIT and opened is not None:
            windows.append((opened, stamp))
            opened = None
    return windows


def read_deployment_cells(metrics: Path) -> list[dict[str, object]]:
    """Collect the cell window of every record that carries a deployment measurement.

    Args:
        metrics: Directory of JSON run records.

    Returns:
        One dict per record, ascending by start, with ``id``, ``start``, ``end`` and ``split``.
    """
    cells: list[dict[str, object]] = []
    for path in sorted(metrics.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record.get("deployment"), dict) or not record["deployment"]:
            continue
        stamp, duration = record.get("timestamp"), record.get("duration_seconds")
        if not stamp or not duration:
            continue
        # `timestamp` is the cell START. Verified against the run log: pruning_aggressive_rep3
        # carries 12:02:23Z with duration 2619 s, and the log shows that cell running
        # 17:02:23 -> 17:46:02 local. Reading it as the end inverts every window by one cell
        # length and manufactures overlaps that never happened.
        begin = datetime.datetime.fromisoformat(stamp).replace(tzinfo=None)
        cells.append(
            {
                "id": record["experiment_id"],
                "start": begin,
                "end": begin + datetime.timedelta(seconds=float(duration)),
                "split": ((record.get("config") or {}).get("data") or {}).get("eval_split"),
            }
        )
    return sorted(cells, key=lambda cell: cell["start"])


def main() -> int:
    """Report every deployment-bearing cell whose window contains a host suspend.

    Returns:
        0 when no cell overlaps a suspend, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args()
    logging.basicConfig(level=arguments.log_level, format="%(message)s")

    if not arguments.metrics.is_dir():
        LOGGER.error("no such directory: %s", arguments.metrics)
        return 1

    windows = read_suspend_windows()
    cells = read_deployment_cells(arguments.metrics)
    LOGGER.info(
        "%d suspend window(s), %d record(s) carrying deployment data", len(windows), len(cells)
    )
    if windows:
        LOGGER.info("  suspend history spans %s .. %s UTC", windows[0][0], windows[-1][1])
    if cells:
        LOGGER.info("  records span            %s .. %s UTC", cells[0]["start"], cells[-1]["end"])
    if not windows or not cells:
        return 0

    # A suspend history that starts after the first record cannot clear the records before it.
    if windows[0][0] > cells[0]["start"]:
        LOGGER.warning(
            "  the event log begins AFTER the first record, so records before %s are unaudited",
            windows[0][0],
        )

    overlaps = [
        (cell, start, stop)
        for cell in cells
        for start, stop in windows
        if cell["start"] < stop and start < cell["end"]
    ]
    if not overlaps:
        LOGGER.info("")
        LOGGER.info("CLEAN: no deployment record's window contains a suspend.")
        return 0

    LOGGER.info("")
    LOGGER.warning(
        "%d cell(s) overlap a suspend -- read their stage timings from the run log", len(overlaps)
    )
    for cell, start, stop in overlaps:
        inside = (min(cell["end"], stop) - max(cell["start"], start)).total_seconds() / 60
        LOGGER.warning("  %s [%s]", cell["id"], cell["split"])
        LOGGER.warning("    cell    %s .. %s", cell["start"], cell["end"])
        LOGGER.warning("    suspend %s .. %s  (%.1f min inside the cell)", start, stop, inside)
    return 1


if __name__ == "__main__":
    sys.exit(main())
