#!/usr/bin/env bash
# Keep a long sweep alive unattended.
#
# WHAT IT IS FOR
#
# A multi-day grid dies for reasons that have nothing to do with the science: a transient
# RemoteProtocolError fetching a cached corpus (seen on the Qwen leg), a MemoryError from the
# accumulation in B-48, a host hiccup. `skip_existing` makes a restart nearly free -- every record
# is written whole and only the in-flight cell is lost -- so the correct response to almost any
# death is to start it again.
#
# TWO MODES, AND THE DIFFERENCE MATTERS
#
#   --isolate-cells (preferred)  Each cell runs in its own child process, so memory is released at
#                                the cell boundary by construction. The parent holds no model state
#                                and its footprint stays flat, so the commit-recycling logic below
#                                should never fire. Crash recovery is then the supervisor's whole
#                                job.
#   without isolation            The runner accumulates ~4 GiB of commit per 1B compression cell and
#                                never returns it (B-48). The supervisor recycles the process on low
#                                commit, preferentially just after a cell boundary where a restart
#                                costs almost nothing.
#
# WHY IT WAITS FOR A BOUNDARY
#
# Recycling on a threshold alone thrashes once the recycle interval approaches the cell duration --
# at 1B those were ~55 and ~54 minutes, so a naive restart could discard 50 minutes of work. Above
# EMERGENCY_COMMIT_GIB the supervisor defers until a cell has just started.
#
# USAGE
#   scripts/supervise_sweep.sh <config> <log> [extra args to run_scale_sweep.py ...]
#
# EXAMPLE
#   scripts/supervise_sweep.sh configs/experiments/qwen_validation.yaml /tmp/qwen.log --isolate-cells

set -u

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <config> <log> [extra args...]" >&2
  exit 2
fi

CONFIG="$1"
LOG="$2"
shift 2
EXTRA=("$@")

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/Scripts/python.exe"
MIN_COMMIT_GIB=7
EMERGENCY_COMMIT_GIB=2
CELL_FRESH_S=360
MIN_UPTIME_S=900
POLL_S=120

# The corpus is cached; a refresh attempt that fails mid-grid would drop cells under
# continue_on_error. Offline is strictly safer for an unattended run.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

commit_free_gib() {
  powershell -NoProfile -Command \
    '$os=Get-CimInstance Win32_OperatingSystem; [math]::Round($os.FreeVirtualMemory/1MB,2)' 2>/dev/null | tr -d '\r'
}

sweep_pid() {
  powershell -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_scale_sweep*' } | Select-Object -First 1).ProcessId" 2>/dev/null | tr -d '\r'
}

uptime_s() {
  powershell -NoProfile -Command \
    "\$p=Get-Process -Id $1 -ErrorAction SilentlyContinue; if (\$p) { [int]((Get-Date) - \$p.StartTime).TotalSeconds } else { 0 }" 2>/dev/null | tr -d '\r'
}

launch() {
  cd "$REPO" || exit 1
  nohup "$PYTHON" -u scripts/run_scale_sweep.py --config "$CONFIG" "${EXTRA[@]}" >> "$LOG" 2>&1 &
  echo "$(date '+%F %T') SUPERVISOR: launched sweep on $CONFIG ${EXTRA[*]}"
  sleep 30
}

kill_sweep() {
  powershell -NoProfile -Command \
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_scale_sweep*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
  sleep 8
}

echo "$(date '+%F %T') SUPERVISOR: watching $CONFIG (log $LOG)"

while true; do
  # Exit once the sweep says every planned cell has a record, so the supervisor cannot resurrect a
  # finished run and spin relaunching a no-op. Both modes emit SWEEP FINISHED; "Completed N run(s)"
  # is NOT a completion signal under --isolate-cells, because each child prints it for its own cell.
  if grep -q "SWEEP FINISHED" "$LOG" 2>/dev/null; then
    echo "$(date '+%F %T') SUPERVISOR: sweep reported complete, exiting"
    exit 0
  fi

  pid=$(sweep_pid)
  if [ -z "$pid" ]; then
    echo "$(date '+%F %T') SUPERVISOR: no sweep running, relaunching"
    launch
    continue
  fi

  free=$(commit_free_gib)
  up=$(uptime_s "$pid")
  free_int=${free%.*}

  if [ -n "$free_int" ] && [ "$free_int" -lt "$MIN_COMMIT_GIB" ] && [ "$up" -gt "$MIN_UPTIME_S" ]; then
    last_start=$(grep -E "\] (run|isolated child for) " "$LOG" 2>/dev/null | tail -1 | cut -c1-19)
    cell_age=999999
    if [ -n "$last_start" ]; then
      start_epoch=$(date -d "$last_start" +%s 2>/dev/null || echo "")
      [ -n "$start_epoch" ] && cell_age=$(( $(date +%s) - start_epoch ))
    fi

    if [ "$cell_age" -lt "$CELL_FRESH_S" ]; then
      echo "$(date '+%F %T') SUPERVISOR: commit free ${free} GiB, cell ${cell_age}s old -- recycling now (cheap)"
      kill_sweep
      launch
    elif [ "$free_int" -lt "$EMERGENCY_COMMIT_GIB" ]; then
      echo "$(date '+%F %T') SUPERVISOR: commit free ${free} GiB critical, cell ${cell_age}s in -- recycling anyway"
      kill_sweep
      launch
    else
      echo "$(date '+%F %T') SUPERVISOR: commit free ${free} GiB, cell ${cell_age}s in -- deferring to the next boundary"
    fi
  fi

  sleep "$POLL_S"
done
