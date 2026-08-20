"""Hold the host awake for the duration of an unattended run.

B-47 again. Setting `standby-timeout-ac 0` was not sufficient: the host still entered Modern
Standby with "Reason: Idle Timeout", because on an S0 machine standby follows the *display* going
idle rather than the STANDBYIDLE timer. Power policy is also mutable -- Windows Update, a docking
event or a group policy refresh can put a timeout back without telling anyone.

This asserts the requirement directly instead. ES_CONTINUOUS | ES_SYSTEM_REQUIRED is the flag a
media player holds while playing, and it survives policy changes because it is a per-process power
request rather than a setting. ES_DISPLAY_REQUIRED is deliberately NOT set: the screen may sleep,
only the system may not.

Run it beside the sweep and leave it. Ctrl-C, or killing it, releases the request.
"""

import ctypes
import datetime
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def main() -> int:
    """Assert the power request and hold it until interrupted.

    Returns:
        0 on a clean release, 1 if the request was refused.
    """
    kernel32 = ctypes.windll.kernel32
    if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        print("SetThreadExecutionState refused the request", flush=True)
        return 1

    started = datetime.datetime.now()
    print(
        f"keep-awake held from {started:%Y-%m-%d %H:%M:%S} (system required, display free)",
        flush=True,
    )
    try:
        while True:
            # Re-assert periodically: the flag is per-thread, and a policy refresh has been seen to
            # clear requests on this host. Cheap insurance at one call a minute.
            kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            time.sleep(60)
    except KeyboardInterrupt:
        held = datetime.datetime.now() - started
        print(f"released after {held}", flush=True)
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
