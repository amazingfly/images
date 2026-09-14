#!/usr/bin/env python3
"""Wait for stage one, switch the shared Space to SDXL, and run stage two."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config_sdxl.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
STOP = False


def stop(_signum, _frame) -> None:
    global STOP
    STOP = True


def stage_one_complete() -> bool:
    path = Path(CONFIG["stage1_state"])
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    return state.get("status") == "complete" and len(state.get("completed", [])) == 12


def main() -> int:
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not STOP and not stage_one_complete():
        print("stage 2 waiting for all 12 matched FLUX results", flush=True)
        time.sleep(60)
    if STOP:
        return 130
    marker = Path(CONFIG["deployment_marker"])
    if not marker.is_file():
        subprocess.run([sys.executable, str(HERE / "deploy_sdxl.py")], check=True)
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(HERE / "supervise.py"),
            "--config",
            str(CONFIG_PATH),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
