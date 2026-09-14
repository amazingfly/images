#!/usr/bin/env python3
"""Run a command with live process-tree and system-memory safety limits."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from run_sdxl_cpu import memory_sample, terminate_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--minimum-available-mb", type=float, default=1200.0)
    parser.add_argument("--maximum-rss-mb", type=float, default=12500.0)
    parser.add_argument("--maximum-swap-used-mb", type=float, default=4096.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main() -> int:
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(args.command, start_new_session=True)
    started = time.monotonic()
    samples = []
    breach = None
    try:
        while process.poll() is None:
            sample = memory_sample(process.pid)
            sample["elapsed_seconds"] = round(time.monotonic() - started, 3)
            samples.append(sample)
            if sample["system_available_mb"] < args.minimum_available_mb:
                breach = "available RAM floor"
            elif sample["process_rss_mb"] > args.maximum_rss_mb:
                breach = "process RSS ceiling"
            elif sample["system_swap_used_mb"] > args.maximum_swap_used_mb:
                breach = "system swap ceiling"
            if breach:
                terminate_group(process)
                break
            time.sleep(args.sample_interval)
    except BaseException:
        terminate_group(process)
        raise

    return_code = process.wait()
    report = {
        "command": args.command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(time.monotonic() - started, 3),
        "return_code": return_code,
        "memory_safety_breach": breach,
        "peak_process_rss_mb": round(
            max((sample["process_rss_mb"] for sample in samples), default=0.0), 1
        ),
        "peak_system_swap_used_mb": round(
            max((sample["system_swap_used_mb"] for sample in samples), default=0.0), 1
        ),
        "minimum_system_available_mb": round(
            min((sample["system_available_mb"] for sample in samples), default=0.0), 1
        ),
        "sample_count": len(samples),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 1 if breach else return_code


if __name__ == "__main__":
    raise SystemExit(main())
