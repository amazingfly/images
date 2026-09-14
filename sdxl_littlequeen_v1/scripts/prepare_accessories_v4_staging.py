#!/usr/bin/env python3
"""Composite one canonical accessory set onto varied Little Queen base images."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/mnt/storage/projects/agentic/images/agy/outputs/"
    "storybook_v2_local100/20260722_103202/base"
)
OUTPUT = ROOT / "training_dataset_accessories_v4" / "staging"
ACCESSORY_SET = ROOT / "storybook_accessories_v2" / "accessory_set.json"
COMPOSITOR = ROOT / "scripts" / "apply_storybook_accessories.py"

# These are intentionally varied in framing, clothing, scene, and pose. The source
# contact sheets were reviewed to exclude obvious inherited crowns and malformed figures.
CANDIDATES = (
    51,
    52,
    53,
    54,
    55,
    56,
    60,
    63,
    64,
    65,
    67,
    68,
    69,
    70,
    72,
    73,
    74,
    76,
    77,
    78,
    81,
    85,
    88,
    91,
    96,
    97,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def paths(candidate: int) -> dict[str, Path]:
    stem = f"candidate_{candidate:03d}"
    return {
        "input": SOURCE / f"candidate_{candidate}.png",
        "image": OUTPUT / "composites" / f"{stem}.png",
        "metadata": OUTPUT / "metadata" / f"{stem}.json",
        "blend_mask": OUTPUT / "masks" / f"{stem}_blend.png",
        "cleanup_mask": OUTPUT / "masks" / f"{stem}_cleanup.png",
        "regalia_mask": OUTPUT / "masks" / f"{stem}_regalia.png",
        "wand_mask": OUTPUT / "masks" / f"{stem}_wand.png",
    }


def compose(candidate: int, force: bool) -> dict:
    item = paths(candidate)
    if not item["input"].is_file():
        raise FileNotFoundError(item["input"])
    if (
        not force
        and item["image"].is_file()
        and item["metadata"].is_file()
        and item["regalia_mask"].is_file()
        and item["wand_mask"].is_file()
    ):
        return {"candidate": candidate, "status": "preserved"}

    for key in ("image", "metadata", "blend_mask", "cleanup_mask", "regalia_mask", "wand_mask"):
        item[key].parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(COMPOSITOR),
        str(item["input"]),
        str(item["image"]),
        "--set",
        str(ACCESSORY_SET),
        "--staff",
        "--metadata",
        str(item["metadata"]),
        "--blend-mask",
        str(item["blend_mask"]),
        "--cleanup-mask",
        str(item["cleanup_mask"]),
        "--regalia-mask",
        str(item["regalia_mask"]),
        "--hand-prop-mask",
        str(item["wand_mask"]),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"candidate {candidate} failed ({result.returncode}):\n{result.stderr[-4000:]}"
        )
    return {"candidate": candidate, "status": "composited"}


def main() -> int:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(compose, candidate, args.force): candidate for candidate in CANDIDATES
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({"candidate": candidate, "error": str(exc)})
                print(f"FAIL candidate {candidate}: {exc}", flush=True)
            else:
                results.append(result)
                print(f"{result['status'].upper()} candidate {candidate}", flush=True)

    report = {
        "version": "littlequeen-accessories-v4-staging",
        "source": str(SOURCE),
        "accessory_set": str(ACCESSORY_SET),
        "candidates": list(CANDIDATES),
        "results": sorted(results, key=lambda item: item["candidate"]),
        "failures": sorted(failures, key=lambda item: item["candidate"]),
    }
    (OUTPUT / "staging_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
