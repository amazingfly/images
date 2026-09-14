#!/usr/bin/env python3
"""Compare matched standard and seed-refine SDXL runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("standard_summary", type=Path)
    parser.add_argument("seed_refine_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def image_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    horizontal = np.diff(gray, axis=1)
    vertical = np.diff(gray, axis=0)
    gradient_rms = float(
        np.sqrt((np.mean(horizontal * horizontal) + np.mean(vertical * vertical)) / 2.0)
    )
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return {
        "luminance_contrast": round(float(np.std(gray)), 6),
        "gradient_rms": round(gradient_rms, 6),
        "laplacian_variance": round(float(np.var(laplacian)), 6),
    }


def mean(values: list[float], digits: int = 3) -> float:
    return round(statistics.fmean(values), digits)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": len(records),
        "total_seconds": round(sum(item["duration_seconds"] for item in records), 3),
        "mean_seconds": mean([item["duration_seconds"] for item in records]),
        "median_seconds": round(statistics.median(item["duration_seconds"] for item in records), 3),
        "peak_process_rss_mb": round(max(item["peak_process_rss_mb"] for item in records), 1),
        "minimum_system_available_mb": round(
            min(item["minimum_system_available_mb"] for item in records), 1
        ),
        "mean_gradient_rms": mean(
            [item["metrics"]["gradient_rms"] for item in records], 6
        ),
        "mean_laplacian_variance": mean(
            [item["metrics"]["laplacian_variance"] for item in records], 6
        ),
        "mean_luminance_contrast": mean(
            [item["metrics"]["luminance_contrast"] for item in records], 6
        ),
    }


def contact_sheets(pairs: list[dict[str, Any]], output_dir: Path) -> list[str]:
    font = ImageFont.load_default(size=18)
    paths: list[str] = []
    thumb_size = (384, 512)
    label_height = 34
    margin = 12
    pairs_per_sheet = 5
    for sheet_number, start in enumerate(range(0, len(pairs), pairs_per_sheet), start=1):
        group = pairs[start : start + pairs_per_sheet]
        width = margin * 3 + thumb_size[0] * 2
        height = margin + len(group) * (label_height + thumb_size[1] + margin)
        sheet = Image.new("RGB", (width, height), "#202124")
        draw = ImageDraw.Draw(sheet)
        for row, pair in enumerate(group):
            top = margin + row * (label_height + thumb_size[1] + margin)
            draw.text(
                (margin, top),
                f"#{pair['index']:02d} seed {pair['seed']}  STANDARD",
                fill="white",
                font=font,
            )
            draw.text(
                (margin * 2 + thumb_size[0], top),
                "SEED + LANCZOS + IMG2IMG",
                fill="white",
                font=font,
            )
            for column, key in enumerate(("standard", "seed_refine")):
                with Image.open(pair[key]["output"]) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB", thumb_size, "black")
                    left = (thumb_size[0] - thumb.width) // 2
                    upper = (thumb_size[1] - thumb.height) // 2
                    canvas.paste(thumb, (left, upper))
                x = margin + column * (thumb_size[0] + margin)
                sheet.paste(canvas, (x, top + label_height))
        path = output_dir / f"paired_comparison_{sheet_number:02d}.jpg"
        sheet.save(path, quality=94, subsampling=0)
        paths.append(str(path))
    return paths


def main() -> int:
    args = parse_args()
    standard = load_summary(args.standard_summary)
    seed_refine = load_summary(args.seed_refine_summary)
    standard_by_index = {item["index"]: item for item in standard["results"]}
    refine_by_index = {item["index"]: item for item in seed_refine["results"]}
    if standard_by_index.keys() != refine_by_index.keys():
        raise ValueError("Runs do not contain the same image indices")

    pairs: list[dict[str, Any]] = []
    standard_records: list[dict[str, Any]] = []
    refine_records: list[dict[str, Any]] = []
    for index in sorted(standard_by_index):
        standard_item = dict(standard_by_index[index])
        refine_item = dict(refine_by_index[index])
        if standard_item["seed"] != refine_item["seed"]:
            raise ValueError(f"Seed mismatch at index {index}")
        if standard_item["prompt"] != refine_item["prompt"]:
            raise ValueError(f"Prompt mismatch at index {index}")
        standard_item["metrics"] = image_metrics(Path(standard_item["output"]))
        refine_item["metrics"] = image_metrics(Path(refine_item["output"]))
        standard_records.append(standard_item)
        refine_records.append(refine_item)
        pairs.append(
            {
                "index": index,
                "seed": standard_item["seed"],
                "standard": standard_item,
                "seed_refine": refine_item,
                "speed_delta_seconds": round(
                    refine_item["duration_seconds"] - standard_item["duration_seconds"], 3
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "standard_summary": str(args.standard_summary.resolve()),
        "seed_refine_summary": str(args.seed_refine_summary.resolve()),
        "standard": aggregate(standard_records),
        "seed_refine": aggregate(refine_records),
        "pairs": pairs,
    }
    report["seed_refine_speed_change_pct"] = round(
        (report["seed_refine"]["mean_seconds"] / report["standard"]["mean_seconds"] - 1.0)
        * 100.0,
        2,
    )
    report["contact_sheets"] = contact_sheets(pairs, args.output_dir)
    report_path = args.output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("standard", "seed_refine", "seed_refine_speed_change_pct")}, indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
