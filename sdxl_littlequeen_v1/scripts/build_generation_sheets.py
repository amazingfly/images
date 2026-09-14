#!/usr/bin/env python3
"""Build labeled contact sheets directly from an SDXL run summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--per-sheet", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = []
    for result in summary.get("results", []):
        if result.get("status") != "completed":
            continue
        source = Path(result["output"])
        if not source.is_file():
            source = summary_path.parent / "output" / source.name
        if not source.is_file():
            continue
        result = dict(result)
        result["output"] = str(source)
        records.append(result)
    records.sort(key=lambda result: int(result["index"]))
    output_dir = args.output_dir or summary_path.parent / "review_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("generated_*.jpg"):
        stale.unlink()

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
    )
    columns = 5
    rows = (args.per_sheet + columns - 1) // columns
    cell_width, image_height, label_height = 220, 294, 30
    cell_height = image_height + label_height
    index = []

    for sheet_number, start in enumerate(range(0, len(records), args.per_sheet), start=1):
        sheet_records = records[start : start + args.per_sheet]
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#202124")
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(sheet_records):
            column, row = offset % columns, offset // columns
            x, y = column * cell_width, row * cell_height
            source = Path(record["output"])
            with Image.open(source) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
                image.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
                sheet.paste(
                    image,
                    (
                        x + (cell_width - image.width) // 2,
                        y + (image_height - image.height) // 2,
                    ),
                )
            label = f"#{int(record['index']):03d}  seed {record['seed']}"
            draw.rectangle(
                (x, y + image_height, x + cell_width, y + cell_height), fill="#111111"
            )
            draw.text((x + 6, y + image_height + 4), label, font=font, fill="#ffffff")
            index.append(
                {
                    "index": int(record["index"]),
                    "seed": int(record["seed"]),
                    "image": str(source),
                    "prompt": record["prompt"],
                }
            )
        sheet.save(output_dir / f"generated_{sheet_number:02d}.jpg", quality=94, subsampling=0)

    (output_dir / "generation_index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} records and contact sheets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
