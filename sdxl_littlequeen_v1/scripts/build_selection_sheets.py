#!/usr/bin/env python3
"""Build labeled contact sheets for manual Little Queen dataset selection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = Path(
    "/mnt/storage/projects/agentic/images/scripts/output_lora_littlequeen_dataset/"
    "curation/first_pass_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "training_dataset" / "selection_sheets"
    )
    parser.add_argument("--per-sheet", type=int, default=20)
    parser.add_argument("--ids-file", type=Path)
    return parser.parse_args()


def image_number(path: Path) -> int:
    match = re.match(r"image_(\d+)_", path.name)
    return int(match.group(1)) if match else 0


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    accepted = [
        record
        for record in report.get("records", [])
        if record.get("decision") == "accept" and Path(record["image_file"]).is_file()
    ]
    accepted.sort(key=lambda record: image_number(Path(record["image_file"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 20)
    columns = 5
    rows = (args.per_sheet + columns - 1) // columns
    cell_width, image_height, label_height = 220, 330, 30
    cell_height = image_height + label_height
    index_records = []

    for accepted_index, record in enumerate(accepted, start=1):
        source = Path(record["image_file"])
        index_records.append(
            {
                "review_id": f"A{accepted_index:03d}",
                "image_number": image_number(source),
                "image_file": str(source),
                "source_caption": record.get("caption", ""),
            }
        )

    if args.ids_file:
        requested_ids = [
            line.strip()
            for line in args.ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {record["review_id"]: record for record in index_records}
        missing = [review_id for review_id in requested_ids if review_id not in by_id]
        if missing:
            raise ValueError(f"Unknown review IDs: {', '.join(missing)}")
        index_records = [by_id[review_id] for review_id in requested_ids]

    for sheet_index, start in enumerate(range(0, len(index_records), args.per_sheet), start=1):
        records = index_records[start : start + args.per_sheet]
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#202124")
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(records):
            column = offset % columns
            row = offset // columns
            x, y = column * cell_width, row * cell_height
            with Image.open(record["image_file"]) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
                image.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
                paste_x = x + (cell_width - image.width) // 2
                paste_y = y + (image_height - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))
            label = f"{record['review_id']}  image_{record['image_number']}"
            draw.rectangle((x, y + image_height, x + cell_width, y + cell_height), fill="#111111")
            draw.text((x + 6, y + image_height + 3), label, font=font, fill="#ffffff")
        destination = args.output_dir / f"accepted_{sheet_index:02d}.jpg"
        sheet.save(destination, quality=92, subsampling=0)

    (args.output_dir / "selection_index.json").write_text(
        json.dumps(index_records, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(index_records)} accepted-image records to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
