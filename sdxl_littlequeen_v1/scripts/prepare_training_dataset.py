#!/usr/bin/env python3
"""Build the manually curated SDXL LoRA dataset without changing source images."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "training_dataset" / "curated_manifest.json"
DEFAULT_REPORT = Path(
    "/mnt/storage/projects/agentic/images/scripts/"
    "output_lora_littlequeen_dataset/curation/first_pass_report.json"
)
DEFAULT_OUTPUT = ROOT / "training_dataset" / "colab" / "train" / "10_lqxl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_index(path: Path) -> int:
    prefix = path.stem.split("_", 2)
    if len(prefix) < 2 or prefix[0] != "image":
        raise ValueError(f"unexpected source image name: {path.name}")
    return int(prefix[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    report = json.loads(args.report.read_text())
    selected = manifest["images"]
    if len(selected) != 30:
        raise SystemExit(f"expected exactly 30 curated images, found {len(selected)}")

    accepted: dict[int, Path] = {}
    for record in report["records"]:
        path = Path(record["image_file"])
        if record["decision"] == "accept":
            accepted[image_index(path)] = path

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("lqxl_*.*"):
        stale.unlink()

    prepared = []
    seen: set[int] = set()
    for position, entry in enumerate(selected, start=1):
        source_index = int(entry["source_index"])
        if source_index in seen:
            raise SystemExit(f"duplicate source index: {source_index}")
        seen.add(source_index)
        source = accepted.get(source_index)
        if source is None:
            raise SystemExit(f"source image {source_index} is not accepted in {args.report}")
        if not source.is_file():
            raise SystemExit(f"missing source image: {source}")

        stem = f"lqxl_{position:03d}"
        image_out = args.output / f"{stem}{source.suffix.lower()}"
        caption_out = args.output / f"{stem}.txt"
        shutil.copy2(source, image_out)
        caption_out.write_text(entry["caption"].strip() + "\n")
        prepared.append(
            {
                "dataset_position": position,
                "source_index": source_index,
                "source_image": str(source),
                "source_sha256": sha256(source),
                "dataset_image": str(image_out),
                "caption": entry["caption"],
            }
        )

    output_manifest = args.output.parent.parent / "prepared_manifest.json"
    output_manifest.write_text(
        json.dumps(
            {
                "trigger_token": manifest["trigger_token"],
                "image_count": len(prepared),
                "repeat_count": 10,
                "images": prepared,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(prepared)} accepted images in {args.output}")
    print(f"Wrote provenance manifest to {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
