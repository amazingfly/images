#!/usr/bin/env python3
"""Prepare the manually curated v2 SDXL LoRA dataset without altering sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "training_dataset_v2" / "curated_manifest.json"
DEFAULT_SOURCE = (
    ROOT
    / "outputs"
    / "identity_dataset100_v2"
    / "colab_fp16_832x1216_20260719"
    / "output"
)
DEFAULT_OUTPUT = ROOT / "training_dataset_v2" / "colab" / "train" / "10_lqxl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    selected = manifest["images"]
    if len(selected) != 30:
        raise SystemExit(f"expected exactly 30 curated images, found {len(selected)}")

    indices = [int(entry["source_index"]) for entry in selected]
    if len(indices) != len(set(indices)):
        raise SystemExit("curated manifest contains duplicate source indices")
    scenes = Counter(entry["scene_id"] for entry in selected)
    if sorted(scenes.values()) != [3, 3, 4, 4, 4, 4, 4, 4]:
        raise SystemExit(f"unexpected scene balance: {dict(scenes)}")

    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("lqxl_*.*"):
        stale.unlink()

    prepared = []
    image_hashes: set[str] = set()
    for position, entry in enumerate(selected, start=1):
        source_index = int(entry["source_index"])
        source = args.source / (
            f"image_{source_index:03d}_seed_{95000 + source_index}.png"
        )
        if not source.is_file():
            raise SystemExit(f"missing source image: {source}")
        with Image.open(source) as image:
            width, height = image.size
            mode = image.mode
            image.verify()
        if (width, height) != (832, 1216):
            raise SystemExit(
                f"unexpected dimensions for {source.name}: {width}x{height}"
            )
        if mode not in {"RGB", "RGBA"}:
            raise SystemExit(f"unexpected image mode for {source.name}: {mode}")

        source_hash = sha256(source)
        if source_hash in image_hashes:
            raise SystemExit(f"duplicate image content: {source.name}")
        image_hashes.add(source_hash)

        stem = f"lqxl_{position:03d}"
        image_out = args.output / f"{stem}.png"
        caption_out = args.output / f"{stem}.txt"
        shutil.copy2(source, image_out)
        caption = entry["caption"].strip()
        if not caption.startswith("lqxl, solo,"):
            raise SystemExit(f"caption {position} does not preserve trigger tokens")
        caption_out.write_text(caption + "\n")
        prepared.append(
            {
                "dataset_position": position,
                "source_index": source_index,
                "scene_id": entry["scene_id"],
                "source_image": str(source),
                "source_sha256": source_hash,
                "source_width": width,
                "source_height": height,
                "dataset_image": str(image_out),
                "caption": caption,
            }
        )

    output_manifest = args.output.parent.parent / "prepared_manifest.json"
    output_manifest.write_text(
        json.dumps(
            {
                "version": manifest["version"],
                "trigger_token": manifest["trigger_token"],
                "image_count": len(prepared),
                "repeat_count": 10,
                "scene_counts": dict(sorted(scenes.items())),
                "images": prepared,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(prepared)} manually curated images in {args.output}")
    print(f"Wrote provenance manifest to {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
