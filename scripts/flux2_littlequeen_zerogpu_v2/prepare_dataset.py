#!/usr/bin/env python3
"""Build the manually curated, identity-bound FLUX.2 Little Queen v2 dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
SPEC = json.loads((HERE / "dataset_spec.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    source_root = Path(SPEC["source_dataset"])
    source_manifest = json.loads(
        (source_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    sources = {int(item["index"]): item for item in source_manifest["images"]}
    output = Path(CONFIG["dataset_dir"])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for target_index, selected in enumerate(SPEC["images"], start=1):
        source_index = int(selected["source_index"])
        source_record = sources[source_index]
        source = source_root / source_record["image"]
        image_name = f"lqk4n2_{target_index:03d}.png"
        caption_name = f"lqk4n2_{target_index:03d}.txt"
        destination = output / image_name
        shutil.copy2(source, destination)
        caption = selected["caption"].strip()
        (output / caption_name).write_text(caption + "\n", encoding="utf-8")
        with Image.open(destination) as image:
            width, height = image.size
        records.append({
            "index": target_index,
            "image": image_name,
            "caption_file": caption_name,
            "caption": caption,
            "width": width,
            "height": height,
            "sha256": sha256(destination),
            "source_index": source_index,
            "source_sha256": source_record["sha256"],
            "source": str(source),
        })

    manifest = {
        "version": "little-queen-flux2-klein-identity-v2",
        "trigger": CONFIG["trigger"],
        "base_model": CONFIG["base_model"],
        "selection": SPEC["exclusion_reason"],
        "excluded_source_indices": SPEC["excluded_source_indices"],
        "caption_strategy": SPEC["caption_strategy"],
        "image_count": len(records),
        "images": records,
    }
    (output / "metadata.jsonl").write_text(
        "\n".join(
            json.dumps({"file_name": item["image"], "text": item["caption"]})
            for item in records
        ) + "\n",
        encoding="utf-8",
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "---\nlicense: other\npretty_name: Little Queen FLUX.2 Klein identity v2\n---\n\n"
        "Private synthetic identity-training dataset with manually reviewed captions.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "image_count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
