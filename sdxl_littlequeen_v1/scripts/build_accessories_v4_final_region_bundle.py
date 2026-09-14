#!/usr/bin/env python3
"""Bundle canonical V4 composites with exact training alpha for final regional tests."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "training_dataset_accessories_v4" / "staging"
TRAINING = ROOT / "training_dataset_accessories_v4" / "colab"
MASKS = ROOT / "training_dataset_accessories_v4" / "regional_exact_masks"
OUTPUT = (
    ROOT
    / "training_dataset_accessories_v4"
    / "colab"
    / "lqaccessories_v4_final_region_bundle.tar.gz"
)
SELECTION = (76, 91)


def extract_alpha(source: Path, destination: Path) -> None:
    image = Image.open(source).convert("RGBA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.getchannel("A").save(destination, format="PNG")


def main() -> int:
    records = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for number in SELECTION:
            item_id = f"candidate_{number:03d}"
            image = STAGING / "composites" / f"{item_id}.png"
            regalia_rgba = (
                TRAINING
                / "train_regalia"
                / "1_lqmoonregalia4"
                / f"regalia_{number:03d}_full.png"
            )
            wand_rgba = (
                TRAINING
                / "train_wand"
                / "1_lqmoonwand4"
                / f"wand_{number:03d}_full.png"
            )
            regalia_mask = MASKS / f"{item_id}_regalia.png"
            wand_mask = MASKS / f"{item_id}_wand.png"
            for path in (image, regalia_rgba, wand_rgba):
                if not path.is_file():
                    raise FileNotFoundError(path)
            extract_alpha(regalia_rgba, regalia_mask)
            extract_alpha(wand_rgba, wand_mask)
            archive.add(image, arcname=f"inputs/{item_id}.png")
            archive.add(regalia_mask, arcname=f"masks/{item_id}_regalia.png")
            archive.add(wand_mask, arcname=f"masks/{item_id}_wand.png")
            records.append({"id": item_id, "seed": 160000 + number})

        manifest = (
            ROOT / "training_dataset_accessories_v4" / "final_regional_selection.json"
        )
        manifest.write_text(
            json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
        )
        archive.add(manifest, arcname="regional_selection.json")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
