#!/usr/bin/env python3
"""Bundle two precise-mask accessory composites for Colab regional validation."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "accessories_v4_regional_inputs"
OUTPUT = ROOT / "training_dataset_accessories_v4" / "colab" / "lqaccessories_v4_region_bundle.tar.gz"
SELECTION = (76, 91)


def main() -> int:
    records = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for number in SELECTION:
            item_id = f"candidate_{number}"
            image = SOURCE / f"{item_id}.png"
            regalia = SOURCE / f"{item_id}_regalia_mask.png"
            wand = SOURCE / f"{item_id}_hand_prop_mask.png"
            metadata = SOURCE / f"{item_id}.json"
            for path in (image, regalia, wand, metadata):
                if not path.is_file():
                    raise FileNotFoundError(path)
            archive.add(image, arcname=f"inputs/{item_id}.png")
            archive.add(regalia, arcname=f"masks/{item_id}_regalia.png")
            archive.add(wand, arcname=f"masks/{item_id}_wand.png")
            records.append({"id": item_id, "seed": 150000 + number})
        manifest = ROOT / "training_dataset_accessories_v4" / "regional_selection.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
        archive.add(manifest, arcname="regional_selection.json")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
