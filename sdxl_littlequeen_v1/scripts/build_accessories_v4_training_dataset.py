#!/usr/bin/env python3
"""Build exact-alpha V4 regalia and Moonstar wand training datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "training_dataset_accessories_v4" / "staging"
SOURCE = Path(
    "/mnt/storage/projects/agentic/images/agy/outputs/"
    "storybook_v2_local100/20260722_103202/base"
)
DATASET = ROOT / "training_dataset_accessories_v4" / "colab"
REGALIA_DIR = DATASET / "train_regalia" / "1_lqmoonregalia4"
WAND_DIR = DATASET / "train_wand" / "1_lqmoonwand4"
MANIFEST = ROOT / "training_dataset_accessories_v4" / "manifest.json"
SELECTION = ROOT / "training_dataset_accessories_v4" / "selection.json"
OUTPUT_SIZE = (768, 1024)

REGALIA_SELECTION = (
    51,
    52,
    54,
    55,
    56,
    60,
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
    78,
    81,
    85,
    88,
    91,
    96,
    97,
)

# These placements retain a readable finial and shaft and include full, partial,
# left-side, and right-side occlusions. Poorly attached/mostly hidden placements
# were deliberately excluded after reviewing the staging contact sheet.
WAND_SELECTION = (51, 54, 55, 64, 67, 68, 70, 76, 91)

REGALIA_CAPTION = (
    "lqmoonregalia4, exactly one symmetrical polished gold Moonstar crown with five "
    "pointed flame-leaf peaks, engraved crescent moon and star filigree, one large "
    "centered oval opalescent rose-amber moonstone, two smaller oval amber side gems, "
    "one matched pair of small gold leaf-drop earrings, one fine gold filigree collar "
    "necklace with a centered rose-amber gem"
)
WAND_CAPTION = (
    "lqmoonwand4, exactly one slender gold Moonstar scepter, one long straight engraved "
    "shaft with crescent moon and star motifs, one broad open crescent finial surrounding "
    "one large faceted rose-amber oval moonstone, five leaf-shaped rose crystals above "
    "the moonstone, two small round rose side gems, one faceted rose crystal point at the base"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def staged_paths(candidate: int, kind: str) -> tuple[Path, Path, Path]:
    stem = f"candidate_{candidate:03d}"
    return (
        STAGING / "composites" / f"{stem}.png",
        SOURCE / f"candidate_{candidate}.png",
        STAGING / "masks" / f"{stem}_{kind}.png",
    )


def exact_visible_mask(composite: Image.Image, source: Image.Image, placed: Image.Image) -> Image.Image:
    difference = np.asarray(ImageChops.difference(composite, source), dtype=np.uint8)
    changed = (difference.max(axis=2) >= 3).astype(np.uint8) * 255
    visible = Image.fromarray(changed).filter(ImageFilter.MaxFilter(3))
    return ImageChops.multiply(placed.convert("L"), visible)


def bounded_crop(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    context_scale: float,
    minimum_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    mask_width = max(1, right - left)
    mask_height = max(1, bottom - top)
    target_aspect = OUTPUT_SIZE[0] / OUTPUT_SIZE[1]
    crop_height = max(minimum_height, round(mask_height * context_scale))
    crop_width = max(round(mask_width * context_scale), round(crop_height * target_aspect))
    crop_height = max(crop_height, round(crop_width / target_aspect))
    image_width, image_height = image_size
    if crop_width > image_width:
        crop_width = image_width
        crop_height = round(crop_width / target_aspect)
    if crop_height > image_height:
        crop_height = image_height
        crop_width = round(crop_height * target_aspect)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = round(min(max(0, center_x - crop_width / 2), image_width - crop_width))
    crop_top = round(min(max(0, center_y - crop_height / 2), image_height - crop_height))
    return crop_left, crop_top, crop_left + crop_width, crop_top + crop_height


def save_rgba(image: Image.Image, mask: Image.Image, destination: Path) -> float:
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    rgba.save(destination, format="PNG", compress_level=6)
    values = np.asarray(mask, dtype=np.float32)
    return float(values.sum() / (255.0 * values.size))


def build_kind(
    kind: str,
    candidates: tuple[int, ...],
    output: Path,
    caption: str,
) -> list[dict]:
    records = []
    for candidate in candidates:
        composite_path, source_path, placed_mask_path = staged_paths(candidate, kind)
        composite = Image.open(composite_path).convert("RGB")
        source = Image.open(source_path).convert("RGB")
        placed = Image.open(placed_mask_path).convert("L")
        mask = exact_visible_mask(composite, source, placed)
        bbox = mask.getbbox()
        if not bbox:
            raise RuntimeError(f"{kind} candidate {candidate} has an empty exact mask")

        stem = f"{kind}_{candidate:03d}"
        full_path = output / f"{stem}_full.png"
        full_fraction = save_rgba(composite, mask, full_path)
        full_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

        crop = bounded_crop(
            bbox,
            composite.size,
            context_scale=2.0 if kind == "regalia" else 1.35,
            minimum_height=480 if kind == "regalia" else 600,
        )
        crop_image = composite.crop(crop).resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
        crop_mask = mask.crop(crop).resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
        crop_path = output / f"{stem}_crop.png"
        crop_fraction = save_rgba(crop_image, crop_mask, crop_path)
        crop_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

        records.append(
            {
                "candidate": candidate,
                "composite": str(composite_path),
                "source": str(source_path),
                "placed_mask": str(placed_mask_path),
                "exact_mask_bbox": list(bbox),
                "full": {
                    "path": str(full_path),
                    "alpha_fraction": round(full_fraction, 6),
                    "sha256": sha256(full_path),
                },
                "crop": {
                    "path": str(crop_path),
                    "crop_box": list(crop),
                    "alpha_fraction": round(crop_fraction, 6),
                    "sha256": sha256(crop_path),
                },
            }
        )
    return records


def main() -> int:
    clean(REGALIA_DIR)
    clean(WAND_DIR)
    regalia = build_kind("regalia", REGALIA_SELECTION, REGALIA_DIR, REGALIA_CAPTION)
    wand = build_kind("wand", WAND_SELECTION, WAND_DIR, WAND_CAPTION)
    selection = {
        "version": "littlequeen-accessories-v4-selection",
        "review_basis": str(STAGING / "review" / "composites.jpg"),
        "regalia_candidates": list(REGALIA_SELECTION),
        "wand_candidates": list(WAND_SELECTION),
        "rejected_for_regalia": [53, 63, 77],
        "rejected_for_wand": sorted(
            set(json.loads((STAGING / "staging_report.json").read_text())["candidates"])
            - set(WAND_SELECTION)
        ),
        "wand_rejection_reason": "mostly hidden, poorly attached, or unreadable hand-prop placement",
    }
    SELECTION.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": "littlequeen-accessories-v4-exact-alpha",
        "training_method": "separate object LoRAs with exact visible-pixel alpha loss",
        "output_resolution": list(OUTPUT_SIZE),
        "regalia_trigger": "lqmoonregalia4",
        "wand_trigger": "lqmoonwand4",
        "regalia_caption": REGALIA_CAPTION,
        "wand_caption": WAND_CAPTION,
        "regalia_examples": len(regalia) * 2,
        "wand_examples": len(wand) * 2,
        "regalia": regalia,
        "wand": wand,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "regalia_examples": manifest["regalia_examples"],
                "wand_examples": manifest["wand_examples"],
                "regalia_dir": str(REGALIA_DIR),
                "wand_dir": str(WAND_DIR),
                "manifest": str(MANIFEST),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
