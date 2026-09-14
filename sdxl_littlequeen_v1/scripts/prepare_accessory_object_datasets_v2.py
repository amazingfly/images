#!/usr/bin/env python3
"""Build decorrelated exact-object datasets for the Moonstar crown and wand."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
V1_DATASET = ROOT / "training_dataset_accessories_v1"
V1_OUTPUT = ROOT / "outputs" / "equipment_dataset100_v1" / "colab_fp16_832x1216_20260720" / "output"
BACKGROUNDS = ROOT / "outputs" / "identity_dataset100_v2" / "colab_fp16_832x1216_20260719" / "output"
DATASET = ROOT / "training_dataset_accessories_v2"
CROWN_OUTPUT = DATASET / "colab" / "train_crown" / "3_lqmooncrest"
WAND_OUTPUT = DATASET / "colab" / "train_wand" / "3_lqmoonwand"
REVIEW = DATASET / "review"
SIZE = (832, 1216)

BACKGROUND_INDICES = (
    1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 16,
    18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31,
    33, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47,
    49, 50, 51, 52, 53, 55, 56, 61, 62, 64, 66, 67,
)

CROWN_DESCRIPTION = (
    "one exact Moonstar flame-leaf diadem, a slim symmetrical polished gold band, "
    "five upward flame-leaf points, one centered round amber moonstone, two pale "
    "leaf inlays, and two small rose-gold side blossoms"
)
WAND_DESCRIPTION = (
    "one exact signature Rosekeeper wand-staff, one long straight slender gold shaft, "
    "one natural pink rose finial, one small oval rose crystal in the shaft, and one "
    "clear faceted teardrop crystal held in a gold cage at the base"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(index: int, seed_base: int) -> Path:
    return V1_OUTPUT / f"image_{index:03d}_seed_{seed_base + index}.png"


def background_path(index: int) -> Path:
    return BACKGROUNDS / f"image_{index:03d}_seed_{95000 + index}.png"


def extract_cutout(source: Path, mask_path: Path, crown_only: bool = False) -> Image.Image:
    image = Image.open(source).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if crown_only:
        crown = Image.new("L", mask.size, 0)
        crown.paste(mask.crop((0, 0, mask.width, 240)), (0, 0))
        mask = crown
    else:
        import cv2
        import numpy as np

        binary = (np.asarray(mask) > 24).astype(np.uint8)
        count, labels, statistics, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count > 1:
            largest = 1 + int(statistics[1:, cv2.CC_STAT_AREA].argmax())
            mask = Image.fromarray((labels == largest).astype(np.uint8) * 255, mode="L")
    bbox = mask.point(lambda value: 255 if value > 24 else 0).getbbox()
    if bbox is None:
        raise RuntimeError(f"empty mask: {mask_path}")
    left, top, right, bottom = bbox
    pad = 8
    bbox = (
        max(0, left - pad),
        max(0, top - pad),
        min(mask.width, right + pad),
        min(mask.height, bottom + pad),
    )
    cutout = image.crop(bbox)
    alpha = mask.crop(bbox).filter(ImageFilter.MaxFilter(3)).filter(
        ImageFilter.GaussianBlur(0.8)
    )
    cutout.putalpha(alpha)
    return cutout


def transformed(cutout: Image.Image, width: int, angle: float) -> Image.Image:
    height = max(1, round(cutout.height * width / cutout.width))
    resized = cutout.resize((width, height), Image.Resampling.LANCZOS)
    return resized.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )


def muted_reference_background(source: Image.Image, variant: int) -> Image.Image:
    background = source.filter(ImageFilter.GaussianBlur(38)).convert("RGB")
    background = ImageEnhance.Color(background).enhance(0.28)
    background = ImageEnhance.Contrast(background).enhance(0.70)
    colors = ((35, 66, 92), (96, 52, 72), (49, 86, 65), (84, 71, 48))
    tint = Image.new("RGB", SIZE, colors[variant % len(colors)])
    return Image.blend(background, tint, 0.42)


def composite_training_image(
    background: Image.Image,
    object_image: Image.Image,
    position: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    canvas = background.convert("RGBA")
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    layer.alpha_composite(object_image, position)
    composite = Image.alpha_composite(canvas, layer).convert("RGB")
    mask = layer.getchannel("A")
    rgba = composite.convert("RGBA")
    rgba.putalpha(mask)
    return rgba, mask


def fit_position(value: int, object_extent: int, canvas_extent: int) -> int:
    return max(4, min(value, canvas_extent - object_extent - 4))


def prepare_concept(
    name: str,
    trigger: str,
    description: str,
    cutout: Image.Image,
    output: Path,
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.iterdir():
        if stale.is_file():
            stale.unlink()

    records = []
    for offset, background_index in enumerate(BACKGROUND_INDICES):
        with Image.open(background_path(background_index)) as opened:
            source_background = ImageOps.exif_transpose(opened).convert("RGB").resize(SIZE)
        contextual = offset < len(BACKGROUND_INDICES) // 2
        if name == "moonstar_crown":
            if contextual:
                width = 178 + (offset * 29) % 86
                angle = (-7, -4, -2, 0, 2, 4, 7)[offset % 7]
                center_x = 416 + (-54, -32, -14, 8, 28, 48)[offset % 6]
                top = 34 + (offset * 17) % 94
                context = "worn above the Little Queen's forehead in a varied story scene"
                background = source_background
            else:
                width = 240 + (offset * 31) % 118
                angle = (-20, -14, -9, -4, 0, 5, 10, 15, 20)[offset % 9]
                center_x = 180 + (offset * 113) % 472
                top = 130 + (offset * 67) % 710
                context = "displayed alone as a clear royal object reference"
                background = muted_reference_background(source_background, offset)
            placed = transformed(cutout, width, angle)
            x = fit_position(center_x - placed.width // 2, placed.width, SIZE[0])
            y = fit_position(top, placed.height, SIZE[1])
        else:
            if contextual:
                width = 88 + (offset * 17) % 48
                angle = (-18, -12, -7, -3, 3, 7, 12, 18)[offset % 8]
                left_side = offset % 2 == 0
                x = 96 + (offset * 19) % 82 if left_side else 582 + (offset * 17) % 72
                y = 90 + (offset * 29) % 190
                context = "held beside the Little Queen with the complete wand-staff visible"
                background = source_background
            else:
                width = 102 + (offset * 13) % 54
                angle = (-38, -28, -18, -9, 0, 9, 18, 28, 38)[offset % 9]
                x = 90 + (offset * 97) % 520
                y = 70 + (offset * 53) % 300
                context = "displayed alone as a clear magical wand-staff reference"
                background = muted_reference_background(source_background, offset)
            placed = transformed(cutout, width, angle)
            x = fit_position(x, placed.width, SIZE[0])
            y = fit_position(y, placed.height, SIZE[1])

        rgba, mask = composite_training_image(background, placed, (x, y))
        image_path = output / f"{name}_{offset + 1:03d}.png"
        caption_path = image_path.with_suffix(".txt")
        rgba.save(image_path, format="PNG", compress_level=6)
        caption = (
            f"{trigger}, {description}, {context}, exactly one accessory, polished anime "
            "storybook object detail"
        )
        caption_path.write_text(caption + "\n", encoding="utf-8")
        records.append(
            {
                "index": offset + 1,
                "background_index": background_index,
                "contextual": contextual,
                "width": width,
                "angle": angle,
                "position": [x, y],
                "mask_fraction": round(sum(mask.getdata()) / (255 * SIZE[0] * SIZE[1]), 6),
                "image": str(image_path),
                "caption": caption,
            }
        )
    return records


def review_sheet(records: list[dict], destination: Path) -> None:
    thumb_size = (166, 243)
    rows, columns = 6, 8
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 24)), "#202124")
    draw = ImageDraw.Draw(sheet)
    for offset, record in enumerate(records):
        with Image.open(record["image"]) as opened:
            rgb = opened.convert("RGB")
            alpha = opened.getchannel("A")
        overlay = Image.new("RGBA", SIZE, (255, 30, 20, 0))
        overlay.putalpha(alpha.point(lambda value: round(value * 0.42)))
        reviewed = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
        reviewed.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = (offset % columns) * thumb_size[0]
        y = (offset // columns) * (thumb_size[1] + 24)
        sheet.paste(reviewed, (x + (thumb_size[0] - reviewed.width) // 2, y))
        draw.text((x + 5, y + thumb_size[1] + 3), f"#{offset + 1:02d}", fill="white")
    sheet.save(destination, quality=94, subsampling=0)


def main() -> int:
    DATASET.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    crown_source = source_path(91, 99000)
    wand_source = source_path(47, 99000)
    crown_cutout = extract_cutout(
        crown_source,
        V1_DATASET / "canonical_masks" / "moonstar_regalia.png",
        crown_only=True,
    )
    wand_cutout = extract_cutout(
        wand_source,
        V1_DATASET / "canonical_masks" / "rosekeeper_staff.png",
    )
    crown_cutout.save(REVIEW / "canonical_crown.png")
    wand_cutout.save(REVIEW / "canonical_wand.png")
    crown_records = prepare_concept(
        "moonstar_crown", "lqmooncrest", CROWN_DESCRIPTION, crown_cutout, CROWN_OUTPUT
    )
    wand_records = prepare_concept(
        "rosekeeper_wand_staff", "lqrosewand2", WAND_DESCRIPTION, wand_cutout, WAND_OUTPUT
    )
    review_sheet(crown_records, REVIEW / "crown_dataset.jpg")
    review_sheet(wand_records, REVIEW / "wand_dataset.jpg")
    manifest = {
        "version": "lq-accessories-v2",
        "method": "exact canonical cutout across decorrelated contextual and object-reference backgrounds",
        "resolution": list(SIZE),
        "sources": {
            "crown": {"path": str(crown_source), "sha256": sha256(crown_source)},
            "wand": {"path": str(wand_source), "sha256": sha256(wand_source)},
        },
        "crown": {"trigger": "lqmooncrest", "records": crown_records},
        "wand": {"trigger": "lqrosewand2", "records": wand_records},
    }
    (DATASET / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(crown_records)} crown and {len(wand_records)} wand examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
