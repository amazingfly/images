#!/usr/bin/env python3
"""Prepare focused alpha-masked LoRA datasets for canonical Little Queen accessories."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    / "outputs"
    / "equipment_dataset100_v1"
    / "colab_fp16_832x1216_20260720"
    / "output"
)
DATASET_ROOT = ROOT / "training_dataset_accessories_v1"
REGALIA_OUTPUT = DATASET_ROOT / "colab" / "train_regalia" / "8_lqmoonregalia"
EQUIPMENT_OUTPUT = DATASET_ROOT / "colab" / "train_equipment" / "6_lqhandgear"
REVIEW_OUTPUT = DATASET_ROOT / "review"
CANONICAL_MASKS = DATASET_ROOT / "canonical_masks"
SIZE = (832, 1216)


@dataclass(frozen=True)
class CropSpec:
    name: str
    box: tuple[int, int, int, int]
    detail: str


@dataclass(frozen=True)
class Concept:
    trigger: str
    name: str
    source_index: int
    seed: int
    mask_shapes: tuple[tuple[str, tuple], ...]
    crops: tuple[CropSpec, ...]
    description: str


REGALIA = Concept(
    trigger="lqmoonregalia",
    name="moonstar_regalia",
    source_index=91,
    seed=99091,
    mask_shapes=(
        ("polygon", ((205, 4), (590, 4), (678, 265), (590, 328), (245, 315), (168, 220))),
        ("ellipse", (139, 326, 237, 532)),
        ("ellipse", (589, 316, 696, 526)),
        ("polygon", ((278, 430), (558, 430), (602, 580), (250, 580))),
        ("ellipse", (137, 635, 287, 803)),
        ("ellipse", (585, 626, 746, 810)),
    ),
    crops=(
        CropSpec("full", (0, 0, 832, 1216), "full-body view showing the complete worn set"),
        CropSpec("wide", (52, 0, 780, 1064), "wide portrait showing crown, jewelry, and wrists"),
        CropSpec("upper", (112, 0, 720, 889), "upper-body portrait showing crown, earrings, necklace, and bracelets"),
        CropSpec("head", (164, 0, 668, 737), "head-and-shoulders portrait showing crown, earrings, and necklace"),
    ),
    description=(
        "one exact matching Moonstar royal regalia set: a slim symmetrical polished gold "
        "diadem with five upward flame-leaf points, one small round amber moonstone centered "
        "above the brow and two pale leaf inlays, matching long hollow gold teardrop earrings "
        "with amber stones, an ornate close gold collar with one oval rose crystal, and "
        "matching narrow engraved gold wrist cuffs worn over the pink gloves"
    ),
)


EQUIPMENT = (
    Concept(
        trigger="lqmoonwand",
        name="moonstar_wand",
        source_index=73,
        seed=99073,
        mask_shapes=(
            ("ellipse", (38, 224, 348, 568)),
            ("polygon", ((269, 398), (325, 389), (371, 797), (304, 812))),
            ("ellipse", (273, 405, 389, 554)),
        ),
        crops=(
            CropSpec("full", (0, 0, 832, 1216), "full-body view with the wand held upright"),
            CropSpec("wide", (0, 116, 728, 1180), "wide portrait with the entire wand and hand visible"),
            CropSpec("medium", (0, 194, 608, 1083), "medium view emphasizing the wand head, shaft, and grip"),
            CropSpec("detail", (0, 190, 504, 927), "close detail of the wand and the hand holding it"),
        ),
        description=(
            "one exact signature Moonstar wand: a single slender straight gold shaft, a "
            "small ivory grip, one open gold crescent cradle around one rose-pink crystal, "
            "and one small five-point star tip"
        ),
    ),
    Concept(
        trigger="lqrosestaff",
        name="rosekeeper_staff",
        source_index=47,
        seed=99047,
        mask_shapes=(
            ("polygon", ((32, 74), (179, 70), (175, 1078), (50, 1082))),
            ("ellipse", (0, 47, 223, 306)),
            ("ellipse", (0, 873, 240, 1167)),
        ),
        crops=(
            CropSpec("full", (0, 0, 832, 1216), "full-body view with the complete staff visible"),
            CropSpec("wide", (0, 0, 728, 1064), "wide portrait emphasizing the complete staff"),
            CropSpec("medium", (0, 0, 608, 889), "medium view emphasizing the rose finial and gold shaft"),
            CropSpec("detail", (0, 34, 504, 771), "close detail of the rose finial and upper shaft"),
        ),
        description=(
            "one exact signature Rosekeeper staff: one long straight slender gold shaft, "
            "one natural pink rose finial at the top, a small oval rose crystal in the "
            "shaft, and one large clear faceted teardrop crystal in a gold cage at the base"
        ),
    ),
    Concept(
        trigger="lqorbscepter",
        name="celestial_orb_scepter",
        source_index=66,
        seed=99066,
        mask_shapes=(
            ("polygon", ((591, 371), (676, 371), (695, 1021), (621, 1027))),
            ("ellipse", (512, 325, 754, 633)),
        ),
        crops=(
            CropSpec("full", (0, 0, 832, 1216), "full-body view with the complete scepter visible"),
            CropSpec("wide", (104, 104, 832, 1168), "wide portrait emphasizing the complete scepter"),
            CropSpec("medium", (224, 206, 832, 1095), "medium view emphasizing the orb, shaft, and hand"),
            CropSpec("detail", (328, 274, 832, 1011), "close detail of the orb head and upper shaft"),
        ),
        description=(
            "one exact signature Celestial Orb scepter: one straight slender gold shaft, "
            "one large luminous round golden orb head engraved with a crescent, and one "
            "small pointed star finial"
        ),
    ),
)


VARIANTS = (
    (False, 0.96, 1.00, 1.00),
    (False, 1.00, 0.96, 1.04),
    (False, 1.04, 1.04, 0.97),
    (True, 0.98, 1.02, 1.02),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(concept: Concept) -> Path:
    return SOURCE_ROOT / f"image_{concept.source_index:03d}_seed_{concept.seed}.png"


def make_mask(concept: Concept) -> Image.Image:
    segmented = CANONICAL_MASKS / f"{concept.name}.png"
    if segmented.is_file():
        with Image.open(segmented) as opened:
            mask = opened.convert("L")
        if mask.size != SIZE:
            raise RuntimeError(f"unexpected segmented mask size: {segmented}: {mask.size}")
        return mask.filter(ImageFilter.MaxFilter(size=7)).filter(
            ImageFilter.GaussianBlur(radius=3)
        )
    mask = Image.new("L", SIZE, 0)
    draw = ImageDraw.Draw(mask)
    for kind, coordinates in concept.mask_shapes:
        if kind == "ellipse":
            draw.ellipse(coordinates, fill=255)
        elif kind == "polygon":
            draw.polygon(coordinates, fill=255)
        else:
            raise ValueError(f"unsupported mask shape: {kind}")
    return mask.filter(ImageFilter.GaussianBlur(radius=5))


def transform(
    image: Image.Image,
    mask: Image.Image,
    crop: CropSpec,
    variant: tuple[bool, float, float, float],
) -> tuple[Image.Image, Image.Image]:
    flip, brightness, contrast, color = variant
    image = image.crop(crop.box).resize(SIZE, Image.Resampling.LANCZOS)
    mask = mask.crop(crop.box).resize(SIZE, Image.Resampling.LANCZOS)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    if flip:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    return image, mask


def caption(concept: Concept, crop: CropSpec) -> str:
    return (
        f"{concept.trigger}, {concept.description}, {crop.detail}, solo, "
        "lqxl Little Queen, auburn hair, pink royal dress, polished anime "
        "storybook illustration"
    )


def prepare_concept(concept: Concept, output: Path, prefix: str) -> list[dict]:
    source = source_path(concept)
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if image.size != SIZE:
        raise RuntimeError(f"unexpected source size for {source}: {image.size}")
    mask = make_mask(concept)
    records = []
    position = 0
    for crop in concept.crops:
        for variant_index, variant in enumerate(VARIANTS, start=1):
            position += 1
            transformed, transformed_mask = transform(image, mask, crop, variant)
            rgba = transformed.convert("RGBA")
            rgba.putalpha(transformed_mask)
            stem = f"{prefix}_{concept.name}_{position:03d}"
            image_out = output / f"{stem}.png"
            caption_out = output / f"{stem}.txt"
            rgba.save(image_out, format="PNG", compress_level=6)
            text = caption(concept, crop)
            caption_out.write_text(text + "\n", encoding="utf-8")
            records.append(
                {
                    "concept": concept.name,
                    "trigger": concept.trigger,
                    "source": str(source),
                    "source_sha256": sha256(source),
                    "crop": crop.name,
                    "variant": variant_index,
                    "image": str(image_out),
                    "caption": text,
                }
            )
    return records


def review_image(concept: Concept) -> Image.Image:
    source = Image.open(source_path(concept)).convert("RGB")
    mask = make_mask(concept)
    overlay = Image.new("RGBA", SIZE, (255, 0, 0, 0))
    overlay.putalpha(mask.point(lambda value: int(value * 0.48)))
    return Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")


def main() -> int:
    for directory in (REGALIA_OUTPUT, EQUIPMENT_OUTPUT, REVIEW_OUTPUT):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    regalia_records = prepare_concept(REGALIA, REGALIA_OUTPUT, "regalia")
    equipment_records = []
    for concept in EQUIPMENT:
        equipment_records.extend(prepare_concept(concept, EQUIPMENT_OUTPUT, "equipment"))

    all_concepts = (REGALIA, *EQUIPMENT)
    for concept in all_concepts:
        review_image(concept).save(REVIEW_OUTPUT / f"mask_{concept.name}.jpg", quality=94)

    manifest = {
        "version": "lq-accessories-v1",
        "method": "alpha-masked focused loss with deterministic crop/color augmentation",
        "resolution": list(SIZE),
        "regalia": {
            "image_count": len(regalia_records),
            "records": regalia_records,
        },
        "equipment": {
            "image_count": len(equipment_records),
            "concept_counts": {
                concept.name: sum(
                    record["concept"] == concept.name for record in equipment_records
                )
                for concept in EQUIPMENT
            },
            "records": equipment_records,
        },
    }
    manifest_path = DATASET_ROOT / "curated_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(regalia_records)} regalia images in {REGALIA_OUTPUT}")
    print(f"Prepared {len(equipment_records)} equipment images in {EQUIPMENT_OUTPUT}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
