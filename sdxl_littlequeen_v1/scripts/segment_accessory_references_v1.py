#!/usr/bin/env python3
"""Create object-level accessory masks from canonical references with SAM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    / "outputs"
    / "equipment_dataset100_v1"
    / "colab_fp16_832x1216_20260720"
    / "output"
)
OUTPUT = ROOT / "training_dataset_accessories_v1" / "canonical_masks"
REVIEW = ROOT / "training_dataset_accessories_v1" / "review" / "sam"
MODEL_ID = "facebook/sam-vit-base"


@dataclass(frozen=True)
class MaskSpec:
    name: str
    source_index: int
    boxes: tuple[tuple[int, int, int, int], ...]


SPECS = (
    MaskSpec(
        "moonstar_regalia",
        91,
        (
            (286, 28, 586, 220),
            (244, 330, 323, 487),
            (526, 332, 600, 481),
            (301, 444, 543, 610),
            (226, 641, 332, 803),
            (467, 628, 573, 786),
        ),
    ),
    MaskSpec("moonstar_wand", 73, ((88, 322, 318, 543),)),
    MaskSpec("rosekeeper_staff", 47, ((62, 54, 222, 1122),)),
    MaskSpec("celestial_orb_scepter", 66, ((508, 422, 710, 966),)),
)


def source_path(index: int) -> Path:
    return SOURCE_ROOT / f"image_{index:03d}_seed_{99000 + index}.png"


def segment(
    model: SamModel,
    processor: SamProcessor,
    image: Image.Image,
    boxes: tuple[tuple[int, int, int, int], ...],
) -> tuple[Image.Image, list[dict]]:
    inputs = processor(image, input_boxes=[[list(box) for box in boxes]], return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores.cpu()[0]
    combined = np.zeros((image.height, image.width), dtype=np.uint8)
    selections = []
    for box_index, box in enumerate(boxes):
        candidate = int(torch.argmax(scores[box_index]).item())
        score = float(scores[box_index, candidate].item())
        selected = masks[box_index, candidate].numpy().astype(bool)
        box_limited = np.zeros_like(selected)
        left, top, right, bottom = box
        box_limited[top:bottom, left:right] = True
        selected &= box_limited
        combined[selected] = 255
        selections.append(
            {
                "box": list(box),
                "candidate": candidate,
                "predicted_iou": round(score, 6),
                "pixels": int(selected.sum()),
            }
        )
    return Image.fromarray(combined, mode="L"), selections


def review(image: Image.Image, mask: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay.putalpha(mask.point(lambda value: int(value * 0.50)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def refine_regalia(image: Image.Image, mask: Image.Image, boxes: tuple) -> Image.Image:
    """Keep the exact crown and tight context around each small worn item."""
    raw = np.asarray(mask) > 0
    refined = np.zeros_like(raw)
    crown_left, crown_top, crown_right, crown_bottom = boxes[0]
    refined[crown_top:crown_bottom, crown_left:crown_right] = raw[
        crown_top:crown_bottom, crown_left:crown_right
    ]
    localized = Image.fromarray((refined.astype(np.uint8) * 255))
    draw = ImageDraw.Draw(localized)
    draw.ellipse((315, 392, 352, 469), fill=255)
    draw.ellipse((473, 390, 502, 466), fill=255)
    draw.polygon(((330, 449), (493, 449), (491, 575), (333, 575)), fill=255)
    draw.ellipse((294, 646, 350, 744), fill=255)
    draw.ellipse((449, 634, 506, 731), fill=255)
    return localized


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID} on CPU", flush=True)
    processor = SamProcessor.from_pretrained(MODEL_ID)
    model = SamModel.from_pretrained(MODEL_ID).eval()
    records = []
    for spec in SPECS:
        path = source_path(spec.source_index)
        image = Image.open(path).convert("RGB")
        mask, selections = segment(model, processor, image, spec.boxes)
        if spec.name == "moonstar_regalia":
            mask = refine_regalia(image, mask, spec.boxes)
        mask_path = OUTPUT / f"{spec.name}.png"
        mask.save(mask_path)
        review(image, mask).save(REVIEW / f"{spec.name}.jpg", quality=94)
        records.append(
            {
                "name": spec.name,
                "source": str(path),
                "mask": str(mask_path),
                "boxes": [list(box) for box in spec.boxes],
                "selections": selections,
            }
        )
        print(f"Segmented {spec.name}: {sum(item['pixels'] for item in selections)} pixels", flush=True)
    (OUTPUT / "manifest.json").write_text(
        json.dumps({"model": MODEL_ID, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
