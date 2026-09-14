#!/usr/bin/env python3
"""Prepare localized, body-aligned regalia and staff datasets for accessory v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "outputs" / "accessories_v3_bootstrap" / "colab_fp16_832x1216_20260721"
SUMMARY = BOOTSTRAP / "test_summary.json"
SELECTION = ROOT / "training_dataset_accessories_v3" / "selection.json"
DATASET = ROOT / "training_dataset_accessories_v3"
REGALIA_OUTPUT = DATASET / "colab" / "train_regalia" / "1_lqmoonregalia3"
STAFF_OUTPUT = DATASET / "colab" / "train_staff" / "1_lqrosekeeper3"
REVIEW = DATASET / "review"
DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"
SIZE = (832, 1216)

REGALIA_DESCRIPTION = (
    "wearing exactly one Moonstar flame-leaf crown, a symmetrical polished gold diadem "
    "with five to seven pointed leaf flames, one centered amber-rose oval moonstone and "
    "smaller matching side gems; one matched pair of long gold leaf-drop earrings; one "
    "gold filigree collar necklace with a centered amber moonstone; matching engraved gold cuffs"
)
STAFF_DESCRIPTION = (
    "holding exactly one complete signature Rosekeeper wand-staff at human scale, with one "
    "natural pink rose finial, one long straight slender gold shaft, one small rose crystal, "
    "and one clear faceted teardrop crystal in a gold cage at the base"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def local_source(result: dict) -> Path:
    return BOOTSTRAP / "output" / Path(result["output"]).name


def detection_boxes(
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    image: Image.Image,
    query: str,
    threshold: float,
) -> list[dict]:
    inputs = processor(images=image, text=query, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    detected = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=0.12,
        target_sizes=[image.size[::-1]],
    )[0]
    return [
        {
            "score": float(score),
            "box": tuple(float(value) for value in box.tolist()),
            "label": label,
        }
        for score, box, label in zip(
            detected["scores"], detected["boxes"], detected["text_labels"], strict=True
        )
    ]


def box_area(box: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def best_face(detections: list[dict], image: Image.Image) -> tuple[float, float, float, float]:
    candidates = [
        item
        for item in detections
        if 0.001 < box_area(item["box"]) / (image.width * image.height) < 0.20
    ]
    if not candidates:
        raise RuntimeError("no usable face detection")
    return max(candidates, key=lambda item: item["score"])["box"]


def best_crown(
    detections: list[dict], face: tuple[float, float, float, float], image: Image.Image
) -> tuple[float, float, float, float]:
    face_left, face_top, face_right, face_bottom = face
    face_center = (face_left + face_right) / 2
    face_width = face_right - face_left
    candidates = []
    for item in detections:
        left, top, right, bottom = item["box"]
        center = (left + right) / 2
        area_fraction = box_area(item["box"]) / (image.width * image.height)
        if not 0.0005 < area_fraction < 0.12:
            continue
        if bottom > face_bottom + 0.35 * (face_bottom - face_top):
            continue
        distance = abs(center - face_center) / max(face_width, 1)
        candidates.append((item["score"] - 0.14 * distance, item["box"]))
    if not candidates:
        face_height = face_bottom - face_top
        return (
            max(0.0, face_left - face_width * 0.45),
            max(0.0, face_top - face_height * 0.95),
            min(float(image.width), face_right + face_width * 0.45),
            min(float(image.height), face_top + face_height * 0.12),
        )
    return max(candidates, key=lambda item: item[0])[1]


def regalia_mask(
    image: Image.Image,
    face: tuple[float, float, float, float],
    crown: tuple[float, float, float, float],
) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    face_left, face_top, face_right, face_bottom = face
    face_width = face_right - face_left
    face_height = face_bottom - face_top
    crown_left, crown_top, crown_right, crown_bottom = crown
    crown_pad_x = max(8, round((crown_right - crown_left) * 0.10))
    crown_pad_y = max(8, round((crown_bottom - crown_top) * 0.12))
    draw.rounded_rectangle(
        (
            crown_left - crown_pad_x,
            crown_top - crown_pad_y,
            crown_right + crown_pad_x,
            crown_bottom + crown_pad_y,
        ),
        radius=max(6, crown_pad_y),
        fill=255,
    )
    earring_width = max(12, face_width * 0.22)
    earring_top = face_top + face_height * 0.52
    earring_bottom = face_bottom + face_height * 0.38
    draw.ellipse(
        (
            face_left - earring_width * 0.45,
            earring_top,
            face_left + earring_width * 0.75,
            earring_bottom,
        ),
        fill=255,
    )
    draw.ellipse(
        (
            face_right - earring_width * 0.75,
            earring_top,
            face_right + earring_width * 0.45,
            earring_bottom,
        ),
        fill=255,
    )
    draw.rounded_rectangle(
        (
            face_left - face_width * 0.28,
            face_bottom - face_height * 0.08,
            face_right + face_width * 0.28,
            face_bottom + face_height * 0.70,
        ),
        radius=max(8, round(face_width * 0.12)),
        fill=255,
    )
    return mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.2))


def crop_box(face: tuple[float, float, float, float], image: Image.Image) -> tuple[int, int, int, int]:
    left, top, right, bottom = face
    face_width = right - left
    face_height = bottom - top
    crop_width = min(image.width, max(390.0, face_width * 3.6))
    crop_height = min(image.height, crop_width * SIZE[1] / SIZE[0])
    if crop_height == image.height:
        crop_width = min(image.width, crop_height * SIZE[0] / SIZE[1])
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2 + face_height * 0.82
    crop_left = min(max(0.0, center_x - crop_width / 2), image.width - crop_width)
    crop_top = min(max(0.0, center_y - crop_height / 2), image.height - crop_height)
    return tuple(round(value) for value in (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def transform_box(
    box: tuple[float, float, float, float], crop: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    left, top, right, bottom = crop
    scale_x = SIZE[0] / (right - left)
    scale_y = SIZE[1] / (bottom - top)
    return (
        (box[0] - left) * scale_x,
        (box[1] - top) * scale_y,
        (box[2] - left) * scale_x,
        (box[3] - top) * scale_y,
    )


def mask_fraction(mask: Image.Image) -> float:
    return float(np.asarray(mask, dtype=np.float32).sum() / (255 * mask.width * mask.height))


def staff_sam_mask(
    image: Image.Image,
    face: tuple[float, float, float, float],
    detections: list[dict],
    processor: SamProcessor,
    model: SamModel,
) -> tuple[Image.Image, dict]:
    image_area = image.width * image.height
    proposed = []
    for item in detections:
        left, top, right, bottom = item["box"]
        width = right - left
        height = bottom - top
        area_fraction = width * height / image_area
        span = max(width / image.width, height / image.height)
        if 0.01 < area_fraction < 0.60 and span > 0.28:
            proposed.append((item["score"] + 0.22 * span - 0.08 * area_fraction, item))
    if not proposed:
        raise RuntimeError("no usable staff detection")
    face_left, face_top, face_right, face_bottom = (round(value) for value in face)
    choices = []
    for _, detection in sorted(proposed, key=lambda item: item[0], reverse=True)[:5]:
        box = tuple(round(value) for value in detection["box"])
        inputs = processor(image, input_boxes=[[[*box]]], return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]
        scores = outputs.iou_scores.cpu()[0, 0]
        for candidate_index in range(masks.shape[0]):
            binary = masks[candidate_index].numpy().astype(bool)
            limited = np.zeros_like(binary)
            left, top, right, bottom = box
            limited[max(0, top):min(image.height, bottom), max(0, left):min(image.width, right)] = True
            binary &= limited
            pixels = int(binary.sum())
            if not pixels:
                continue
            ys, xs = np.where(binary)
            span = max(
                (xs.max() - xs.min() + 1) / image.width,
                (ys.max() - ys.min() + 1) / image.height,
            )
            fraction = pixels / image_area
            face_overlap = int(binary[face_top:face_bottom, face_left:face_right].sum()) / pixels
            if fraction < 0.003:
                fraction_penalty = 1.5
            elif fraction > 0.14:
                fraction_penalty = 2.0 + 5.0 * (fraction - 0.14)
            else:
                fraction_penalty = 0.0
            choice_score = (
                float(scores[candidate_index])
                + 0.25 * detection["score"]
                + 0.7 * span
                - 2.5 * face_overlap
                - fraction_penalty
            )
            choices.append(
                (
                    choice_score,
                    candidate_index,
                    binary,
                    fraction,
                    span,
                    face_overlap,
                    detection,
                    box,
                )
            )
    if not choices:
        raise RuntimeError("SAM produced no usable staff mask")
    selected = max(choices, key=lambda item: item[0])
    mask = Image.fromarray(selected[2].astype(np.uint8) * 255, mode="L")
    mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.0))
    metadata = {
        "detection_score": round(float(selected[6]["score"]), 6),
        "detection_box": list(selected[7]),
        "sam_candidate": selected[1],
        "mask_fraction": round(mask_fraction(mask), 6),
        "raw_fraction": round(selected[3], 6),
        "span": round(float(selected[4]), 6),
        "face_overlap": round(selected[5], 6),
    }
    return mask, metadata


def save_rgba(image: Image.Image, mask: Image.Image, destination: Path) -> None:
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    rgba.save(destination, format="PNG", compress_level=6)


def make_review(records: list[dict], destination: Path) -> None:
    thumb = (208, 304)
    columns = 6
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb[0], rows * (thumb[1] + 24)), "#202124")
    draw = ImageDraw.Draw(sheet)
    for offset, record in enumerate(records):
        with Image.open(record["image"]) as opened:
            rgb = opened.convert("RGB")
            alpha = opened.getchannel("A")
        overlay = Image.new("RGBA", rgb.size, (255, 20, 20, 0))
        overlay.putalpha(alpha.point(lambda value: round(value * 0.48)))
        reviewed = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
        reviewed.thumbnail(thumb, Image.Resampling.LANCZOS)
        x = (offset % columns) * thumb[0]
        y = (offset // columns) * (thumb[1] + 24)
        sheet.paste(reviewed, (x + (thumb[0] - reviewed.width) // 2, y))
        draw.text((x + 5, y + thumb[1] + 3), record["id"], fill="white")
    sheet.save(destination, quality=94, subsampling=0)


def main() -> int:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    results = {int(item["index"]): item for item in summary["results"]}
    clean_directory(REGALIA_OUTPUT)
    clean_directory(STAFF_OUTPUT)
    REVIEW.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DETECTOR_ID} on CPU", flush=True)
    detector_processor = AutoProcessor.from_pretrained(DETECTOR_ID)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID).eval()
    print(f"Loading {SAM_ID} on CPU", flush=True)
    sam_processor = SamProcessor.from_pretrained(SAM_ID)
    sam = SamModel.from_pretrained(SAM_ID).eval()

    regalia_records = []
    for source_index in selection["regalia_indices"]:
        result = results[int(source_index)]
        source = local_source(result)
        image = Image.open(source).convert("RGB")
        face = best_face(detection_boxes(detector_processor, detector, image, "a face.", 0.20), image)
        crown = best_crown(
            detection_boxes(detector_processor, detector, image, "a gold crown.", 0.16),
            face,
            image,
        )
        mask = regalia_mask(image, face, crown)
        caption = result["prompt"].replace("lqmooncrest", "lqmoonregalia3")
        caption = f"lqmoonregalia3, {REGALIA_DESCRIPTION}, " + caption.replace("lqmoonregalia3,", "", 1).strip()
        base_name = f"regalia_source_{int(source_index):03d}"
        full_path = REGALIA_OUTPUT / f"{base_name}_full.png"
        save_rgba(image, mask, full_path)
        full_path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        regalia_records.append(
            {
                "id": f"r{int(source_index):03d}f",
                "source_index": int(source_index),
                "variant": "full",
                "source": str(source),
                "source_sha256": sha256(source),
                "image": str(full_path),
                "mask_fraction": round(mask_fraction(mask), 6),
                "face_box": [round(value, 2) for value in face],
                "crown_box": [round(value, 2) for value in crown],
                "caption": caption,
            }
        )
        crop = crop_box(face, image)
        cropped_image = image.crop(crop).resize(SIZE, Image.Resampling.LANCZOS)
        cropped_face = transform_box(face, crop)
        cropped_crown = transform_box(crown, crop)
        cropped_mask = regalia_mask(cropped_image, cropped_face, cropped_crown)
        crop_path = REGALIA_OUTPUT / f"{base_name}_crop.png"
        save_rgba(cropped_image, cropped_mask, crop_path)
        crop_path.with_suffix(".txt").write_text(
            caption.replace("full-body", "upper-body detail") + "\n", encoding="utf-8"
        )
        regalia_records.append(
            {
                "id": f"r{int(source_index):03d}c",
                "source_index": int(source_index),
                "variant": "upper_body_crop",
                "source": str(source),
                "source_sha256": sha256(source),
                "image": str(crop_path),
                "mask_fraction": round(mask_fraction(cropped_mask), 6),
                "face_box": [round(value, 2) for value in cropped_face],
                "crown_box": [round(value, 2) for value in cropped_crown],
                "caption": caption,
            }
        )
        print(f"Prepared regalia source {source_index}", flush=True)

    staff_records = []
    for source_index in selection["staff_indices"]:
        result = results[int(source_index)]
        source = local_source(result)
        image = Image.open(source).convert("RGB")
        face = best_face(detection_boxes(detector_processor, detector, image, "a face.", 0.20), image)
        staff_detections = detection_boxes(
            detector_processor, detector, image, "a rose-topped magic wand staff.", 0.10
        )
        mask, metadata = staff_sam_mask(image, face, staff_detections, sam_processor, sam)
        if not 0.002 < metadata["mask_fraction"] < 0.20:
            raise RuntimeError(f"staff mask outside safety range for source {source_index}: {metadata}")
        caption = result["prompt"].replace("lqrosewand2", "lqrosekeeper3")
        caption = f"lqrosekeeper3, {STAFF_DESCRIPTION}, " + caption.replace("lqrosekeeper3,", "", 1).strip()
        destination = STAFF_OUTPUT / f"staff_source_{int(source_index):03d}.png"
        save_rgba(image, mask, destination)
        destination.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        staff_records.append(
            {
                "id": f"s{int(source_index):03d}",
                "source_index": int(source_index),
                "source": str(source),
                "source_sha256": sha256(source),
                "image": str(destination),
                "caption": caption,
                **metadata,
            }
        )
        print(f"Prepared staff source {source_index}: {metadata}", flush=True)

    make_review(regalia_records, REVIEW / "regalia_masks.jpg")
    make_review(staff_records, REVIEW / "staff_masks.jpg")
    manifest = {
        "version": "lq-accessories-v3",
        "method": "curated worn/held contextual examples with localized alpha-masked loss",
        "detector": DETECTOR_ID,
        "segmenter": SAM_ID,
        "selection": selection,
        "regalia": {"trigger": "lqmoonregalia3", "description": REGALIA_DESCRIPTION, "records": regalia_records},
        "staff": {"trigger": "lqrosekeeper3", "description": STAFF_DESCRIPTION, "records": staff_records},
    }
    (DATASET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(regalia_records)} regalia and {len(staff_records)} staff samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
