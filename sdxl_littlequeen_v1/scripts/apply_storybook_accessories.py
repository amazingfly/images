#!/usr/bin/env python3
"""Place canonical Little Queen accessories and emit localized blend masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET = ROOT / "storybook_accessories_v1" / "accessory_set.json"
DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--set", dest="set_path", type=Path, default=DEFAULT_SET)
    parser.add_argument("--staff", action="store_true", help="Place the canonical staff.")
    parser.add_argument("--no-crown", action="store_true")
    parser.add_argument("--no-jewelry", action="store_true")
    parser.add_argument(
        "--face-box",
        nargs=4,
        type=float,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Normalized face box override.",
    )
    parser.add_argument(
        "--staff-anchor",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Normalized hand/grip anchor override.",
    )
    parser.add_argument(
        "--staff-hand",
        choices=("auto", "leftmost", "rightmost"),
        default="auto",
    )
    parser.add_argument("--staff-angle", type=float)
    parser.add_argument("--crown-scale", type=float, default=1.0)
    parser.add_argument("--staff-scale", type=float, default=1.0)
    parser.add_argument(
        "--staff-anchor-outset",
        type=float,
        help="Override horizontal hand-anchor offset in person widths.",
    )
    parser.add_argument(
        "--remove-existing-crown",
        action="store_true",
        help="Locally inpaint a detected generated crown before placing the canonical crown.",
    )
    parser.add_argument(
        "--existing-crown-min-score",
        type=float,
        default=0.4,
        help="Minimum detector score required before automatic crown removal.",
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--blend-mask", type=Path)
    parser.add_argument("--cleanup-mask", type=Path)
    parser.add_argument(
        "--regalia-mask",
        type=Path,
        help="Write a precise full-accessory mask for crown and jewelry pixels.",
    )
    parser.add_argument(
        "--hand-prop-mask",
        type=Path,
        help="Write a precise visible-pixel mask for the wand or staff.",
    )
    parser.add_argument(
        "--hand-prop-full-mask",
        type=Path,
        help="Write the full wand/staff mask before hand occlusion.",
    )
    parser.add_argument(
        "--hand-prop-visible-mask",
        type=Path,
        help="Write only wand/staff pixels that remain visible around the hand.",
    )
    parser.add_argument(
        "--hand-occlusion-mask",
        type=Path,
        help="Write the detected hand mask composited in front of the wand/staff.",
    )
    parser.add_argument(
        "--hand-grip-mask",
        type=Path,
        help="Write a tight hand-and-shaft mask for a final grip-only refinement.",
    )
    parser.add_argument(
        "--no-person-occlusion",
        action="store_true",
        help="Draw the staff above the character instead of behind her silhouette.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_area(box: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def contains(outer: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    return outer[0] <= point[0] <= outer[2] and outer[1] <= point[1] <= outer[3]


@lru_cache(maxsize=1)
def detector_components() -> tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    processor = AutoProcessor.from_pretrained(DETECTOR_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID)
        .to(device)
        .eval()
    )
    return processor, model


@lru_cache(maxsize=1)
def sam_components() -> tuple[SamProcessor, SamModel]:
    processor = SamProcessor.from_pretrained(SAM_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained(SAM_ID).to(device).eval()
    return processor, model


def clear_model_caches() -> None:
    detector_components.cache_clear()
    sam_components.cache_clear()


def detect(image: Image.Image) -> list[dict]:
    processor, model = detector_components()
    inputs = processor(
        images=image,
        text=(
            "face. person. hand. crown. hair ornament. flower in hair. "
            "wooden rod. staff. wand. sword. earring. necklace."
        ),
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.20,
        text_threshold=0.12,
        target_sizes=[image.size[::-1]],
    )[0]
    return [
        {
            "label": str(label),
            "score": float(score),
            "box": tuple(float(value) for value in box.tolist()),
        }
        for score, box, label in zip(
            result["scores"], result["boxes"], result["text_labels"], strict=True
        )
    ]


def choose_person(detections: list[dict], image: Image.Image) -> tuple[float, float, float, float]:
    image_area = image.width * image.height
    candidates = [
        item
        for item in detections
        if "person" in item["label"]
        and 0.04 < box_area(item["box"]) / image_area < 0.90
    ]
    if not candidates:
        raise RuntimeError("no usable person detection")
    return max(candidates, key=lambda item: item["score"] + box_area(item["box"]) / image_area)["box"]


def choose_face(
    detections: list[dict],
    person: tuple[float, float, float, float],
    image: Image.Image,
) -> tuple[float, float, float, float]:
    image_area = image.width * image.height
    candidates = [
        item
        for item in detections
        if "face" in item["label"]
        and 0.001 < box_area(item["box"]) / image_area < 0.20
        and contains(person, center(item["box"]))
    ]
    if not candidates:
        raise RuntimeError("no usable face detection; pass --face-box")
    return max(candidates, key=lambda item: item["score"])["box"]


def estimate_hand(person: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    p_left, p_top, p_right, p_bottom = person
    p_width = p_right - p_left
    p_height = p_bottom - p_top
    center_x = (p_left + p_right) / 2
    hand_x = center_x + p_width * 0.22
    hand_y = p_top + p_height * 0.55
    hw = p_width * 0.10
    hh = p_height * 0.08
    return (hand_x - hw / 2, hand_y - hh / 2, hand_x + hw / 2, hand_y + hh / 2)


def choose_hand(
    detections: list[dict],
    person: tuple[float, float, float, float],
    mode: str,
) -> tuple[tuple[float, float, float, float], str]:
    candidates = [
        item
        for item in detections
        if "hand" in item["label"] and contains(person, center(item["box"]))
    ]
    if not candidates:
        return estimate_hand(person), "estimated"
    person_center_x = center(person)[0]
    if mode == "leftmost":
        return min(candidates, key=lambda item: center(item["box"])[0])["box"], "detected"
    if mode == "rightmost":
        return max(candidates, key=lambda item: center(item["box"])[0])["box"], "detected"
    return (
        max(
            candidates,
            key=lambda item: abs(center(item["box"])[0] - person_center_x)
            + 30 * item["score"],
        )["box"],
        "detected",
    )



def choose_existing_crown(
    detections: list[dict],
    face: tuple[float, float, float, float] | None,
    person: tuple[float, float, float, float],
    image: Image.Image,
) -> tuple[float, float, float, float] | None:
    reference_center_x = center(face if face else person)[0]
    reference_width = (face[2] - face[0]) if face else (person[2] - person[0])
    face_bottom = face[3] if face else person[1] + (person[3] - person[1]) * 0.42
    face_top = face[1] if face else person[1] + (person[3] - person[1]) * 0.08
    face_height = (face[3] - face[1]) if face else (person[3] - person[1]) * 0.18
    candidates = []
    for item in detections:
        if "crown" not in item["label"]:
            continue
        box = item["box"]
        box_center_x = center(box)[0]
        if box[3] > face_bottom or box[1] > face_top + face_height * 0.25:
            continue
        distance = abs(box_center_x - reference_center_x) / max(reference_width, 1)
        area_fraction = box_area(box) / (image.width * image.height)
        if area_fraction > 0.18:
            continue
        candidates.append((item["score"] - 0.12 * distance, box))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def estimate_face(
    person: tuple[float, float, float, float],
    crown: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    person_left, person_top, person_right, person_bottom = person
    person_width = person_right - person_left
    person_height = person_bottom - person_top
    face_width = person_width * 0.58
    face_height = person_height * 0.20
    center_x = center(crown)[0] if crown else center(person)[0]
    face_top = (
        crown[3] + face_height * 0.45
        if crown
        else person_top + person_height * 0.18
    )
    return (
        center_x - face_width / 2,
        face_top,
        center_x + face_width / 2,
        face_top + face_height,
    )


def sam_box_mask(
    image: Image.Image,
    box: tuple[float, float, float, float],
    feather: float = 0.0,
) -> Image.Image:
    processor, model = sam_components()
    rounded_box = [round(value) for value in box]
    inputs = processor(
        image, input_boxes=[[[*rounded_box]]], return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]
    scores = outputs.iou_scores.cpu()[0, 0]
    best = int(torch.argmax(scores))
    array = (masks[best].numpy() > 0).astype(np.uint8) * 255
    result = Image.fromarray(array)
    return result.filter(ImageFilter.GaussianBlur(feather)) if feather else result


def remove_crown(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Replace only the old crown silhouette with nearby hair/background color."""
    rgb = np.asarray(image.convert("RGB"))
    binary_mask = (np.asarray(mask) > 127).astype(np.uint8) * 255
    cleaned = cv2.inpaint(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        binary_mask,
        3,
        cv2.INPAINT_TELEA,
    )
    return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))


def constrain_mask_to_box(
    mask: Image.Image,
    box: tuple[float, float, float, float],
    expansion: float = 0.3,
) -> Image.Image:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    bounds = (
        max(0, round(left - width * expansion)),
        max(0, round(top - height * expansion)),
        min(mask.width, round(right + width * expansion)),
        min(mask.height, round(bottom + height * expansion)),
    )
    allowed = Image.new("L", mask.size, 0)
    ImageDraw.Draw(allowed).rectangle(bounds, fill=255)
    return ImageChops.multiply(mask, allowed)


def load_asset(base: Path, relative: str) -> tuple[Path, Image.Image]:
    path = (base / relative).resolve()
    image = Image.open(path).convert("RGBA")
    alpha = image.getchannel("A")
    if alpha.getextrema()[0] != 0:
        raise RuntimeError(f"asset lacks transparent background: {path}")
    return path, image


def scaled(asset: Image.Image, target_width: float | None = None, target_height: float | None = None) -> Image.Image:
    if (target_width is None) == (target_height is None):
        raise ValueError("provide exactly one target dimension")
    ratio = asset.width / asset.height
    if target_width is not None:
        size = (max(1, round(target_width)), max(1, round(target_width / ratio)))
    else:
        size = (max(1, round(target_height * ratio)), max(1, round(target_height)))
    return asset.resize(size, Image.Resampling.LANCZOS)


def paste_at(layer: Image.Image, asset: Image.Image, left: float, top: float) -> tuple[int, int, int, int]:
    x = round(left)
    y = round(top)
    layer.alpha_composite(asset, dest=(x, y))
    return (x, y, x + asset.width, y + asset.height)


def paste_asset_alpha(mask: Image.Image, asset: Image.Image, left: int, top: int) -> None:
    alpha = asset.getchannel("A")
    mask.paste(ImageChops.lighter(mask.crop((left, top, left + asset.width, top + asset.height)), alpha), (left, top))


def mark_attachment(mask: Image.Image, box: tuple[int, int, int, int], kind: str) -> None:
    draw = ImageDraw.Draw(mask)
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    if kind == "crown":
        draw.rounded_rectangle(
            (left, bottom - max(8, height * 0.17), right, bottom + max(4, height * 0.05)),
            radius=max(4, round(height * 0.06)),
            fill=255,
        )
    elif kind == "earring":
        radius = max(4, round(height * 0.13))
        draw.ellipse((left - radius, top - radius, left + radius, top + radius), fill=255)
    elif kind == "necklace":
        draw.rounded_rectangle(
            (left, top - max(3, height * 0.10), right, top + max(5, height * 0.20)),
            radius=max(3, round(height * 0.08)),
            fill=255,
        )


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    set_path = args.set_path.resolve()
    config = json.loads(set_path.read_text(encoding="utf-8"))
    assets_base = set_path.parent
    image = Image.open(args.input).convert("RGB")
    original = image.convert("RGBA")
    detections = detect(image)
    person = choose_person(detections, image)
    face_source = "detected"
    if args.face_box:
        face = (
            args.face_box[0] * image.width,
            args.face_box[1] * image.height,
            args.face_box[2] * image.width,
            args.face_box[3] * image.height,
        )
        face_source = "manual"
    else:
        try:
            face = choose_face(detections, person, image)
        except RuntimeError:
            preliminary_crown = choose_existing_crown(detections, None, person, image)
            face = estimate_face(person, preliminary_crown)
            face_source = "estimated_from_crown" if preliminary_crown else "estimated_from_person"
    existing_crown = choose_existing_crown(detections, face, person, image)
    existing_crown_score = next(
        (
            float(item["score"])
            for item in detections
            if existing_crown is not None
            and "crown" in item["label"]
            and item["box"] == existing_crown
        ),
        None,
    )
    remove_existing_crown = bool(
        existing_crown
        and existing_crown_score is not None
        and existing_crown_score
        >= getattr(args, "existing_crown_min_score", 0.4)
        and args.remove_existing_crown
        and not args.no_crown
    )
    inherited_accessories = [
        item
        for item in detections
        if any(
            label in item["label"]
            for label in (
                "crown",
                "hair ornament",
                "flower in hair",
                "wooden rod",
                "staff",
                "wand",
                "sword",
                "earring",
                "necklace",
            )
        )
    ]

    geometry = config["geometry"]
    asset_records: dict[str, dict] = {}
    placements: dict[str, object] = {}
    under = original.copy()
    blend_mask = Image.new("L", image.size, 0)
    cleanup_mask = Image.new("L", image.size, 0)
    regalia_mask = Image.new("L", image.size, 0)
    hand_prop_mask = Image.new("L", image.size, 0)
    hand_prop_full_mask = Image.new("L", image.size, 0)
    hand_prop_visible_mask = Image.new("L", image.size, 0)
    hand_occlusion_mask = Image.new("L", image.size, 0)
    hand_grip_mask = Image.new("L", image.size, 0)
    if existing_crown:
        cleanup_mask = sam_box_mask(image, existing_crown)
        cleanup_mask = cleanup_mask.filter(ImageFilter.MaxFilter(5))
        if remove_existing_crown:
            original = remove_crown(image, cleanup_mask).convert("RGBA")
            under = original.copy()
        cleanup_mask = cleanup_mask.filter(ImageFilter.GaussianBlur(1.2))

    if args.staff:
        staff_path, staff_asset = load_asset(assets_base, config["assets"]["staff"])
        person_height = person[3] - person[1]
        if args.staff_anchor:
            anchor = (args.staff_anchor[0] * image.width, args.staff_anchor[1] * image.height)
            hand_box = None
            hand_source = "manual"
        else:
            hand_box, hand_source = choose_hand(detections, person, args.staff_hand)
            anchor = center(hand_box)
        person_center_x = center(person)[0]
        outset_fraction = (
            args.staff_anchor_outset
            if getattr(args, "staff_anchor_outset", None) is not None
            else geometry.get("staff_anchor_outset_in_person_widths", 0.0)
        )
        outset = outset_fraction * (person[2] - person[0])
        if outset:
            direction = -1.0 if anchor[0] < person_center_x else 1.0
            anchor = (anchor[0] + direction * outset, anchor[1])
        staff_angle = (
            args.staff_angle
            if args.staff_angle is not None
            else float(geometry.get("staff_angle_degrees", 6.0))
        )
        if geometry.get("staff_angle_outward", False):
            staff_angle = abs(staff_angle) if anchor[0] < person_center_x else -abs(staff_angle)
        grip_fraction = geometry["staff_grip_fraction_from_top"]
        target_height = person_height * geometry["staff_height_in_person_heights"] * args.staff_scale
        available_height = min(
            image.height * 0.96,
            anchor[1] / max(grip_fraction, 0.01),
            (image.height - anchor[1]) / max(1.0 - grip_fraction, 0.01),
        )
        staff_asset = scaled(staff_asset, target_height=min(target_height, available_height))
        if staff_angle:
            staff_asset = staff_asset.rotate(
                staff_angle,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
        if staff_asset.height > available_height:
            staff_asset = scaled(staff_asset, target_height=available_height)
        left = anchor[0] - staff_asset.width / 2
        top = anchor[1] - staff_asset.height * grip_fraction
        left = min(max(-staff_asset.width * 0.15, left), image.width - staff_asset.width * 0.85)
        top = min(max(0.0, top), image.height - staff_asset.height)
        staff_box = paste_at(under, staff_asset, left, top)
        paste_asset_alpha(hand_prop_mask, staff_asset, staff_box[0], staff_box[1])
        paste_asset_alpha(hand_prop_full_mask, staff_asset, staff_box[0], staff_box[1])
        grip_radius = max(10, round((person[2] - person[0]) * 0.055))
        grip = ImageDraw.Draw(blend_mask)
        grip.ellipse(
            (
                anchor[0] - grip_radius,
                anchor[1] - grip_radius,
                anchor[0] + grip_radius,
                anchor[1] + grip_radius,
            ),
            fill=255,
        )
        placements["staff"] = {
            "box": staff_box,
            "grip_anchor": [round(anchor[0], 2), round(anchor[1], 2)],
            "detected_hand_box": hand_box,
            "hand_source": hand_source,
            "anchor_outset_in_person_widths": outset_fraction,
            "angle": staff_angle,
        }
        asset_records["staff"] = {"path": str(staff_path), "sha256": sha256(staff_path)}

    if args.staff and not args.no_person_occlusion:
        if hand_box is not None:
            hand_occlusion_mask = sam_box_mask(image, hand_box, feather=0.8)
            hand_occlusion_mask = constrain_mask_to_box(
                hand_occlusion_mask, hand_box
            )
        else:
            hand_occlusion_mask = sam_box_mask(image, person, feather=0.8)
        under = Image.composite(original, under, hand_occlusion_mask)
        hand_prop_mask = ImageChops.multiply(
            hand_prop_mask, ImageOps.invert(hand_occlusion_mask)
        )

    face_left, face_top, face_right, face_bottom = face
    face_width = face_right - face_left
    face_height = face_bottom - face_top
    face_center_x = (face_left + face_right) / 2
    person_width = person[2] - person[0]

    if not args.no_crown:
        crown_path, crown_asset = load_asset(assets_base, config["assets"]["crown"])
        crown_width = min(
            face_width * geometry["crown_width_in_face_widths"] * args.crown_scale,
            person_width * geometry["crown_max_width_in_person_widths"],
            image.width * geometry["crown_max_width_in_image_widths"],
        )
        crown_asset = scaled(crown_asset, target_width=crown_width)
        crown_bottom = face_top + face_height * geometry["crown_band_below_face_top"]
        crown_box = paste_at(
            under,
            crown_asset,
            face_center_x - crown_asset.width / 2,
            crown_bottom - crown_asset.height,
        )
        paste_asset_alpha(regalia_mask, crown_asset, crown_box[0], crown_box[1])
        mark_attachment(blend_mask, crown_box, "crown")
        placements["crown"] = {"box": crown_box}
        asset_records["crown"] = {"path": str(crown_path), "sha256": sha256(crown_path)}

    if not args.no_jewelry:
        earring_path, earring_asset = load_asset(assets_base, config["assets"]["earring"])
        earring_height = face_height * geometry["earring_height_in_face_heights"]
        right_earring = scaled(earring_asset, target_height=earring_height)
        left_earring = ImageOps.mirror(right_earring)
        earring_top = face_top + face_height * geometry["earring_top_below_face_top"]
        outset = face_width * geometry["earring_outset_in_face_widths"]
        left_box = paste_at(
            under,
            left_earring,
            face_left - outset - left_earring.width / 2,
            earring_top,
        )
        right_box = paste_at(
            under,
            right_earring,
            face_right + outset - right_earring.width / 2,
            earring_top,
        )
        paste_asset_alpha(regalia_mask, left_earring, left_box[0], left_box[1])
        paste_asset_alpha(regalia_mask, right_earring, right_box[0], right_box[1])
        mark_attachment(blend_mask, left_box, "earring")
        mark_attachment(blend_mask, right_box, "earring")
        placements["earrings"] = {"left_box": left_box, "right_box": right_box}
        asset_records["earring"] = {"path": str(earring_path), "sha256": sha256(earring_path)}

        necklace_path, necklace_asset = load_asset(assets_base, config["assets"]["necklace"])
        necklace_width = min(
            face_width * geometry["necklace_width_in_face_widths"],
            person_width * geometry["necklace_max_width_in_person_widths"],
        )
        necklace_asset = scaled(necklace_asset, target_width=necklace_width)
        necklace_top = face_bottom + face_height * geometry["necklace_top_below_face_bottom"]
        necklace_box = paste_at(
            under,
            necklace_asset,
            face_center_x - necklace_asset.width / 2,
            necklace_top,
        )
        paste_asset_alpha(regalia_mask, necklace_asset, necklace_box[0], necklace_box[1])
        mark_attachment(blend_mask, necklace_box, "necklace")
        placements["necklace"] = {"box": necklace_box}
        asset_records["necklace"] = {"path": str(necklace_path), "sha256": sha256(necklace_path)}

    composited_rgb = under.convert("RGB")
    if remove_existing_crown:
        regalia_mask = ImageChops.lighter(regalia_mask, cleanup_mask)
    difference = np.asarray(ImageChops.difference(composited_rgb, image), dtype=np.uint8)
    changed = Image.fromarray((difference.max(axis=2) >= 3).astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(3)
    )
    regalia_mask = ImageChops.multiply(regalia_mask, changed)
    hand_prop_visible_mask = ImageChops.multiply(hand_prop_mask, changed)
    hand_prop_full_mask = ImageChops.multiply(
        hand_prop_full_mask,
        ImageChops.lighter(changed, hand_occlusion_mask),
    )
    hand_prop_mask = ImageChops.lighter(
        hand_prop_visible_mask,
        hand_occlusion_mask.filter(ImageFilter.MaxFilter(5)),
    )
    if args.staff and hand_box is not None:
        hand_width = float(hand_box[2]) - float(hand_box[0])
        hand_height = float(hand_box[3]) - float(hand_box[1])
        grip_region = Image.new("L", image.size, 0)
        grip_left = round(float(hand_box[0]) - hand_width * 0.65)
        grip_top = round(float(hand_box[1]) - hand_height * 0.45)
        grip_right = round(float(hand_box[2]) + hand_width * 0.65)
        grip_bottom = round(float(hand_box[3]) + hand_height * 0.45)
        ImageDraw.Draw(grip_region).rounded_rectangle(
            (
                grip_left,
                grip_top,
                grip_right,
                grip_bottom,
            ),
            radius=max(4, round(min(hand_width, hand_height) * 0.4)),
            fill=255,
        )
        hand_grip_mask = grip_region.filter(ImageFilter.GaussianBlur(1.0))
    blend_mask = blend_mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(2.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    composited_rgb.save(args.output, format="PNG", compress_level=6)
    blend_path = args.blend_mask or args.output.with_name(args.output.stem + "_blend_mask.png")
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    blend_mask.save(blend_path, format="PNG")
    cleanup_path = args.cleanup_mask or args.output.with_name(args.output.stem + "_cleanup_mask.png")
    cleanup_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_mask.save(cleanup_path, format="PNG")
    regalia_mask = regalia_mask.filter(ImageFilter.MaxFilter(5)).filter(
        ImageFilter.GaussianBlur(0.8)
    )
    hand_prop_mask = hand_prop_mask.filter(ImageFilter.MaxFilter(5)).filter(
        ImageFilter.GaussianBlur(0.8)
    )
    regalia_path = args.regalia_mask or args.output.with_name(
        args.output.stem + "_regalia_mask.png"
    )
    hand_prop_path = args.hand_prop_mask or args.output.with_name(
        args.output.stem + "_hand_prop_mask.png"
    )
    hand_prop_full_path = getattr(args, "hand_prop_full_mask", None) or args.output.with_name(
        args.output.stem + "_hand_prop_full_mask.png"
    )
    hand_prop_visible_path = getattr(
        args, "hand_prop_visible_mask", None
    ) or args.output.with_name(args.output.stem + "_hand_prop_visible_mask.png")
    hand_occlusion_path = getattr(args, "hand_occlusion_mask", None) or args.output.with_name(
        args.output.stem + "_hand_occlusion_mask.png"
    )
    hand_grip_path = getattr(args, "hand_grip_mask", None) or args.output.with_name(
        args.output.stem + "_hand_grip_mask.png"
    )
    regalia_path.parent.mkdir(parents=True, exist_ok=True)
    hand_prop_path.parent.mkdir(parents=True, exist_ok=True)
    regalia_mask.save(regalia_path, format="PNG")
    hand_prop_mask.save(hand_prop_path, format="PNG")
    hand_prop_full_mask.save(hand_prop_full_path, format="PNG")
    hand_prop_visible_mask.save(hand_prop_visible_path, format="PNG")
    hand_occlusion_mask.save(hand_occlusion_path, format="PNG")
    hand_grip_mask.save(hand_grip_path, format="PNG")
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": config["version"],
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "blend_mask": str(blend_path.resolve()),
        "cleanup_mask": str(cleanup_path.resolve()),
        "regalia_mask": str(regalia_path.resolve()),
        "hand_prop_mask": str(hand_prop_path.resolve()),
        "hand_prop_full_mask": str(hand_prop_full_path.resolve()),
        "hand_prop_visible_mask": str(hand_prop_visible_path.resolve()),
        "hand_occlusion_mask": str(hand_occlusion_path.resolve()),
        "hand_grip_mask": str(hand_grip_path.resolve()),
        "image_size": list(image.size),
        "person_box": person,
        "face_box": face,
        "face_source": face_source,
        "existing_crown_box": existing_crown,
        "existing_crown_score": existing_crown_score,
        "inherited_accessory_detections": inherited_accessories,
        "existing_crown_removed": remove_existing_crown,
        "placements": placements,
        "assets": asset_records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not getattr(args, "quiet", False):
        print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
