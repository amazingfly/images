#!/usr/bin/env python3
"""Validate automatic Little Queen accessory placement and exact-mask refinement."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


VALIDATOR_VERSION = "littlequeen-accessory-validator-v4.7"
GRIP_MODEL_ID = "openai/clip-vit-base-patch32"
GRIP_TEXTS = (
    "a small hand tightly closed around a vertical magic wand",
    "visible fingers firmly wrapped around the shaft of a wand",
    "an open palm beside a vertical magic wand",
    "a magic wand floating next to a hand",
    "a malformed hand merely touching a wand",
)
GRIP_POSITIVE_TEXT_COUNT = 2


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def grip_crop(image_path: Path, metadata_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hand_box = metadata["placements"]["staff"].get("detected_hand_box")
    if not hand_box:
        raise RuntimeError(f"no detected hand box in {metadata_path}")
    left, top, right, bottom = (float(value) for value in hand_box)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    side = max(right - left, bottom - top) * 4.2
    side = min(max(side, 144.0), min(image.size) * 0.42)
    crop_box = (
        max(0, round(center_x - side / 2)),
        max(0, round(center_y - side / 2)),
        min(image.width, round(center_x + side / 2)),
        min(image.height, round(center_y + side / 2)),
    )
    return image.crop(crop_box)


@lru_cache(maxsize=1)
def grip_clip_components():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(GRIP_MODEL_ID)
    model = CLIPModel.from_pretrained(GRIP_MODEL_ID).to("cpu").eval()
    return processor, model, torch


def score_grip_semantics(
    items: list[tuple[str, Path, Path]],
) -> dict[str, dict[str, Any]]:
    if not items:
        return {}
    processor, model, torch = grip_clip_components()
    crops = [grip_crop(image_path, metadata_path) for _, image_path, metadata_path in items]
    inputs = processor(
        text=list(GRIP_TEXTS),
        images=crops,
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        probabilities = model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()
    results = {}
    for (item_id, _, _), row in zip(items, probabilities, strict=True):
        positive = float(row[:GRIP_POSITIVE_TEXT_COUNT].sum())
        results[item_id] = {
            "positive_probability": round(positive, 6),
            "text_probabilities": {
                text: round(float(probability), 6)
                for text, probability in zip(GRIP_TEXTS, row, strict=True)
            },
        }
    return results


def apply_grip_semantic_validation(
    run_root: Path,
    records: list[dict[str, Any]],
    threshold: float,
) -> None:
    items = []
    for record in records:
        artifacts = record.get("artifacts", {})
        if "final" not in artifacts or "metadata" not in artifacts:
            continue
        items.append(
            (
                record["id"],
                run_root / artifacts["final"],
                run_root / artifacts["metadata"],
            )
        )
    scores = score_grip_semantics(items)
    for record in records:
        semantic = scores.get(record["id"])
        if semantic is None:
            continue
        validation = record["validation"]
        validation["metrics"]["grip_semantic"] = semantic
        if semantic["positive_probability"] < threshold:
            validation["failures"].append("semantic grip check found an open or floating wand")
        validation["failures"] = list(dict.fromkeys(validation["failures"]))
        validation["accepted"] = not validation["failures"]
        validation["score"] = max(
            0.0,
            round(
                100.0
                - 22.0 * len(validation["failures"])
                - 4.0 * len(validation["advisories"]),
                1,
            ),
        )


def mask_metrics(mask: np.ndarray) -> dict[str, Any]:
    binary = (mask >= 8).astype(np.uint8)
    pixels = int(binary.sum())
    height, width = binary.shape
    if not pixels:
        return {
            "pixels": 0,
            "fraction": 0.0,
            "bbox": None,
            "component_count": 0,
            "largest_component_fraction": 0.0,
        }
    ys, xs = np.where(binary)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    return {
        "pixels": pixels,
        "fraction": round(pixels / (width * height), 6),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
        "component_count": len(areas),
        "largest_component_fraction": round(max(areas) / pixels, 6),
    }


def hand_shape_metrics(
    mask: np.ndarray,
    hand_box: list[float] | tuple[float, float, float, float],
) -> dict[str, float]:
    height, width = mask.shape
    left = max(0, int(math.floor(float(hand_box[0]))))
    top = max(0, int(math.floor(float(hand_box[1]))))
    right = min(width, int(math.ceil(float(hand_box[2]))))
    bottom = min(height, int(math.ceil(float(hand_box[3]))))
    binary = (mask[top:bottom, left:right] >= 8).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "bbox_fill": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "aspect_ratio": 0.0,
        }
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    x, y, contour_width, contour_height = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    perimeter = float(cv2.arcLength(contour, closed=True))
    return {
        "bbox_fill": round(
            area / max(float(contour_width * contour_height), 1.0),
            6,
        ),
        "solidity": round(area / max(hull_area, 1.0), 6),
        "circularity": round(
            4.0 * math.pi * area / max(perimeter * perimeter, 1.0),
            6,
        ),
        "aspect_ratio": round(
            min(contour_width, contour_height) / max(contour_width, contour_height),
            6,
        ),
    }


def minimum_mask_distance(mask: np.ndarray, point: tuple[float, float]) -> float:
    ys, xs = np.where(mask >= 8)
    if not len(xs):
        return float("inf")
    squared = (xs.astype(np.float32) - point[0]) ** 2 + (
        ys.astype(np.float32) - point[1]
    ) ** 2
    return float(math.sqrt(float(squared.min())))


def box_inside(
    box: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
    padding: int = 2,
) -> bool:
    left, top, right, bottom = box
    return (
        left >= padding
        and top >= padding
        and right <= width - padding
        and bottom <= height - padding
    )


def validate_preflight(
    image_path: Path,
    regalia_mask_path: Path,
    wand_mask_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    image = load_rgb(image_path)
    height, width = image.shape[:2]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    regalia = load_mask(regalia_mask_path)
    wand_refinement = load_mask(wand_mask_path)
    visible_wand_path = wand_mask_path.with_name(
        wand_mask_path.name.replace("_wand_mask", "_wand_visible_mask")
    )
    wand = (
        load_mask(visible_wand_path)
        if visible_wand_path.is_file()
        else wand_refinement
    )
    regalia_stats = mask_metrics(regalia)
    wand_stats = mask_metrics(wand)
    wand_refinement_stats = mask_metrics(wand_refinement)
    failures: list[str] = []
    advisories: list[str] = []

    person = [float(value) for value in metadata["person_box"]]
    face = [float(value) for value in metadata["face_box"]]
    person_width = person[2] - person[0]
    person_height = person[3] - person[1]
    face_width = face[2] - face[0]
    face_height = face[3] - face[1]
    person_height_fraction = person_height / height
    face_width_fraction = face_width / width

    placements = metadata["placements"]
    crown_box = placements.get("crown", {}).get("box")
    wand_placement = placements.get("staff", {})
    wand_box = wand_placement.get("box")
    grip = tuple(float(value) for value in wand_placement.get("grip_anchor", (0, 0)))
    hand_box = wand_placement.get("detected_hand_box")
    hand_source = wand_placement.get("hand_source", "unknown")
    full_wand_path = wand_mask_path.with_name(
        wand_mask_path.name.replace("_wand_mask", "_wand_full_mask")
    )
    hand_mask_path = wand_mask_path.with_name(
        wand_mask_path.name.replace("_wand_mask", "_hand_occlusion_mask")
    )
    full_wand = load_mask(full_wand_path) if full_wand_path.is_file() else None
    hand_mask = load_mask(hand_mask_path) if hand_mask_path.is_file() else None
    full_wand_stats = mask_metrics(full_wand) if full_wand is not None else None

    if not 0.45 <= person_height_fraction <= 0.99:
        failures.append("person framing is not a usable full-body scale")
    if person[1] < -0.01 * height or person[3] > 1.015 * height:
        failures.append("person is clipped by the image boundary")
    if face_width_fraction < 0.055 or face_height < 48:
        failures.append("face is too small for stable crown placement")
    if metadata.get("face_source") == "estimated_from_person":
        advisories.append("face box was estimated from the person")

    if not crown_box or not box_inside(crown_box, width, height):
        failures.append("canonical crown is clipped")
        crown_width_ratio = 0.0
        crown_center_offset = float("inf")
    else:
        crown_width_ratio = (crown_box[2] - crown_box[0]) / max(face_width, 1)
        crown_center_offset = abs(
            (crown_box[0] + crown_box[2]) / 2 - (face[0] + face[2]) / 2
        ) / max(face_width, 1)
        if not 1.08 <= crown_width_ratio <= 1.58:
            failures.append("crown scale is inconsistent with the face")
        if crown_center_offset > 0.12:
            failures.append("crown is not centered on the face")

    if not 0.0025 <= regalia_stats["fraction"] <= 0.055:
        failures.append("regalia mask area is outside the safe range")

    if not wand_box or wand_stats["bbox"] is None:
        failures.append("wand mask is empty")
        wand_span_fraction = 0.0
    else:
        visible_wand_box = wand_stats["bbox"]
        if not box_inside(visible_wand_box, width, height):
            failures.append("visible canonical wand pixels touch the image boundary")
        wand_span_fraction = (
            visible_wand_box[3] - visible_wand_box[1]
        ) / max(person_height, 1)
        if wand_span_fraction < 0.27:
            failures.append("too little of the wand remains visible")
        if not 0.0018 <= wand_stats["fraction"] <= 0.045:
            failures.append("wand mask area is outside the safe range")
        grip_distance = minimum_mask_distance(wand, grip)
        if grip_distance > max(42.0, person_width * 0.12):
            failures.append("visible wand is not attached near the selected hand")

    hand_outboard_fraction = None
    hand_height_fraction = None
    hand_wand_overlap_pixels = None
    hand_shape = None
    wand_above_hand_fraction = None
    wand_below_hand_fraction = None
    wand_face_overlap_pixels = 0
    inherited_off_axis_prop = None
    inherited_head_ornament = None
    if hand_source != "detected" or not hand_box:
        failures.append(f"wand anchor source is {hand_source}")
    else:
        hand_center = (
            (float(hand_box[0]) + float(hand_box[2])) / 2,
            (float(hand_box[1]) + float(hand_box[3])) / 2,
        )
        person_center_x = (person[0] + person[2]) / 2
        hand_outboard_fraction = abs(hand_center[0] - person_center_x) / max(
            person_width, 1
        )
        hand_height_fraction = (hand_center[1] - person[1]) / max(person_height, 1)
        if hand_outboard_fraction < 0.28:
            failures.append("selected hand is not far enough outside the body silhouette")
        if not 0.40 <= hand_height_fraction <= 0.72:
            failures.append("selected hand is not at a reliable wand-holding height")
        if full_wand is not None and hand_mask is not None:
            full_region = full_wand >= 8
            hand_region = hand_mask >= 8
            hand_shape = hand_shape_metrics(hand_mask, hand_box)
            hand_wand_overlap_pixels = int((full_region & hand_region).sum())
            if hand_wand_overlap_pixels < 20:
                failures.append("hand mask does not visibly occlude the wand grip")
            visible_y, _ = np.where(wand >= 8)
            if len(visible_y):
                wand_above_hand_fraction = float(
                    (visible_y < hand_center[1] - 2).sum() / len(visible_y)
                )
                wand_below_hand_fraction = float(
                    (visible_y > hand_center[1] + 2).sum() / len(visible_y)
                )
                if wand_above_hand_fraction < 0.34 or wand_below_hand_fraction < 0.05:
                    failures.append("wand shaft is not visibly continuous across the hand")

    face_left = max(0, int(math.floor(face[0])))
    face_top = max(0, int(math.floor(face[1])))
    face_right = min(width, int(math.ceil(face[2])))
    face_bottom = min(height, int(math.ceil(face[3])))
    wand_face_overlap_pixels = int(
        (wand[face_top:face_bottom, face_left:face_right] >= 8).sum()
    )
    if wand_face_overlap_pixels > 600:
        failures.append("canonical wand overlaps the face")

    inherited = metadata.get("inherited_accessory_detections", [])
    if wand_box:
        canonical_center_x = (float(wand_box[0]) + float(wand_box[2])) / 2
        for detection in inherited:
            label = str(detection.get("label", ""))
            score_value = float(detection.get("score", 0.0))
            if score_value < 0.40 or not any(
                word in label for word in ("wooden rod", "staff", "wand", "sword")
            ):
                continue
            box = [float(value) for value in detection["box"]]
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            if box_width > person_width * 0.72 or box_height < person_height * 0.16:
                continue
            center_x = (box[0] + box[2]) / 2
            if abs(center_x - canonical_center_x) > person_width * 0.24:
                inherited_off_axis_prop = {
                    "label": label,
                    "score": round(score_value, 6),
                    "box": [round(value, 2) for value in box],
                }
                failures.append("base contains a second prop away from the canonical wand")
                break

    for detection in inherited:
        label = str(detection.get("label", ""))
        score_value = float(detection.get("score", 0.0))
        if score_value < 0.28 or not any(
            phrase in label for phrase in ("hair ornament", "flower in hair")
        ):
            continue
        box = [float(value) for value in detection["box"]]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        if (
            face[0] - face_width * 0.55
            <= center_x
            <= face[2] + face_width * 0.55
            and face[1] - face_height * 1.1
            <= center_y
            <= face[3] + face_height * 0.2
        ):
            inherited_head_ornament = {
                "label": label,
                "score": round(score_value, 6),
                "box": [round(value, 2) for value in box],
            }
            failures.append("base contains an inherited hair ornament")
            break

    if wand_stats["largest_component_fraction"] < 0.62:
        failures.append("visible wand is too fragmented by the character")
    existing_crown_score = metadata.get("existing_crown_score")
    if (
        existing_crown_score is not None
        and float(existing_crown_score) >= 0.30
        and not metadata.get("existing_crown_removed")
    ):
        failures.append("base likely contains an inherited crown")

    score = 100.0
    score -= 22.0 * len(failures)
    score -= 5.0 * len(advisories)
    score = max(0.0, round(score, 1))
    metrics = {
        "image_size": [width, height],
        "person_height_fraction": round(person_height_fraction, 6),
        "face_width_fraction": round(face_width_fraction, 6),
        "crown_width_in_face_widths": round(crown_width_ratio, 6),
        "crown_center_offset_in_face_widths": (
            round(crown_center_offset, 6) if math.isfinite(crown_center_offset) else None
        ),
        "regalia_mask": regalia_stats,
        "wand_mask": wand_stats,
        "wand_refinement_mask": wand_refinement_stats,
        "full_wand_mask": full_wand_stats,
        "wand_visible_span_in_person_heights": round(wand_span_fraction, 6),
        "wand_grip_distance_pixels": (
            round(minimum_mask_distance(wand, grip), 3)
            if wand_stats["pixels"]
            else None
        ),
        "hand_source": hand_source,
        "hand_outboard_in_person_widths": (
            round(hand_outboard_fraction, 6)
            if hand_outboard_fraction is not None
            else None
        ),
        "hand_height_in_person_heights": (
            round(hand_height_fraction, 6)
            if hand_height_fraction is not None
            else None
        ),
        "hand_wand_overlap_pixels": hand_wand_overlap_pixels,
        "hand_shape": hand_shape,
        "wand_pixels_above_hand_fraction": (
            round(wand_above_hand_fraction, 6)
            if wand_above_hand_fraction is not None
            else None
        ),
        "wand_pixels_below_hand_fraction": (
            round(wand_below_hand_fraction, 6)
            if wand_below_hand_fraction is not None
            else None
        ),
        "wand_face_overlap_pixels": wand_face_overlap_pixels,
        "inherited_off_axis_prop": inherited_off_axis_prop,
        "inherited_head_ornament": inherited_head_ornament,
        "existing_crown_score": existing_crown_score,
        "existing_crown_removed": bool(metadata.get("existing_crown_removed")),
    }
    return {
        "validator_version": VALIDATOR_VERSION,
        "stage": "preflight",
        "accepted": not failures,
        "score": score,
        "failures": failures,
        "advisories": advisories,
        "metrics": metrics,
    }


def edge_retention(reference: np.ndarray, result: np.ndarray, mask: np.ndarray) -> float:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    ref_edges = cv2.Canny(ref_gray, 60, 140) > 0
    result_edges = cv2.Canny(result_gray, 60, 140) > 0
    region = mask >= 8
    reference_region = ref_edges & region
    if not reference_region.any():
        return 1.0
    nearby_result = cv2.dilate(result_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float((reference_region & nearby_result).sum() / reference_region.sum())


def masked_correlation(reference: np.ndarray, result: np.ndarray, mask: np.ndarray) -> float:
    region = mask >= 8
    ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)[region].astype(np.float32)
    out = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)[region].astype(np.float32)
    if len(ref) < 2 or ref.std() < 1e-5 or out.std() < 1e-5:
        return 0.0
    return float(np.corrcoef(ref, out)[0, 1])


def validate_final(
    composite_path: Path,
    final_path: Path,
    regalia_mask_path: Path,
    wand_mask_path: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    composite = load_rgb(composite_path)
    final = load_rgb(final_path)
    regalia = load_mask(regalia_mask_path)
    wand = load_mask(wand_mask_path)
    grip_mask_path = wand_mask_path.with_name(
        wand_mask_path.name.replace("_wand_mask", "_hand_grip_mask")
    )
    grip = load_mask(grip_mask_path) if grip_mask_path.is_file() else None
    union = np.maximum(regalia, wand)
    if grip is not None:
        union = np.maximum(union, grip)
    region = union >= 8
    outside = union == 0
    difference = np.abs(final.astype(np.int16) - composite.astype(np.int16)).max(axis=2)
    failures = list(preflight["failures"])
    advisories = list(preflight["advisories"])

    outside_changed_fraction = float((difference[outside] > 1).mean())
    inside_mean_change = float(difference[region].mean()) if region.any() else 0.0
    correlation = masked_correlation(composite, final, union)
    retained_edges = edge_retention(composite, final, union)
    sharpness = float(
        cv2.Laplacian(cv2.cvtColor(final, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    )

    if outside_changed_fraction > 0.00003:
        failures.append("pixels outside the exact accessory masks changed")
    if not 3.0 <= inside_mean_change <= 30.0:
        failures.append("regional refinement changed accessory pixels too little or too much")
    if correlation < 0.82:
        failures.append("accessory structure changed beyond the safe limit")
    if retained_edges < 0.68:
        failures.append("too many canonical accessory edges were lost")
    if sharpness < 18.0:
        failures.append("finished page is unusually blurry")

    failures = list(dict.fromkeys(failures))
    advisories = list(dict.fromkeys(advisories))
    score = 100.0
    score -= 22.0 * len(failures)
    score -= 4.0 * len(advisories)
    score = max(0.0, round(score, 1))
    return {
        "validator_version": VALIDATOR_VERSION,
        "stage": "final",
        "accepted": not failures,
        "score": score,
        "failures": failures,
        "advisories": advisories,
        "metrics": {
            **preflight["metrics"],
            "outside_mask_changed_fraction": round(outside_changed_fraction, 8),
            "inside_mask_mean_max_channel_change": round(inside_mean_change, 6),
            "masked_luminance_correlation": round(correlation, 6),
            "canonical_edge_retention": round(retained_edges, 6),
            "final_laplacian_variance": round(sharpness, 6),
            "grip_mask": mask_metrics(grip) if grip is not None else None,
        },
    }


def validate_record(run_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    artifacts = record["artifacts"]
    resolve = lambda key: run_root / artifacts[key]
    preflight = validate_preflight(
        resolve("composite"),
        resolve("regalia_mask"),
        resolve("wand_mask"),
        resolve("metadata"),
    )
    if "final" not in artifacts or not resolve("final").is_file():
        return preflight
    return validate_final(
        resolve("composite"),
        resolve("final"),
        resolve("regalia_mask"),
        resolve("wand_mask"),
        preflight,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--semantic-grip-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = []
    for record in summary["records"]:
        if "composite" not in record.get("artifacts", {}):
            continue
        results.append(
            {
                "id": record["id"],
                "validation": validate_record(args.run_root, record),
            }
        )
    if args.semantic_grip_threshold is not None:
        validation_records = []
        source_by_id = {record["id"]: record for record in summary["records"]}
        for result in results:
            source = source_by_id[result["id"]]
            validation_records.append(
                {
                    "id": result["id"],
                    "artifacts": source["artifacts"],
                    "validation": result["validation"],
                }
            )
        apply_grip_semantic_validation(
            args.run_root,
            validation_records,
            args.semantic_grip_threshold,
        )
        by_id = {record["id"]: record["validation"] for record in validation_records}
        for result in results:
            result["validation"] = by_id[result["id"]]
    report = {
        "validator_version": VALIDATOR_VERSION,
        "record_count": len(results),
        "accepted_count": sum(item["validation"]["accepted"] for item in results),
        "records": results,
    }
    destination = args.output or args.run_root / "validation_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
