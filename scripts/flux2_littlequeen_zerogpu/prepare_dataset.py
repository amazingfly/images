#!/usr/bin/env python3
"""Build the curated FLUX.2 Klein Little Queen identity dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path("/mnt/storage/projects/agentic/images/sdxl_littlequeen_v1")
IDENTITY_DIR = ROOT / "training_dataset_v2/colab/train/10_lqxl"
IDENTITY_MANIFEST = ROOT / "training_dataset_v2/curated_manifest.json"
STORY_ROOT = ROOT / "outputs/storybook_mvp_v1/the_witches_trick_20260728/base"
DEFAULT_OUTPUT = ROOT / "training_dataset_flux2_klein_v1"
TRIGGER = "LQK4N"


STORY_ADDITIONS = [
    {
        "scene": 45,
        "candidate": 1,
        "caption": (
            "LQK4N. Full body three-quarter view of the young queen seated on a stone bench, "
            "with very long loose wavy hair and straight bangs, a small gold pointed crown, "
            "a short pink dress with puff sleeves and gold trim, hands resting beside her, in a "
            "rose-covered castle garden."
        ),
    },
    {
        "scene": 47,
        "candidate": 6,
        "caption": (
            "LQK4N. Wide full body rear view of the young queen walking through open palace garden "
            "doors, with very long loose hair flowing down her back, a small gold crown, and "
            "a knee-length pink dress, surrounded by drifting flower petals."
        ),
    },
    {
        "scene": 92,
        "candidate": 9,
        "caption": (
            "LQK4N. Full body action view of the young queen striding along a castle garden path, "
            "with very long windblown hair and straight bangs, a gold pointed crown, a dark "
            "green dress with pink puff sleeves and gold trim, holding a slender gold wand, with a "
            "small rabbit beside her."
        ),
    },
    {
        "scene": 105,
        "candidate": 3,
        "caption": (
            "LQK4N. Medium full body three-quarter portrait of the young queen standing inside a "
            "large decorative frame of gold chains and floating hearts, with hair gathered "
            "into a low side bun and straight bangs, an intricate gold crown with a rose, and a "
            "layered pink dress with puff sleeves and ornate gold trim."
        ),
    },
    {
        "scene": 106,
        "candidate": 3,
        "caption": (
            "LQK4N. Medium full body three-quarter portrait of the young queen in a rose garden, with "
            "very long loose wavy hair and straight bangs, a tall intricate gold crown, and a "
            "pink and warm gold dress with puff sleeves, standing with a gentle smile."
        ),
    },
    {
        "scene": 110,
        "candidate": 5,
        "caption": (
            "LQK4N. Dynamic full body action view of the young queen casting a stream of golden fire, "
            "with long windblown hair and straight bangs, a narrow gold crown, a warm gold "
            "ball gown with pink ruffles, and both arms extended among dark castle towers."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_caption(caption: str) -> str:
    body = caption.removeprefix("lqxl,").strip()
    return f"{TRIGGER}. {body[0].upper()}{body[1:].rstrip('.')}."


def add_image(
    output: Path,
    index: int,
    source: Path,
    caption: str,
    provenance: dict,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    image_name = f"lqk4n_{index:03d}.png"
    caption_name = f"lqk4n_{index:03d}.txt"
    image_path = output / image_name
    caption_path = output / caption_name
    shutil.copy2(source, image_path)
    caption_path.write_text(caption.strip() + "\n", encoding="utf-8")
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "index": index,
        "image": image_name,
        "caption_file": caption_name,
        "caption": caption.strip(),
        "width": width,
        "height": height,
        "sha256": sha256(image_path),
        "source": str(source),
        **provenance,
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob("lqk4n_*"):
        if path.is_file() or path.is_symlink():
            path.unlink()

    identity = json.loads(IDENTITY_MANIFEST.read_text(encoding="utf-8"))
    records = []
    for item in identity["images"]:
        index = len(records) + 1
        source = IDENTITY_DIR / f"lqxl_{index:03d}.png"
        records.append(
            add_image(
                output,
                index,
                source,
                normalize_caption(item["caption"]),
                {
                    "source_group": "sdxl_identity_v2_manual_curation",
                    "source_index": item["source_index"],
                    "scene_id": item["scene_id"],
                },
            )
        )

    for item in STORY_ADDITIONS:
        index = len(records) + 1
        source = STORY_ROOT / f"scene_{item['scene']:03d}_candidate_{item['candidate']:02d}.png"
        records.append(
            add_image(
                output,
                index,
                source,
                item["caption"],
                {
                    "source_group": "the_witches_trick_manual_identity_addition",
                    "source_scene": item["scene"],
                    "source_candidate": item["candidate"],
                },
            )
        )

    manifest = {
        "version": "little-queen-flux2-klein-identity-v1",
        "trigger": TRIGGER,
        "base_model": "black-forest-labs/FLUX.2-klein-base-4B",
        "selection": (
            "Thirty manually curated SDXL identity images plus six manually reviewed story images "
            "adding rear, seated, action, magic, and non-pink outfit coverage."
        ),
        "caption_strategy": {
            "implicit_identity": [
                "facial identity and proportions",
                "large dark brown eyes",
                "auburn hair color",
                "young childlike body proportions",
                "polished anime storybook rendering",
            ],
            "explicit_variables": [
                "framing and pose",
                "hair arrangement",
                "outfit construction and color",
                "crown construction",
                "held objects and companions",
                "background and action",
            ],
        },
        "image_count": len(records),
        "images": records,
    }
    metadata_rows = [
        json.dumps({"file_name": item["image"], "text": item["caption"]})
        for item in records
    ]
    (output / "metadata.jsonl").write_text(
        "\n".join(metadata_rows) + "\n", encoding="utf-8"
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "---\nlicense: other\npretty_name: Little Queen FLUX.2 Klein identity training set\n---\n\n"
        "Private synthetic training dataset for the Little Queen identity LoRA.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "image_count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
