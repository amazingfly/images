#!/usr/bin/env python3
"""Build the deterministic scene-cycling prompt set for identity dataset v2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "config_identity_dataset100_prompts.json"
IMAGE_COUNT = 100
FIRST_SEED = 95001

IDENTITY_PREFIX = (
    "lqxl, solo, one pointed gold crown with rose jewels, long wavy chestnut hair "
    "with straight bangs, petite storybook child, round face, large glossy ruby-brown "
    "eyes, fair peach skin, narrow shoulders, rose-pink puff-sleeve princess dress "
    "with gold trim"
)
FRAMING_PREFIX = (
    "wide full-body view, whole figure inside frame from head to shoes, both hands visible"
)
STYLE_SUFFIX = "polished anime storybook illustration"

# Framing plus the complete identity description is 74 CLIP-L content tokens,
# so both remain in chunk one. The short scene and style use a second chunk.
SCENES = (
    ("palace_courtyard", "standing in a sunlit palace courtyard", 88),
    ("crystal_lake", "walking beside a quiet crystal lake", 87),
    ("castle_greenhouse", "standing in a flowering castle greenhouse", 87),
    ("village_street", "turning on a bright village street", 87),
    ("castle_balcony", "beneath a starry castle balcony", 86),
    ("autumn_forest", "walking through an enchanted autumn forest", 87),
    ("throne_hall", "inside a luminous marble throne hall", 87),
    ("sunrise_hill", "on a windy hill at sunrise", 87),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = []
    for offset in range(IMAGE_COUNT):
        scene_id, scene, token_count = SCENES[offset % len(SCENES)]
        prompts.append(
            {
                "seed": FIRST_SEED + offset,
                "scene_id": scene_id,
                "scene": scene,
                "clip_l_content_tokens": token_count,
                "prompt": (
                    f"{FRAMING_PREFIX}, {IDENTITY_PREFIX}, {scene}, {STYLE_SUFFIX}"
                ),
            }
        )

    scene_counts = Counter(record["scene_id"] for record in prompts)
    payload = {
        "description": "Little Queen SDXL identity-strengthening dataset v2",
        "image_count": IMAGE_COUNT,
        "first_seed": FIRST_SEED,
        "scene_counts": dict(scene_counts),
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {args.output}")
    print("Scene counts: " + ", ".join(f"{key}={value}" for key, value in scene_counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
