#!/usr/bin/env python3
"""Build 100 varied prompts for the local Little Queen SDXL v2 run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "config_lora_v2_local100_prompts.json"
FIRST_SEED = 98001

LOCATIONS = (
    ("rose_garden", "in a quiet rose garden"),
    ("palace_courtyard", "in a sunlit palace courtyard"),
    ("marble_hall", "inside a bright marble hall"),
    ("cloudy_hill", "beneath a cloudy sky on a hill"),
    ("forest_stream", "beside a forest stream"),
    ("village_street", "on a cheerful village street"),
    ("crystal_lake", "near a still crystal lake"),
    ("castle_courtyard", "in an open castle courtyard"),
    ("evening_stars", "under the evening stars"),
    ("autumn_forest", "among autumn trees"),
    ("greenhouse", "inside a flowering greenhouse"),
    ("sunset_balcony", "on a balcony at sunset"),
    ("snowy_courtyard", "in a snowy castle courtyard"),
    ("lantern_festival", "at a lantern festival"),
    ("crystal_cavern", "inside a glowing crystal cavern"),
    ("storybook_library", "inside a grand storybook library"),
    ("seaside_path", "on a bright seaside path"),
    ("waterfall", "near a gentle waterfall"),
    ("wildflower_meadow", "in a wildflower meadow"),
    ("moonlit_bridge", "on a moonlit stone bridge"),
    ("garden_steps", "on ancient garden steps"),
    ("village_market", "in a colorful village market"),
    ("windy_cliff", "on a windy cliff above the sea"),
    ("dragon_sanctuary", "in a friendly dragon sanctuary"),
    ("flower_maze", "inside a tall flower maze"),
)

ACTIONS = (
    (
        "calm",
        "wide full-body front view, entire crown and both shoes inside frame, "
        "both hands visible, standing calmly",
    ),
    (
        "walking",
        "full-body three-quarter view, entire crown and both shoes inside frame, "
        "both hands visible, taking one graceful step",
    ),
    (
        "turning",
        "wide full-body motion view, entire crown and both shoes inside frame, "
        "both hands visible, turning as her hair and skirt flow",
    ),
    (
        "magic",
        "wide full-body action view, entire crown and both shoes inside frame, "
        "both hands visible, raising both hands as a bright magical aura begins",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = []
    offset = 0
    for location_id, location in LOCATIONS:
        for action_id, action in ACTIONS:
            prompts.append(
                {
                    "seed": FIRST_SEED + offset,
                    "location_id": location_id,
                    "action_id": action_id,
                    "prompt": (
                        f"{action}, solo, lqxl Little Queen, {location}, "
                        "polished anime storybook illustration"
                    ),
                }
            )
            offset += 1

    assert len(prompts) == 100
    payload = {
        "description": "Little Queen SDXL v2 local production run",
        "image_count": len(prompts),
        "first_seed": FIRST_SEED,
        "location_counts": dict(Counter(item["location_id"] for item in prompts)),
        "action_counts": dict(Counter(item["action_id"] for item in prompts)),
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
