#!/usr/bin/env python3
"""Build 100 outfit-disentanglement prompts for the stacked local test."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "config_outfit_v1_local100_prompts.json"
FIRST_SEED = 101001

LOCATIONS = (
    ("rose_garden", "in a quiet rose garden"),
    ("palace_courtyard", "in a sunlit palace courtyard"),
    ("marble_hall", "inside a bright marble hall"),
    ("cloudy_hill", "beneath a cloudy sky on a green hill"),
    ("forest_stream", "beside a clear forest stream"),
    ("village_street", "on a cheerful village street"),
    ("crystal_lake", "near a still crystal lake"),
    ("castle_courtyard", "in an open castle courtyard"),
    ("evening_stars", "under the evening stars"),
    ("autumn_forest", "among glowing autumn trees"),
    ("greenhouse", "inside a flowering greenhouse"),
    ("sunset_balcony", "on a palace balcony at sunset"),
    ("snowy_courtyard", "in a snowy castle courtyard"),
    ("lantern_festival", "at a lantern festival"),
    ("crystal_cavern", "inside a glowing crystal cavern"),
    ("storybook_library", "inside a grand storybook library"),
    ("seaside_path", "on a bright seaside path"),
    ("waterfall", "near a gentle waterfall"),
    ("wildflower_meadow", "in a wildflower meadow"),
    ("moonlit_bridge", "on a moonlit stone bridge"),
)

VARIATIONS = (
    (
        "calm_no_prop",
        "wide full-body front view, entire head and both shoes inside frame, "
        "empty hands visible, standing calmly",
    ),
    (
        "walking_rose_crown",
        "full-body three-quarter view, entire gold rose crown and both shoes "
        "inside frame, empty hands visible, taking one graceful step",
    ),
    (
        "turning_moon_circlet",
        "wide full-body motion view, entire silver moon circlet and both shoes "
        "inside frame, turning as her hair and skirt flow",
    ),
    (
        "magic_star_wand",
        "wide full-body action view, entire head, star wand, and both shoes "
        "inside frame, raising one star wand as a bright magical aura begins",
    ),
    (
        "reading_no_crown",
        "full-body seated view, entire uncovered head and both shoes inside "
        "frame, no crown, reading an open storybook held in both hands",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    prompts = []
    offset = 0
    for location_id, location in LOCATIONS:
        for variation_id, framing in VARIATIONS:
            prompts.append(
                {
                    "seed": FIRST_SEED + offset,
                    "location_id": location_id,
                    "variation_id": variation_id,
                    "prompt": (
                        f"{framing}, solo, lqmoonfit, lqxl Little Queen, "
                        f"{location}, polished anime storybook illustration"
                    ),
                }
            )
            offset += 1

    assert len(prompts) == 100
    payload = {
        "description": "Stacked Little Queen identity plus Moonstar outfit LoRA test",
        "image_count": len(prompts),
        "first_seed": FIRST_SEED,
        "outfit_description_in_prompts": False,
        "location_counts": dict(Counter(item["location_id"] for item in prompts)),
        "variation_counts": dict(Counter(item["variation_id"] for item in prompts)),
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
