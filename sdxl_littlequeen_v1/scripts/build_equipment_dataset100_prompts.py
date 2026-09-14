#!/usr/bin/env python3
"""Build the 10-scene by 10-pose Moonstar regalia dataset prompts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "config_equipment_dataset100_prompts.json"
FIRST_SEED = 99001

SUBJECT = "solo, lqxl Little Queen"
OUTFIT = (
    "Moonstar dress: rose-pink bodice, high collar, puff sleeves, eight-petal skirt, "
    "ivory underskirt, gold piping, star-crescent hem, rose-gem "
    "brooch, white stockings, pink strap shoes"
)
CROWN = (
    "one five-point gold crown, rose gem, moonstones, star-crescent filigree"
)
FRAMING = (
    "full body, crown and shoes framed, clear face and arms"
)
STYLE = (
    "polished anime storybook art, crisp regalia, luminous color"
)

EQUIPMENT = (
    (
        "moonstar_wand",
        "holding one slender gold Moonstar wand: filigree grip, rose crystal, crescent "
        "cradle, star tip, moonstone chains",
    ),
    (
        "royal_scepter",
        "holding one gold royal scepter: filigree shaft, rose-moonstone orb, crescent "
        "wings, star finial, ivy engraving",
    ),
    (
        "starlight_staff",
        "holding one gold starlight staff: engraved shaft, crescent head, suspended rose "
        "crystal, star-map inlays, ribbon tassel",
    ),
    (
        "ceremonial_spellblade",
        "holding one ceremonial spellblade: short ivory-gold blade, crescent guard, "
        "rose-gem pommel, etched stars and moon phases",
    ),
    (
        "crescent_glaive",
        "holding one crescent glaive: slender gold-ivory shaft, single curved blade, "
        "rose-crystal socket, constellation engraving",
    ),
)

SCENES = (
    ("rose_courtyard", "in a sunlit palace rose courtyard"),
    ("throne_hall", "inside a luminous marble throne hall"),
    ("flower_greenhouse", "inside a flowering castle greenhouse"),
    ("moonlit_balcony", "on a moonlit palace balcony"),
    ("crystal_lake", "beside a still crystal lake at dawn"),
    ("autumn_forest", "on an autumn forest path with drifting leaves"),
    ("star_observatory", "inside a royal star observatory"),
    ("lantern_festival", "in a village square during a lantern festival"),
    ("ancient_ruins", "among moon-temple ruins at golden hour"),
    ("snow_garden", "in a snowy palace garden with warm lanterns"),
)

POSES = (
    (
        "formal_front",
        "front-facing presentation, item upright, free hand open",
    ),
    (
        "walking_three_quarter",
        "graceful step in three-quarter view, item lowered",
    ),
    (
        "gentle_curtsy",
        "three-quarter curtsy, item angled across her, free hand lifting skirt",
    ),
    (
        "seated_step",
        "seated upright, item across her lap, free hand relaxed",
    ),
    (
        "profile_lookback",
        "side profile looking back, item lowered, hair moving",
    ),
    (
        "inviting_reach",
        "free hand reaching forward, item raised by shoulder",
    ),
    (
        "guarded_action",
        "grounded action stance, item forward, determined expression",
    ),
    (
        "magic_overhead",
        "item overhead inside one starlight ring, feet apart",
    ),
    (
        "kneeling_vow",
        "kneeling on one knee, item upright, hand over heart",
    ),
    (
        "spinning_motion",
        "mid-turn, skirt and hair flowing, item swept aside",
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
    for scene_index, (scene_id, scene) in enumerate(SCENES):
        for pose_index, (pose_id, pose) in enumerate(POSES):
            equipment_id, equipment = EQUIPMENT[(scene_index + pose_index) % len(EQUIPMENT)]
            prompts.append(
                {
                    "seed": FIRST_SEED + offset,
                    "scene_id": scene_id,
                    "pose_id": pose_id,
                    "equipment_id": equipment_id,
                    "prompt_sections": {
                        "subject_outfit_crown": f"{SUBJECT}, {OUTFIT}, {CROWN}",
                        "equipment": equipment,
                        "pose_scene_style": f"{FRAMING}, {pose}, {scene}, {STYLE}",
                    },
                    "prompt": (
                        f"{SUBJECT}, {OUTFIT}, {CROWN}, {equipment}, {FRAMING}, {pose}, "
                        f"{scene}, {STYLE}"
                    ),
                }
            )
            offset += 1

    assert len(prompts) == 100
    payload = {
        "description": "Little Queen Moonstar outfit and variable equipment dataset",
        "training_intent": (
            "Bind the invariant dress and crown to a future outfit trigger while "
            "captioning equipment, pose, and scene as variables. Train signature "
            "equipment separately if exact item interchangeability is required."
        ),
        "image_count": len(prompts),
        "first_seed": FIRST_SEED,
        "scene_counts": dict(Counter(item["scene_id"] for item in prompts)),
        "pose_counts": dict(Counter(item["pose_id"] for item in prompts)),
        "equipment_counts": dict(Counter(item["equipment_id"] for item in prompts)),
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
