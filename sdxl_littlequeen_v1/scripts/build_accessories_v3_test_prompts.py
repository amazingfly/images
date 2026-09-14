#!/usr/bin/env python3
"""Build controlled identity-only and stacked validation prompts for accessory v3."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "config_accessories_v3_test_prompts.json"
CROWN = (
    "lqmoonregalia3, wearing exactly one Moonstar flame-leaf crown: a symmetrical polished "
    "gold diadem with five to seven pointed leaf flames, one centered amber-rose oval moonstone "
    "and smaller matching side gems; one matched pair of long gold leaf-drop earrings; one "
    "gold filigree collar necklace with a centered amber moonstone; matching engraved gold cuffs"
)
STAFF = (
    "lqrosekeeper3, holding exactly one complete signature Rosekeeper wand-staff at human scale: "
    "one natural pink rose finial, one long straight slender gold shaft, one small rose crystal, "
    "and one clear faceted teardrop crystal in a gold cage at the base; fingers closed naturally "
    "around the shaft"
)
STYLE = "solo, one girl, lqxl Little Queen, polished anime storybook illustration"


def record(case_id: str, seed: int, prompt: str, regalia: float, staff: float, outfit: float = 0.0) -> dict:
    return {
        "case_id": case_id,
        "seed": seed,
        "identity_weight": 0.9,
        "outfit_weight": outfit,
        "regalia_weight": regalia,
        "equipment_weight": staff,
        "prompt": prompt,
    }


def main() -> int:
    plain_regalia = (
        f"full-body front view, entire crown, both ears, neckline, wrists, and both shoes framed, "
        f"inside a plain blue stone hall, {STYLE}, wearing a simple blue travel dress, {CROWN}, "
        "empty hands, no wand, no staff, no floral archway"
    )
    plain_staff = (
        f"full-body presentation, entire head, complete staff, both hands, and both shoes framed, "
        f"inside a plain gray practice room, {STYLE}, wearing a simple green travel dress, bare "
        f"head, {STAFF}, no floral archway, no rose garden"
    )
    prompts = [
        record(
            "baseline_plain", 128001,
            f"full-body front view, entire head and both shoes framed, inside a plain blue stone hall, {STYLE}, wearing a simple blue travel dress, bare head, empty hands, no crown, no jewelry, no wand, no staff",
            0.0, 0.0,
        ),
        record("regalia_w025", 128002, plain_regalia, 0.25, 0.0),
        record("regalia_w045", 128002, plain_regalia, 0.45, 0.0),
        record("regalia_w065", 128002, plain_regalia, 0.65, 0.0),
        record(
            "regalia_woodland", 128003,
            f"waist-up three-quarter portrait, entire crown, both ears, neckline, and wrists framed, in a mossy green woodland clearing, {STYLE}, wearing a simple ivory dress, {CROWN}, empty hands, no wand, no staff, no floral archway",
            0.45, 0.0,
        ),
        record(
            "regalia_seaside", 128004,
            f"full-body turning view, entire crown, wrists, and both shoes framed, on a windy blue seaside cliff, {STYLE}, wearing a simple burgundy dress, {CROWN}, empty hands, no wand, no staff, no floral archway",
            0.45, 0.0,
        ),
        record("staff_w025", 128010, plain_staff, 0.0, 0.25),
        record("staff_w045", 128010, plain_staff, 0.0, 0.45),
        record("staff_w065", 128010, plain_staff, 0.0, 0.65),
        record(
            "staff_snow", 128011,
            f"full-body walking view, entire head, complete staff, both hands, and both shoes framed, staff lowered diagonally at her side, on a snowy pine path, {STYLE}, wearing a simple navy winter dress, bare head, {STAFF}, no floral archway, no rose garden",
            0.0, 0.45,
        ),
        record(
            "staff_desert", 128012,
            f"full-body casting stance, entire head, complete staff, both hands, and both shoes framed, on a dry desert plain, {STYLE}, wearing a simple ivory travel dress, bare head, {STAFF}, no floral archway, no rose garden",
            0.0, 0.45,
        ),
        record(
            "both_plain", 128020,
            f"full-body guard stance, entire crown, complete staff, both hands, and both shoes framed, inside a plain gray stone gate, {STYLE}, wearing a simple violet dress, {CROWN}, {STAFF}, no floral archway, no rose garden",
            0.45, 0.45,
        ),
        record(
            "both_outfit_stack", 128021,
            f"full-body guard stance, entire crown, complete staff, both hands, and both shoes framed, on a simple lakeside dock, {STYLE}, lqmoonfit, {CROWN}, {STAFF}, no floral archway, no rose garden",
            0.45, 0.45, 0.55,
        ),
        record(
            "regalia_outfit_stack", 128022,
            f"waist-up front portrait, entire crown, both ears, neckline, and wrists framed, inside a plain wood-paneled library, {STYLE}, lqmoonfit, {CROWN}, empty hands, no wand, no staff, no floral archway",
            0.45, 0.0, 0.55,
        ),
    ]
    OUTPUT.write_text(json.dumps({"name": "littlequeen_accessories_v3_validation", "prompts": prompts}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
