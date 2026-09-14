#!/usr/bin/env python3
"""Build decorrelated, body-aligned bootstrap prompts for accessory v3."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "config_accessories_v3_bootstrap_prompts.json"

SCENES = (
    ("blue_stone_hall", "inside a plain blue stone hall with undecorated walls"),
    ("snowy_pines", "on a quiet path among snow-covered pine trees"),
    ("desert_plain", "on a broad dry desert plain beneath a pale sky"),
    ("seaside_cliff", "on a windy gray seaside cliff above blue water"),
    ("village_square", "in a simple timber village square in morning light"),
    ("library", "inside a quiet wood-paneled library with bookcases"),
    ("green_meadow", "in an open green meadow with distant hills"),
    ("mountain_bridge", "on a weathered stone bridge in the mountains"),
    ("autumn_road", "on a country road beneath orange autumn trees"),
    ("moon_observatory", "inside a clean stone observatory beneath a moonlit dome"),
    ("lakeside_dock", "on a wooden dock beside a still blue lake"),
    ("market_street", "on a colorful market street with canvas awnings"),
)

DRESSES = (
    "a simple sapphire-blue travel dress with silver piping",
    "a simple forest-green day dress with cream trim",
    "a simple ivory winter dress with navy trim",
    "a simple burgundy court dress with pale-gold trim",
    "a simple violet storybook dress with white trim",
    "a simple charcoal-gray travel dress with red trim",
)

REGALIA_POSES = (
    "head-and-shoulders front portrait, entire crown, both ears, and neckline visible",
    "head-and-shoulders three-quarter portrait, entire crown, both ears, and neckline visible",
    "waist-up front view, entire crown, both ears, neckline, and both wrists visible",
    "waist-up three-quarter view, entire crown, both ears, neckline, and both wrists visible",
    "full-body front view, entire crown, both hands, and both shoes framed",
    "full-body turning view, entire crown, both hands, and both shoes framed",
)

STAFF_POSES = (
    "full-body front view, holding the complete upright wand-staff beside her in her right hand",
    "full-body three-quarter view, holding the complete upright wand-staff beside her in her left hand",
    "full-body walking view, carrying the complete wand-staff lowered diagonally in one hand",
    "full-body guard stance, gripping the complete wand-staff vertically with both hands",
    "full-body casting stance, pointing the complete wand-staff forward with one hand",
    "knees-up presentation view, holding the complete wand-staff upright close beside her shoulder",
)

CROWN = (
    "lqmooncrest, wearing exactly one Moonstar flame-leaf diadem fitted on her head: a slim "
    "symmetrical polished gold band, five upward flame-leaf points, one centered round amber "
    "moonstone, two pale leaf inlays, and two small rose-gold side blossoms"
)
JEWELRY = (
    "wearing one matched pair of small gold crescent-drop earrings with rose-pink crystals, "
    "one narrow gold collar necklace with one centered amber moonstone and tiny rose pendant, "
    "and one matching slender engraved gold cuff on each wrist"
)
STAFF = (
    "lqrosewand2, holding exactly one signature Rosekeeper wand-staff with a human-scale "
    "grip: one long straight slender gold shaft, one natural pink rose finial, one small oval "
    "rose crystal in the shaft, and one clear faceted teardrop crystal held in a gold cage at "
    "the base; her fingers close naturally around the shaft"
)


def regalia_records() -> list[dict]:
    records = []
    weights = (0.32, 0.38, 0.44)
    for offset in range(48):
        scene_id, scene = SCENES[offset % len(SCENES)]
        pose = REGALIA_POSES[offset % len(REGALIA_POSES)]
        dress = DRESSES[(offset * 5 + offset // len(SCENES)) % len(DRESSES)]
        weight = weights[(offset // 2) % len(weights)]
        prompt = (
            f"{pose}, {scene}, solo, one girl, lqxl Little Queen, {dress}, {CROWN}, {JEWELRY}, "
            "empty hands, no wand, no staff, no weapon, no floral archway, polished anime "
            "storybook illustration"
        )
        records.append(
            {
                "case_id": f"regalia_{offset + 1:03d}_{scene_id}",
                "concept": "regalia",
                "scene_id": scene_id,
                "pose_id": f"regalia_pose_{offset % len(REGALIA_POSES) + 1}",
                "seed": 126001 + offset,
                "identity_weight": 0.9,
                "outfit_weight": 0.0,
                "regalia_weight": weight,
                "equipment_weight": 0.0,
                "prompt": prompt,
            }
        )
    return records


def staff_records() -> list[dict]:
    records = []
    weights = (0.26, 0.32, 0.38)
    for offset in range(60):
        scene_id, scene = SCENES[(offset * 5 + 2) % len(SCENES)]
        pose = STAFF_POSES[offset % len(STAFF_POSES)]
        dress = DRESSES[(offset * 7 + 1) % len(DRESSES)]
        weight = weights[(offset // 2) % len(weights)]
        prompt = (
            f"{pose}, entire head, complete wand-staff, both hands, and both shoes inside the "
            f"frame, {scene}, solo, one girl, lqxl Little Queen, {dress}, bare head, no crown, "
            f"{STAFF}, no floral archway, no rose garden, polished anime storybook illustration"
        )
        records.append(
            {
                "case_id": f"staff_{offset + 1:03d}_{scene_id}",
                "concept": "staff",
                "scene_id": scene_id,
                "pose_id": f"staff_pose_{offset % len(STAFF_POSES) + 1}",
                "seed": 126101 + offset,
                "identity_weight": 0.9,
                "outfit_weight": 0.0,
                "regalia_weight": 0.0,
                "equipment_weight": weight,
                "prompt": prompt,
            }
        )
    return records


def main() -> int:
    records = regalia_records() + staff_records()
    payload = {
        "name": "littlequeen_accessories_v3_contextual_bootstrap",
        "method": "v2-assisted worn and held candidates with decorrelated scenes and clothing",
        "prompts": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} prompts to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
