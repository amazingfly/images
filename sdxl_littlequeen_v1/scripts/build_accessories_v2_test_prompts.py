#!/usr/bin/env python3
"""Build a weight and scene matrix for the decorrelated crown and wand-staff LoRAs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "config_accessories_v2_test_prompts.json"
CROWN = (
    "lqmooncrest, wearing one exact Moonstar flame-leaf diadem: a slim symmetrical "
    "polished gold band, five upward flame-leaf points, one centered round amber "
    "moonstone, two pale leaf inlays, two small rose-gold side blossoms"
)
STAFF = (
    "lqrosewand2, holding exactly one signature Rosekeeper wand-staff in one hand: "
    "one long straight slender gold shaft, one natural pink rose finial, one small "
    "oval rose crystal, one clear faceted teardrop crystal in a gold cage at the base"
)
EMPTY = "empty hands, no wand, no staff, no scepter, no weapon"

CASES = (
    (124001, "crown_w035", 0.35, 0.0, "full-body front view, entire head and both shoes framed, inside a plain blue stone hall", EMPTY),
    (124001, "crown_w055", 0.55, 0.0, "full-body front view, entire head and both shoes framed, inside a plain blue stone hall", EMPTY),
    (124001, "crown_w075", 0.75, 0.0, "full-body front view, entire head and both shoes framed, inside a plain blue stone hall", EMPTY),
    (124002, "crown_woodland", 0.55, 0.0, "head-and-shoulders three-quarter portrait, both ears visible, in a green woodland clearing", EMPTY),
    (124003, "crown_seaside", 0.55, 0.0, "wide full-body turning view, entire head and both shoes framed, on a blue seaside cliff", EMPTY),
    (124010, "staff_w025", 0.55, 0.25, "wide full-body presentation, entire head, staff, and both shoes framed, inside a plain blue stone hall", STAFF),
    (124010, "staff_w045", 0.55, 0.45, "wide full-body presentation, entire head, staff, and both shoes framed, inside a plain blue stone hall", STAFF),
    (124010, "staff_w065", 0.55, 0.65, "wide full-body presentation, entire head, staff, and both shoes framed, inside a plain blue stone hall", STAFF),
    (124011, "staff_walking", 0.55, 0.45, "wide full-body walking view, entire head, staff, and both shoes framed, staff lowered at her side, beside a snowy forest path", STAFF),
    (124012, "staff_casting", 0.55, 0.45, "wide full-body action view, entire head, staff, and both shoes framed, pointing the staff forward, among desert ruins", STAFF),
    (124013, "staff_seated", 0.55, 0.45, "full-body seated view, entire head and staff framed, staff resting diagonally across her lap, inside a quiet library", STAFF),
    (124014, "staff_close", 0.55, 0.45, "three-quarter portrait from knees upward, crown and complete upright staff framed, at a lakeside dock", STAFF),
    (124015, "both_strong", 0.72, 0.62, "wide full-body guard stance, entire head, staff, and both shoes framed, inside a moonlit castle gate", STAFF),
)


def main() -> int:
    prompts = []
    for seed, case_id, crown_weight, staff_weight, scene, hand_item in CASES:
        prompts.append(
            {
                "seed": seed,
                "case_id": case_id,
                "item_id": "rosekeeper_wand_staff" if staff_weight else "none",
                "regalia_weight": crown_weight,
                "equipment_weight": staff_weight,
                "prompt": (
                    f"{scene}, solo, lqxl Little Queen, lqmoonfit, {CROWN}, {hand_item}, "
                    "polished anime storybook illustration"
                ),
            }
        )
    OUTPUT.write_text(
        json.dumps(
            {
                "description": "Decorrelated v2 Moonstar crown and Rosekeeper wand-staff test",
                "image_count": len(prompts),
                "prompts": prompts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(prompts)} v2 test prompts to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
