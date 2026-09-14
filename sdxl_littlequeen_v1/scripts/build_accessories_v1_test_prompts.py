#!/usr/bin/env python3
"""Build the Colab consistency test matrix for canonical regalia and hand gear."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "config_accessories_v1_test_prompts.json"

REGALIA = (
    "lqmoonregalia, the exact matching Moonstar regalia: slim symmetrical polished "
    "gold diadem with five upward flame-leaf points, one centered round amber "
    "moonstone and two pale leaf inlays, matching long hollow gold teardrop earrings, "
    "an ornate close gold collar with one oval rose crystal, matching narrow gold "
    "engraved gold wrist cuffs worn over the pink gloves"
)
ITEMS = {
    "none": "empty hands, no wand, no staff, no scepter, no weapon",
    "moonstar_wand": (
        "lqmoonwand, holding exactly one signature Moonstar wand: one slender "
        "straight gold shaft, small ivory grip, one open gold crescent around one "
        "rose-pink crystal, one small five-point star tip"
    ),
    "rosekeeper_staff": (
        "lqrosestaff, holding exactly one signature Rosekeeper staff: one long "
        "straight slender gold shaft, one natural pink rose finial, one small oval "
        "rose crystal, one clear faceted teardrop crystal in a gold cage at the base"
    ),
    "celestial_orb_scepter": (
        "lqorbscepter, holding exactly one signature Celestial Orb scepter: one "
        "straight slender gold shaft, one large luminous round golden orb engraved "
        "with a crescent, one small pointed star finial"
    ),
}

STRESS_CASES = (
    (112001, "regalia_close", "none", 0.85, 0.0, "head-and-shoulders royal portrait, both ears and both wrists visible, inside a bright palace salon"),
    (112002, "regalia_front", "none", 0.85, 0.0, "wide full-body front view, entire crown and both shoes framed, standing in a rose courtyard"),
    (112003, "regalia_three_quarter", "none", 1.00, 0.0, "full-body three-quarter view, entire crown and both shoes framed, inside a grand library"),
    (112004, "regalia_turning", "none", 1.00, 0.0, "wide full-body turning pose, entire crown and both shoes framed, on a moonlit balcony"),
    (112005, "wand_front", "moonstar_wand", 0.90, 0.95, "wide full-body front view, entire crown, wand, and both shoes framed, wand upright beside her, in a palace courtyard"),
    (112006, "wand_walking", "moonstar_wand", 0.90, 0.95, "full-body three-quarter walking view, entire crown, wand, and both shoes framed, wand lowered at her side, beside a forest stream"),
    (112007, "wand_casting", "moonstar_wand", 0.95, 1.05, "wide full-body action view, entire crown, wand, and both shoes framed, pointing the wand forward while casting one ring of starlight, in a crystal cavern"),
    (112008, "wand_overhead", "moonstar_wand", 0.95, 1.05, "wide full-body action view, entire crown, wand, and both shoes framed, raising the wand overhead, beneath evening stars"),
    (112009, "wand_seated", "moonstar_wand", 1.00, 1.15, "full-body seated view, entire crown and wand framed, wand resting diagonally across her lap, inside a storybook library"),
    (112010, "wand_guard", "moonstar_wand", 1.00, 1.15, "wide full-body defensive stance, entire crown, wand, and both shoes framed, wand held across her body, on ancient garden steps"),
    (112011, "rose_staff_front", "rosekeeper_staff", 0.95, 1.00, "wide full-body front view, entire crown, staff, and both shoes framed, staff upright, in an autumn garden"),
    (112012, "rose_staff_walk", "rosekeeper_staff", 0.95, 1.10, "full-body three-quarter walking view, entire crown, staff, and both shoes framed, staff lowered, at a lantern festival"),
    (112013, "orb_scepter_front", "celestial_orb_scepter", 0.95, 1.00, "wide full-body front view, entire crown, scepter, and both shoes framed, scepter upright, in a celestial observatory"),
    (112014, "orb_scepter_raise", "celestial_orb_scepter", 0.95, 1.10, "wide full-body action view, entire crown, scepter, and both shoes framed, raising the scepter in one hand, near a crystal lake"),
    (112015, "regalia_strong", "none", 1.15, 0.0, "head-and-shoulders three-quarter portrait, both earrings and necklace clearly visible, in a snowy palace garden"),
    (112016, "wand_strong", "moonstar_wand", 1.05, 1.25, "wide full-body presentation view, entire crown, wand, and both shoes framed, wand upright and clearly separated from her body, in a wildflower meadow"),
)

CALIBRATION_SCENE = (
    "full-body three-quarter storybook view, entire crown, both hands, and both shoes "
    "framed, inside a simple sunlit stone library"
)
CALIBRATION_CASES = (
    (120001, "regalia_w025", "none", 0.25, 0.0, CALIBRATION_SCENE),
    (120001, "regalia_w040", "none", 0.40, 0.0, CALIBRATION_SCENE),
    (120001, "regalia_w055", "none", 0.55, 0.0, CALIBRATION_SCENE),
    (120002, "regalia_w040_garden", "none", 0.40, 0.0, "head-and-shoulders portrait, both ears and necklace visible, in a green woodland garden"),
    (120003, "regalia_w050_balcony", "none", 0.50, 0.0, "wide full-body turning view, entire crown and both shoes framed, on a blue seaside balcony"),
    (120010, "wand_w015", "moonstar_wand", 0.40, 0.15, CALIBRATION_SCENE + ", holding the wand upright and clearly separated from her body"),
    (120010, "wand_w025", "moonstar_wand", 0.40, 0.25, CALIBRATION_SCENE + ", holding the wand upright and clearly separated from her body"),
    (120010, "wand_w035", "moonstar_wand", 0.40, 0.35, CALIBRATION_SCENE + ", holding the wand upright and clearly separated from her body"),
    (120010, "wand_w045", "moonstar_wand", 0.40, 0.45, CALIBRATION_SCENE + ", holding the wand upright and clearly separated from her body"),
    (120011, "wand_w030_walk", "moonstar_wand", 0.40, 0.30, "wide full-body walking view, entire crown, wand, and both shoes framed, wand lowered at her side, beside a forest stream"),
    (120012, "wand_w030_cast", "moonstar_wand", 0.40, 0.30, "wide full-body action view, entire crown, wand, and both shoes framed, pointing the wand forward in a stone courtyard"),
    (120020, "rose_staff_w030", "rosekeeper_staff", 0.40, 0.30, "wide full-body front view, entire crown, staff, and both shoes framed, one staff upright, in an autumn orchard"),
    (120030, "orb_scepter_w030", "celestial_orb_scepter", 0.40, 0.30, "wide full-body front view, entire crown, scepter, and both shoes framed, one scepter upright, in a blue celestial observatory"),
)

CHECKPOINT_SCENE = (
    "wide full-body front view, entire crown, wand, both hands, and both shoes framed, "
    "holding the wand upright and clearly separated from her body, inside a plain blue "
    "stone hall"
)
CHECKPOINT_CASES = (
    (121001, "regalia300_w040", "none", 0.40, 0.0, "full-body front view, entire crown and both shoes framed, inside a plain blue stone hall", "regalia300", "equipment"),
    (121001, "regalia300_w060", "none", 0.60, 0.0, "full-body front view, entire crown and both shoes framed, inside a plain blue stone hall", "regalia300", "equipment"),
    (121010, "wand300_w030", "moonstar_wand", 0.40, 0.30, CHECKPOINT_SCENE, "regalia300", "equipment300"),
    (121010, "wand300_w045", "moonstar_wand", 0.40, 0.45, CHECKPOINT_SCENE, "regalia300", "equipment300"),
    (121010, "wand600_w030", "moonstar_wand", 0.40, 0.30, CHECKPOINT_SCENE, "regalia300", "equipment600"),
    (121010, "wand600_w045", "moonstar_wand", 0.40, 0.45, CHECKPOINT_SCENE, "regalia300", "equipment600"),
    (121010, "wand900_w030", "moonstar_wand", 0.40, 0.30, CHECKPOINT_SCENE, "regalia300", "equipment"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("stress", "calibration", "checkpoints"), default="stress"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.profile == "stress":
        cases = STRESS_CASES
    elif args.profile == "calibration":
        cases = CALIBRATION_CASES
    else:
        cases = CHECKPOINT_CASES
    prompts = []
    for case in cases:
        seed, case_id, item_id, regalia_weight, equipment_weight, scene = case[:6]
        regalia_adapter, equipment_adapter = (
            case[6:8] if len(case) == 8 else ("regalia", "equipment")
        )
        prompts.append(
            {
                "seed": seed,
                "case_id": case_id,
                "item_id": item_id,
                "regalia_weight": regalia_weight,
                "equipment_weight": equipment_weight,
                "regalia_adapter": regalia_adapter,
                "equipment_adapter": equipment_adapter,
                "prompt": (
                    f"{scene}, solo, lqxl Little Queen, lqmoonfit, {REGALIA}, "
                    f"{ITEMS[item_id]}, polished anime storybook illustration"
                ),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "description": f"Canonical Little Queen regalia and equipment {args.profile} grid",
                "image_count": len(prompts),
                "prompts": prompts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(prompts)} {args.profile} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
