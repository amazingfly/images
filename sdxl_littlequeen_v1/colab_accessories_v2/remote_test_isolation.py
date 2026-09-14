#!/usr/bin/env python3
"""Run identity-only isolation tests for the v2 crown and wand-staff LoRAs."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_PATH = Path("/content/remote_test_accessories_base.py")
spec = importlib.util.spec_from_file_location("accessory_test_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load test base: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.ROOT = Path("/content/lqaccessories_v2_isolation")
base.OUTPUT = base.ROOT / "test_output"
base.PROMPTS = Path("/content/lqaccessories_v2_isolation_prompts.json")
base.ARCHIVE = Path("/content/lqaccessories_v2_isolation_results.tar.gz")
base.MODEL = base.ROOT / "models" / "sd_xl_base_1.0.safetensors"
base.IDENTITY = base.ROOT / "test_models" / "lqxl_sdxl_v2.safetensors"
base.OUTFIT = base.ROOT / "test_models" / "lqmoonfit_sdxl_v1.safetensors"
base.REGALIA = base.ROOT / "test_models" / "lqmooncrest_sdxl_v2.safetensors"
base.EQUIPMENT = base.ROOT / "test_models" / "lqrosewand_sdxl_v2.safetensors"
base.REGALIA_300 = base.ROOT / "unused_regalia_checkpoint.safetensors"
base.EQUIPMENT_300 = base.ROOT / "unused_equipment_checkpoint.safetensors"
base.EQUIPMENT_600 = base.ROOT / "unused_equipment_midpoint.safetensors"
base.OPTIONAL_UPLOADS = ()
base.UPLOADS = (
    (
        Path("/content/lqxl_sdxl_v2.safetensors.part_"),
        base.IDENTITY,
        170_552_852,
        "3799128d4bfd4fbc7848d5b1de099a3e99cf3f718845c98efa7d4d48a7953994",
    ),
    (
        Path("/content/lqmoonfit_sdxl_v1.safetensors.part_"),
        base.OUTFIT,
        170_545_956,
        "985e82df407690831ef8d27a9c6c3b061689278c334e3694e21eaf8aa88c36d7",
    ),
    (
        Path("/content/lqmooncrest_sdxl_v2.safetensors.part_"),
        base.REGALIA,
        42_865_452,
        "78c7bc739910fe2bcdb933b425e97092866499940f96dcea44bbee67ab11e091",
    ),
    (
        Path("/content/lqrosewand_sdxl_v2.safetensors.part_"),
        base.EQUIPMENT,
        85_424_948,
        "38b489dc87ede3986ed4ff22230e1ae5c0c35a2001d802cc32b3c9012eb64524",
    ),
)
base.NEGATIVE = (
    "cropped head, cropped crown, cropped feet, out of frame, two girls, multiple people, "
    "duplicate, twin, adult woman, extra limbs, extra hands, malformed hands, multiple crowns, "
    "floating crown, multiple staffs, duplicate staff, extra wand, broken shaft, bent shaft, "
    "staff through body, fused equipment, rose garden, floral archway, photorealistic, text, "
    "watermark, blurry, low detail"
)
raise SystemExit(base.main())
