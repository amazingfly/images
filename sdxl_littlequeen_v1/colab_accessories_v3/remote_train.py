#!/usr/bin/env python3
"""Configure contextual, localized v3 regalia and staff LoRA training."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


BASE_PATH = Path("/content/remote_train_accessories_base.py")
spec = importlib.util.spec_from_file_location("accessory_train_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load training base: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.ROOT = Path("/content/lqaccessories_v3")
base.BUNDLE = Path("/content/lqaccessories_v3_bundle.tar.gz")
base.BUNDLE_PART_PREFIX = Path("/content/lqaccessories_v3_bundle.tar.gz.part_")
base.SD_SCRIPTS = base.ROOT / "sd-scripts"
base.LOGS = base.ROOT / "logs"
base.MODEL_DIR = base.ROOT / "models"
base.JOBS = (
    {
        "id": "regalia",
        "config": base.ROOT / "dataset_regalia.toml",
        "output": base.ROOT / "output_regalia",
        "name": "lqmoonregalia_sdxl_v3",
        "dim": 8,
        "alpha": 4,
        "steps": 300,
        "seed": 127001,
    },
    {
        "id": "staff",
        "config": base.ROOT / "dataset_staff.toml",
        "output": base.ROOT / "output_staff",
        "name": "lqrosekeeper_sdxl_v3",
        "dim": 8,
        "alpha": 4,
        "steps": 320,
        "seed": 127002,
    },
)

original_extract_bundle = base.extract_bundle


def extract_bundle() -> None:
    original_extract_bundle()
    fixed_cache = Path("/content/isolated_cache_v3_fixed.py")
    if fixed_cache.is_file():
        shutil.copy2(fixed_cache, base.ROOT / "isolated_cache.py")


base.extract_bundle = extract_bundle

original_training_command = base.training_command


def training_command(model: Path, job: dict) -> list[str]:
    command = original_training_command(model, job)
    return [
        argument.replace("--save_every_n_steps=300", "--save_every_n_steps=100")
        .replace("--learning_rate=0.0001", "--learning_rate=0.00002")
        .replace("--unet_lr=0.0001", "--unet_lr=0.00002")
        for argument in command
    ]


base.training_command = training_command
raise SystemExit(base.main())
