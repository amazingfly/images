#!/usr/bin/env python3
"""Configure the proven accessory trainer for decorrelated v2 object LoRAs."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_PATH = Path("/content/remote_train_accessories_base.py")
spec = importlib.util.spec_from_file_location("accessory_train_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load training base: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.ROOT = Path("/content/lqaccessories_v2")
base.BUNDLE = Path("/content/lqaccessories_v2_bundle.tar.gz")
base.BUNDLE_PART_PREFIX = Path("/content/lqaccessories_v2_bundle.tar.gz.part_")
base.SD_SCRIPTS = base.ROOT / "sd-scripts"
base.LOGS = base.ROOT / "logs"
base.MODEL_DIR = base.ROOT / "models"
base.JOBS = (
    {
        "id": "crown",
        "config": base.ROOT / "dataset_crown.toml",
        "output": base.ROOT / "output_crown",
        "name": "lqmooncrest_sdxl_v2",
        "dim": 8,
        "alpha": 4,
        "steps": 400,
        "seed": 123001,
    },
    {
        "id": "wand",
        "config": base.ROOT / "dataset_wand.toml",
        "output": base.ROOT / "output_wand",
        "name": "lqrosewand_sdxl_v2",
        "dim": 16,
        "alpha": 8,
        "steps": 500,
        "seed": 123002,
    },
)

original_training_command = base.training_command


def training_command(model: Path, job: dict) -> list[str]:
    command = original_training_command(model, job)
    command = [
        argument.replace("--save_every_n_steps=300", "--save_every_n_steps=250")
        .replace("--learning_rate=0.0001", "--learning_rate=0.00005")
        .replace("--unet_lr=0.0001", "--unet_lr=0.00005")
        for argument in command
    ]
    return command


base.training_command = training_command
raise SystemExit(base.main())
