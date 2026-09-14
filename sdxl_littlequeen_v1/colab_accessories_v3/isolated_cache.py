#!/usr/bin/env python3
"""Run Kohya SDXL cache tools while releasing unneeded model components."""

from __future__ import annotations

import gc
import runpy
import sys
from pathlib import Path

import torch


ROOT = Path("/content/lqaccessories_v3")
SD_SCRIPTS = ROOT / "sd-scripts"


def release(*models: object) -> None:
    for model in models:
        if model is not None and hasattr(model, "to"):
            model.to("meta")
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"latents", "text"}:
        raise SystemExit("usage: isolated_cache.py {latents|text} [kohya arguments]")
    mode = sys.argv.pop(1)
    sys.path.insert(0, str(SD_SCRIPTS))
    sys.path.insert(0, str(SD_SCRIPTS / "tools"))

    from library import sdxl_train_util

    original_loader = sdxl_train_util.load_target_model

    def isolated_loader(args, accelerator, model_version, weight_dtype):
        loaded = original_loader(args, accelerator, model_version, weight_dtype)
        stable_format, text_encoder1, text_encoder2, vae, unet, scale, info = loaded
        if mode == "latents":
            release(text_encoder1, text_encoder2, unet)
            text_encoder1 = text_encoder2 = unet = None
        else:
            release(vae, unet)
            vae = unet = None
        print(
            f"isolated {mode} cache: "
            f"{torch.cuda.memory_allocated() / (1024 * 1024):.1f} MiB allocated",
            flush=True,
        )
        return stable_format, text_encoder1, text_encoder2, vae, unet, scale, info

    sdxl_train_util.load_target_model = isolated_loader
    tool = "cache_latents.py" if mode == "latents" else "cache_text_encoder_outputs.py"
    runpy.run_path(str(SD_SCRIPTS / "tools" / tool), run_name="__main__")


if __name__ == "__main__":
    main()
