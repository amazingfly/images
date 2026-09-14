#!/usr/bin/env python3
"""Compare v2 checkpoints and run a minimal-prompt SDXL identity test on Colab."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/lqxl_v2_test")
LOGS = ROOT / "logs"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
WIDTH = 768
HEIGHT = 1024
STEPS = 28
GUIDANCE_SCALE = 5.0
LORA_WEIGHT = 1.00

COMPARE_LORAS = {
    "step900": Path("/content/lqxl_v2_step900.safetensors"),
    "final1200": Path("/content/lqxl_v2_final1200.safetensors"),
}
SELECTED_LORA = Path("/content/lqxl_v2_selected.safetensors")
COMPARE_PROMPTS = [
    (97001, "solo, lqxl Little Queen, full-body, standing in a quiet garden"),
    (97002, "solo, lqxl Little Queen, full-body, standing inside a bright hall"),
    (97003, "solo, lqxl Little Queen, full-body, standing beneath a cloudy sky"),
    (97004, "solo, lqxl Little Queen, full-body, standing beside a forest stream"),
    (97005, "solo, lqxl Little Queen, full-body, standing on a village street"),
]
FINAL_PROMPTS = [
    (97101, "solo, lqxl Little Queen, full-body, standing in a quiet garden"),
    (97102, "solo, lqxl Little Queen, full-body, standing inside a bright hall"),
    (97103, "solo, lqxl Little Queen, full-body, standing beneath a cloudy sky"),
    (97104, "solo, lqxl Little Queen, full-body, standing beside a forest stream"),
    (97105, "solo, lqxl Little Queen, full-body, standing on a village street"),
    (97106, "solo, lqxl Little Queen, full-body, standing near a still lake"),
    (97107, "solo, lqxl Little Queen, full-body, standing in an open courtyard"),
    (97108, "solo, lqxl Little Queen, full-body, standing under the evening stars"),
    (97109, "solo, lqxl Little Queen, full-body, standing on a windy hill"),
    (97110, "solo, lqxl Little Queen, full-body, standing among autumn trees"),
]
NEGATIVE_PROMPT = (
    "two people, multiple people, duplicate, twins, clone, cropped head, "
    "cropped feet, out of frame, extra limbs, missing limbs, malformed hands, "
    "adult, photorealistic, text, watermark, blurry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compare", "final"), required=True)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_sample() -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        return {key: "" for key in ("gpu_used_mb", "gpu_total_mb", "gpu_util_pct", "gpu_temp_c")}
    values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    return dict(
        zip(
            ("gpu_used_mb", "gpu_total_mb", "gpu_util_pct", "gpu_temp_c"),
            values,
            strict=True,
        )
    )


def monitor_resources(stop: threading.Event, destination: Path) -> None:
    fields = ["timestamp_utc", "gpu_used_mb", "gpu_total_mb", "gpu_util_pct", "gpu_temp_c"]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            writer.writerow(
                {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **gpu_sample()}
            )
            stream.flush()
            stop.wait(2)


def main() -> int:
    args = parse_args()
    output = ROOT / ("compare" if args.mode == "compare" else "minimal10")
    archive = Path(f"/content/lqxl_v2_{args.mode}_results.tar.gz")
    shutil.rmtree(output, ignore_errors=True)
    shutil.rmtree(LOGS, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "diffusers==0.34.0",
            "transformers==4.49.0",
            "accelerate==1.4.0",
            "peft==0.15.2",
            "safetensors>=0.4.5",
        ]
    )

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu_name = torch.cuda.get_device_name(0)
    run(["nvidia-smi"])
    if args.mode == "compare":
        loras = COMPARE_LORAS
        prompts = COMPARE_PROMPTS
    else:
        loras = {"selected": SELECTED_LORA}
        prompts = FINAL_PROMPTS
    for lora in loras.values():
        if not lora.is_file():
            raise FileNotFoundError(lora)

    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_resources,
        args=(stop, LOGS / f"{args.mode}_resource_usage.csv"),
        daemon=True,
    )
    monitor.start()
    results = []
    try:
        for label, lora in loras.items():
            adapter_output = output / label
            adapter_output.mkdir(parents=True, exist_ok=True)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                add_watermarker=False,
                low_cpu_mem_usage=True,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
            pipe.load_lora_weights(str(lora), adapter_name=label)
            pipe.set_adapters([label], adapter_weights=[LORA_WEIGHT])
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()
            for index, (seed, prompt) in enumerate(prompts, start=1):
                destination = adapter_output / f"image_{index:02d}_seed_{seed}.png"
                started = time.monotonic()
                print(f"[{label} {index:02d}/{len(prompts)}] {prompt}", flush=True)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                with torch.inference_mode():
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        width=WIDTH,
                        height=HEIGHT,
                        num_inference_steps=STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=generator,
                    ).images[0]
                image.save(destination, format="PNG", compress_level=6)
                results.append(
                    {
                        "adapter": label,
                        "index": index,
                        "seed": seed,
                        "prompt": prompt,
                        "output": str(destination),
                        "duration_seconds": round(time.monotonic() - started, 2),
                    }
                )
                del image
                torch.cuda.empty_cache()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        stop.set()
        monitor.join(timeout=10)

    summary = {
        "status": "complete",
        "mode": args.mode,
        "model": MODEL_ID,
        "precision": "fp16",
        "resolution": [WIDTH, HEIGHT],
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "lora_weight": LORA_WEIGHT,
        "loras": {label: {"path": str(path), "sha256": sha256(path)} for label, path in loras.items()},
        "gpu": gpu_name,
        "positive_prompts_only_describe_trigger_base_class_and_vague_scene": True,
        "results": results,
    }
    summary_path = ROOT / f"{args.mode}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    archive.unlink(missing_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(output, arcname="output")
        bundle.add(summary_path, arcname="run_summary.json")
        bundle.add(LOGS, arcname="logs")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {archive} ({archive.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
