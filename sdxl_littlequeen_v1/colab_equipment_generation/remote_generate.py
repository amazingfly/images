#!/usr/bin/env python3
"""Generate the Moonstar outfit/equipment dataset on a Colab T4."""

from __future__ import annotations

import argparse
import csv
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


ROOT = Path("/content/lqxl_equipment_generation")
OUTPUT = ROOT / "output"
LOGS = ROOT / "logs"
PROMPTS = Path("/content/lqxl_equipment_prompts.json")
LORA_PART_PREFIX = Path("/content/lqxl_sdxl_v2.safetensors.part_")
LORA = ROOT / "models" / "lqxl_sdxl_v2.safetensors"
ARCHIVE = Path("/content/lqxl_equipment_results.tar.gz")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_SHA256 = "3799128d4bfd4fbc7848d5b1de099a3e99cf3f718845c98efa7d4d48a7953994"
LORA_BYTES = 170_552_852
WIDTH = 832
HEIGHT = 1216
STEPS = 28
GUIDANCE_SCALE = 5.0
LORA_WEIGHT = 0.90

NEGATIVE_PROMPT = (
    "close-up crop, cropped body, cropped head, cropped crown, cropped feet, out of "
    "frame, two girls, multiple people, twin, clone, adult woman, mature body, extra "
    "arms, extra hands, extra fingers, missing limbs, fused hands, malformed face, "
    "asymmetrical eyes, duplicate crown, extra crown, floating crown, two weapons, "
    "multiple weapons, duplicate equipment, fused weapon, bent weapon, broken weapon, "
    "weapon through body, photorealistic, text, watermark, blurry, low detail"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def install_dependencies() -> None:
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
            "compel==2.0.3",
            "safetensors>=0.4.5",
        ]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_uploaded_lora() -> None:
    if LORA.is_file() and LORA.stat().st_size == LORA_BYTES and sha256(LORA) == LORA_SHA256:
        print(f"Using verified cached LoRA: {LORA}", flush=True)
        return
    parts = sorted(Path("/content").glob(LORA_PART_PREFIX.name + "*"))
    if not parts:
        raise FileNotFoundError(f"No uploaded LoRA chunks match {LORA_PART_PREFIX}*")
    LORA.parent.mkdir(parents=True, exist_ok=True)
    temporary = LORA.with_suffix(".assembling")
    with temporary.open("wb") as destination:
        for part in parts:
            print(f"Appending {part.name}", flush=True)
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
    if temporary.stat().st_size != LORA_BYTES:
        raise RuntimeError(f"LoRA size mismatch: {temporary.stat().st_size} != {LORA_BYTES}")
    actual = sha256(temporary)
    if actual != LORA_SHA256:
        raise RuntimeError(f"LoRA checksum mismatch: {actual}")
    temporary.replace(LORA)


def valid_png(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return path.stat().st_size > 1024
    except (OSError, ValueError):
        return False


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
    keys = ("gpu_used_mb", "gpu_total_mb", "gpu_util_pct", "gpu_temp_c")
    if result.returncode or not result.stdout.strip():
        return {key: "" for key in keys}
    return dict(zip(keys, (value.strip() for value in result.stdout.splitlines()[0].split(",")), strict=True))


def monitor_resources(stop: threading.Event) -> None:
    destination = LOGS / "resource_usage.csv"
    fields = ["timestamp_utc", "gpu_used_mb", "gpu_total_mb", "gpu_util_pct", "gpu_temp_c"]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            writer.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **gpu_sample()})
            stream.flush()
            stop.wait(2)


def write_summary(summary: dict) -> None:
    (ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_archive() -> None:
    ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="output")
        archive.add(ROOT / "run_summary.json", arcname="run_summary.json")
        archive.add(LOGS, arcname="logs")
    print(f"Wrote result archive: {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)", flush=True)


def main() -> int:
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.reset:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        shutil.rmtree(LOGS, ignore_errors=True)
        (ROOT / "run_summary.json").unlink(missing_ok=True)
        ARCHIVE.unlink(missing_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if not PROMPTS.is_file():
        raise FileNotFoundError(PROMPTS)

    install_dependencies()
    assemble_uploaded_lora()

    import torch
    from compel import Compel, ReturnedEmbeddingsType
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Colab session")
    gpu_name = torch.cuda.get_device_name(0)
    run(["nvidia-smi"])

    records = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    if len(records) != 100:
        raise RuntimeError(f"Expected 100 prompts, found {len(records)}")

    print(f"Loading {MODEL_ID} in FP16", flush=True)
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
    pipe.load_lora_weights(str(LORA), adapter_name="lqxl")
    pipe.set_adapters(["lqxl"], adapter_weights=[LORA_WEIGHT])
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )
    negative, negative_pooled = compel(NEGATIVE_PROMPT)

    try:
        summary = json.loads((ROOT / "run_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {
            "name": "littlequeen_moonstar_equipment_dataset100_colab_fp16",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "precision": "fp16",
            "quantized": False,
            "resolution": [WIDTH, HEIGHT],
            "sampler": "DPM++ 2M Karras",
            "steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "lora": {"name": "lqxl_sdxl_v2", "sha256": LORA_SHA256, "weight": LORA_WEIGHT},
            "long_prompt_conditioning": "Compel, no truncation",
            "gpu": gpu_name,
            "results": [],
        }
    completed = {
        int(result["index"]): result
        for result in summary.get("results", [])
        if result.get("status") == "completed" and valid_png(Path(result["output"]))
    }

    stop = threading.Event()
    monitor = threading.Thread(target=monitor_resources, args=(stop,), daemon=True)
    monitor.start()
    try:
        for index, record in enumerate(records, start=1):
            if index in completed:
                print(f"[{index:03d}] skipping completed image", flush=True)
                continue
            destination = OUTPUT / f"image_{index:03d}_seed_{record['seed']}.png"
            with torch.inference_mode():
                positive, pooled = compel(record["prompt"])
                positive, padded_negative = compel.pad_conditioning_tensors_to_same_length(
                    [positive, negative]
                )
                started = time.monotonic()
                print(
                    f"[{index:03d}] generating seed {record['seed']} "
                    f"({record['scene_id']}/{record['pose_id']}/{record['equipment_id']})",
                    flush=True,
                )
                generator = torch.Generator(device="cuda").manual_seed(int(record["seed"]))
                image = pipe(
                    prompt_embeds=positive,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=padded_negative,
                    negative_pooled_prompt_embeds=negative_pooled,
                    width=WIDTH,
                    height=HEIGHT,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                ).images[0]
            image.save(destination, format="PNG", compress_level=6)
            if not valid_png(destination):
                raise RuntimeError(f"Invalid output image: {destination}")
            result = {
                "index": index,
                "seed": int(record["seed"]),
                "scene_id": record["scene_id"],
                "pose_id": record["pose_id"],
                "equipment_id": record["equipment_id"],
                "prompt": record["prompt"],
                "output": str(destination),
                "width": WIDTH,
                "height": HEIGHT,
                "duration_seconds": round(time.monotonic() - started, 2),
                "status": "completed",
            }
            summary["results"] = [
                prior for prior in summary.get("results", []) if int(prior["index"]) != index
            ]
            summary["results"].append(result)
            summary["results"].sort(key=lambda item: int(item["index"]))
            summary["status"] = "running"
            write_summary(summary)
            print(f"[{index:03d}] completed in {result['duration_seconds']:.1f}s", flush=True)
            del image, positive, pooled, padded_negative
            torch.cuda.empty_cache()
    finally:
        stop.set()
        monitor.join(timeout=10)

    summary["status"] = "complete"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(summary)
    build_archive()
    print(f"Completed all 100 images on {gpu_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
