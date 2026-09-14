#!/usr/bin/env python3
"""Generate the native-resolution Little Queen dataset on a Colab T4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/lqxl_dataset_generation")
OUTPUT = ROOT / "output"
LOGS = ROOT / "logs"
PROMPTS = Path("/content/lqxl_generation_prompts.json")
LORA = ROOT / "models" / "lqxl_sdxl_v1.safetensors"
ARCHIVE = Path("/content/lqxl_generation_results.tar.gz")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DRIVE_FILE_ID = "1u01FIhCTDsxKIxOXfy9QvQbEclHwaacG"
LORA_SHA256 = "5a21f64d626b7916c5e08189a127b9e36e36c8014a97b1101d394acf7f19db01"
LORA_BYTES = 170_553_908
WIDTH = 832
HEIGHT = 1216
STEPS = 28
GUIDANCE_SCALE = 5.0
LORA_WEIGHT = 0.80

NEGATIVE_PROMPT = (
    "close-up crop, cropped body, out of frame, cropped crown, cropped feet, "
    "missing arms, missing forearms, missing hands, hidden hands, extra limbs, "
    "extra arms, extra fingers, fused hands, extra crown, duplicate crown, "
    "floating crown, two girls, multiple people, twin, clone, adult woman, "
    "mature body, broad adult shoulders, low neckline, cleavage, photorealistic, "
    "blonde hair, black hair, blue dress, silver crown, malformed face, "
    "asymmetrical eyes, crossed eyes, open mouth, visible teeth, text, watermark, "
    "low detail, blurry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
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


def download_drive_lora() -> None:
    if (
        LORA.is_file()
        and LORA.stat().st_size == LORA_BYTES
        and sha256(LORA) == LORA_SHA256
    ):
        print(f"Using verified cached Drive LoRA: {LORA}", flush=True)
        return
    LORA.parent.mkdir(parents=True, exist_ok=True)
    temporary = LORA.with_suffix(".download")
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={DRIVE_FILE_ID}&export=download&confirm=t"
    )
    print(f"Downloading the shared Drive LoRA file {DRIVE_FILE_ID}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    if temporary.stat().st_size != LORA_BYTES:
        raise RuntimeError(
            f"Drive LoRA size mismatch: {temporary.stat().st_size} != {LORA_BYTES}"
        )
    actual = sha256(temporary)
    if actual != LORA_SHA256:
        raise RuntimeError(f"Drive LoRA checksum mismatch: {actual}")
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


def monitor_resources(stop: threading.Event) -> None:
    destination = LOGS / "resource_usage.csv"
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


def write_summary(summary: dict) -> None:
    (ROOT / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def build_archive() -> None:
    ARCHIVE.unlink(missing_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="output")
        archive.add(ROOT / "run_summary.json", arcname="run_summary.json")
        if LOGS.is_dir():
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
    download_drive_lora()

    import torch
    from compel import Compel, ReturnedEmbeddingsType
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Colab session")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name:
        print(f"WARNING: requested T4 but received {gpu_name}", flush=True)
    run(["nvidia-smi"])

    prompt_data = json.loads(PROMPTS.read_text(encoding="utf-8"))
    records = prompt_data["prompts"]
    if len(records) != 100:
        raise RuntimeError(f"Expected 100 prompts, found {len(records)}")
    target_records = records[: args.limit] if args.limit else records

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

    unique_prompts = list(dict.fromkeys(record["prompt"] for record in records))
    print(f"Encoding {len(unique_prompts)} unique long prompts without truncation", flush=True)
    conditioning: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        for prompt in unique_prompts:
            positive, pooled = compel(prompt)
            negative, negative_pooled = compel(NEGATIVE_PROMPT)
            positive, negative = compel.pad_conditioning_tensors_to_same_length(
                [positive, negative]
            )
            conditioning[prompt] = (
                positive.cpu(),
                pooled.cpu(),
                negative.cpu(),
                negative_pooled.cpu(),
            )

    summary_path = ROOT / "run_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {
            "name": "littlequeen_sdxl_identity_dataset100_colab_fp16",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "precision": "fp16",
            "quantized": False,
            "resolution": [WIDTH, HEIGHT],
            "sampler": "DPM++ 2M Karras",
            "steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "lora": {
                "drive_folder": "1HET7p9xFXZvtN9pzCV6AdoY_tLGOjRT7",
                "drive_file_id": DRIVE_FILE_ID,
                "sha256": LORA_SHA256,
                "weight": LORA_WEIGHT,
            },
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
        for index, record in enumerate(target_records, start=1):
            if index in completed:
                print(f"[{index:03d}] skipping completed image", flush=True)
                continue
            destination = OUTPUT / f"image_{index:03d}_seed_{record['seed']}.png"
            positive, pooled, negative, negative_pooled = conditioning[record["prompt"]]
            started = time.monotonic()
            print(f"[{index:03d}] generating seed {record['seed']} ({record['scene_id']})", flush=True)
            generator = torch.Generator(device="cuda").manual_seed(int(record["seed"]))
            with torch.inference_mode():
                image = pipe(
                    prompt_embeds=positive,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative,
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
            del image
            torch.cuda.empty_cache()
    finally:
        stop.set()
        monitor.join(timeout=10)

    summary["status"] = "smoke_complete" if args.limit else "complete"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(summary)
    build_archive()
    print(f"Completed {len(target_records)} requested images on {gpu_name}", flush=True)
    return 0


if __name__ == "__main__":
    main()
