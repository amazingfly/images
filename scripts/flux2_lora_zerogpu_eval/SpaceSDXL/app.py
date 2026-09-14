#!/usr/bin/env python3
"""Deterministic SDXL Little Queen identity evaluation on ZeroGPU."""

from __future__ import annotations

import spaces

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


BASE_MODEL = os.environ.get(
    "SDXL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
)
RESULT_REPO = os.environ.get(
    "RESULT_REPO", "amazingfly/little-queen-lora-eval-results"
)
LORA_REPO = os.environ.get(
    "SDXL_LORA_REPO", "amazingfly/little-queen-sdxl-lora-v2"
)
LORA_WEIGHT = os.environ.get("SDXL_LORA_WEIGHT", "lqxl_sdxl_v2.safetensors")
TRIGGER = os.environ.get("SDXL_TRIGGER", "lqxl Little Queen")
STYLE_PREFIX = os.environ.get(
    "STYLE_PREFIX",
    "Polished children's storybook illustration, luminous painterly digital art, "
    "clear readable composition, expressive natural anatomy.",
)
NEGATIVE_PROMPT = os.environ.get(
    "SDXL_NEGATIVE_PROMPT",
    "two people, multiple people, duplicate, twins, clone, cropped head, cropped "
    "feet, out of frame, extra limbs, missing limbs, malformed hands, adult, "
    "photorealistic, text, watermark, blurry",
)
WIDTH = int(os.environ.get("SDXL_WIDTH", "768"))
HEIGHT = int(os.environ.get("SDXL_HEIGHT", "1024"))
NUM_INFERENCE_STEPS = int(os.environ.get("SDXL_NUM_INFERENCE_STEPS", "28"))
GUIDANCE_SCALE = float(os.environ.get("SDXL_GUIDANCE_SCALE", "5.0"))
GPU_DURATION_SECONDS = int(os.environ.get("GPU_DURATION_SECONDS", "60"))
TOKEN = os.environ.get("HF_TOKEN", "").strip()
WORK_ROOT = Path("/tmp/little_queen_sdxl_eval")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,100}$")
VARIANT = "sdxl_v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_job(raw: str | dict) -> dict:
    job = json.loads(raw) if isinstance(raw, str) else dict(raw)
    required = ("id", "variant", "scene", "seed", "lora_scale", "prompt")
    missing = [field for field in required if field not in job]
    if missing:
        raise ValueError(f"job is missing fields: {', '.join(missing)}")
    if not ID_RE.fullmatch(str(job["id"])):
        raise ValueError("job.id is invalid")
    if job["variant"] != VARIANT:
        raise ValueError(f"unsupported variant: {job['variant']}")
    scale = float(job["lora_scale"])
    if not 0.0 < scale <= 1.5:
        raise ValueError("lora_scale must be between 0 and 1.5")
    prompt = str(job["prompt"]).strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    return {
        "id": str(job["id"]),
        "variant": str(job["variant"]),
        "scene": int(job["scene"]),
        "seed": int(job["seed"]),
        "lora_scale": scale,
        "prompt": prompt,
    }


def result_paths(job_id: str) -> tuple[str, str]:
    return f"images/{job_id}.png", f"metadata/{job_id}.json"


def existing_result(api: HfApi, job: dict) -> dict | None:
    image_remote, metadata_remote = result_paths(job["id"])
    files = set(api.list_repo_files(RESULT_REPO, repo_type="dataset"))
    if image_remote not in files or metadata_remote not in files:
        return None
    metadata_path = hf_hub_download(
        RESULT_REPO, metadata_remote, repo_type="dataset", token=TOKEN
    )
    record = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return record if record.get("job") == job and record.get("ok") else None


if not TOKEN:
    raise RuntimeError("HF_TOKEN Space secret is missing")

# This is the exact reviewed identity recipe: FP16 SDXL base, DPM++ Karras,
# 28 steps, CFG 5, and the final 1200-step identity LoRA at weight 1.0.
PIPE = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    add_watermarker=False,
    low_cpu_mem_usage=True,
    token=TOKEN,
)
PIPE.scheduler = DPMSolverMultistepScheduler.from_config(
    PIPE.scheduler.config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)
PIPE.load_lora_weights(
    LORA_REPO,
    weight_name=LORA_WEIGHT,
    adapter_name=VARIANT,
    token=TOKEN,
)
PIPE.set_adapters([VARIANT], adapter_weights=[1.0])
PIPE.enable_vae_slicing()
PIPE.to("cuda")


@spaces.GPU(duration=GPU_DURATION_SECONDS, size="large")
def render_gpu(job_json: str, request_id: str) -> str:
    job = validate_job(job_json)
    if request_id != job["id"]:
        raise ValueError("request_id must equal the deterministic job id")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    output = WORK_ROOT / f"{job['id']}.png"
    PIPE.set_adapters([VARIANT], adapter_weights=[job["lora_scale"]])
    effective_prompt = f"{TRIGGER}, {STYLE_PREFIX} {job['prompt']}"
    generator = torch.Generator(device="cuda").manual_seed(job["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = PIPE(
        prompt=effective_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    temporary = output.with_suffix(".tmp.png")
    result.images[0].save(temporary, format="PNG")
    temporary.replace(output)
    payload = {
        "path": str(output),
        "effective_prompt": effective_prompt,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 3
        ),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }
    return json.dumps(payload, sort_keys=True)


def run_job(request_id: str, job_json: str) -> dict:
    job = validate_job(job_json)
    if request_id != job["id"]:
        raise ValueError("request_id must equal the deterministic job id")
    api = HfApi(token=TOKEN)
    prior = existing_result(api, job)
    if prior:
        return {
            "status": "already_complete",
            "request_id": request_id,
            "job_id": job["id"],
            "variant": job["variant"],
            "image_sha256": prior["image"]["sha256"],
        }
    rendered = json.loads(render_gpu(json.dumps(job, sort_keys=True), request_id))
    image_path = Path(rendered["path"])
    if not image_path.is_file():
        raise RuntimeError("GPU renderer did not produce its image")
    image_remote, metadata_remote = result_paths(job["id"])
    record = {
        "ok": True,
        "version": "little-queen-sdxl-zerogpu-eval-v1",
        "completed_at": now(),
        "request_id": request_id,
        "job": job,
        "base_model": BASE_MODEL,
        "adapter": {"repo": LORA_REPO, "weight_name": LORA_WEIGHT},
        "settings": {
            "dtype": "float16",
            "scheduler": "DPM-Solver++ with Karras sigmas",
            "width": WIDTH,
            "height": HEIGHT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "trigger": TRIGGER,
            "style_prefix": STYLE_PREFIX,
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "effective_prompt": rendered["effective_prompt"],
        "image": {
            "path": image_remote,
            "bytes": image_path.stat().st_size,
            "sha256": sha256(image_path),
        },
        "runtime": {key: value for key, value in rendered.items() if key != "path"},
    }
    metadata_path = WORK_ROOT / f"{job['id']}.json"
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    api.create_commit(
        repo_id=RESULT_REPO,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=image_remote, path_or_fileobj=image_path),
            CommitOperationAdd(
                path_in_repo=metadata_remote, path_or_fileobj=metadata_path
            ),
        ],
        commit_message=f"Complete deterministic SDXL evaluation job {job['id']}",
    )
    return {
        "status": "completed",
        "request_id": request_id,
        "job_id": job["id"],
        "variant": job["variant"],
        "image_sha256": record["image"]["sha256"],
        "elapsed_seconds": record["runtime"]["elapsed_seconds"],
    }


with gr.Blocks() as demo:
    gr.Markdown("# Private Little Queen SDXL Identity Evaluator")
    request = gr.Textbox(label="Deterministic request ID")
    job = gr.Textbox(label="Job JSON", lines=10)
    output = gr.JSON(label="Durable result")
    button = gr.Button("Run one matched SDXL evaluation job")
    button.click(
        run_job,
        inputs=[request, job],
        outputs=output,
        api_name="run_job",
        concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1).launch()
