#!/usr/bin/env python3
"""Deterministic FLUX.2 Little Queen LoRA evaluation on ZeroGPU."""

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
from diffusers import Flux2KleinPipeline
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


BASE_MODEL = os.environ.get(
    "BASE_MODEL", "black-forest-labs/FLUX.2-klein-base-4B"
)
RESULT_REPO = os.environ.get(
    "RESULT_REPO", "amazingfly/little-queen-lora-eval-results"
)
V1_REPO = os.environ.get(
    "V1_REPO", "amazingfly/little-queen-flux2-klein-lora-v1"
)
V1_WEIGHT = os.environ.get(
    "V1_WEIGHT", "littlequeen-flux2-klein-v1.safetensors"
)
V2_REPO = os.environ.get(
    "V2_REPO", "amazingfly/little-queen-flux2-klein-lora-v2"
)
V2_WEIGHT = os.environ.get(
    "V2_WEIGHT", "littlequeen-flux2-klein-v2.safetensors"
)
TRIGGER = os.environ.get("TRIGGER", "LQK4N")
STYLE_PREFIX = os.environ.get(
    "STYLE_PREFIX",
    "Polished children's storybook illustration, luminous painterly digital art, "
    "clear readable composition, expressive natural anatomy.",
)
WIDTH = int(os.environ.get("WIDTH", "832"))
HEIGHT = int(os.environ.get("HEIGHT", "1216"))
NUM_INFERENCE_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "4.0"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "512"))
GPU_DURATION_SECONDS = int(os.environ.get("GPU_DURATION_SECONDS", "60"))
TOKEN = os.environ.get("HF_TOKEN", "").strip()
WORK_ROOT = Path("/tmp/little_queen_lora_eval")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,100}$")
VARIANTS = {
    "flux_v1": {"repo": V1_REPO, "weight_name": V1_WEIGHT},
    "flux_v2": {"repo": V2_REPO, "weight_name": V2_WEIGHT},
}


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
    if job["variant"] not in VARIANTS:
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
        RESULT_REPO,
        metadata_remote,
        repo_type="dataset",
        token=TOKEN,
    )
    record = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return record if record.get("job") == job and record.get("ok") else None


if not TOKEN:
    raise RuntimeError("HF_TOKEN Space secret is missing")

# ZeroGPU optimizes root-level CUDA placement through its CUDA emulation layer.
# Both adapters share this one BF16 pipeline, ensuring a controlled comparison.
PIPE = Flux2KleinPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    token=TOKEN,
)
PIPE.vae.enable_slicing()
PIPE.vae.enable_tiling()
PIPE.load_lora_weights(
    V1_REPO,
    weight_name=V1_WEIGHT,
    adapter_name="flux_v1",
    token=TOKEN,
)
PIPE.load_lora_weights(
    V2_REPO,
    weight_name=V2_WEIGHT,
    adapter_name="flux_v2",
    token=TOKEN,
)
PIPE.to("cuda")


@spaces.GPU(duration=GPU_DURATION_SECONDS, size="large")
def render_gpu(job_json: str, request_id: str) -> str:
    job = validate_job(job_json)
    if request_id != job["id"]:
        raise ValueError("request_id must equal the deterministic job id")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    output = WORK_ROOT / f"{job['id']}.png"
    PIPE.set_adapters(job["variant"], adapter_weights=job["lora_scale"])
    effective_prompt = f"{TRIGGER}. {STYLE_PREFIX} {job['prompt']}"
    generator = torch.Generator(device="cuda").manual_seed(job["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = PIPE(
        prompt=effective_prompt,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
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
        "version": "little-queen-lora-zerogpu-eval-v1",
        "completed_at": now(),
        "request_id": request_id,
        "job": job,
        "base_model": BASE_MODEL,
        "adapter": VARIANTS[job["variant"]],
        "settings": {
            "dtype": "bfloat16",
            "width": WIDTH,
            "height": HEIGHT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "trigger": TRIGGER,
            "style_prefix": STYLE_PREFIX,
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
        commit_message=f"Complete deterministic evaluation job {job['id']}",
    )
    # Do not return Hub-relative file paths through Gradio JSON. Gradio treats
    # strings ending in image extensions as Space files and tries to download
    # them, while the durable artifact belongs to the private Hub result repo.
    return {
        "status": "completed",
        "request_id": request_id,
        "job_id": job["id"],
        "variant": job["variant"],
        "image_sha256": record["image"]["sha256"],
        "elapsed_seconds": record["runtime"]["elapsed_seconds"],
    }


with gr.Blocks() as demo:
    gr.Markdown("# Private Little Queen LoRA Evaluator")
    request = gr.Textbox(label="Deterministic request ID")
    job = gr.Textbox(label="Job JSON", lines=10)
    output = gr.JSON(label="Durable result")
    button = gr.Button("Run one matched evaluation job")
    button.click(
        run_job,
        inputs=[request, job],
        outputs=output,
        api_name="run_job",
        concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1).launch()
