#!/usr/bin/env python3
"""Segmented FLUX.2 Klein LoRA training endpoint for Hugging Face ZeroGPU."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import spaces
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
DATASET_REPO = os.environ.get("DATASET_REPO", "amazingfly/little-queen-flux2-klein-dataset-v3")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "amazingfly/little-queen-flux2-klein-lora-v3")
TRIGGER = "LQK4N"
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "1200"))
SEGMENT_STEPS = int(os.environ.get("SEGMENT_STEPS", "215"))
GPU_DURATION_FIXED_SECONDS = float(os.environ.get("GPU_DURATION_FIXED_SECONDS", "32"))
GPU_SECONDS_PER_STEP = float(os.environ.get("GPU_SECONDS_PER_STEP", "0.8"))
TAIL_GPU_DURATION_FIXED_SECONDS = float(os.environ.get("TAIL_GPU_DURATION_FIXED_SECONDS", "26"))
TAIL_GPU_SECONDS_PER_STEP = float(os.environ.get("TAIL_GPU_SECONDS_PER_STEP", "0.8"))
GPU_MAX_DURATION_SECONDS = int(os.environ.get("GPU_MAX_DURATION_SECONDS", "200"))
WORK_ROOT = Path("/tmp/lq_flux2_training_v3")
DATA_DIR = WORK_ROOT / "dataset"
OUTPUT_DIR = WORK_ROOT / "output"
TRAINER = Path(__file__).with_name("train_flux2_klein_segmented.py")
STEP_RE = re.compile(r"^checkpoint-(\d+)$")
LOG = logging.getLogger("littlequeen-flux2-space")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def latest_checkpoint(root: Path) -> tuple[int, Path | None]:
    checkpoints = []
    if root.is_dir():
        for path in root.iterdir():
            match = STEP_RE.match(path.name)
            if path.is_dir() and match:
                checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])


def download_training_inputs(token: str) -> tuple[int, str, str, dict, dict]:
    started = time.monotonic()
    timings = {}
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_started = time.monotonic()
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        local_dir=DATA_DIR,
        token=token,
    )
    timings["dataset_snapshot_seconds"] = round(time.monotonic() - phase_started, 3)
    phase_started = time.monotonic()
    snapshot_download(
        OUTPUT_REPO,
        repo_type="model",
        local_dir=OUTPUT_DIR,
        token=token,
        allow_patterns=[
            "checkpoint-*/*",
            "training_state.json",
            "pytorch_lora_weights.safetensors",
            "README.md",
        ],
    )
    timings["checkpoint_snapshot_seconds"] = round(time.monotonic() - phase_started, 3)
    phase_started = time.monotonic()
    model_path = snapshot_download(BASE_MODEL, token=token)
    timings["model_snapshot_seconds"] = round(time.monotonic() - phase_started, 3)
    step, checkpoint = latest_checkpoint(OUTPUT_DIR)
    state_path = OUTPUT_DIR / "training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    timings["download_inputs_seconds"] = round(time.monotonic() - started, 3)
    LOG.info("phase timings input_download=%s", json.dumps(timings, sort_keys=True))
    return step, checkpoint.name if checkpoint else "none", model_path, state, timings


def upload_checkpoint(
    token: str,
    step: int,
    segment_log: Path,
    complete: bool,
    request_id: str,
) -> tuple[str, dict]:
    started = time.monotonic()
    timings = {}
    api = HfApi(token=token)
    checkpoint = OUTPUT_DIR / f"checkpoint-{step}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"expected checkpoint is missing: {checkpoint}")
    state = {
        "version": "little-queen-flux2-klein-training-v3",
        "updated_at": utc_now(),
        "base_model": BASE_MODEL,
        "dataset_repo": DATASET_REPO,
        "output_repo": OUTPUT_REPO,
        "trigger": TRIGGER,
        "latest_step": step,
        "total_steps": TOTAL_STEPS,
        "complete": complete,
        "checkpoint": checkpoint.name,
        "last_request_id": request_id,
    }
    (OUTPUT_DIR / "training_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    lora = checkpoint / "pytorch_lora_weights.safetensors"
    primary_patterns = [f"{checkpoint.name}/**", "training_state.json"]
    if complete:
        if not lora.is_file():
            raise RuntimeError(f"final LoRA is missing from {checkpoint}")
        shutil.copy2(lora, OUTPUT_DIR / "littlequeen-flux2-klein-v3.safetensors")
        primary_patterns.append("littlequeen-flux2-klein-v3.safetensors")
    timings["checkpoint_prepare_seconds"] = round(time.monotonic() - started, 3)

    # Commit the resumable state and its pointer together. An interruption can
    # expose either the prior complete checkpoint or this one. The final adapter
    # is also part of this commit before the state can say training is complete.
    phase_started = time.monotonic()
    api.upload_folder(
        repo_id=OUTPUT_REPO,
        repo_type="model",
        folder_path=OUTPUT_DIR,
        allow_patterns=primary_patterns,
        commit_message=f"Save resumable training state at step {step}",
    )
    timings["checkpoint_primary_upload_seconds"] = round(
        time.monotonic() - phase_started, 3
    )
    phase_started = time.monotonic()
    try:
        api.upload_file(
            repo_id=OUTPUT_REPO,
            repo_type="model",
            path_or_fileobj=segment_log,
            path_in_repo=f"logs/segment_{step:06d}.log",
            commit_message=f"Upload training log for step {step}",
        )
        if lora.is_file():
            api.upload_file(
                repo_id=OUTPUT_REPO,
                repo_type="model",
                path_or_fileobj=lora,
                path_in_repo=f"checkpoints/littlequeen-flux2-klein-v3-step{step:06d}.safetensors",
                commit_message=f"Save inference LoRA at step {step}",
            )
        checkpoint_dirs = sorted(
            {
                name.split("/", 1)[0]
                for name in api.list_repo_files(OUTPUT_REPO, repo_type="model")
                if STEP_RE.match(name.split("/", 1)[0])
            },
            key=lambda name: int(name.split("-")[1]),
        )
        for obsolete in checkpoint_dirs[:-3]:
            api.delete_folder(
                obsolete,
                repo_id=OUTPUT_REPO,
                repo_type="model",
                commit_message=f"Prune superseded resumable checkpoint {obsolete}",
            )
    except Exception as error:
        LOG.warning("primary state is durable; auxiliary upload or pruning failed: %s", error)
    timings["checkpoint_auxiliary_seconds"] = round(time.monotonic() - phase_started, 3)
    timings["checkpoint_total_seconds"] = round(time.monotonic() - started, 3)
    LOG.info("phase timings checkpoint_upload=%s", json.dumps(timings, sort_keys=True))
    return checkpoint.name, timings


def gpu_duration(
    current_step: int,
    segment_target: int,
    _request_id: str,
    _model_path: str,
    aggressive_tail: bool,
) -> int:
    steps = max(1, segment_target - current_step)
    if aggressive_tail:
        fixed_seconds = TAIL_GPU_DURATION_FIXED_SECONDS
        seconds_per_step = TAIL_GPU_SECONDS_PER_STEP
    else:
        fixed_seconds = GPU_DURATION_FIXED_SECONDS
        seconds_per_step = GPU_SECONDS_PER_STEP
    estimated = math.ceil(fixed_seconds + seconds_per_step * steps)
    return min(GPU_MAX_DURATION_SECONDS, estimated)


@spaces.GPU(duration=gpu_duration, size="large")
def train_gpu_segment(
    current_step: int,
    segment_target: int,
    request_id: str,
    model_path: str,
    aggressive_tail: bool,
) -> str:
    gpu_started = time.monotonic()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN Space secret is missing")
    segment_log = WORK_ROOT / f"segment_{segment_target:06d}.log"
    command = [
        sys.executable,
        str(TRAINER),
        "--pretrained_model_name_or_path",
        model_path,
        "--dataset_name",
        str(DATA_DIR),
        "--image_column",
        "image",
        "--caption_column",
        "text",
        "--instance_prompt",
        TRIGGER,
        "--output_dir",
        str(OUTPUT_DIR),
        "--resolution",
        "1024",
        "--use_aspect_ratio_buckets",
        "--center_crop",
        "--train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "0.00007",
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--rank",
        "32",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0.05",
        "--mixed_precision",
        "bf16",
        "--allow_tf32",
        "--cache_latents",
        "--seed",
        "24681357",
        "--max_train_steps",
        str(TOTAL_STEPS),
        "--stop_after_step",
        str(segment_target),
        "--checkpointing_steps",
        str(segment_target),
        "--checkpoints_total_limit",
        "3",
        "--skip_final_inference",
        "--report_to",
        "tensorboard",
    ]
    if current_step:
        command.extend(["--resume_from_checkpoint", "latest"])

    LOG.info("starting segment %s -> %s (%s)", current_step, segment_target, request_id)
    process_started = time.monotonic()
    with segment_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "HF_TOKEN": token, "PYTHONUNBUFFERED": "1"},
        )
    process_seconds = round(time.monotonic() - process_started, 3)
    if process.returncode != 0:
        tail = segment_log.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"trainer exited {process.returncode}\n{tail}")

    result = {
        "segment_log": str(segment_log),
        "gpu_function_seconds": round(time.monotonic() - gpu_started, 3),
        "trainer_process_seconds": process_seconds,
        "declared_gpu_duration_seconds": gpu_duration(
            current_step, segment_target, request_id, model_path, aggressive_tail
        ),
        "aggressive_tail": aggressive_tail,
    }
    LOG.info("phase timings gpu=%s", json.dumps(result, sort_keys=True))
    return json.dumps(result, sort_keys=True)


def parse_trainer_timings(segment_log: Path) -> dict:
    timings = {}
    pattern = re.compile(r"TIMING phase=([a-z0-9_]+) seconds=([0-9.]+)")
    for match in pattern.finditer(segment_log.read_text(encoding="utf-8", errors="replace")):
        timings[match.group(1)] = float(match.group(2))
    return timings


def train_segment(
    request_id: str = "", requested_steps: int = 0, aggressive_tail: bool = False
) -> dict:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN Space secret is missing")
    started_at = utc_now()
    started = time.monotonic()
    current_step, restored, model_path, prior_state, input_timings = download_training_inputs(token)
    if request_id and prior_state.get("last_request_id") == request_id:
        return {
            "status": "duplicate_request_already_checkpointed",
            "request_id": request_id,
            "current_step": current_step,
            "total_steps": TOTAL_STEPS,
            "restored_checkpoint": restored,
            "timestamp": utc_now(),
        }
    if current_step >= TOTAL_STEPS:
        return {
            "status": "complete",
            "current_step": current_step,
            "total_steps": TOTAL_STEPS,
            "restored_checkpoint": restored,
            "request_id": request_id,
            "timestamp": utc_now(),
        }

    segment_size = min(max(int(requested_steps or SEGMENT_STEPS), 1), SEGMENT_STEPS)
    segment_target = min(current_step + segment_size, TOTAL_STEPS)
    gpu_result = json.loads(
        train_gpu_segment(
            current_step, segment_target, request_id, model_path, aggressive_tail
        )
    )
    segment_log = Path(gpu_result["segment_log"])
    saved_step, saved_path = latest_checkpoint(OUTPUT_DIR)
    if saved_step < segment_target or saved_path is None:
        raise RuntimeError(
            f"segment target {segment_target} was not durably checkpointed; latest is {saved_step}"
        )
    complete = saved_step >= TOTAL_STEPS
    uploaded, upload_timings = upload_checkpoint(
        token, saved_step, segment_log, complete, request_id
    )

    for path in OUTPUT_DIR.glob("checkpoint-*"):
        if path.name != uploaded and path.is_dir():
            shutil.rmtree(path)
    elapsed = round(time.monotonic() - started, 2)
    timings = {
        **input_timings,
        **{key: value for key, value in gpu_result.items() if key.endswith("_seconds")},
        **parse_trainer_timings(segment_log),
        **upload_timings,
    }
    timings["endpoint_total_seconds"] = elapsed
    optimizer_seconds = timings.get("optimizer_training")
    if optimizer_seconds is not None:
        timings["endpoint_non_training_seconds"] = round(elapsed - optimizer_seconds, 3)
    result = {
        "status": "complete" if complete else "checkpointed",
        "request_id": request_id,
        "started_step": current_step,
        "current_step": saved_step,
        "total_steps": TOTAL_STEPS,
        "restored_checkpoint": restored,
        "uploaded_checkpoint": uploaded,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "timings": timings,
        "timestamp": utc_now(),
    }
    LOG.info("segment result %s", result)
    return result


def status() -> dict:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return {"status": "HF_TOKEN Space secret is missing", "timestamp": utc_now()}
    try:
        path = hf_hub_download(
            OUTPUT_REPO,
            "training_state.json",
            repo_type="model",
            token=token,
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        return {"status": "not_started", "detail": str(error), "timestamp": utc_now()}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
with gr.Blocks() as demo:
    gr.Markdown("# Little Queen FLUX.2 Klein segmented trainer")
    request = gr.Textbox(label="Request ID")
    requested_steps = gr.Number(value=SEGMENT_STEPS, precision=0, visible=False)
    aggressive_tail = gr.Checkbox(value=False, visible=False)
    train_button = gr.Button("Run one durable segment", variant="primary")
    status_button = gr.Button("Status")
    output = gr.JSON(label="Training state")
    train_button.click(
        train_segment,
        inputs=[request, requested_steps, aggressive_tail],
        outputs=output,
        api_name="train_segment",
    )
    status_button.click(status, outputs=output, api_name="status")

demo.queue(default_concurrency_limit=1, max_size=4).launch()
