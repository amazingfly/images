#!/usr/bin/env python3
"""Install a pinned kohya trainer and train the Little Queen v2 SDXL LoRA."""

from __future__ import annotations

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
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/content/lqxl_v2")
BUNDLE = Path("/content/lqxl_v2_bundle.tar.gz")
SD_SCRIPTS = ROOT / "sd-scripts"
OUTPUT = ROOT / "output"
LOGS = ROOT / "logs"
MODEL_DIR = ROOT / "models"
SD_SCRIPTS_COMMIT = "656587768f0d36a90702fd329a5826e585d9cf8d"
MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_FILE = "sd_xl_base_1.0.safetensors"
EXPECTED_MODEL_SHA256 = (
    "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_logged(command: list[str], destination: Path, *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    with destination.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, remainder = line.partition(":")
        value = remainder.strip().split(maxsplit=1)
        if value:
            values[key] = int(value[0])
    return values


def gpu_sample() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        return {"gpu_memory_used_mb": "", "gpu_memory_total_mb": "", "gpu_utilization_pct": "", "gpu_temperature_c": ""}
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    return dict(
        zip(
            ("gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_utilization_pct", "gpu_temperature_c"),
            fields,
            strict=True,
        )
    )


def monitor_resources(stop: threading.Event, destination: Path) -> None:
    fieldnames = [
        "timestamp_utc",
        "host_available_mb",
        "host_swap_used_mb",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_utilization_pct",
        "gpu_temperature_c",
    ]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        while not stop.is_set():
            memory = meminfo()
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "host_available_mb": round(memory.get("MemAvailable", 0) / 1024, 1),
                "host_swap_used_mb": round(
                    (memory.get("SwapTotal", 0) - memory.get("SwapFree", 0)) / 1024,
                    1,
                ),
                **gpu_sample(),
            }
            writer.writerow(row)
            stream.flush()
            stop.wait(10)


def extract_bundle() -> None:
    if not BUNDLE.is_file():
        raise FileNotFoundError(BUNDLE)
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "r:gz") as archive:
        archive.extractall(ROOT)


def install_trainer() -> None:
    if not (SD_SCRIPTS / ".git").is_dir():
        run(["git", "clone", "https://github.com/kohya-ss/sd-scripts.git", str(SD_SCRIPTS)])
    run(["git", "fetch", "origin", SD_SCRIPTS_COMMIT], cwd=SD_SCRIPTS)
    run(["git", "checkout", "--detach", SD_SCRIPTS_COMMIT], cwd=SD_SCRIPTS)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            "requirements.txt",
        ],
        cwd=SD_SCRIPTS,
    )


def download_model() -> Path:
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=MODEL_DIR,
        )
    )
    actual = sha256(downloaded)
    if actual != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"SDXL checksum mismatch: {actual}")
    return downloaded


def training_command(model: Path) -> list[str]:
    return [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process=1",
        str(SD_SCRIPTS / "sdxl_train_network.py"),
        f"--pretrained_model_name_or_path={model}",
        f"--dataset_config={ROOT / 'dataset_config.toml'}",
        f"--output_dir={OUTPUT}",
        "--output_name=lqxl_sdxl_v2",
        "--save_model_as=safetensors",
        "--save_precision=fp16",
        "--save_every_n_steps=300",
        "--network_module=networks.lora",
        "--network_dim=32",
        "--network_alpha=16",
        "--network_train_unet_only",
        "--learning_rate=0.0001",
        "--unet_lr=0.0001",
        "--optimizer_type=AdamW8bit",
        "--lr_scheduler=cosine_with_restarts",
        "--lr_scheduler_num_cycles=1",
        "--lr_warmup_steps=60",
        "--max_train_steps=1200",
        "--mixed_precision=fp16",
        "--lowram",
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--no_half_vae",
        "--sdpa",
        "--min_snr_gamma=5",
        "--noise_offset=0.035",
        "--max_grad_norm=1.0",
        "--seed=83219",
        "--max_data_loader_n_workers=0",
        f"--logging_dir={LOGS / 'tensorboard'}",
        "--log_with=tensorboard",
    ]


def cache_command(mode: str, model: Path) -> list[str]:
    command = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process=1",
        str(ROOT / "isolated_cache.py"),
        mode,
        "--sdxl",
        f"--pretrained_model_name_or_path={model}",
        f"--dataset_config={ROOT / 'dataset_config.toml'}",
        "--mixed_precision=fp16",
        "--lowram",
        "--max_data_loader_n_workers=0",
        "--seed=83219",
    ]
    if mode == "latents":
        command.extend(("--vae_batch_size=1", "--no_half_vae"))
    return command


def main() -> int:
    extract_bundle()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    final_lora = OUTPUT / "lqxl_sdxl_v2.safetensors"
    result_path = ROOT / "training_result.json"
    try:
        prior_result = json.loads(result_path.read_text())
    except (OSError, ValueError):
        prior_result = {}
    if prior_result.get("status") == "complete" and final_lora.is_file():
        print("Training is already complete; preserving the existing LoRA.", flush=True)
        print(json.dumps(prior_result, indent=2), flush=True)
        return 0

    pid_path = ROOT / "training.pid"
    try:
        prior_pid = int(pid_path.read_text().strip())
        os.kill(prior_pid, 0)
    except (OSError, ValueError):
        pass
    else:
        if prior_pid != os.getpid():
            raise RuntimeError(f"training process {prior_pid} is already active")
    pid_path.write_text(f"{os.getpid()}\n")

    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        run(["nvidia-smi"])
        install_trainer()
        model = download_model()

        stop = threading.Event()
        monitor = threading.Thread(
            target=monitor_resources,
            args=(stop, LOGS / "resource_usage.csv"),
            daemon=True,
        )
        monitor.start()
        started = time.monotonic()
        status = "failed"
        try:
            run_logged(
                cache_command("latents", model),
                LOGS / "cache_latents.log",
                cwd=SD_SCRIPTS,
            )
            run_logged(
                cache_command("text", model),
                LOGS / "cache_text.log",
                cwd=SD_SCRIPTS,
            )
            run_logged(
                training_command(model),
                LOGS / "training.log",
                cwd=SD_SCRIPTS,
            )
            status = "complete"
        finally:
            stop.set()
            monitor.join(timeout=20)

        if not final_lora.is_file():
            raise RuntimeError(f"training ended without final LoRA: {final_lora}")
        report = {
            "status": status,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "sd_scripts_commit": SD_SCRIPTS_COMMIT,
            "base_model_sha256": EXPECTED_MODEL_SHA256,
            "lora": str(final_lora),
            "lora_bytes": final_lora.stat().st_size,
            "lora_sha256": sha256(final_lora),
            "checkpoints": sorted(path.name for path in OUTPUT.glob("*.safetensors")),
        }
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
