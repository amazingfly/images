#!/usr/bin/env python3
"""Train focused SDXL LoRAs for Little Queen worn regalia and hand equipment."""

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


ROOT = Path("/content/lqaccessories_v1")
BUNDLE = Path("/content/lqaccessories_v1_bundle.tar.gz")
BUNDLE_PART_PREFIX = Path("/content/lqaccessories_v1_bundle.tar.gz.part_")
SD_SCRIPTS = ROOT / "sd-scripts"
LOGS = ROOT / "logs"
MODEL_DIR = ROOT / "models"
MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_FILE = "sd_xl_base_1.0.safetensors"
EXPECTED_MODEL_SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
SD_SCRIPTS_COMMIT = "656587768f0d36a90702fd329a5826e585d9cf8d"

JOBS = (
    {
        "id": "regalia",
        "config": ROOT / "dataset_regalia.toml",
        "output": ROOT / "output_regalia",
        "name": "lqmoonregalia_sdxl_v1",
        "dim": 16,
        "alpha": 8,
        "steps": 600,
        "seed": 113001,
    },
    {
        "id": "equipment",
        "config": ROOT / "dataset_equipment.toml",
        "output": ROOT / "output_equipment",
        "name": "lqhandgear_sdxl_v1",
        "dim": 32,
        "alpha": 16,
        "steps": 900,
        "seed": 113002,
    },
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_logged(command: list[str], destination: Path, *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    with destination.open("w", encoding="utf-8") as log:
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


def assemble_parts(prefix: Path, destination: Path) -> None:
    parts = sorted(prefix.parent.glob(prefix.name + "*"))
    if not parts:
        raise FileNotFoundError(f"no uploaded chunks match {prefix}*")
    temporary = destination.with_suffix(destination.suffix + ".assembling")
    with temporary.open("wb") as output:
        for part in parts:
            print(f"Appending {part.name}", flush=True)
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    temporary.replace(destination)


def extract_bundle() -> None:
    if not BUNDLE.is_file():
        assemble_parts(BUNDLE_PART_PREFIX, BUNDLE)
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "r:gz") as archive:
        archive.extractall(ROOT)


def install_trainer() -> None:
    if not (SD_SCRIPTS / ".git").is_dir():
        run(["git", "clone", "https://github.com/kohya-ss/sd-scripts.git", str(SD_SCRIPTS)])
    run(["git", "fetch", "origin", SD_SCRIPTS_COMMIT], cwd=SD_SCRIPTS)
    run(["git", "checkout", "--detach", SD_SCRIPTS_COMMIT], cwd=SD_SCRIPTS)
    run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt"],
        cwd=SD_SCRIPTS,
    )


def download_model() -> Path:
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=MODEL_DIR))
    actual = sha256(downloaded)
    if actual != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"SDXL checksum mismatch: {actual}")
    return downloaded


def meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, remainder = line.partition(":")
        fields = remainder.strip().split(maxsplit=1)
        if fields:
            values[key] = int(fields[0])
    return values


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
    return dict(zip(keys, (field.strip() for field in result.stdout.splitlines()[0].split(",")), strict=True))


def monitor_resources(stop: threading.Event) -> None:
    fields = ["timestamp_utc", "host_available_mb", "host_swap_used_mb", "gpu_used_mb", "gpu_total_mb", "gpu_util_pct", "gpu_temp_c"]
    with (LOGS / "resource_usage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            memory = meminfo()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "host_available_mb": round(memory.get("MemAvailable", 0) / 1024, 1),
                    "host_swap_used_mb": round((memory.get("SwapTotal", 0) - memory.get("SwapFree", 0)) / 1024, 1),
                    **gpu_sample(),
                }
            )
            stream.flush()
            stop.wait(10)


def cache_command(mode: str, model: Path, job: dict) -> list[str]:
    command = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process=1",
        str(ROOT / "isolated_cache.py"),
        mode,
        "--sdxl",
        f"--pretrained_model_name_or_path={model}",
        f"--dataset_config={job['config']}",
        "--mixed_precision=fp16",
        "--lowram",
        "--max_data_loader_n_workers=0",
        f"--seed={job['seed']}",
    ]
    if mode == "latents":
        command.extend(("--vae_batch_size=1", "--no_half_vae"))
    return command


def training_command(model: Path, job: dict) -> list[str]:
    return [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process=1",
        str(SD_SCRIPTS / "sdxl_train_network.py"),
        f"--pretrained_model_name_or_path={model}",
        f"--dataset_config={job['config']}",
        f"--output_dir={job['output']}",
        f"--output_name={job['name']}",
        "--save_model_as=safetensors",
        "--save_precision=fp16",
        "--save_every_n_steps=300",
        "--network_module=networks.lora",
        f"--network_dim={job['dim']}",
        f"--network_alpha={job['alpha']}",
        "--network_train_unet_only",
        "--learning_rate=0.0001",
        "--unet_lr=0.0001",
        "--optimizer_type=AdamW8bit",
        "--lr_scheduler=cosine_with_restarts",
        "--lr_scheduler_num_cycles=1",
        f"--lr_warmup_steps={max(20, job['steps'] // 20)}",
        f"--max_train_steps={job['steps']}",
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
        "--noise_offset=0.02",
        "--max_grad_norm=1.0",
        f"--seed={job['seed']}",
        "--max_data_loader_n_workers=0",
        f"--logging_dir={LOGS / ('tensorboard_' + job['id'])}",
        "--log_with=tensorboard",
    ]


def main() -> int:
    extract_bundle()
    LOGS.mkdir(parents=True, exist_ok=True)
    result_path = ROOT / "training_result.json"
    try:
        prior = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prior = {}
    finals = {job["id"]: job["output"] / f"{job['name']}.safetensors" for job in JOBS}
    if prior.get("status") == "complete" and all(path.is_file() for path in finals.values()):
        print("Both accessory LoRAs are already complete; preserving artifacts.", flush=True)
        print(json.dumps(prior, indent=2), flush=True)
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
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        run(["nvidia-smi"])
        install_trainer()
        model = download_model()
        for job in JOBS:
            job["output"].mkdir(parents=True, exist_ok=True)

        stop = threading.Event()
        monitor = threading.Thread(target=monitor_resources, args=(stop,), daemon=True)
        monitor.start()
        started = time.monotonic()
        try:
            for job in JOBS:
                final = finals[job["id"]]
                if final.is_file():
                    print(f"Preserving completed {job['id']} LoRA: {final}", flush=True)
                    continue
                run_logged(cache_command("latents", model, job), LOGS / f"cache_latents_{job['id']}.log", cwd=SD_SCRIPTS)
                run_logged(cache_command("text", model, job), LOGS / f"cache_text_{job['id']}.log", cwd=SD_SCRIPTS)
                run_logged(training_command(model, job), LOGS / f"training_{job['id']}.log", cwd=SD_SCRIPTS)
        finally:
            stop.set()
            monitor.join(timeout=20)

        reports = {}
        for job in JOBS:
            final = finals[job["id"]]
            if not final.is_file():
                raise RuntimeError(f"missing final {job['id']} LoRA: {final}")
            reports[job["id"]] = {
                "lora": str(final),
                "bytes": final.stat().st_size,
                "sha256": sha256(final),
                "steps": job["steps"],
                "network_dim": job["dim"],
                "network_alpha": job["alpha"],
                "checkpoints": sorted(path.name for path in job["output"].glob("*.safetensors")),
            }
        report = {
            "status": "complete",
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "sd_scripts_commit": SD_SCRIPTS_COMMIT,
            "base_model_sha256": EXPECTED_MODEL_SHA256,
            "training_method": "alpha-masked U-Net-only LoRA",
            "models": reports,
        }
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
