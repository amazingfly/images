#!/usr/bin/env python3
"""Run quantized SDXL images serially with per-process memory telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
KIB_PER_MIB = 1024.0
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing run directory and skip its completed valid images.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path if path.is_absolute() else (ROOT / path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not config.get("prompts") and config.get("prompts_file"):
        prompts_path = resolve(config["prompts_file"])
        with prompts_path.open(encoding="utf-8") as handle:
            prompt_config = json.load(handle)
        config["prompts"] = prompt_config.get("prompts", prompt_config)
    prompts = config.get("prompts", [])
    if not prompts:
        raise ValueError("The config must contain at least one prompt")
    return config


def proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def reject_conflicting_generators() -> None:
    own_pid = os.getpid()
    patterns = ("generate_images_lora.py", "run_littlequeen_dataset.py")
    conflicts = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        command = proc_cmdline(int(entry.name))
        if any(pattern in command for pattern in patterns):
            conflicts.append(f"{entry.name}: {command.strip()}")
    if conflicts:
        raise RuntimeError(
            "Refusing to overlap SDXL with an SD1.5 generator:\n" + "\n".join(conflicts)
        )


def read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        parts = remainder.strip().split(maxsplit=1)
        if not parts:
            continue
        number = parts[0]
        try:
            values[key] = int(number)
        except ValueError:
            continue
    return values


def process_tree(root_pid: int) -> list[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            parents[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sorted(descendants)


def memory_sample(root_pid: int) -> dict[str, float]:
    process_totals = {key: 0 for key in ("VmRSS", "VmHWM", "VmSwap", "RssAnon", "RssFile")}
    pids = process_tree(root_pid)
    for pid in pids:
        values = read_key_values(Path(f"/proc/{pid}/status"))
        for key in process_totals:
            process_totals[key] += values.get(key, 0)

    system = read_key_values(Path("/proc/meminfo"))
    swap_used_kib = system.get("SwapTotal", 0) - system.get("SwapFree", 0)
    return {
        "process_count": float(len(pids)),
        "process_rss_mb": process_totals["VmRSS"] / KIB_PER_MIB,
        "process_hwm_mb": process_totals["VmHWM"] / KIB_PER_MIB,
        "process_swap_mb": process_totals["VmSwap"] / KIB_PER_MIB,
        "process_anon_mb": process_totals["RssAnon"] / KIB_PER_MIB,
        "process_file_mb": process_totals["RssFile"] / KIB_PER_MIB,
        "system_available_mb": system.get("MemAvailable", 0) / KIB_PER_MIB,
        "system_swap_used_mb": swap_used_kib / KIB_PER_MIB,
    }


def valid_png(path: Path) -> bool:
    try:
        return path.stat().st_size > 1024 and path.read_bytes()[:8] == PNG_SIGNATURE
    except OSError:
        return False


def lora_prompt_suffix(loras: list[dict[str, Any]]) -> str:
    return "".join(
        f"<lora:{item['name']}:{float(item['weight']):g}>" for item in loras
    )


def build_command(
    config: dict[str, Any], prompt_record: dict[str, Any], output: Path
) -> list[str]:
    generation = config["generation"]
    active_loras = prompt_record.get("loras", config.get("loras", []))
    prompt = prompt_record["prompt"] + lora_prompt_suffix(active_loras)
    command = [
        str(resolve(config["binary"])),
        "--model",
        str(resolve(config["model"])),
        "--prompt",
        prompt,
        "--negative-prompt",
        prompt_record.get("negative_prompt", config.get("negative_prompt", "")),
        "--output",
        str(output),
        "--width",
        str(generation["width"]),
        "--height",
        str(generation["height"]),
        "--steps",
        str(generation["steps"]),
        "--cfg-scale",
        str(generation["cfg_scale"]),
        "--sampling-method",
        generation["sampling_method"],
        "--scheduler",
        generation["scheduler"],
        "--threads",
        str(generation["threads"]),
        "--rng",
        generation["rng"],
        "--seed",
        str(prompt_record["seed"]),
        "--lora-model-dir",
        str(resolve(config["lora_dir"])),
        "--lora-apply-mode",
        generation.get("lora_apply_mode", "at_runtime"),
        "--verbose",
    ]
    if generation.get("mmap", True):
        command.append("--mmap")
    if generation.get("vae_tiling", True):
        command.append("--vae-tiling")
    if generation.get("diffusion_fa", False):
        command.append("--diffusion-fa")
    if generation.get("vae_conv_direct", False):
        command.append("--vae-conv-direct")
    hires = generation.get("hires")
    if hires:
        command.extend(
            [
                "--hires",
                "--hires-width",
                str(hires["width"]),
                "--hires-height",
                str(hires["height"]),
                "--hires-steps",
                str(hires["steps"]),
                "--hires-upscaler",
                hires.get("upscaler", "Lanczos"),
                "--hires-denoising-strength",
                str(hires["denoising_strength"]),
            ]
        )
    command.extend(str(value) for value in prompt_record.get("extra_args", []))
    return command


def terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def render_one(
    config: dict[str, Any], record: dict[str, Any], index: int, run_dir: Path, log_dir: Path
) -> dict[str, Any]:
    output = run_dir / record.get(
        "output_relative",
        f"image_{index:02d}_seed_{record['seed']}.png",
    )
    log_stem = record.get("log_stem", f"image_{index:02d}")
    process_log = log_dir / f"{log_stem}.log"
    memory_log = log_dir / f"{log_stem}_memory.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    process_log.parent.mkdir(parents=True, exist_ok=True)
    memory_log.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(config, record, output)
    memory_config = config["memory"]
    interval = float(memory_config.get("sample_interval_seconds", 1.0))
    start = time.monotonic()
    breach: str | None = None
    samples: list[dict[str, float]] = []

    print(f"[{index:02d}] starting seed {record['seed']}", flush=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(config["generation"]["threads"])
    with process_log.open("w", encoding="utf-8") as output_log:
        output_log.write("COMMAND: " + " ".join(command) + "\n")
        output_log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=output_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        try:
            while process.poll() is None:
                sample = memory_sample(process.pid)
                sample["elapsed_seconds"] = time.monotonic() - start
                samples.append(sample)
                if sample["system_available_mb"] < float(
                    memory_config["minimum_available_mb"]
                ):
                    breach = "system available RAM fell below the configured floor"
                elif sample["process_rss_mb"] > float(
                    memory_config["maximum_process_rss_mb"]
                ):
                    breach = "process-tree RSS exceeded the configured ceiling"
                elif sample["system_swap_used_mb"] > float(
                    memory_config["maximum_swap_used_mb"]
                ):
                    breach = "system swap use exceeded the configured ceiling"
                if breach:
                    output_log.write(f"MEMORY SAFETY STOP: {breach}\n")
                    output_log.flush()
                    terminate_group(process)
                    break
                time.sleep(interval)
        except BaseException:
            terminate_group(process)
            raise
        return_code = process.wait()

    fieldnames = [
        "elapsed_seconds",
        "process_count",
        "process_rss_mb",
        "process_hwm_mb",
        "process_swap_mb",
        "process_anon_mb",
        "process_file_mb",
        "system_available_mb",
        "system_swap_used_mb",
    ]
    with memory_log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    def peak(field: str) -> float:
        return max((sample[field] for sample in samples), default=0.0)

    result = {
        "index": index,
        "seed": record["seed"],
        "prompt": record["prompt"],
        "output": str(output),
        "process_log": str(process_log),
        "memory_log": str(memory_log),
        "return_code": return_code,
        "memory_safety_breach": breach,
        "duration_seconds": round(time.monotonic() - start, 3),
        "peak_process_rss_mb": round(peak("process_rss_mb"), 1),
        "peak_process_hwm_mb": round(peak("process_hwm_mb"), 1),
        "peak_process_swap_mb": round(peak("process_swap_mb"), 1),
        "peak_system_swap_used_mb": round(peak("system_swap_used_mb"), 1),
        "minimum_system_available_mb": round(
            min((sample["system_available_mb"] for sample in samples), default=0.0), 1
        ),
        "valid_png": valid_png(output),
    }
    result["status"] = (
        "completed"
        if return_code == 0 and result["valid_png"] and breach is None
        else "failed"
    )
    print(
        f"[{index:02d}] {result['status']} in {result['duration_seconds']:.1f}s; "
        f"peak RSS {result['peak_process_rss_mb']:.1f} MiB; "
        f"minimum available {result['minimum_system_available_mb']:.1f} MiB",
        flush=True,
    )
    return result


def validate_inputs(config: dict[str, Any]) -> None:
    binary = resolve(config["binary"])
    model = resolve(config["model"])
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Missing executable: {binary}")
    if not model.is_file():
        raise FileNotFoundError(f"Missing quantized SDXL model: {model}")
    lora_dir = resolve(config["lora_dir"])
    loras = list(config.get("loras", []))
    for prompt in config.get("prompts", []):
        loras.extend(prompt.get("loras", []))
    for item in {
        (entry["name"], float(entry["weight"])): entry
        for entry in loras
    }.values():
        lora = lora_dir / f"{item['name']}.safetensors"
        if not lora.is_file():
            raise FileNotFoundError(f"Missing LoRA: {lora}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    reject_conflicting_generators()
    validate_inputs(config)
    if args.start_index < 1 or args.start_index > len(config["prompts"]):
        raise ValueError("--start-index must identify a configured prompt")
    prompts = config["prompts"][args.start_index - 1 :]
    if args.limit:
        prompts = prompts[: args.limit]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve(config["output_root"]) / run_id
    log_dir = resolve(config["log_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    log_dir.mkdir(parents=True, exist_ok=args.resume)

    summary_path = run_dir / "run_summary.json"
    if args.resume and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("finished_at", None)
        summary["status"] = "running"
        summary["resumed_at"] = datetime.now().isoformat(timespec="seconds")
    else:
        summary = {
            "name": config["name"],
            "run_id": run_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "model": str(resolve(config["model"])),
            "model_size_bytes": resolve(config["model"]).stat().st_size,
            "generation": config["generation"],
            "memory_limits": config["memory"],
            "results": [],
        }

    completed = {
        int(result["index"]): result
        for result in summary.get("results", [])
        if result.get("status") == "completed"
        and valid_png(Path(result.get("output", "")))
    }
    for index, record in enumerate(prompts, start=args.start_index):
        if index in completed:
            print(f"[{index:02d}] skipping completed valid image", flush=True)
            continue
        result = render_one(config, record, index, run_dir, log_dir)
        summary["results"] = [
            prior for prior in summary.get("results", []) if int(prior["index"]) != index
        ]
        summary["results"].append(result)
        summary["results"].sort(key=lambda item: int(item["index"]))
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if result["status"] != "completed":
            summary["status"] = "failed"
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            return 1

    summary["status"] = "completed"
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(prompts)} image(s): {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
