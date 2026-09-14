#!/usr/bin/env python3
"""Quota-aware, restart-safe supervisor for Little Queen ZeroGPU evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gradio_client import Client
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
QUOTA_RETRY_RE = re.compile(
    r"try again in\s+(?:(\d+)\s+day[s]?,?\s*)?(\d{1,2}):(\d{2}):(\d{2})",
    re.IGNORECASE,
)
QUOTA_PATTERNS = (
    "quota",
    "exceeded your gpu quota",
    "daily limit",
    "not enough quota",
    "gpu task aborted",
)
STOP = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def quota_retry_after_seconds(error: BaseException) -> int | None:
    match = QUOTA_RETRY_RE.search(str(error))
    if not match:
        return None
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def is_quota_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(pattern in message for pattern in QUOTA_PATTERNS)


def sleep_interruptibly(seconds: float) -> None:
    deadline = time.monotonic() + max(0, seconds)
    while not STOP and time.monotonic() < deadline:
        time.sleep(min(30, deadline - time.monotonic()))


def local_paths(output_dir: Path, job_id: str) -> tuple[Path, Path]:
    return output_dir / "images" / f"{job_id}.png", output_dir / "metadata" / f"{job_id}.json"


def remote_paths(job_id: str) -> tuple[str, str]:
    return f"images/{job_id}.png", f"metadata/{job_id}.json"


def recover_job(config: dict, token: str, output_dir: Path, job: dict) -> bool:
    api = HfApi(token=token)
    image_remote, metadata_remote = remote_paths(job["id"])
    files = set(api.list_repo_files(config["result_repo"], repo_type="dataset"))
    if image_remote not in files or metadata_remote not in files:
        return False
    image_local, metadata_local = local_paths(output_dir, job["id"])
    image_local.parent.mkdir(parents=True, exist_ok=True)
    metadata_local.parent.mkdir(parents=True, exist_ok=True)
    cached_image = hf_hub_download(
        config["result_repo"], image_remote, repo_type="dataset", token=token
    )
    cached_metadata = hf_hub_download(
        config["result_repo"], metadata_remote, repo_type="dataset", token=token
    )
    shutil.copy2(cached_image, image_local)
    shutil.copy2(cached_metadata, metadata_local)
    record = json.loads(metadata_local.read_text(encoding="utf-8"))
    return bool(record.get("ok") and record.get("job") == job)


def build_sheet(output_dir: Path, jobs: list[dict]) -> Path:
    rows = []
    paired = {}
    for job in jobs:
        key = (job["scene"], float(job["lora_scale"]))
        paired.setdefault(key, {})[job["variant"]] = job
    for key in sorted(paired):
        variants = paired[key]
        if {"flux_v1", "flux_v2"}.issubset(variants):
            rows.append((key, variants))
    if not rows:
        raise RuntimeError("no complete v1/v2 rows are available for a sheet")
    thumb_width, thumb_height, label_height = 416, 608, 46
    canvas = Image.new(
        "RGB", (thumb_width * 2, (thumb_height + label_height) * len(rows)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row, ((scene, scale), variants) in enumerate(rows):
        y = row * (thumb_height + label_height)
        for column, variant in enumerate(("flux_v1", "flux_v2")):
            image_path, _ = local_paths(output_dir, variants[variant]["id"])
            with Image.open(image_path) as source:
                image = source.convert("RGB").resize((thumb_width, thumb_height))
            x = column * thumb_width
            canvas.paste(image, (x, y))
            draw.text(
                (x + 8, y + thumb_height + 12),
                f"{variant}  scene {scene:03d}  weight {scale:.1f}",
                fill="black",
            )
    path = output_dir / "stage1_flux_v1_vs_v2.jpg"
    temporary = path.with_suffix(".tmp.jpg")
    canvas.save(temporary, format="JPEG", quality=94)
    temporary.replace(path)
    return path


def build_cross_model_sheet(
    output_dir: Path, jobs: list[dict], baseline_jobs: list[dict]
) -> Path:
    """Pair the promoted FLUX v2 weight-1.0 images with matched SDXL renders."""
    flux_by_scene = {
        int(job["scene"]): job
        for job in baseline_jobs
        if job["variant"] == "flux_v2" and float(job["lora_scale"]) == 1.0
    }
    sdxl_by_scene = {int(job["scene"]): job for job in jobs}
    scenes = sorted(set(flux_by_scene) & set(sdxl_by_scene))
    if not scenes:
        raise RuntimeError("no matched FLUX v2/SDXL scenes are available for a sheet")
    thumb_width, thumb_height, label_height = 416, 608, 46
    canvas = Image.new(
        "RGB", (thumb_width * 2, (thumb_height + label_height) * len(scenes)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row, scene in enumerate(scenes):
        y = row * (thumb_height + label_height)
        entries = (
            ("flux_v2 weight 1.0", flux_by_scene[scene]),
            ("sdxl_v2 weight 1.0", sdxl_by_scene[scene]),
        )
        for column, (label, job) in enumerate(entries):
            image_path, _ = local_paths(output_dir, job["id"])
            with Image.open(image_path) as source:
                image = source.convert("RGB").resize((thumb_width, thumb_height))
            x = column * thumb_width
            canvas.paste(image, (x, y))
            draw.text(
                (x + 8, y + thumb_height + 12),
                f"{label}  scene {scene:03d}",
                fill="black",
            )
    path = output_dir / "stage2_flux_v2_vs_sdxl.jpg"
    temporary = path.with_suffix(".tmp.jpg")
    canvas.save(temporary, format="JPEG", quality=94)
    temporary.replace(path)
    return path


def build_configured_sheet(
    config: dict, manifest: dict, output_dir: Path, jobs: list[dict]
) -> Path:
    stage = manifest.get("stage")
    if stage == "flux_v1_vs_flux_v2":
        return build_sheet(output_dir, jobs)
    if stage == "flux_v2_vs_sdxl_v2":
        baseline_path = Path(config["baseline_manifest"])
        if not baseline_path.is_absolute():
            baseline_path = HERE / baseline_path
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        return build_cross_model_sheet(output_dir, jobs, baseline["jobs"])
    raise RuntimeError(f"unsupported evaluation stage: {stage!r}")


def main() -> int:
    global STOP
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = args.config.parent / config["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    token = Path(config["token_path"]).read_text(encoding="utf-8").strip()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.get("log_filename", "supervisor.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    state_path = output_dir / config.get("state_filename", "supervisor_state.json")
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"version": config["version"], "next_attempt_at": None}
    )

    def stop(_signum, _frame) -> None:
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    client: Client | None = None
    failures = 0
    while not STOP:
        completed = [job for job in jobs if recover_job(config, token, output_dir, job)]
        pending = [job for job in jobs if job not in completed]
        state.update(
            {
                "updated_at": utc_now(),
                "completed": [job["id"] for job in completed],
                "pending": [job["id"] for job in pending],
            }
        )
        atomic_json(state_path, state)
        if not pending:
            sheet = build_configured_sheet(config, manifest, output_dir, jobs)
            state.update(
                {
                    "status": "complete",
                    "completed_at": utc_now(),
                    "comparison_sheet": str(sheet),
                    "next_attempt_at": None,
                }
            )
            atomic_json(state_path, state)
            logging.info("evaluation complete: %s", sheet)
            return 0
        if args.once and completed:
            return 0
        next_attempt = state.get("next_attempt_at")
        if next_attempt:
            wait = (
                datetime.fromisoformat(next_attempt) - datetime.now(timezone.utc)
            ).total_seconds()
            if wait > 0:
                logging.info("sleeping %.0fs for persisted quota reset", wait)
                if args.once:
                    return 75
                sleep_interruptibly(wait)
                continue
            state["next_attempt_at"] = None
            atomic_json(state_path, state)
        job = pending[0]
        try:
            if client is None:
                client = Client(config["space_repo"], hf_token=token)
            logging.info("requesting %s (%d/%d complete)", job["id"], len(completed), len(jobs))
            result = client.predict(
                request_id=job["id"],
                job_json=json.dumps(job, sort_keys=True),
                api_name="/run_job",
            )
            logging.info("result %s", json.dumps(result, sort_keys=True)[:2000])
            if not recover_job(config, token, output_dir, job):
                raise RuntimeError("Space returned without a durable matching Hub result")
            failures = 0
            state["next_attempt_at"] = None
            atomic_json(state_path, state)
        except Exception as error:
            client = None
            logging.exception("evaluation request failed: %s", error)
            if is_quota_error(error):
                retry = quota_retry_after_seconds(error) or 86400
                deadline = datetime.now(timezone.utc) + timedelta(
                    seconds=retry + int(config["quota_retry_buffer_seconds"])
                )
                state["next_attempt_at"] = deadline.isoformat(timespec="seconds")
                state["last_quota_error"] = str(error)
                atomic_json(state_path, state)
                if args.once:
                    return 75
                continue
            failures += 1
            retry = (
                int(config["short_retry_seconds"])
                if failures <= int(config["short_retry_count"])
                else int(config["general_retry_seconds"])
            )
            if args.once:
                return 1
            sleep_interruptibly(retry)
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
