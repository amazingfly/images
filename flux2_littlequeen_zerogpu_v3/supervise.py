#!/usr/bin/env python3
"""Durably supervise multi-day segmented ZeroGPU training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gradio_client import Client
from huggingface_hub import HfApi, hf_hub_download


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
QUOTA_PATTERNS = (
    "quota",
    "exceeded your gpu quota",
    "daily limit",
    "not enough quota",
    "gpu task aborted",
)
STOP = False
QUOTA_REMAINING_RE = re.compile(r"requested\s+vs\.\s+(\d+)s\s+left", re.IGNORECASE)
QUOTA_RETRY_RE = re.compile(r"try again in\s+(\d+):(\d{2}):(\d{2})", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--retry-now",
        action="store_true",
        help="Ignore and clear a persisted ZeroGPU retry deadline.",
    )
    return parser.parse_args()


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z"
    )
    stream_formatter = logging.Formatter("%(levelname)s %(message)s")
    logger = logging.getLogger("littlequeen-flux2-supervisor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    return logger


def read_token(path: Path) -> str:
    token = path.expanduser().read_text(encoding="utf-8").strip()
    if not token.startswith("hf_"):
        raise RuntimeError(f"invalid Hugging Face token file: {path}")
    return token


def read_write_token(path: Path) -> tuple[str, str]:
    """Wait for the configured token to become write-capable."""
    while not STOP:
        try:
            token = read_token(path)
            owner, role = token_role(token)
            if role in {"write", "fineGrained"}:
                return token, owner
            logging.getLogger("littlequeen-flux2-supervisor").warning(
                "Hugging Face token for %s is %r, waiting 900 seconds for write access at %s",
                owner,
                role,
                path,
            )
        except Exception as error:
            logging.getLogger("littlequeen-flux2-supervisor").warning(
                "Hugging Face token is not ready: %s", error
            )
        time.sleep(900)
    raise RuntimeError("stopped while waiting for Hugging Face write token")


def token_role(token: str) -> tuple[str, str | None]:
    who = HfApi(token=token).whoami()
    return (
        str(who.get("name", "")),
        who.get("auth", {}).get("accessToken", {}).get("role"),
    )


def repository_state(config: dict, token: str) -> dict:
    try:
        path = hf_hub_download(
            config["output_repo"],
            "training_state.json",
            repo_type="model",
            token=token,
            force_download=True,
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {"latest_step": 0, "total_steps": config["total_steps"], "complete": False}


def connect_client(config: dict, token: str, logger: logging.Logger) -> Client:
    api = HfApi(token=token)
    while not STOP:
        try:
            runtime = api.get_space_runtime(config["space_repo"])
            stage = str(runtime.stage)
            logger.info(
                "Space stage=%s hardware=%s requested_hardware=%s",
                stage,
                runtime.hardware,
                runtime.requested_hardware,
            )
            if stage == "RUNNING":
                return Client(
                    config["space_repo"], hf_token=token, verbose=False, download_files=False
                )
            if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
                logger.error("Space cannot start: %s", runtime.raw.get("errorMessage", stage))
                interruptible_sleep(config["general_retry_seconds"], logger, "Space is in an error state")
            else:
                interruptible_sleep(30, logger, f"waiting for Space stage {stage}")
        except Exception as error:
            logger.warning("could not inspect or connect to Space: %s", error)
            interruptible_sleep(60, logger, "waiting to reconnect to Space")
    raise RuntimeError("stopped while waiting for Space")


def is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(pattern in message for pattern in QUOTA_PATTERNS)


def quota_remaining_seconds(error: Exception) -> int:
    match = QUOTA_REMAINING_RE.search(str(error))
    return int(match.group(1)) if match else 0


def declared_gpu_seconds(steps: int, config: dict, *, tail: bool = False) -> int:
    prefix = "tail_" if tail else ""
    estimate = (
        config[f"{prefix}gpu_duration_fixed_seconds"]
        + config[f"{prefix}gpu_seconds_per_step"] * steps
    )
    return min(config["gpu_max_duration_seconds"], int(estimate + 0.999))


def admission_quota_seconds(steps: int, config: dict, *, tail: bool = False) -> int:
    return int(
        declared_gpu_seconds(steps, config, tail=tail)
        * config["quota_admission_multiplier"]
        + 0.999
    )


def quota_tail_steps(error: Exception, config: dict) -> int:
    remaining = quota_remaining_seconds(error)
    increment = int(config["tail_step_increment"])
    minimum = int(config["tail_min_steps"])
    maximum = int(config["segment_steps"])
    maximum -= maximum % increment
    for steps in range(maximum, minimum - 1, -increment):
        if admission_quota_seconds(steps, config, tail=True) <= remaining:
            return steps
    return 0


def quota_retry_after_seconds(error: Exception) -> int | None:
    match = QUOTA_RETRY_RE.search(str(error))
    if not match:
        return None
    hours, minutes, seconds = (int(value) for value in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def load_supervisor_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"version": "littlequeen-flux2-supervisor-v3"}


def save_supervisor_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_quota_deadline(path: Path, state: dict, error: Exception, config: dict) -> None:
    retry_after = quota_retry_after_seconds(error)
    if retry_after is None:
        retry_after = config["quota_initial_sleep_seconds"]
    deadline = datetime.now(timezone.utc) + timedelta(
        seconds=retry_after + config["quota_retry_buffer_seconds"]
    )
    state.update(
        {
            "next_attempt_at": deadline.isoformat(timespec="seconds"),
            "quota_error_at": utc_now(),
            "quota_remaining_seconds": quota_remaining_seconds(error),
            "last_quota_error": str(error),
        }
    )
    save_supervisor_state(path, state)


def clear_quota_deadline(path: Path, state: dict) -> None:
    state["next_attempt_at"] = None
    save_supervisor_state(path, state)


def record_segment_result(output_dir: Path, state_path: Path, state: dict, result: dict) -> None:
    record = {"recorded_at": utc_now(), **result}
    state["last_successful_segment"] = record
    save_supervisor_state(state_path, state)
    with (output_dir / "segment_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def wait_for_persisted_deadline(
    path: Path, state: dict, logger: logging.Logger, ignore: bool = False
) -> None:
    raw_deadline = state.get("next_attempt_at")
    if ignore:
        clear_quota_deadline(path, state)
        return
    if not raw_deadline:
        return
    try:
        deadline = datetime.fromisoformat(raw_deadline)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        clear_quota_deadline(path, state)
        return
    seconds = max(0, int((deadline - datetime.now(timezone.utc)).total_seconds()))
    if seconds:
        interruptible_sleep_until(deadline, logger, "persisted ZeroGPU quota reset")
    if not STOP:
        clear_quota_deadline(path, state)


def interruptible_sleep(seconds: int, logger: logging.Logger, reason: str) -> None:
    deadline = time.monotonic() + seconds
    logger.info("sleeping %s seconds: %s", seconds, reason)
    while not STOP and time.monotonic() < deadline:
        time.sleep(min(5, max(0.1, deadline - time.monotonic())))


def interruptible_sleep_until(deadline: datetime, logger: logging.Logger, reason: str) -> None:
    remaining = max(0, int((deadline - datetime.now(timezone.utc)).total_seconds()))
    logger.info(
        "sleeping until %s (%s seconds): %s",
        deadline.isoformat(timespec="seconds"),
        remaining,
        reason,
    )
    while not STOP:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(5, max(0.1, remaining)))


def handle_signal(signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    logger = configure_logging(output_dir / "training.log")
    supervisor_state_path = output_dir / "supervisor_state.json"
    supervisor_state = load_supervisor_state(supervisor_state_path)
    token_path = args.token_file or Path(config["token_path"])
    token, owner = read_write_token(token_path)
    if owner != config["owner"]:
        raise RuntimeError(f"token owner {owner!r} does not match configured owner {config['owner']!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    wait_for_persisted_deadline(
        supervisor_state_path, supervisor_state, logger, ignore=args.retry_now
    )
    logger.info(
        "supervisor started space=%s output=%s target_steps=%s segment_steps=%s",
        config["space_repo"],
        config["output_repo"],
        config["total_steps"],
        config["segment_steps"],
    )
    client = connect_client(config, token, logger)
    while not STOP:
        state = repository_state(config, token)
        current = int(state.get("latest_step", 0))
        total = int(state.get("total_steps", config["total_steps"]))
        logger.info("progress %s/%s complete=%s", current, total, state.get("complete", False))
        if state.get("complete") or current >= total:
            logger.info("training complete at step %s/%s", current, total)
            return 0

        request_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        try:
            logger.info("requesting segment request_id=%s from_step=%s", request_id, current)
            result = client.predict(
                request_id, config["segment_steps"], False, api_name="/train_segment"
            )
            logger.info("segment response %s", json.dumps(result, sort_keys=True))
            clear_quota_deadline(supervisor_state_path, supervisor_state)
            record_segment_result(
                output_dir, supervisor_state_path, supervisor_state, result
            )
            if args.once:
                return 0
            continue
        except Exception as error:
            logger.error("segment request failed: %s: %s", type(error).__name__, error)
            if args.once:
                return 2
            if is_quota_error(error):
                set_quota_deadline(supervisor_state_path, supervisor_state, error, config)
                tail_error = error
                tried_tail_steps = set()
                tail_succeeded = False
                while True:
                    tail_steps = quota_tail_steps(tail_error, config)
                    if not tail_steps or tail_steps in tried_tail_steps:
                        break
                    tried_tail_steps.add(tail_steps)
                    tail_request_id = f"{request_id}-tail{tail_steps}"
                    try:
                        logger.info(
                            "using quota remainder for %s-step tail segment request_id=%s "
                            "declared_gpu_seconds=%s admission_quota_seconds=%s",
                            tail_steps,
                            tail_request_id,
                            declared_gpu_seconds(tail_steps, config, tail=True),
                            admission_quota_seconds(tail_steps, config, tail=True),
                        )
                        result = client.predict(
                            tail_request_id, tail_steps, True, api_name="/train_segment"
                        )
                        logger.info("tail segment response %s", json.dumps(result, sort_keys=True))
                        record_segment_result(
                            output_dir, supervisor_state_path, supervisor_state, result
                        )
                        tail_succeeded = True
                        break
                    except Exception as tail_error:
                        logger.error(
                            "tail segment request failed: %s: %s",
                            type(tail_error).__name__,
                            tail_error,
                        )
                        if not is_quota_error(tail_error):
                            break
                        set_quota_deadline(
                            supervisor_state_path, supervisor_state, tail_error, config
                        )
                if tail_succeeded:
                    continue
                wait_for_persisted_deadline(
                    supervisor_state_path, supervisor_state, logger
                )
                continue

            recovered = False
            for attempt in range(1, config["short_retry_count"] + 1):
                interruptible_sleep(
                    config["short_retry_seconds"],
                    logger,
                    f"short transient retry {attempt}/{config['short_retry_count']}",
                )
                if STOP:
                    break
                try:
                    client = connect_client(config, token, logger)
                    result = client.predict(
                        request_id, config["segment_steps"], False, api_name="/train_segment"
                    )
                    logger.info("transient retry response %s", json.dumps(result, sort_keys=True))
                    recovered = True
                    break
                except Exception as retry_error:
                    logger.error(
                        "transient retry %s failed: %s: %s",
                        attempt,
                        type(retry_error).__name__,
                        retry_error,
                    )
                    if is_quota_error(retry_error):
                        set_quota_deadline(
                            supervisor_state_path, supervisor_state, retry_error, config
                        )
                        wait_for_persisted_deadline(
                            supervisor_state_path,
                            supervisor_state,
                            logger,
                        )
                        recovered = True
                        break
            if recovered:
                continue
            interruptible_sleep(
                config["general_retry_seconds"],
                logger,
                "Space unavailable after short retries",
            )
            client = connect_client(config, token, logger)

    logger.info("supervisor stopped by signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
