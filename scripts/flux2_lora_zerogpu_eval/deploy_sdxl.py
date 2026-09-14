#!/usr/bin/env python3
"""Deploy the reviewed SDXL identity checkpoint and second-stage ZeroGPU app."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config_sdxl.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="publish the checkpoint and manifest without changing the active Space",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_token() -> str:
    path = Path(CONFIG["token_path"])
    token = path.read_text(encoding="utf-8").strip()
    if not token.startswith("hf_"):
        raise RuntimeError(f"invalid Hugging Face token file: {path}")
    return token


def require_owner(api: HfApi) -> None:
    who = api.whoami()
    owner = who.get("name", "")
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    if owner != CONFIG["owner"] or role not in {"write", "fineGrained"}:
        raise RuntimeError(
            f"write token for {CONFIG['owner']} required; got owner={owner!r} role={role!r}"
        )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    token = read_token()
    api = HfApi(token=token)
    require_owner(api)
    adapter = CONFIG["adapter"]
    source = Path(adapter["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_sha256 = sha256(source)
    if actual_sha256 != adapter["sha256"]:
        raise RuntimeError(
            f"SDXL checkpoint hash mismatch: expected {adapter['sha256']}, got {actual_sha256}"
        )

    model_repo = adapter["repo"]
    if not api.repo_exists(model_repo, repo_type="model"):
        api.create_repo(model_repo, repo_type="model", private=True)
    if not api.file_exists(
        model_repo, adapter["weight_name"], repo_type="model", token=token
    ):
        api.upload_file(
            repo_id=model_repo,
            repo_type="model",
            path_or_fileobj=source,
            path_in_repo=adapter["weight_name"],
            commit_message="Publish reviewed Little Queen SDXL v2 identity LoRA",
        )

    manifest_path = HERE / CONFIG["manifest"]
    api.upload_file(
        repo_id=CONFIG["result_repo"],
        repo_type="dataset",
        path_or_fileobj=manifest_path,
        path_in_repo=f"manifests/{manifest_path.name}",
        commit_message="Publish locked FLUX v2/SDXL comparison manifest",
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "model_repo": model_repo,
                    "checkpoint_sha256": actual_sha256,
                    "manifest": str(manifest_path),
                    "space_unchanged": True,
                },
                indent=2,
            )
        )
        return 0
    with tempfile.TemporaryDirectory(prefix="lq_sdxl_eval_space_") as temporary:
        staging = Path(temporary)
        for source_file in (HERE / "SpaceSDXL").iterdir():
            if source_file.is_file():
                (staging / source_file.name).write_bytes(source_file.read_bytes())
        api.upload_folder(
            repo_id=CONFIG["space_repo"],
            repo_type="space",
            folder_path=staging,
            commit_message="Switch evaluator to reviewed SDXL identity comparison",
            delete_patterns=["*.py", "requirements.txt", "README.md"],
        )
    api.add_space_secret(CONFIG["space_repo"], "HF_TOKEN", token)
    variables = {
        "SDXL_BASE_MODEL": CONFIG["base_model"],
        "RESULT_REPO": CONFIG["result_repo"],
        "SDXL_LORA_REPO": adapter["repo"],
        "SDXL_LORA_WEIGHT": adapter["weight_name"],
        "SDXL_TRIGGER": CONFIG["trigger"],
        "STYLE_PREFIX": CONFIG["style_prefix"],
        "SDXL_NEGATIVE_PROMPT": CONFIG["negative_prompt"],
        "SDXL_WIDTH": CONFIG["width"],
        "SDXL_HEIGHT": CONFIG["height"],
        "SDXL_NUM_INFERENCE_STEPS": CONFIG["num_inference_steps"],
        "SDXL_GUIDANCE_SCALE": CONFIG["guidance_scale"],
        "GPU_DURATION_SECONDS": CONFIG["gpu_duration_seconds"],
    }
    for key, value in variables.items():
        api.add_space_variable(CONFIG["space_repo"], key, str(value))
    runtime = api.get_space_runtime(CONFIG["space_repo"])
    if runtime.requested_hardware != "zero-a10g" and runtime.hardware != "zero-a10g":
        api.request_space_hardware(CONFIG["space_repo"], "zero-a10g")
    marker = {
        "status": "deployed",
        "deployed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "space_repo": CONFIG["space_repo"],
        "result_repo": CONFIG["result_repo"],
        "model_repo": model_repo,
        "checkpoint_sha256": actual_sha256,
        "manifest": str(manifest_path),
    }
    atomic_json(Path(CONFIG["deployment_marker"]), marker)
    print(json.dumps(marker, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
