#!/usr/bin/env python3
"""Deploy the private Little Queen ZeroGPU evaluation endpoint."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


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


def main() -> int:
    token = read_token()
    api = HfApi(token=token)
    require_owner(api)
    result_repo = CONFIG["result_repo"]
    if not api.repo_exists(result_repo, repo_type="dataset"):
        api.create_repo(result_repo, repo_type="dataset", private=True)
    manifest_path = HERE / CONFIG["manifest"]
    api.upload_file(
        repo_id=result_repo,
        repo_type="dataset",
        path_or_fileobj=manifest_path,
        path_in_repo=f"manifests/{manifest_path.name}",
        commit_message="Publish locked FLUX v1/v2 evaluation manifest",
    )
    with tempfile.TemporaryDirectory(prefix="lq_eval_space_") as temporary:
        staging = Path(temporary)
        for source in (HERE / "Space").iterdir():
            if source.is_file():
                (staging / source.name).write_bytes(source.read_bytes())
        api.upload_folder(
            repo_id=CONFIG["space_repo"],
            repo_type="space",
            folder_path=staging,
            commit_message="Repurpose completed v1 trainer as LoRA evaluator",
            delete_patterns=["*.py", "requirements.txt", "README.md"],
        )
    api.add_space_secret(CONFIG["space_repo"], "HF_TOKEN", token)
    variables = {
        "BASE_MODEL": CONFIG["base_model"],
        "RESULT_REPO": CONFIG["result_repo"],
        "V1_REPO": CONFIG["adapters"]["flux_v1"]["repo"],
        "V1_WEIGHT": CONFIG["adapters"]["flux_v1"]["weight_name"],
        "V2_REPO": CONFIG["adapters"]["flux_v2"]["repo"],
        "V2_WEIGHT": CONFIG["adapters"]["flux_v2"]["weight_name"],
        "TRIGGER": CONFIG["trigger"],
        "STYLE_PREFIX": CONFIG["style_prefix"],
        "WIDTH": CONFIG["width"],
        "HEIGHT": CONFIG["height"],
        "NUM_INFERENCE_STEPS": CONFIG["num_inference_steps"],
        "GUIDANCE_SCALE": CONFIG["guidance_scale"],
        "MAX_SEQUENCE_LENGTH": CONFIG["max_sequence_length"],
        "GPU_DURATION_SECONDS": CONFIG["gpu_duration_seconds"],
    }
    for key, value in variables.items():
        api.add_space_variable(CONFIG["space_repo"], key, str(value))
    runtime = api.get_space_runtime(CONFIG["space_repo"])
    if runtime.requested_hardware != "zero-a10g" and runtime.hardware != "zero-a10g":
        api.request_space_hardware(CONFIG["space_repo"], "zero-a10g")
    print(
        json.dumps(
            {
                "status": "deployed",
                "space": f"https://huggingface.co/spaces/{CONFIG['space_repo']}",
                "results": f"https://huggingface.co/datasets/{result_repo}",
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
