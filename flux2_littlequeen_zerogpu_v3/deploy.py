#!/usr/bin/env python3
"""Create or update the Little Queen FLUX.2 Klein Hub repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
PINNED_DIFFUSERS_COMMIT = "3a2f35d4efa4c059c8bfb3bc0d6c906264895c81"
TRAINER_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/"
    f"{PINNED_DIFFUSERS_COMMIT}/examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
)


class WriteTokenRequired(RuntimeError):
    """The unattended deployment cannot proceed with the current credential."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--skip-hardware", action="store_true")
    return parser.parse_args()


def read_token(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Hugging Face token file does not exist: {path}")
    token = path.read_text(encoding="utf-8").strip()
    if not token.startswith("hf_"):
        raise RuntimeError(f"Hugging Face token file is invalid: {path}")
    return token


def require_write_token(api: HfApi) -> str:
    who = api.whoami()
    owner = who.get("name", "")
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    if role not in {"write", "fineGrained"}:
        raise WriteTokenRequired(
            f"Hugging Face token for {owner or 'unknown user'} is {role!r}, not write-capable. "
            "Create a write token at https://huggingface.co/settings/tokens and replace the token file."
        )
    return owner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_file_matches(api: HfApi, repo_id: str, repo_type: str, local: Path, remote: str) -> bool:
    try:
        cached = Path(
            hf_hub_download(
                repo_id,
                remote,
                repo_type=repo_type,
                token=api.token,
                force_download=True,
            )
        )
        return sha256(cached) == sha256(local)
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    token_path = (args.token_file or Path(config["token_path"])).expanduser()
    token = read_token(token_path)
    api = HfApi(token=token)
    owner = require_write_token(api)
    if owner != config["owner"]:
        raise RuntimeError(f"token owner {owner!r} does not match configured owner {config['owner']!r}")

    dataset_dir = Path(config["dataset_dir"])
    if not (dataset_dir / "dataset_manifest.json").is_file():
        raise RuntimeError(f"prepared dataset is missing: {dataset_dir}")

    if not api.repo_exists(config["dataset_repo"], repo_type="dataset"):
        api.create_repo(config["dataset_repo"], repo_type="dataset", private=True)
    if not api.repo_exists(config["output_repo"], repo_type="model"):
        api.create_repo(config["output_repo"], repo_type="model", private=True)
    if not api.repo_exists(config["space_repo"], repo_type="space"):
        api.create_repo(
            config["space_repo"],
            repo_type="space",
            private=True,
            space_sdk="gradio",
            space_hardware="zero-a10g" if not args.skip_hardware else None,
        )
    if not remote_file_matches(
        api,
        config["dataset_repo"],
        "dataset",
        dataset_dir / "dataset_manifest.json",
        "dataset_manifest.json",
    ):
        api.upload_folder(
            repo_id=config["dataset_repo"],
            repo_type="dataset",
            folder_path=dataset_dir,
            commit_message="Upload curated Little Queen FLUX.2 Klein training dataset",
        )
    else:
        print("curated dataset already matches Hub manifest; skipping upload")

    model_readme = HERE / "model_README.md"
    api.upload_file(
        repo_id=config["output_repo"],
        repo_type="model",
        path_or_fileobj=model_readme,
        path_in_repo="README.md",
        commit_message="Initialize Little Queen FLUX.2 Klein LoRA repository",
    )

    with tempfile.TemporaryDirectory(prefix="lq_flux2_space_") as temp:
        staging = Path(temp)
        for path in (HERE / "Space").iterdir():
            if path.name == "__pycache__" or path.suffix == ".pyc":
                continue
            destination = staging / path.name
            if path.is_dir():
                shutil.copytree(path, destination)
            else:
                shutil.copy2(path, destination)
        source = staging / "official_trainer.py"
        from urllib.request import urlopen

        with urlopen(TRAINER_URL, timeout=120) as response:
            source.write_bytes(response.read())
        destination = staging / "train_flux2_klein_segmented.py"
        import subprocess

        subprocess.run(
            [str(Path.home() / "miniforge3/bin/python"), str(HERE / "patch_trainer.py"), str(source), str(destination)],
            check=True,
        )
        source.unlink()
        api.upload_folder(
            repo_id=config["space_repo"],
            repo_type="space",
            folder_path=staging,
            commit_message="Deploy segmented FLUX.2 Klein ZeroGPU trainer",
        )

    api.add_space_secret(config["space_repo"], "HF_TOKEN", token)
    for key in (
        "dataset_repo",
        "output_repo",
        "total_steps",
        "segment_steps",
        "gpu_duration_fixed_seconds",
        "gpu_seconds_per_step",
        "tail_gpu_duration_fixed_seconds",
        "tail_gpu_seconds_per_step",
        "gpu_max_duration_seconds",
    ):
        api.add_space_variable(config["space_repo"], key.upper(), str(config[key]))
    if not args.skip_hardware:
        runtime = api.get_space_runtime(config["space_repo"])
        if runtime.requested_hardware != "zero-a10g" and runtime.hardware != "zero-a10g":
            api.request_space_hardware(config["space_repo"], "zero-a10g")
    print(
        json.dumps(
            {
                "status": "deployed",
                "space": f"https://huggingface.co/spaces/{config['space_repo']}",
                "dataset": f"https://huggingface.co/datasets/{config['dataset_repo']}",
                "model": f"https://huggingface.co/{config['output_repo']}",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WriteTokenRequired as error:
        print(f"CREDENTIAL_WAIT: {error}")
        raise SystemExit(3)
