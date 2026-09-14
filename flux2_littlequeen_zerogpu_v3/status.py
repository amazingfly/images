#!/usr/bin/env python3
"""Show local and Hub status without exposing credentials."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


HERE = Path(__file__).resolve().parent
config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
token = Path(config["token_path"]).read_text(encoding="utf-8").strip()
result = {"space_repo": config["space_repo"], "output_repo": config["output_repo"]}
supervisor_state = Path(config["output_dir"]) / "supervisor_state.json"
if supervisor_state.is_file():
    result["supervisor"] = json.loads(supervisor_state.read_text(encoding="utf-8"))
try:
    runtime = HfApi(token=token).get_space_runtime(config["space_repo"])
    result["space_stage"] = runtime.stage
    result["space_hardware"] = runtime.hardware
except Exception as error:
    result["space_error"] = str(error)
try:
    state = hf_hub_download(
        config["output_repo"],
        "training_state.json",
        repo_type="model",
        token=token,
        force_download=True,
    )
    result["training"] = json.loads(Path(state).read_text(encoding="utf-8"))
except EntryNotFoundError:
    result["training"] = {"latest_step": 0, "total_steps": config["total_steps"],
                          "status": "awaiting first saved checkpoint"}
except Exception as error:
    result["training_error"] = str(error)
print(json.dumps(result, indent=2, default=str))
