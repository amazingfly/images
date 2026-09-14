#!/bin/bash
set -e
set -o pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
exec > >(tee -a pipeline_lora.log) 2>&1
exec 9>/tmp/agentic-images-lora-pipeline.lock

if ! flock -n 9; then
    echo "Another LoRA image pipeline run is already active. Exiting at $(date)."
    exit 1
fi

STAGE="initializing"
trap 'echo "LoRA pipeline failed during ${STAGE} at $(date)." >&2' ERR

echo "LoRA pipeline started at $(date)"

CONFIG_FILE="config_lora_littlequeen.json"
SHOULD_GENERATE_PROMPTS=$(python3 - <<'PY'
import json
import os

with open("config_lora_littlequeen.json", "r") as f:
    config = json.load(f)

prompts_file = config["prompts_file"]
regenerate = bool(config.get("pipeline", {}).get("regenerate_prompts", False))
print("yes" if regenerate or not os.path.exists(prompts_file) else "no")
PY
)

if [ "$SHOULD_GENERATE_PROMPTS" = "yes" ]; then
    STAGE="prompt generation"
    echo "Starting Prompt Generation..."
    python3 -u scripts/generate_prompts.py "$CONFIG_FILE"
    echo "Prompt Generation complete."

    echo "Sleeping for 15 seconds to allow RAM to clear..."
    sleep 15
else
    echo "Skipping Prompt Generation; using existing LoRA prompts file."
fi

STAGE="LoRA image generation"
echo "Starting LoRA Image Generation..."
python3 -u scripts/generate_images_lora.py "$CONFIG_FILE"
echo "LoRA image generation and background upscaling complete. Pipeline finished at $(date)."
