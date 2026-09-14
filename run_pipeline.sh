#!/bin/bash
set -e
set -o pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
exec > >(tee -a pipeline.log) 2>&1
exec 9>/tmp/agentic-images-pipeline.lock

if ! flock -n 9; then
    echo "Another image pipeline run is already active. Exiting at $(date)."
    exit 1
fi

STAGE="initializing"
trap 'echo "Pipeline failed during ${STAGE} at $(date)." >&2' ERR

echo "Pipeline started at $(date)"

SHOULD_GENERATE_PROMPTS=$(python3 - <<'PY'
import json
import os

with open("config.json", "r") as f:
    config = json.load(f)

prompts_file = config["prompts_file"]
regenerate = bool(config.get("pipeline", {}).get("regenerate_prompts", False))
print("yes" if regenerate or not os.path.exists(prompts_file) else "no")
PY
)

if [ "$SHOULD_GENERATE_PROMPTS" = "yes" ]; then
    # Run Prompt Generation
    STAGE="prompt generation"
    echo "Starting Prompt Generation..."
    python3 -u scripts/generate_prompts.py
    echo "Prompt Generation complete."

    # Sleep to ensure RAM is reclaimed
    echo "Sleeping for 15 seconds to allow RAM to clear..."
    sleep 15
else
    echo "Skipping Prompt Generation; using existing prompts file."
fi


# Run Image Generation
STAGE="image generation"
echo "Starting Image Generation..."
python3 -u scripts/generate_images.py
echo "Image generation and background upscaling complete. Pipeline finished at $(date)."
