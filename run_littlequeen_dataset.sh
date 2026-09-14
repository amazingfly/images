#!/bin/bash
set -e
set -o pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

CONFIG_FILE="${1:-config_lora_littlequeen_dataset.json}"
LOG_FILE="${2:-pipeline_littlequeen_dataset.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
exec 9>/tmp/agentic-images-littlequeen-dataset-pipeline.lock

if ! flock -n 9; then
    echo "Another Little Queen dataset pipeline run is already active. Exiting at $(date)."
    exit 1
fi

echo "Little Queen dataset pipeline started at $(date)"
echo "Config: $CONFIG_FILE"
echo "Log: $LOG_FILE"

python3 -u scripts/run_littlequeen_dataset.py "$CONFIG_FILE"

echo "Little Queen dataset pipeline finished at $(date)"
