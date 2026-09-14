#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${ROOT}/scripts/run_sdxl_cpu.py" "${ROOT}/config_lora5.json" "$@"
