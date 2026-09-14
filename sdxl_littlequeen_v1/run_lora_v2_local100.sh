#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

./scripts/build_lora_v2_local100_prompts.py
exec ./scripts/run_sdxl_cpu.py config_lora_v2_local100.json "$@"
