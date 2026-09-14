#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

./scripts/build_outfit_v1_local100_prompts.py
exec ./scripts/run_sdxl_cpu.py config_outfit_v1_local100.json "$@"
