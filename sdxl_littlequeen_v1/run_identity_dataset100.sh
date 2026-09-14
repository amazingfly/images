#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

./scripts/build_identity_dataset100_prompts.py
exec ./scripts/run_sdxl_cpu.py config_identity_dataset100.json "$@"
