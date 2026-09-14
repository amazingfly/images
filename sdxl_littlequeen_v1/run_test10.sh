#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "${ROOT}/scripts/run_sdxl_cpu.py" "${ROOT}/config_test10.json" "$@"
