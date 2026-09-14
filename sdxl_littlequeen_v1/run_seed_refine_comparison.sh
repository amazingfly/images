#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
SEED_RUN="${RUN_ID}_seed_refine"
STANDARD_RUN="${RUN_ID}_standard"
REPORT_DIR="${ROOT}/outputs/seed_refine_comparison/${RUN_ID}_report"

"${ROOT}/scripts/run_sdxl_cpu.py" \
  "${ROOT}/config_compare_seed_refine10.json" --run-id "${SEED_RUN}"
"${ROOT}/scripts/run_sdxl_cpu.py" \
  "${ROOT}/config_compare_standard10.json" --run-id "${STANDARD_RUN}"
"${ROOT}/scripts/compare_seed_refine.py" \
  "${ROOT}/outputs/seed_refine_comparison/standard/${STANDARD_RUN}/run_summary.json" \
  "${ROOT}/outputs/seed_refine_comparison/seed_refine/${SEED_RUN}/run_summary.json" \
  "${REPORT_DIR}"
