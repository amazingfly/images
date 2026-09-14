#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-compare}"
SELECTED="${2:-900}"
SESSION="${COLAB_SESSION:-lqxl-v2-eval}"
GPU="${COLAB_GPU:-T4}"
REMOTE_SCRIPT="/content/remote_test_v2.py"

upload_chunked() {
  local source="$1"
  local remote_name="$2"
  local chunk_dir
  chunk_dir="$(mktemp -d)"
  split -b 40M -d -a 2 "${source}" "${chunk_dir}/part_"
  for part in "${chunk_dir}"/part_*; do
    colab upload -s "${SESSION}" "${part}" "/content/${remote_name}.$(basename "${part}")"
  done
  rm -rf "${chunk_dir}"
}

case "${MODE}" in
  compare)
    LAUNCHER="${ROOT}/colab_v2_test/launch_compare.py"
    REMOTE_ARCHIVE="/content/lqxl_v2_compare_results.tar.gz"
    LOCAL_ROOT="${ROOT}/outputs/lqxl_sdxl_v2_eval/compare"
    ;;
  final)
    LAUNCHER="${ROOT}/colab_v2_test/launch_final.py"
    REMOTE_ARCHIVE="/content/lqxl_v2_final_results.tar.gz"
    LOCAL_ROOT="${ROOT}/outputs/lqxl_sdxl_v2_eval/minimal10_${SELECTED}"
    ;;
  *) printf 'Usage: %s [compare|final] [900|1200]\n' "$0" >&2; exit 2 ;;
esac

if timeout 30s colab sessions </dev/null 2>&1 | grep -Fq "${SESSION}"; then
  printf 'Reusing active Colab session %s\n' "${SESSION}"
else
  colab new -s "${SESSION}" --gpu "${GPU}"
fi

colab upload -s "${SESSION}" "${ROOT}/colab_v2_test/remote_test.py" "${REMOTE_SCRIPT}"
if [[ "${MODE}" == "compare" && "${SKIP_LORA_UPLOAD:-0}" != "1" ]]; then
  upload_chunked \
    "${ROOT}/training_dataset_v2/colab/downloads/checkpoints/lqxl_sdxl_v2-step00000900.safetensors" \
    lqxl_v2_step900.safetensors
  upload_chunked \
    "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" \
    lqxl_v2_final1200.safetensors
elif [[ "${MODE}" == "final" ]]; then
  case "${SELECTED}" in
    900) LORA="${ROOT}/training_dataset_v2/colab/downloads/checkpoints/lqxl_sdxl_v2-step00000900.safetensors" ;;
    1200) LORA="${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" ;;
    *) printf 'Selected checkpoint must be 900 or 1200\n' >&2; exit 2 ;;
  esac
  upload_chunked "${LORA}" lqxl_v2_selected.safetensors
fi

colab exec -s "${SESSION}" -f "${LAUNCHER}" --timeout 7200
mkdir -p "${LOCAL_ROOT}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" "${REMOTE_ARCHIVE}" "${LOCAL_ROOT}/results.tar.gz"
rm -rf "${LOCAL_ROOT}/output" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/run_summary.json"
tar -xzf "${LOCAL_ROOT}/results.tar.gz" -C "${LOCAL_ROOT}"

if [[ "${MODE}" == "final" ]]; then
  timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
fi
printf 'Downloaded %s results to %s\n' "${MODE}" "${LOCAL_ROOT}"
