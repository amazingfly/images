#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-accessories-v4-region}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${ROOT}/training_dataset_accessories_v4/colab/lqaccessories_v4_region_bundle.tar.gz"
RESULTS="${ROOT}/outputs/accessories_v4_region/colab_strength_sweep_20260726"
SUCCESS=0
ALLOCATED=0

upload_chunked() {
  local source="$1"
  local remote_prefix="$2"
  local chunk_dir
  chunk_dir="$(mktemp -d)"
  split -b 40M -d -a 2 "${source}" "${chunk_dir}/part_"
  for part in "${chunk_dir}"/part_*; do
    colab upload -s "${SESSION}" "${part}" "/content/${remote_prefix}$(basename "${part}")"
  done
  rm -rf "${chunk_dir}"
}

cleanup() {
  if [[ "${ALLOCATED}" -eq 0 ]]; then
    return
  elif [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

"${ROOT}/scripts/build_accessories_v4_region_bundle.py"
colab new -s "${SESSION}" --gpu "${GPU}"
ALLOCATED=1
upload_chunked "${BUNDLE}" "lqaccessories_v4_region_bundle.tar.gz.part_"
upload_chunked "${ROOT}/models/loras/lqmoonregalia_sdxl_v3.safetensors" "lqmoonregalia_sdxl_v3.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqrosekeeper_sdxl_v3.safetensors" "lqrosekeeper_sdxl_v3.safetensors.part_"
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v4_region/remote_test.py" /content/remote_accessories_v4_region.py
colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v4_region/launch.py" --timeout 21600

mkdir -p "${RESULTS}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_v4_region_results.tar.gz "${RESULTS}/results.tar.gz"
rm -rf "${RESULTS}/regalia" "${RESULTS}/final" "${RESULTS}/review" "${RESULTS}/summary.json"
tar -xzf "${RESULTS}/results.tar.gz" -C "${RESULTS}"
SUCCESS=1
printf 'Downloaded precise-region validation to %s\n' "${RESULTS}"
