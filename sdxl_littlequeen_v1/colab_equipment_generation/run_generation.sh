#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lqxl-equipment-fp16}"
GPU="${COLAB_GPU:-T4}"
REMOTE_SCRIPT="/content/remote_equipment_generate.py"
REMOTE_PROMPTS="/content/lqxl_equipment_prompts.json"
REMOTE_ARCHIVE="/content/lqxl_equipment_results.tar.gz"
LOCAL_ROOT="${ROOT}/outputs/equipment_dataset100_v1/colab_fp16_832x1216_20260720"
DOWNLOAD="${LOCAL_ROOT}/lqxl_equipment_results.tar.gz"

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

"${ROOT}/scripts/build_equipment_dataset100_prompts.py"

if timeout 30s colab sessions </dev/null 2>&1 | grep -Fq "${SESSION}"; then
  printf 'Reusing active Colab session %s\n' "${SESSION}"
else
  colab new -s "${SESSION}" --gpu "${GPU}"
fi

colab upload -s "${SESSION}" "${ROOT}/colab_equipment_generation/remote_generate.py" "${REMOTE_SCRIPT}"
colab upload -s "${SESSION}" "${ROOT}/config_equipment_dataset100_prompts.json" "${REMOTE_PROMPTS}"
upload_chunked "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "lqxl_sdxl_v2.safetensors.part_"

colab exec -s "${SESSION}" -f "${ROOT}/colab_equipment_generation/launch_full.py" --timeout 21600

mkdir -p "${LOCAL_ROOT}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" "${REMOTE_ARCHIVE}" "${DOWNLOAD}"
rm -rf "${LOCAL_ROOT}/output" "${LOCAL_ROOT}/logs" "${LOCAL_ROOT}/run_summary.json"
tar -xzf "${DOWNLOAD}" -C "${LOCAL_ROOT}"
timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
printf 'Downloaded Colab results to %s\n' "${LOCAL_ROOT}"
