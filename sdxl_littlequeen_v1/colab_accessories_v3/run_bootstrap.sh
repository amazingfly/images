#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-accessories-v3-bootstrap}"
GPU="${COLAB_GPU:-T4}"
LOCAL_ROOT="${ROOT}/outputs/accessories_v3_bootstrap/colab_fp16_832x1216_20260721"
SUCCESS=0

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
  if [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

"${ROOT}/scripts/build_accessory_bootstrap_prompts_v3.py"
colab new -s "${SESSION}" --gpu "${GPU}"
upload_chunked "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "lqxl_sdxl_v2.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors" "lqmoonfit_sdxl_v1.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqmooncrest_sdxl_v2.safetensors" "lqmooncrest_sdxl_v2.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqrosewand_sdxl_v2.safetensors" "lqrosewand_sdxl_v2.safetensors.part_"
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v1/remote_test.py" /content/remote_test_accessories_base.py
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v3/remote_bootstrap.py" /content/remote_accessories_v3_bootstrap.py
colab upload -s "${SESSION}" "${ROOT}/config_accessories_v3_bootstrap_prompts.json" /content/lqaccessories_v3_bootstrap_prompts.json

colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v3/launch_bootstrap.py" --timeout 21600
mkdir -p "${LOCAL_ROOT}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_v3_bootstrap_results.tar.gz "${LOCAL_ROOT}/results.tar.gz"
rm -rf "${LOCAL_ROOT}/output" "${LOCAL_ROOT}/test_summary.json"
tar -xzf "${LOCAL_ROOT}/results.tar.gz" -C "${LOCAL_ROOT}"
SUCCESS=1
printf 'Downloaded contextual accessory v3 bootstrap candidates.\n'
