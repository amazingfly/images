#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-accessories-test-v1}"
TEST_ROOT="${ROOT}/outputs/accessories_v1_test/colab_fp16_832x1216_20260721_checkpoints"

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

"${ROOT}/scripts/build_accessories_v1_test_prompts.py" --profile checkpoints
upload_chunked "${ROOT}/training_dataset_accessories_v1/colab/downloads/checkpoints_regalia/lqmoonregalia_sdxl_v1-step00000300.safetensors" "lqmoonregalia_step300.safetensors.part_"
upload_chunked "${ROOT}/training_dataset_accessories_v1/colab/downloads/checkpoints_equipment/lqhandgear_sdxl_v1-step00000300.safetensors" "lqhandgear_step300.safetensors.part_"
upload_chunked "${ROOT}/training_dataset_accessories_v1/colab/downloads/checkpoints_equipment/lqhandgear_sdxl_v1-step00000600.safetensors" "lqhandgear_step600.safetensors.part_"
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v1/remote_test.py" /content/remote_test_accessories_v1.py
colab upload -s "${SESSION}" "${ROOT}/config_accessories_v1_test_prompts.json" /content/lqaccessories_test_prompts.json
colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v1/launch_test.py" --timeout 21600

mkdir -p "${TEST_ROOT}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_test_results.tar.gz "${TEST_ROOT}/results.tar.gz"
rm -rf "${TEST_ROOT}/output" "${TEST_ROOT}/test_summary.json"
tar -xzf "${TEST_ROOT}/results.tar.gz" -C "${TEST_ROOT}"
printf 'Downloaded checkpoint comparison to %s\n' "${TEST_ROOT}"
