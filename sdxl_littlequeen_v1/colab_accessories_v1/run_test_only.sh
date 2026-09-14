#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-accessories-test-v1}"
GPU="${COLAB_GPU:-T4}"
TEST_PROFILE="${ACCESSORY_TEST_PROFILE:-stress}"
TEST_TAG="${ACCESSORY_TEST_TAG:-${TEST_PROFILE}}"
TEST_ROOT="${ROOT}/outputs/accessories_v1_test/colab_fp16_832x1216_20260721_${TEST_TAG}"
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
  if [[ "${SUCCESS}" -eq 1 && "${KEEP_COLAB_ON_SUCCESS:-0}" -eq 1 ]]; then
    printf 'Preserving Colab session %s for follow-up tests.\n' "${SESSION}" >&2
  elif [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after test failure.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

"${ROOT}/scripts/build_accessories_v1_test_prompts.py" --profile "${TEST_PROFILE}"
colab new -s "${SESSION}" --gpu "${GPU}"
upload_chunked "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "lqxl_sdxl_v2.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors" "lqmoonfit_sdxl_v1.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqmoonregalia_sdxl_v1.safetensors" "lqmoonregalia_sdxl_v1.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqhandgear_sdxl_v1.safetensors" "lqhandgear_sdxl_v1.safetensors.part_"
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v1/remote_test.py" /content/remote_test_accessories_v1.py
colab upload -s "${SESSION}" "${ROOT}/config_accessories_v1_test_prompts.json" /content/lqaccessories_test_prompts.json
colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v1/launch_test.py" --timeout 21600

mkdir -p "${TEST_ROOT}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_test_results.tar.gz "${TEST_ROOT}/results.tar.gz"
rm -rf "${TEST_ROOT}/output" "${TEST_ROOT}/test_summary.json"
tar -xzf "${TEST_ROOT}/results.tar.gz" -C "${TEST_ROOT}"
SUCCESS=1
printf 'Downloaded accessory test grid to %s\n' "${TEST_ROOT}"
