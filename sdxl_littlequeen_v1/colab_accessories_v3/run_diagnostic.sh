#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-accessories-v3-diagnostic}"
GPU="${COLAB_GPU:-T4}"
TEST_ROOT="${ROOT}/outputs/accessories_v3_diagnostic/colab_fp16_832x1216_20260722"
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

allocated=0
for delay in 0 30 60 120; do
  if [[ "${delay}" -gt 0 ]]; then sleep "${delay}"; fi
  if colab new -s "${SESSION}" --gpu "${GPU}"; then
    allocated=1
    break
  fi
done
if [[ "${allocated}" -ne 1 ]]; then
  printf 'Unable to allocate Colab session after retries.\n' >&2
  exit 1
fi

upload_chunked "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "lqxl_sdxl_v2.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors" "lqmoonfit_sdxl_v1.safetensors.part_"
mkdir -p "${TEST_ROOT}"
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v1/remote_test.py" /content/remote_test_accessories_base.py
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v3/remote_test.py" /content/remote_test_accessories_v3.py
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v3/launch_test.py" /content/launch_test_accessories_v3.py
colab upload -s "${SESSION}" "${ROOT}/config_accessories_v3_diagnostic_prompts.json" /content/lqaccessories_v3_test_prompts.json
colab upload -s "${SESSION}" "${ROOT}/models/loras/lqmoonregalia_sdxl_v3.safetensors" /content/lqmoonregalia_sdxl_v3.safetensors
colab upload -s "${SESSION}" "${ROOT}/models/loras/lqrosekeeper_sdxl_v3.safetensors" /content/lqrosekeeper_sdxl_v3.safetensors

colab exec -s "${SESSION}" --timeout 21600 --command \
  "mkdir -p /content/lqaccessories_v3/output_regalia /content/lqaccessories_v3/output_staff && cp /content/lqmoonregalia_sdxl_v3.safetensors /content/lqaccessories_v3/output_regalia/ && cp /content/lqrosekeeper_sdxl_v3.safetensors /content/lqaccessories_v3/output_staff/"
colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v3/launch_test.py" --timeout 21600

COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_v3_test_results.tar.gz "${TEST_ROOT}/results.tar.gz"
rm -rf "${TEST_ROOT}/output" "${TEST_ROOT}/test_summary.json"
tar -xzf "${TEST_ROOT}/results.tar.gz" -C "${TEST_ROOT}"
SUCCESS=1
printf 'Downloaded the V3 same-seed diagnostic grid.\n'
