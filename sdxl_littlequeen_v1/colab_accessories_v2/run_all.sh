#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-accessories-v2}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${ROOT}/training_dataset_accessories_v2/colab/lqaccessories_v2_bundle.tar.gz"
DOWNLOADS="${ROOT}/training_dataset_accessories_v2/colab/downloads"
TEST_ROOT="${ROOT}/outputs/accessories_v2_test/colab_fp16_832x1216_20260721"
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
    printf 'Preserving successful Colab session %s.\n' "${SESSION}"
  elif [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

"${ROOT}/scripts/prepare_accessory_object_datasets_v2.py"
"${ROOT}/scripts/build_accessories_v2_test_prompts.py"
rm -f "${BUNDLE}"
tar -czf "${BUNDLE}" \
  -C "${ROOT}/training_dataset_accessories_v2/colab" train_crown train_wand \
  -C "${ROOT}/training_dataset_accessories_v2" manifest.json \
  -C "${ROOT}/colab_accessories_v2" dataset_crown.toml dataset_wand.toml isolated_cache.py

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

upload_chunked "${BUNDLE}" "lqaccessories_v2_bundle.tar.gz.part_"
upload_chunked "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "lqxl_sdxl_v2.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors" "lqmoonfit_sdxl_v1.safetensors.part_"
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v1/remote_train.py" /content/remote_train_accessories_base.py
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v1/remote_test.py" /content/remote_test_accessories_base.py
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v2/remote_train.py" /content/remote_train_accessories_v2.py
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v2/remote_test.py" /content/remote_test_accessories_v2.py
colab upload -s "${SESSION}" "${ROOT}/config_accessories_v2_test_prompts.json" /content/lqaccessories_v2_test_prompts.json

colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v2/launch_train.py" --timeout 21600

mkdir -p "${DOWNLOADS}/checkpoints_crown" "${DOWNLOADS}/checkpoints_wand" \
  "${ROOT}/models/loras" "${ROOT}/logs/training_accessories_v2" "${TEST_ROOT}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_v2/training_result.json "${DOWNLOADS}/training_result.json"

for file in resource_usage.csv cache_latents_crown.log cache_text_crown.log training_crown.log cache_latents_wand.log cache_text_wand.log training_wand.log; do
  colab download -s "${SESSION}" "/content/lqaccessories_v2/logs/${file}" "${ROOT}/logs/training_accessories_v2/${file}"
done
colab download -s "${SESSION}" \
  /content/lqaccessories_v2/output_crown/lqmooncrest_sdxl_v2-step00000250.safetensors \
  "${DOWNLOADS}/checkpoints_crown/lqmooncrest_sdxl_v2-step00000250.safetensors"
colab download -s "${SESSION}" \
  /content/lqaccessories_v2/output_wand/lqrosewand_sdxl_v2-step00000250.safetensors \
  "${DOWNLOADS}/checkpoints_wand/lqrosewand_sdxl_v2-step00000250.safetensors"
colab download -s "${SESSION}" \
  /content/lqaccessories_v2/output_crown/lqmooncrest_sdxl_v2.safetensors \
  "${ROOT}/models/loras/lqmooncrest_sdxl_v2.safetensors"
colab download -s "${SESSION}" \
  /content/lqaccessories_v2/output_wand/lqrosewand_sdxl_v2.safetensors \
  "${ROOT}/models/loras/lqrosewand_sdxl_v2.safetensors"
cp "${ROOT}/models/loras/lqmooncrest_sdxl_v2.safetensors" "${DOWNLOADS}/"
cp "${ROOT}/models/loras/lqrosewand_sdxl_v2.safetensors" "${DOWNLOADS}/"
sha256sum "${ROOT}/models/loras/lqmooncrest_sdxl_v2.safetensors" >"${ROOT}/models/loras/lqmooncrest_sdxl_v2.safetensors.sha256"
sha256sum "${ROOT}/models/loras/lqrosewand_sdxl_v2.safetensors" >"${ROOT}/models/loras/lqrosewand_sdxl_v2.safetensors.sha256"

colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v2/launch_test.py" --timeout 21600
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_v2_test_results.tar.gz "${TEST_ROOT}/results.tar.gz"
rm -rf "${TEST_ROOT}/output" "${TEST_ROOT}/test_summary.json"
tar -xzf "${TEST_ROOT}/results.tar.gz" -C "${TEST_ROOT}"
SUCCESS=1
printf 'Downloaded v2 crown and wand-staff LoRAs plus the validation grid.\n'
