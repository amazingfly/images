#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-accessories-v1}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${ROOT}/training_dataset_accessories_v1/colab/lqaccessories_v1_bundle.tar.gz"
DOWNLOADS="${ROOT}/training_dataset_accessories_v1/colab/downloads"
TEST_ROOT="${ROOT}/outputs/accessories_v1_test/colab_fp16_832x1216_20260721"
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
    printf 'Preserving successful Colab session %s for visual review.\n' "${SESSION}"
  elif [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

"${ROOT}/scripts/segment_accessory_references_v1.py"
"${ROOT}/scripts/prepare_accessory_training_datasets_v1.py"
"${ROOT}/scripts/build_accessories_v1_test_prompts.py"
rm -f "${BUNDLE}"
tar -czf "${BUNDLE}" \
  -C "${ROOT}/training_dataset_accessories_v1/colab" train_regalia train_equipment \
  -C "${ROOT}/training_dataset_accessories_v1" curated_manifest.json \
  -C "${ROOT}/colab_accessories_v1" dataset_regalia.toml dataset_equipment.toml isolated_cache.py

if timeout 30s colab sessions </dev/null 2>&1 | grep -Fq "${SESSION}"; then
  printf 'Reusing active Colab session %s\n' "${SESSION}"
else
  colab new -s "${SESSION}" --gpu "${GPU}"
fi

upload_chunked "${BUNDLE}" "lqaccessories_v1_bundle.tar.gz.part_"
upload_chunked "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "lqxl_sdxl_v2.safetensors.part_"
upload_chunked "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors" "lqmoonfit_sdxl_v1.safetensors.part_"
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v1/remote_train.py" /content/remote_train_accessories_v1.py
colab upload -s "${SESSION}" "${ROOT}/colab_accessories_v1/remote_test.py" /content/remote_test_accessories_v1.py
colab upload -s "${SESSION}" "${ROOT}/config_accessories_v1_test_prompts.json" /content/lqaccessories_test_prompts.json

colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v1/launch_train.py" --timeout 21600

mkdir -p "${DOWNLOADS}/checkpoints_regalia" "${DOWNLOADS}/checkpoints_equipment" \
  "${ROOT}/models/loras" "${ROOT}/logs/training_accessories_v1" "${TEST_ROOT}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_v1/training_result.json "${DOWNLOADS}/training_result.json"

for file in resource_usage.csv cache_latents_regalia.log cache_text_regalia.log training_regalia.log cache_latents_equipment.log cache_text_equipment.log training_equipment.log; do
  colab download -s "${SESSION}" "/content/lqaccessories_v1/logs/${file}" "${ROOT}/logs/training_accessories_v1/${file}"
done
for step in 00000300; do
  colab download -s "${SESSION}" \
    "/content/lqaccessories_v1/output_regalia/lqmoonregalia_sdxl_v1-step${step}.safetensors" \
    "${DOWNLOADS}/checkpoints_regalia/lqmoonregalia_sdxl_v1-step${step}.safetensors"
done
for step in 00000300 00000600; do
  colab download -s "${SESSION}" \
    "/content/lqaccessories_v1/output_equipment/lqhandgear_sdxl_v1-step${step}.safetensors" \
    "${DOWNLOADS}/checkpoints_equipment/lqhandgear_sdxl_v1-step${step}.safetensors"
done
colab download -s "${SESSION}" /content/lqaccessories_v1/output_regalia/lqmoonregalia_sdxl_v1.safetensors "${ROOT}/models/loras/lqmoonregalia_sdxl_v1.safetensors"
colab download -s "${SESSION}" /content/lqaccessories_v1/output_equipment/lqhandgear_sdxl_v1.safetensors "${ROOT}/models/loras/lqhandgear_sdxl_v1.safetensors"
cp "${ROOT}/models/loras/lqmoonregalia_sdxl_v1.safetensors" "${DOWNLOADS}/"
cp "${ROOT}/models/loras/lqhandgear_sdxl_v1.safetensors" "${DOWNLOADS}/"
sha256sum "${ROOT}/models/loras/lqmoonregalia_sdxl_v1.safetensors" >"${ROOT}/models/loras/lqmoonregalia_sdxl_v1.safetensors.sha256"
sha256sum "${ROOT}/models/loras/lqhandgear_sdxl_v1.safetensors" >"${ROOT}/models/loras/lqhandgear_sdxl_v1.safetensors.sha256"

# Training artifacts are local before inference starts, so a test-only failure
# cannot strand completed models on an ephemeral runtime.
colab exec -s "${SESSION}" -f "${ROOT}/colab_accessories_v1/launch_test.py" --timeout 21600
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lqaccessories_test_results.tar.gz "${TEST_ROOT}/results.tar.gz"
rm -rf "${TEST_ROOT}/output" "${TEST_ROOT}/test_summary.json"
tar -xzf "${TEST_ROOT}/results.tar.gz" -C "${TEST_ROOT}"
SUCCESS=1
printf 'Downloaded accessory LoRAs and test grid.\n'
