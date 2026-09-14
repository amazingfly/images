#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lqmoonfit-sdxl-v1}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${ROOT}/training_dataset_outfit_v1/colab/lqmoonfit_v1_training_bundle.tar.gz"
DOWNLOADS="${ROOT}/training_dataset_outfit_v1/colab/downloads"
REMOTE_BUNDLE="/content/lqmoonfit_v1_bundle.tar.gz"
REMOTE_SCRIPT="/content/remote_train_outfit_v1.py"
SUCCESS=0

cleanup() {
  if [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

"${ROOT}/scripts/prepare_outfit_training_dataset_v1.py"
rm -f "${BUNDLE}"
tar -czf "${BUNDLE}" \
  -C "${ROOT}/training_dataset_outfit_v1/colab" train prepared_manifest.json \
  -C "${ROOT}/colab_outfit_v1" dataset_config.toml isolated_cache.py

if timeout 30s colab sessions </dev/null 2>&1 | grep -Fq "${SESSION}"; then
  printf 'Reusing active Colab session %s\n' "${SESSION}"
else
  colab new -s "${SESSION}" --gpu "${GPU}"
fi

colab upload -s "${SESSION}" "${BUNDLE}" "${REMOTE_BUNDLE}"
colab upload -s "${SESSION}" "${ROOT}/colab_outfit_v1/remote_train.py" "${REMOTE_SCRIPT}"
colab exec -s "${SESSION}" -f "${ROOT}/colab_outfit_v1/launch_remote.py" --timeout 21600

mkdir -p "${DOWNLOADS}/checkpoints" "${ROOT}/models/loras" "${ROOT}/logs/training_outfit_v1"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" \
  /content/lqmoonfit_v1/training_result.json "${DOWNLOADS}/training_result.json"
colab download -s "${SESSION}" /content/lqmoonfit_v1/logs/resource_usage.csv "${ROOT}/logs/training_outfit_v1/resource_usage.csv"
colab download -s "${SESSION}" /content/lqmoonfit_v1/logs/cache_latents.log "${ROOT}/logs/training_outfit_v1/cache_latents.log"
colab download -s "${SESSION}" /content/lqmoonfit_v1/logs/cache_text.log "${ROOT}/logs/training_outfit_v1/cache_text.log"
colab download -s "${SESSION}" /content/lqmoonfit_v1/logs/training.log "${ROOT}/logs/training_outfit_v1/training.log"
for step in 00000300 00000600; do
  colab download -s "${SESSION}" \
    "/content/lqmoonfit_v1/output/lqmoonfit_sdxl_v1-step${step}.safetensors" \
    "${DOWNLOADS}/checkpoints/lqmoonfit_sdxl_v1-step${step}.safetensors"
done
colab download -s "${SESSION}" /content/lqmoonfit_v1/output/lqmoonfit_sdxl_v1.safetensors "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors"
cp "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors" "${DOWNLOADS}/lqmoonfit_sdxl_v1.safetensors"
sha256sum "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors" >"${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors.sha256"
SUCCESS=1
printf 'Downloaded trained outfit LoRA to %s\n' "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors"
