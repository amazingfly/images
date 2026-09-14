#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lqxl-sdxl-v2}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${ROOT}/training_dataset_v2/colab/lqxl_v2_training_bundle.tar.gz"
DOWNLOADS="${ROOT}/training_dataset_v2/colab/downloads"
REMOTE_BUNDLE="/content/lqxl_v2_bundle.tar.gz"
REMOTE_SCRIPT="/content/remote_train_v2.py"
SUCCESS=0

cleanup() {
  if [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

"${ROOT}/scripts/prepare_training_dataset_v2.py"
rm -f "${BUNDLE}"
tar -czf "${BUNDLE}" \
  -C "${ROOT}/training_dataset_v2/colab" train prepared_manifest.json \
  -C "${ROOT}/colab_v2" dataset_config.toml isolated_cache.py

if timeout 30s colab sessions </dev/null 2>&1 | grep -Fq "${SESSION}"; then
  printf 'Reusing active Colab session %s\n' "${SESSION}"
else
  colab new -s "${SESSION}" --gpu "${GPU}"
fi

colab upload -s "${SESSION}" "${BUNDLE}" "${REMOTE_BUNDLE}"
colab upload -s "${SESSION}" "${ROOT}/colab_v2/remote_train.py" "${REMOTE_SCRIPT}"
colab exec -s "${SESSION}" -f "${ROOT}/colab_v2/launch_remote.py" --timeout 21600

mkdir -p "${DOWNLOADS}" "${ROOT}/models/loras" "${ROOT}/logs/training_v2"
mkdir -p "${DOWNLOADS}/checkpoints"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" \
  /content/lqxl_v2/training_result.json "${DOWNLOADS}/training_result.json"
colab download -s "${SESSION}" /content/lqxl_v2/logs/resource_usage.csv "${ROOT}/logs/training_v2/resource_usage.csv"
colab download -s "${SESSION}" /content/lqxl_v2/logs/cache_latents.log "${ROOT}/logs/training_v2/cache_latents.log"
colab download -s "${SESSION}" /content/lqxl_v2/logs/cache_text.log "${ROOT}/logs/training_v2/cache_text.log"
colab download -s "${SESSION}" /content/lqxl_v2/logs/training.log "${ROOT}/logs/training_v2/training.log"
for step in 00000300 00000600 00000900; do
  colab download -s "${SESSION}" \
    "/content/lqxl_v2/output/lqxl_sdxl_v2-step${step}.safetensors" \
    "${DOWNLOADS}/checkpoints/lqxl_sdxl_v2-step${step}.safetensors"
done
colab download -s "${SESSION}" /content/lqxl_v2/output/lqxl_sdxl_v2.safetensors "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors"
cp "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "${DOWNLOADS}/lqxl_sdxl_v2.safetensors"
sha256sum "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" >"${ROOT}/models/loras/lqxl_sdxl_v2.safetensors.sha256"
SUCCESS=1
printf 'Downloaded trained LoRA to %s\n' "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors"
