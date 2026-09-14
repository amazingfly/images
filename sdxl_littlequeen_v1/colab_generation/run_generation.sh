#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-smoke}"
SESSION="${COLAB_SESSION:-lqxl-dataset-fp16}"
GPU="${COLAB_GPU:-T4}"
REMOTE_SCRIPT="/content/remote_generate.py"
REMOTE_PROMPTS="/content/lqxl_generation_prompts.json"
REMOTE_ARCHIVE="/content/lqxl_generation_results.tar.gz"
LOCAL_ROOT="${ROOT}/outputs/identity_dataset100_v2/colab_fp16_832x1216_20260719"
DOWNLOAD="${LOCAL_ROOT}/lqxl_generation_results.tar.gz"

case "${MODE}" in
  smoke) LAUNCHER="${ROOT}/colab_generation/launch_smoke.py" ;;
  full) LAUNCHER="${ROOT}/colab_generation/launch_full.py" ;;
  *) printf 'Usage: %s [smoke|full]\n' "$0" >&2; exit 2 ;;
esac

"${ROOT}/scripts/build_identity_dataset100_prompts.py"

if timeout 30s colab sessions </dev/null 2>&1 | grep -Fq "${SESSION}"; then
  printf 'Reusing active Colab session %s\n' "${SESSION}"
else
  colab new -s "${SESSION}" --gpu "${GPU}"
fi

colab upload -s "${SESSION}" "${ROOT}/colab_generation/remote_generate.py" "${REMOTE_SCRIPT}"
colab upload -s "${SESSION}" "${ROOT}/config_identity_dataset100_prompts.json" "${REMOTE_PROMPTS}"
colab exec -s "${SESSION}" -f "${LAUNCHER}" --timeout 21600

mkdir -p "${LOCAL_ROOT}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" "${REMOTE_ARCHIVE}" "${DOWNLOAD}"
rm -rf "${LOCAL_ROOT}/output" "${LOCAL_ROOT}/logs"
tar -xzf "${DOWNLOAD}" -C "${LOCAL_ROOT}"

if [[ "${MODE}" == "full" ]]; then
  timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
fi
printf 'Downloaded %s Colab results to %s\n' "${MODE}" "${LOCAL_ROOT}"
