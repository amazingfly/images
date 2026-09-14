#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${HERE}/config.json"
PYTHON="${HOME}/miniforge3/bin/python"
OUTPUT_DIR="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "${CONFIG}")"
LOG="${OUTPUT_DIR}/launcher.log"
mkdir -p "${OUTPUT_DIR}"

SUPERVISOR_STATE="${OUTPUT_DIR}/supervisor_state.json"
if "${PYTHON}" -c '
import json, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
state_path = pathlib.Path(sys.argv[2])
if not state_path.is_file():
    raise SystemExit(1)
state = json.loads(state_path.read_text())
segment = state.get("last_successful_segment") or {}
complete = segment.get("status") == "complete" and int(segment.get("current_step", -1)) >= int(config["total_steps"])
raise SystemExit(0 if complete else 1)
' "${CONFIG}" "${SUPERVISOR_STATE}"; then
  echo "training already complete according to ${SUPERVISOR_STATE}; nothing to do"
  exit 0
fi

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

exec > >(while IFS= read -r line; do printf '%s %s\n' "$(timestamp)" "$line"; done | tee -a "${LOG}") 2>&1

echo "preparing curated dataset"
"${PYTHON}" "${HERE}/prepare_identity_v3.py"

TOKEN_PATH="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["token_path"])' "${CONFIG}")"
while true; do
  set +e
  "${PYTHON}" "${HERE}/deploy.py"
  deploy_status=$?
  set -e
  if (( deploy_status == 0 )); then
    break
  fi
  if (( deploy_status == 3 )); then
    echo "deployment is waiting for a write-capable token at ${TOKEN_PATH}; checking again in 60 seconds"
  else
    echo "deployment failed with status ${deploy_status}; retrying in 60 seconds"
  fi
  sleep 60
done

echo "starting multi-day ZeroGPU supervisor"
exec "${PYTHON}" "${HERE}/supervise.py"
