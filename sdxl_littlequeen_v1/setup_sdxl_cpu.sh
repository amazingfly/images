#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDCPP_DIR="${ROOT}/.tooling/stable-diffusion.cpp"
SDCPP_COMMIT="ea4e566ccffa10f853ecc3f29e74b1820bc91beb"
SOURCE_MODEL="${ROOT}/models/source/sd_xl_base_1.0.safetensors"
QUANTIZED_MODEL="${ROOT}/models/quantized/sd_xl_base_1.0-q8_0.gguf"
LCM_LORA="${ROOT}/models/loras/lcm-lora-sdxl.safetensors"
SDXL_SHA256="31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
LCM_SHA256="a764e6859b6e04047cd761c08ff0cee96413a8e004c9f07707530cd776b19141"

mkdir -p "${ROOT}/.tooling" "${ROOT}/models/source" "${ROOT}/models/quantized" "${ROOT}/models/loras" "${ROOT}/logs"

if [[ ! -d "${SDCPP_DIR}/.git" ]]; then
  git clone --filter=blob:none --recurse-submodules https://github.com/leejet/stable-diffusion.cpp.git "${SDCPP_DIR}"
fi

if [[ "$(git -C "${SDCPP_DIR}" rev-parse HEAD)" != "${SDCPP_COMMIT}" ]]; then
  git -C "${SDCPP_DIR}" fetch --depth 1 origin "${SDCPP_COMMIT}"
  git -C "${SDCPP_DIR}" checkout --detach "${SDCPP_COMMIT}"
  git -C "${SDCPP_DIR}" submodule update --init --recursive
fi

cmake -S "${SDCPP_DIR}" -B "${SDCPP_DIR}/build-cpu" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_BUILD_EXAMPLES=ON \
  -DSD_CUDA=OFF \
  -DSD_HIPBLAS=OFF \
  -DSD_VULKAN=OFF \
  -DSD_OPENCL=OFF \
  -DSD_SYCL=OFF \
  -DGGML_NATIVE=ON
cmake --build "${SDCPP_DIR}/build-cpu" --target sd-cli -j 6

curl -fL --retry 5 --retry-delay 5 --continue-at - \
  --output "${SOURCE_MODEL}" \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
echo "${SDXL_SHA256}  ${SOURCE_MODEL}" | sha256sum --check

curl -fL --retry 5 --retry-delay 5 --continue-at - \
  --output "${LCM_LORA}" \
  https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors
echo "${LCM_SHA256}  ${LCM_LORA}" | sha256sum --check

if [[ ! -s "${QUANTIZED_MODEL}" ]]; then
  python3 "${ROOT}/scripts/guarded_command.py" \
    --report "${ROOT}/logs/quantize_q8_memory.json" \
    --minimum-available-mb 1200 \
    --maximum-rss-mb 12500 \
    --maximum-swap-used-mb 4096 \
    -- \
    /usr/bin/time -v "${SDCPP_DIR}/build-cpu/bin/sd-cli" \
      --mode convert \
      --model "${SOURCE_MODEL}" \
      --output "${QUANTIZED_MODEL}" \
      --type q8_0 \
      --verbose \
    >"${ROOT}/logs/quantize_q8.log" 2>&1
fi

sha256sum "${QUANTIZED_MODEL}" >"${QUANTIZED_MODEL}.sha256"
echo "SDXL CPU model setup complete: ${QUANTIZED_MODEL}"
