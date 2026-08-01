#!/bin/bash
# build_quantize.sh — build only llama-quantize, CPU-only (no GPU backend).
#
# Quantization does not need a GPU, so this lets a host without CUDA (e.g. the
# phone/adb server) build the quantizer used by scripts/setup/get_models.sh.
# The GPU server already gets llama-quantize from build_server_cuda.sh.
#
# Usage (from the repository root):
#   ./scripts/setup/build_quantize.sh
#   -> build_quantize/bin/llama-quantize

set -eu

cd "$(dirname "$0")/../.."

# LLAMA_CURL defaults to ON and needs libcurl headers, which the AE Docker image
# does not carry (and a clean Ubuntu only has the curl CLI). Nothing here
# downloads models over HTTP, so turn it off like every other build script.
cmake -S . -B build_quantize \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DLLAMA_CURL=OFF \
    -DMZCACHE_BUILD_SERVER=ON \
    -DMZCACHE_SVM_KV_CHUNK=OFF

cmake --build build_quantize -j"$(nproc)" --target llama-quantize

echo ""
echo "Built:"
ls -lh build_quantize/bin/llama-quantize
