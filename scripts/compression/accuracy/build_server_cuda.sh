#!/usr/bin/env bash
# build_server_cuda.sh — GPU/server-side build of the F1-accuracy tools.
#
# Builds mz_accuracy, mz_load_state_accuracy and mz_gen_state against a CUDA
# llama.cpp. These are plain llama.cpp state tools (prefill + save/load .kv +
# score F1); they do NOT use the mzcache OpenCL core, so MZCACHE_SVM_KV_CHUNK
# stays OFF and only MZCACHE_BUILD_SERVER is turned on.
#
# Requirements: CUDA toolkit + an NVIDIA GPU, cmake, a C++17 compiler.
# Usage: ./scripts/compression/accuracy/build_server_cuda.sh [build_dir]
#        (default build_dir: cuda_build)
set -euo pipefail

BUILD_DIR="${1:-cuda_build}"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DMZCACHE_BUILD_SERVER=ON \
    -DMZCACHE_SVM_KV_CHUNK=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF

cmake --build "$BUILD_DIR" \
    --target mz_accuracy mz_load_state_accuracy mz_gen_state -j"$(nproc)"

echo
echo "Built:"
echo "  $BUILD_DIR/bin/mz_accuracy"
echo "  $BUILD_DIR/bin/mz_load_state_accuracy"
echo "  $BUILD_DIR/bin/mz_gen_state"
