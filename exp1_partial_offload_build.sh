#!/bin/bash
# exp1_partial_offload_build.sh — partial-offload swap baseline (OpenCL/GPU).
#
# Builds the self-contained vendored partial_offload tree under
# baselines/partial_offload/ (a snapshot of the partial_offload branch @ 90248ae)
# with its own core + mzcache lib. Uses the branch's own flags: OpenCL backend,
# MZCACHE_SVM_KV_CHUNK=OFF (whole-layer KV). The example runs with -ngl 100.
# Out-of-source build (build dir at repo root, gitignored); installs + pushes.
#
# Usage: ADB_SERIAL=<serial> ./exp1_partial_offload_build.sh <install_dir> [new]
# Requires $ANDROID_NDK (r25+).

set -e

if [ -z "$1" ]; then
  echo "Usage: ADB_SERIAL=<serial> $0 <install_dir> [new]"
  exit 1
fi
if [ -z "$ADB_SERIAL" ]; then
  echo "Set ADB_SERIAL to the target device serial (see 'adb devices')."
  exit 1
fi

INSTALL_DIR=$1
SRC_DIR=baselines/partial_offload
BUILD_DIR=exp1_partial_offload_build   # ends in *build/ -> gitignored

if [ "$2" == "new" ]; then
  rm -rf "$BUILD_DIR"
fi

# Self-contained OpenCL SDK — same scheme as mzcache_build.sh: vendored CL
# headers + the device's own libOpenCL.so, so the NDK is never modified.
OPENCL_SDK=$PWD/opencl_sdk
CL_VENDORED=$PWD/examples/llama.android/llama/src/main/cpp/include/CL
if [ ! -f "$OPENCL_SDK/lib/libOpenCL.so" ]; then
  echo "[0/4] Preparing local OpenCL SDK (vendored headers + device libOpenCL.so)..."
  mkdir -p "$OPENCL_SDK/include/CL" "$OPENCL_SDK/lib"
  cp "$CL_VENDORED"/*.h "$OPENCL_SDK/include/CL/"
  adb -s "$ADB_SERIAL" pull /vendor/lib64/libOpenCL.so "$OPENCL_SDK/lib/libOpenCL.so"
fi

echo "[1/4] Configuring vendored partial_offload baseline (OpenCL, whole-layer KV)..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DGGML_OPENCL=ON \
  -DOpenCL_INCLUDE_DIR="$OPENCL_SDK/include" \
  -DOpenCL_LIBRARY="$OPENCL_SDK/lib/libOpenCL.so" \
  -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_CURL=OFF \
  -DMZCACHE_SVM_KV_CHUNK=OFF \

echo "[2/4] Building with Ninja..."
cmake --build "$BUILD_DIR"

echo "[3/4] Installing to: $INSTALL_DIR"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR" --config Release

echo "[4/4] Pushing to device: /data/local/tmp/mzcache/"
adb -s "$ADB_SERIAL" push "$INSTALL_DIR" /data/local/tmp/mzcache/
echo "Done."
