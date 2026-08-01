#!/bin/bash
# cpu_build.sh — CPU-only Android build for the exp1_android_swap baseline.
#
# Builds vanilla llama.cpp (MZCACHE_SVM_KV_CHUNK=OFF, GGML_OPENCL=OFF) with the
# token-embedding mlock enabled (MZ_MLOCK_TOKEN_EMBD=ON), matching the
# baseline-test "android swap" baseline. Installs and pushes to the device.
#
# Usage: ADB_SERIAL=<serial> ./cpu_build.sh <install_dir> [new]
#   <install_dir>: absolute/relative path binaries install to, then pushed to
#                  /data/local/tmp/mzcache/ on the device.
#   [new]:         remove the build directory and configure from scratch.
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
BUILD_DIR=cpu_build

if [ "$2" == "new" ]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[1/4] Configuring CPU-only build (no OpenCL) with token_embd mlock..."
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DGGML_OPENCL=OFF \
  -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_CURL=OFF \
  -DMZCACHE_SVM_KV_CHUNK=OFF \
  -DMZ_MLOCK_TOKEN_EMBD=ON \

echo "[2/4] Building with Ninja..."
ninja
cd ..

echo "[3/4] Installing to: $INSTALL_DIR"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR" --config Release

echo "[4/4] Pushing to device: /data/local/tmp/mzcache/"
adb -s "$ADB_SERIAL" push "$INSTALL_DIR" /data/local/tmp/mzcache/
echo "Done."
