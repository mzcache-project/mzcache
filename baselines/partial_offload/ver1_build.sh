# mzcache - OpenCL Build Script for Android GPU 

#!/bin/bash

# Usage: ./build_android_gpu.sh /absolute/path/to/install_dir [new]
# [new]: clean build directory and build from scratch

set -e  # Exit on error

if [ -z "$1" ]; then
  echo "❌ Usage: $0 <install_dir> [new]"
  echo "If you want to build from scratch, add "new" at the end"
  echo "ex) ./opencl_build.sh <install_dir> new"
  exit 1
fi

INSTALL_DIR=$1
DEVICE_CODE=$2   
BUILD_DIR=build-android-gpu

case $DEVICE_CODE in
  g) ADB_SERIAL="R3CY405R8YZ" ;;
  o) ADB_SERIAL="75a91e7e" ;;
  *)
    echo "❌  Invalid device code: $DEVICE_CODE (use 'g' or 'o')"
    exit 1
    ;;
esac

# Step 0: Optionally clean old build directory
if [ "$3" == "new" ]; then
  if [ -d "$BUILD_DIR" ]; then
    echo "🧹 Removing existing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
  fi
  mkdir "$BUILD_DIR"
fi
cd "$BUILD_DIR"

echo "🔧 [1/3] Configuring GPU build with OpenCL..."
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_HEXAGON=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_CURL=OFF \
  -DMZCACHE_SVM_KV_CHUNK=ON \
  -DGGML_OPENCL_PROFILING=ON \

echo "🧱 [2/3] Building with Ninja..."
ninja
cd ..

echo "📦 [3/3] Installing to: $INSTALL_DIR"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR" --config Release

echo "✅ GPU build and installation complete!"

echo "📤 [4/4] Pushing to device: /data/local/tmp/hongseung/"
adb -s "$ADB_SERIAL" push "$INSTALL_DIR" /data/local/tmp/hongseung/
# adb -s R3CY405R8YZ push "$INSTALL_DIR" /data/local/tmp/hongseung/

echo "Done."

# LD_LIBRARY_PATH =./gpu_install/lib
# ./gpu_install/bin/weight_load_test -m Qwen3-0.6B-FP16.gguf -ngl 100
# pa2q:/data/local/tmp/mzcache # ls
# Qwen3-0.6B-FP16.gguf  drop_cache.sh  gpu_install

