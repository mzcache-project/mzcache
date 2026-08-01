# mzCache - OpenCL Build Script for Android GPU 

#!/bin/bash

# Usage: ./minsung_build.sh <local_install_dir> <device_code> [new]
# Example: ./minsung_build.sh ./android-build g

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "❌ Usage: $0 <local_install_dir> <device_code> [new]"
  echo "   device_code: 'g' or 'o'"
  echo "   [new]: clean build (optional)"
  echo ""
  echo "Example: $0 ./android-build g"
  echo "Example: $0 ./android-build g new"
  exit 1
fi

INSTALL_DIR=$1
DEVICE_CODE=$2   
BUILD_DIR=build-android-gpu

case $DEVICE_CODE in
  g) ADB_SERIAL="R3CY405R8YZ" ;;
  # g) ADB_SERIAL="192.168.1.97:5555" ;;
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
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "🔧 [1/4] Configuring GPU build with OpenCL..."
cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_CURL=OFF \
  -DMZCACHE_SVM_KV_CHUNK=OFF

echo "🧱 [2/4] Building with Ninja..."
ninja
cd ..

echo "📦 [3/4] Installing locally to: $INSTALL_DIR"
# Create absolute path
INSTALL_DIR_ABS=$(readlink -f "$INSTALL_DIR")
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR_ABS" --config Release

echo "📤 [4/4] Pushing to device: /data/local/tmp/mzcache/"
adb -s "$ADB_SERIAL" push "$INSTALL_DIR_ABS" /data/local/tmp/mzcache/

echo "✅ Complete! Binary location on device:"
echo "   /data/local/tmp/mzcache/$(basename $INSTALL_DIR_ABS)/bin/"