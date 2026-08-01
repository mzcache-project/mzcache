#!/bin/bash
# mzcache — OpenCL build script for Android (arm64-v8a, on-device CLI binaries)
#
# Usage: ADB_SERIAL=<device-serial> ./mzcache_build.sh <install_dir> [new]
#   <install_dir>: directory the binaries are installed to (a plain name like
#                  mzcache_install_flexgen8bit is fine), then pushed to
#                  /data/local/tmp/mzcache/<install_dir>/ on the device
#   [new]:         remove the build directory and configure from scratch
#
# Requires $ANDROID_NDK to point at an NDK (r25+) installation.

set -e  # Exit on error

if [ -z "$1" ]; then
  echo "Usage: ADB_SERIAL=<serial> $0 <install_dir> [new]"
  exit 1
fi

if [ -z "$ADB_SERIAL" ]; then
  echo "Set ADB_SERIAL to the target device serial (see 'adb devices')."
  exit 1
fi

if [ -z "${ANDROID_NDK:-}" ] || [ ! -d "$ANDROID_NDK" ]; then
  echo "Set ANDROID_NDK to the NDK root (scripts/setup/get_ndk.sh prints the export line)."
  echo "It is needed in every shell that builds, not just the one that fetched the NDK."
  exit 1
fi

INSTALL_DIR=$1
BUILD_DIR=mzcache_build

# Step 0: Optionally clean old build directory
if [ "$2" == "new" ]; then
  if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
  fi
fi
# --- Self-contained OpenCL SDK (the NDK is never modified) -------------------
# ggml-opencl's find_package(OpenCL) would otherwise need CL headers and a
# libOpenCL.so injected into the NDK sysroot. Instead we point CMake at a local
# prefix: the headers are vendored in-repo (Khronos, Apache-2.0) and the
# library is the device's own vendor blob, pulled once over adb (it is not
# redistributable, so it can never live in the repo).
OPENCL_SDK=$PWD/opencl_sdk
CL_VENDORED=$PWD/examples/llama.android/llama/src/main/cpp/include/CL
if [ ! -f "$OPENCL_SDK/lib/libOpenCL.so" ]; then
  echo "[0/4] Preparing local OpenCL SDK (vendored headers + device libOpenCL.so)..."
  mkdir -p "$OPENCL_SDK/include/CL" "$OPENCL_SDK/lib"
  cp "$CL_VENDORED"/*.h "$OPENCL_SDK/include/CL/"
  adb -s "$ADB_SERIAL" pull /vendor/lib64/libOpenCL.so "$OPENCL_SDK/lib/libOpenCL.so"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[1/4] Configuring GPU build with OpenCL..."
cmake .. -G Ninja \
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
  -DGGML_OPENCL_PROFILING=OFF \
  -DMZCACHE_SVM_KV_CHUNK=ON \
  -DMZCACHE_COMPRESSION="${MZCACHE_COMPRESSION:-FLEXGEN_8BIT}" \

echo "[2/4] Building with Ninja..."
ninja
cd ..

echo "[3/4] Installing to: $INSTALL_DIR"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR" --config Release

echo "[4/4] Pushing to device: /data/local/tmp/mzcache/"
# adb push only nests the source directory under the destination when that
# destination already exists; without this mkdir the first build on a fresh phone
# lands as /data/local/tmp/mzcache/bin/... and every later script reports the
# install dir as missing.
adb -s "$ADB_SERIAL" shell mkdir -p /data/local/tmp/mzcache
adb -s "$ADB_SERIAL" push "$INSTALL_DIR" /data/local/tmp/mzcache/

echo "Done."
