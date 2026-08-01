#!/bin/bash
# get_ndk.sh — download, sha256-verify, and unpack the pinned Android NDK.
#
# The AE builds are verified against this exact STOCK NDK. No modification of
# the NDK is needed: the OpenCL headers are vendored in-repo and libOpenCL.so
# is pulled from the device by the build scripts (mzcache_build.sh /
# exp1_partial_offload_build.sh), so find_package(OpenCL) never looks in the
# NDK sysroot.
#
# Usage: ./scripts/setup/get_ndk.sh [dest_dir]
#   default dest: the repository root (android-ndk-* is gitignored)
#   prints the `export ANDROID_NDK=...` line to use afterwards.
set -e

NDK_REL=r29
SHA256=4abbbcdc842f3d4879206e9695d52709603e52dd68d3c1fff04b3b5e7a308ecf
URL=https://dl.google.com/android/repository/android-ndk-${NDK_REL}-linux.zip

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DEST=${1:-$ROOT}
NDK_DIR=$DEST/android-ndk-$NDK_REL

if [ -d "$NDK_DIR" ]; then
    echo "already present: $NDK_DIR"
else
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    echo "downloading $URL (~780 MB)..."
    curl -fL -o "$TMP/ndk.zip" "$URL"
    echo "$SHA256  $TMP/ndk.zip" | sha256sum -c -
    echo "unpacking to $DEST ..."
    unzip -q "$TMP/ndk.zip" -d "$DEST"
fi

echo ""
echo "export ANDROID_NDK=$NDK_DIR"
