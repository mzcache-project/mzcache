#!/bin/bash
# setup_memstress.sh — build mmap_touch with the Android NDK, push it, and
# create the on-device stress files used by the OS-paging pressure ramp.
#
# Usage:
#   ANDROID_NDK=<ndk> ADB_SERIAL=<serial> ./setup_memstress.sh [num_bins=60]
#
# num_bins * 128 MiB of storage is consumed on the device (60 -> 7.5 GiB).
# One 128 MiB random file is generated and copied num_bins times: page-cache
# pressure works per inode, so identical contents in distinct files behave the
# same as distinct random files, and copying is much faster than /dev/urandom.

source "$(dirname "$0")/../../common.sh"

NUM_BINS=${1:-60}
: "${ANDROID_NDK:?set ANDROID_NDK to an NDK (r25+) install}"

CLANG=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang
[ -x "$CLANG" ] || { echo "ERROR: $CLANG not found" >&2; exit 1; }

require_device

HERE=$(cd "$(dirname "$0")" && pwd)
echo "[1/3] building mmap_touch (file-cache pressure) + anon_hog (dirty anon pressure)"
"$CLANG" -O2 -Wall -o /tmp/mmap_touch "$HERE/mmap_touch.c"
"$CLANG" -O2 -Wall -o /tmp/anon_hog   "$HERE/anon_hog.c"

echo "[2/3] pushing to $DEVICE_DIR/memstress/"
adbsh "mkdir -p $DEVICE_DIR/memstress"
adb -s "$ADB_SERIAL" push /tmp/mmap_touch "$DEVICE_DIR/memstress/"
adb -s "$ADB_SERIAL" push /tmp/anon_hog   "$DEVICE_DIR/memstress/"
adbsh "chmod +x $DEVICE_DIR/memstress/mmap_touch $DEVICE_DIR/memstress/anon_hog"

echo "[3/3] creating $NUM_BINS x 128 MiB stress files (skips existing)"
adbsh "cd $DEVICE_DIR/memstress && \
  if [ ! -e stress_000.bin ]; then dd if=/dev/urandom of=stress_000.bin bs=1048576 count=128 2>/dev/null; fi && \
  i=1; while [ \$i -lt $NUM_BINS ]; do \
    f=\$(printf 'stress_%03d.bin' \$i); \
    [ -e \$f ] || cp stress_000.bin \$f; \
    i=\$((i+1)); \
  done && ls stress_*.bin | wc -l"

echo "Done. Pressure tool ready: $DEVICE_DIR/memstress/mmap_touch <count> [size_mb] [hold_sec]"
