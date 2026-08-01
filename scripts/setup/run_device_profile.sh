#!/bin/bash
# run_device_profile.sh — one-time device profile (mandatory gate for mzCache runs).
#
# mzCache derives its two scheduling constants (per_layer_balance,
# decomp_load_ratio) from a one-time device profile instead of hardcoding them.
# All mz_* binaries refuse to start if `mzcache_device_profile_<COMPRESSION>.txt`
# is absent from the working directory, so run this once per device (and once per
# compression flag — a profile taken under a different MZCACHE_COMPRESSION
# build is rejected).
#
# Usage:
#   ADB_SERIAL=<serial> ./run_device_profile.sh [file_mb=512] [passes=3]
#
# Takes ~3 minutes and leaves a 512 MB mz_profile_test.bin on the device.
# Delete the on-device profile to force re-profiling:
#   adb shell 'rm /data/local/tmp/mzcache/mzcache_device_profile_*.txt'

source "$(dirname "$0")/../common.sh"

FILE_MB=${1:-512}
PASSES=${2:-3}

require_device
require_on_device "$DEVICE_DIR/$MZ_INSTALL/bin/mz_device_profile" \
    "build and push mzCache first: ADB_SERIAL=$ADB_SERIAL ./mzcache_build.sh $MZ_INSTALL"

echo "Profiling (file_mb=$FILE_MB passes=$PASSES, ~3 min)..."
adbsh "cd $DEVICE_DIR && ./$MZ_INSTALL/bin/mz_device_profile $FILE_MB $PASSES"

# The binary names the profile after the compression backend it was built with, so
# print the one just written (newest) — globbing would dump every backend's profile
# under a single header once a second backend has been profiled.
PROF=$(adbsh "ls -t $DEVICE_DIR/mzcache_device_profile_*.txt 2>/dev/null | head -1" | tr -d '\r')
echo ""
echo "--- $PROF ---"
adbsh "cat '$PROF'"
