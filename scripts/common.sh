#!/bin/bash
# common.sh — shared environment and helpers for the mzCache AE scripts.
#
# Source this from every runner:
#   source "$(dirname "$0")/../common.sh"
#
# Required environment:
#   ADB_SERIAL     target device serial (see `adb devices`)
#
# Optional overrides (defaults shown):
#   DEVICE_DIR     on-device working directory        /data/local/tmp/mzcache
#   GGUF_DIR       on-device model directory          /data/local/tmp/gguf
#   MODEL_TAG      state-file model prefix            qwen3_0.6B   (exaone4_1.2B for EXAONE)
#   GGUF           on-device model path               $GGUF_DIR/Qwen3-0.6B-FP16.gguf
#   MZ_INSTALL     mzCache install dir name           mzcache_install_flexgen8bit
#   PO_INSTALL     partial-offload install dir name   exp1_partial_offload
#   ANDSW_INSTALL  android-swap install dir name      exp1_android_swap
#
# All experiment binaries read state/layer files relative to their cwd, so every
# on-device command must run from $DEVICE_DIR.

set -u

: "${ADB_SERIAL:?set ADB_SERIAL to the target device serial (see 'adb devices')}"

DEVICE_DIR=${DEVICE_DIR:-/data/local/tmp/mzcache}
GGUF_DIR=${GGUF_DIR:-/data/local/tmp/gguf}
MODEL_TAG=${MODEL_TAG:-qwen3_0.6B}
GGUF=${GGUF:-$GGUF_DIR/Qwen3-0.6B-FP16.gguf}
MZ_INSTALL=${MZ_INSTALL:-mzcache_install_flexgen8bit}
PO_INSTALL=${PO_INSTALL:-exp1_partial_offload}
ANDSW_INSTALL=${ANDSW_INSTALL:-exp1_android_swap}

# Repository root and results directory (scripts live in <root>/scripts/...).
AE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS_ROOT=${RESULTS_ROOT:-$AE_ROOT/results}

adbsh() {  # plain shell on the device
    adb -s "$ADB_SERIAL" shell "$@"
}

# Root shell on the device (su required for governor/THP/zram/battery
# sysfs/mlock). The payload is wrapped in single quotes on the device side so
# that $vars inside it are expanded by su's inner shell, not the adb outer
# shell. Consequence: the payload must not itself contain single quotes —
# use double quotes inside.
adbsu() {
    adb -s "$ADB_SERIAL" shell "su -c '$*'"
}

require_device() {
    if ! adb -s "$ADB_SERIAL" get-state >/dev/null 2>&1; then
        echo "ERROR: device $ADB_SERIAL not reachable via adb" >&2
        exit 1
    fi
}

require_su() {
    if [ "$(adbsu id -u 2>/dev/null | tr -d '\r')" != "0" ]; then
        echo "ERROR: 'su' is not available on $ADB_SERIAL — this step needs a rooted device" >&2
        exit 1
    fi
}

require_on_device() {  # require_on_device <path> <hint>
    if ! adbsh "test -e $1" >/dev/null 2>&1; then
        echo "ERROR: missing on device: $1" >&2
        echo "  hint: $2" >&2
        exit 1
    fi
}

drop_caches() {
    adbsu 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

device_model() {
    adbsh getprop ro.product.model | tr -d '\r' | tr ' ' '_'
}

# require_profile_gate — mzCache (FLEXGEN/FLEXGEN_8BIT) binaries refuse to start
# without the one-time device profile in $DEVICE_DIR (see scripts/setup/run_device_profile.sh).
# The profile is written per compression backend as mzcache_device_profile_<BACKEND>.txt
# (see mzcache/src/mzcache_profile.cpp), so match the glob rather than a fixed name.
require_profile_gate() {
    if ! adbsh "ls $DEVICE_DIR/mzcache_device_profile_*.txt" >/dev/null 2>&1; then
        echo "ERROR: no device profile on $ADB_SERIAL: $DEVICE_DIR/mzcache_device_profile_*.txt" >&2
        echo "  hint: run scripts/setup/run_device_profile.sh once on this device" >&2
        exit 1
    fi
}

# require_weight_layers — mzCache's weight swap-out reloads layers from
# $DEVICE_DIR/layers/<model-name>_layer_<i>.bin; without them the engine
# refuses to unload weights (degrades to compression-only swap-out) and the
# experiment silently changes. Generate once per model with
# scripts/setup/gen_weight_layers.sh. The file names use the model's own name
# (from the GGUF), not $MODEL_TAG.
require_weight_layers() {
    local name
    case $MODEL_TAG in
        qwen3*)   name=Qwen3-0.6B ;;
        exaone4*) name=EXAONE-4.0-1.2B ;;
        *) echo "ERROR: no layer-file mapping for MODEL_TAG '$MODEL_TAG'" >&2; exit 1 ;;
    esac
    require_on_device "$DEVICE_DIR/layers/${name}_layer_0.bin" \
        "generate the per-layer weight files once: ADB_SERIAL=$ADB_SERIAL ./scripts/setup/gen_weight_layers.sh"
}
