#!/bin/bash
# gen_weight_layers.sh — one-time per model: generate the per-layer weight
# files mzCache's weight swap-out reloads from.
#
# mzCache unloads transformer layers from GPU memory at swap-out and reloads
# them from $DEVICE_DIR/layers/<model>_layer_<i>.bin. The engine refuses to
# unload a layer whose reload file is missing (it degrades to compression-only
# swap-out), so these files must exist BEFORE any weight-unloading experiment
# (Figures 9-14). The Android app (Figure 15) generates its own copy on first
# launch; this script provisions the CLI working directory.
#
# Runs examples/mz_dump_layers on the device (the dump must happen there — it
# needs the model loaded through the OpenCL build), once per model. Idempotent.
#
# Usage:
#   ADB_SERIAL=<serial> ./gen_weight_layers.sh [gguf ...]
#     default gguf set: $GGUF_DIR/Qwen3-0.6B-FP16.gguf and
#                       $GGUF_DIR/EXAONE-4.0-1.2B-FP16.gguf
#     install dir: $MZ_INSTALL (any mzCache build works)
#
# Storage: Qwen3-0.6B 28 x 30 MB (~840 MB), EXAONE-4.0-1.2B 30 x 68 MB (~2 GB).

source "$(dirname "$0")/../common.sh"

require_device

BIN=$DEVICE_DIR/$MZ_INSTALL/bin/mz_dump_layers
require_on_device "$BIN" \
    "build and push mzCache first: ADB_SERIAL=$ADB_SERIAL ./mzcache_build.sh $MZ_INSTALL"

if [ $# -ge 1 ]; then
    GGUFS=("$@")
else
    GGUFS=("$GGUF_DIR/Qwen3-0.6B-FP16.gguf" "$GGUF_DIR/EXAONE-4.0-1.2B-FP16.gguf")
fi

for gguf in "${GGUFS[@]}"; do
    require_on_device "$gguf" "push the model first (scripts/setup/push_files.sh)"
    echo "=== $gguf ==="
    adbsh "cd $DEVICE_DIR && ./$MZ_INSTALL/bin/mz_dump_layers $gguf" \
        | grep -E "^wrote |^done:|ERROR" || { echo "ERROR: dump failed for $gguf" >&2; exit 1; }
done

echo ""
echo "layer files on device:"
adbsh "ls -l $DEVICE_DIR/layers/ | head -3; ls $DEVICE_DIR/layers/ | wc -l"
