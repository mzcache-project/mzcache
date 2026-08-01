#!/bin/bash
# run_mzcache.sh — Figure 9, mzCache series: swap-in TTFT sweep.
#
# For each (context, remaining-fraction) the device runs mz_prefill_repeated:
# load state -> swapout(1-fp) -> swapin_generate (restore overlapped with an
# 8-token prefill). The reported "prefill time" of each cycle is the swap-in
# TTFT.
#
# Usage:
#   ADB_SERIAL=<serial> ./run_mzcache.sh [options]
#     -c "8193 16385 32700"       context sizes
#     -f "0 0.25 0.5 0.75 1.0"    remaining-in-memory fractions (1.0 = no-op)
#     -r 3                        repeats (cycles) per config
#     -o <dir>                    output dir (default results/ttft)
#
# Env: GGUF, MODEL_TAG, MZ_INSTALL (see scripts/common.sh).
# Requires: fa states on device, one-time device profile (scripts/setup/).
# Output: <out>/ttft_mzcache_<device>_<model_tag>.csv
#         schema: system,device,model,ctx,remaining_pct,rep,ttft_ms

source "$(dirname "$0")/../common.sh"

CTXS="8193 16385 32700"
FPS="0 0.25 0.5 0.75 1.0"
REPS=3
OUT=$RESULTS_ROOT/ttft

while getopts "c:f:r:o:" opt; do
    case $opt in
        c) CTXS=$OPTARG ;;
        f) FPS=$OPTARG ;;
        r) REPS=$OPTARG ;;
        o) OUT=$OPTARG ;;
        *) exit 1 ;;
    esac
done

require_device
require_profile_gate
require_weight_layers
require_on_device "$DEVICE_DIR/$MZ_INSTALL/bin/mz_prefill_repeated" \
    "build and push mzCache first: ADB_SERIAL=$ADB_SERIAL ./mzcache_build.sh $MZ_INSTALL"

DEV=$(device_model)
mkdir -p "$OUT/raw"
RUNTAG="mzcache_${DEV}_${MODEL_TAG}"
DEV_LOG_DIR=results_ae
DEV_LOG=mz_ttft_${RUNTAG}.log

# start fresh so the pulled log contains exactly this sweep
adbsh "mkdir -p $DEVICE_DIR/$DEV_LOG_DIR && rm -f $DEVICE_DIR/$DEV_LOG_DIR/$DEV_LOG"

for ctx in $CTXS; do
    require_on_device "$DEVICE_DIR/states/${MODEL_TAG}_fa_${ctx}.kv" \
        "push or generate the flash-attn state for ctx=$ctx (scripts/setup/push_files.sh)"
    for fp in $FPS; do
        echo "=== mzCache ctx=$ctx fp=$fp (${REPS} cycles) ==="
        drop_caches; sleep 2
        timeout 900 adb -s "$ADB_SERIAL" shell \
            "cd $DEVICE_DIR && ./$MZ_INSTALL/bin/mz_prefill_repeated $GGUF $ctx $fp $REPS 0 8 $DEV_LOG_DIR $DEV_LOG" \
            > "$OUT/raw/${RUNTAG}_${ctx}_fp${fp}.stdout" 2>&1 \
            || { echo "WARN: run failed (ctx=$ctx fp=$fp) — see $OUT/raw/${RUNTAG}_${ctx}_fp${fp}.stdout"; }
        sleep 5   # thermal cooldown between configs
    done
done

adb -s "$ADB_SERIAL" pull "$DEVICE_DIR/$DEV_LOG_DIR/$DEV_LOG" "$OUT/raw/" >/dev/null

CSV=$OUT/ttft_${RUNTAG}.csv
echo "system,device,model,ctx,remaining_pct,rep,ttft_ms" > "$CSV"
# device log rows: model ctx remaining_ratio repeats tokens_per_step cycle prefill_us
awk -v dev="$DEV" -v model="$MODEL_TAG" '
    !/^#/ && NF >= 7 {
        printf "mzcache,%s,%s,%d,%d,%d,%.1f\n", dev, model, $2, $3*100, $6, $7/1000.0
    }' "$OUT/raw/$DEV_LOG" >> "$CSV"

echo "Wrote $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
echo "Note: the first cycle of each config can include a cold clSVMAlloc penalty;"
echo "the plot script uses the per-config median across cycles."
