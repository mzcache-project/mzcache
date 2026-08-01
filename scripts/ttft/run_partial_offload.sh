#!/bin/bash
# run_partial_offload.sh — Figure 9, Partial Offload series: restore TTFT sweep.
#
# Runs the vendored partial-offload baseline (baselines/partial_offload, GPU)
# at each (context, resident-ratio). The binary converts (ctx, ratio) to a
# (weight_layers, kv_layers) combination internally and reports "TTFT: ... ms"
# including KV + weight reload from storage.
#
# Qwen3-0.6B (28 layers) uses the author's hand-tuned table; EXAONE-4.0-1.2B
# (30 layers) uses a computed equal-fraction reconstruction (see lookup_layers in
# exp1_partial_offload.cpp). Both are selected by the model in the GGUF path.
#
# Usage:
#   ADB_SERIAL=<serial> ./run_partial_offload.sh [options]
#     -c "8193 16385 32700"       context sizes
#     -f "0 0.25 0.5 0.75 1.0"    resident-footprint ratios
#     -r 3                        repeats per config
#     -o <dir>                    output dir (default results/ttft)
#     -d each|once|never          drop_caches policy (default each; the paper
#                                 scenario restores from cold storage)
#
# Output: <out>/ttft_partial_offload_<device>_<model_tag>.csv
#         schema: system,device,model,ctx,remaining_pct,rep,ttft_ms

source "$(dirname "$0")/../common.sh"

CTXS="8193 16385 32700"
RATIOS="0 0.25 0.5 0.75 1.0"
REPS=3
OUT=$RESULTS_ROOT/ttft
DROP=each

while getopts "c:f:r:o:d:" opt; do
    case $opt in
        c) CTXS=$OPTARG ;;
        f) RATIOS=$OPTARG ;;
        r) REPS=$OPTARG ;;
        o) OUT=$OPTARG ;;
        d) DROP=$OPTARG ;;
        *) exit 1 ;;
    esac
done

require_device
require_su   # drop_caches
require_on_device "$DEVICE_DIR/$PO_INSTALL/bin/exp1_partial_offload" \
    "build and push it first: ADB_SERIAL=$ADB_SERIAL ./exp1_partial_offload_build.sh $PO_INSTALL"

DEV=$(device_model)
mkdir -p "$OUT/raw"
RUNTAG="partial_offload_${DEV}_${MODEL_TAG}"
CSV=$OUT/ttft_${RUNTAG}.csv
echo "system,device,model,ctx,remaining_pct,rep,ttft_ms" > "$CSV"

[ "$DROP" = once ] && { drop_caches; sleep 2; }

for ctx in $CTXS; do
    require_on_device "$DEVICE_DIR/states/${MODEL_TAG}_${ctx}.kv" \
        "push or generate the non-fa state for ctx=$ctx (scripts/setup/push_files.sh)"
    for ratio in $RATIOS; do
        for rep in $(seq 1 "$REPS"); do
            [ "$DROP" = each ] && { drop_caches; sleep 2; }
            log="$OUT/raw/${RUNTAG}_${ctx}_r${ratio}_rep${rep}.log"
            echo "=== PO ctx=$ctx ratio=$ratio rep=$rep ==="
            timeout 900 adb -s "$ADB_SERIAL" shell \
                "cd $DEVICE_DIR && taskset 80 ./$PO_INSTALL/bin/exp1_partial_offload $GGUF $ctx $ratio" \
                > "$log" 2>&1 || { echo "WARN: run failed — see $log"; continue; }
            # line format: "TTFT: <us> us (<ms> ms)" — take the ms value in parens
            ttft=$(grep -m1 'TTFT:' "$log" | sed -nE 's/.*\(([0-9.]+) *ms\).*/\1/p')
            if [ -n "$ttft" ]; then
                echo "partial_offload,$DEV,$MODEL_TAG,$ctx,$(awk -v r="$ratio" 'BEGIN{printf "%d", r*100}'),$rep,$ttft" >> "$CSV"
                echo "    TTFT ${ttft} ms"
            else
                echo "WARN: no 'TTFT:' line in $log"
            fi
            sleep 8   # thermal cooldown
        done
    done
done

echo "Wrote $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
