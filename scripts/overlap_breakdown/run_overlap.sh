#!/bin/bash
# run_overlap.sh — Figures 10 and 11(a): swap-in breakdown + overlap ablation.
#
# Two experiments on one device (paper setting: Galaxy S25+, Qwen3-0.6B,
# remaining=0.5, ctx {8193, 16385, 32700}):
#
#  (1) Breakdown (Fig 10) — mz_prefill with prefill_type=3 (mz_NO) prints
#      Alloc/Restore/Prefill separately; compared against the partial-offload
#      baseline's "Total Alloc:" / "Total Read:" / "  Prefill:" lines at the
#      same remaining ratio.
#  (2) Overlap ladder (Fig 11a) — prefill_type {3,2,1,0} = mz_NO / mz_AR
#      (alloc-reload overlapped) / mz_RP (reload-prefill overlapped) / mz_ARP
#      (fully overlapped, = mzCache). Total swap-in = alloc + restore + prefill
#      as printed, for every type.
#
# Protocol (do not skip): per config, 1 discarded warm-up + N measured reps
# (median in the plot). The first run after a push/boot pays a cold clSVMAlloc
# penalty of several hundred ms. Thermal drift inflates late runs by 10-20% —
# if numbers look high, cool down for 5 minutes and re-measure.
#
# Usage:
#   ADB_SERIAL=<serial> ./run_overlap.sh [options]
#     -c "8193 16385 32700"   context sizes
#     -R 0.5                  remaining-in-memory ratio
#     -r 3                    measured reps (excludes the warm-up)
#     -o <dir>                output dir (default results/overlap_breakdown)
#
# Output:
#   <out>/overlap_mzcache.csv   ctx,type,rep,alloc_ms,restore_ms,prefill_ms,total_ms
#   <out>/breakdown_po.csv      ctx,rep,alloc_ms,reload_ms,prefill_ms
#   <out>/raw/                  full per-run logs

source "$(dirname "$0")/../common.sh"

CTXS="8193 16385 32700"
REM=0.5
REPS=3
OUT=$RESULTS_ROOT/overlap_breakdown

while getopts "c:R:r:o:" opt; do
    case $opt in
        c) CTXS=$OPTARG ;;
        R) REM=$OPTARG ;;
        r) REPS=$OPTARG ;;
        o) OUT=$OPTARG ;;
        *) exit 1 ;;
    esac
done

require_device
require_su
require_profile_gate
require_weight_layers
require_on_device "$DEVICE_DIR/$MZ_INSTALL/bin/mz_prefill" \
    "build and push mzCache first: ADB_SERIAL=$ADB_SERIAL ./mzcache_build.sh $MZ_INSTALL"
require_on_device "$DEVICE_DIR/$PO_INSTALL/bin/exp1_partial_offload" \
    "build and push it first: ADB_SERIAL=$ADB_SERIAL ./exp1_partial_offload_build.sh $PO_INSTALL"

mkdir -p "$OUT/raw"
MZ_CSV=$OUT/overlap_mzcache.csv
PO_CSV=$OUT/breakdown_po.csv
echo "ctx,type,rep,alloc_ms,restore_ms,prefill_ms,total_ms" > "$MZ_CSV"
echo "ctx,rep,alloc_ms,reload_ms,prefill_ms" > "$PO_CSV"

drop_caches; sleep 3

# --- (1)+(2) mzCache: prefill_type selects the overlap combination ------------
for ctx in $CTXS; do
    for type in 3 2 1 0; do
        for rep in $(seq 0 "$REPS"); do            # rep 0 = discarded warm-up
            log="$OUT/raw/mz_${ctx}_t${type}_r${rep}.log"
            timeout 900 adb -s "$ADB_SERIAL" shell \
                "cd $DEVICE_DIR && echo | ./$MZ_INSTALL/bin/mz_prefill $GGUF $ctx $REM $type" \
                > "$log" 2>&1 || { echo "WARN: run failed — see $log"; continue; }
            a=$(grep -m1 'Alloc time:'   "$log" | grep -oE '[0-9]+' | head -1)
            r=$(grep -m1 'Restore time:' "$log" | grep -oE '[0-9]+' | head -1)
            p=$(grep -m1 'Prefill time:' "$log" | grep -oE '[0-9]+' | head -1)
            echo "MZ ctx=$ctx type=$type rep=$rep alloc=${a:-NA}ms restore=${r:-NA}ms prefill=${p:-NA}ms"
            if [ "$rep" -gt 0 ] && [ -n "$a" ] && [ -n "$r" ] && [ -n "$p" ]; then
                echo "$ctx,$type,$rep,$a,$r,$p,$((a + r + p))" >> "$MZ_CSV"
            fi
            sleep 5
        done
    done
done

# --- (1) partial offload: breakdown is always printed ------------------------
for ctx in $CTXS; do
    for rep in $(seq 0 "$REPS"); do                # rep 0 = discarded warm-up
        log="$OUT/raw/po_${ctx}_r${rep}.log"
        timeout 900 adb -s "$ADB_SERIAL" shell \
            "cd $DEVICE_DIR && taskset 80 ./$PO_INSTALL/bin/exp1_partial_offload $GGUF $ctx $REM" \
            > "$log" 2>&1 || { echo "WARN: run failed — see $log"; continue; }
        a=$(grep -m1 'Total Alloc:' "$log" | grep -oE '[0-9]+' | head -1)
        r=$(grep -m1 'Total Read:'  "$log" | grep -oE '[0-9]+' | head -1)
        p=$(grep -m1 '  Prefill: '  "$log" | grep -oE '[0-9]+' | head -1)
        echo "PO ctx=$ctx rep=$rep alloc=${a:-NA}us read=${r:-NA}us prefill=${p:-NA}us"
        if [ "$rep" -gt 0 ] && [ -n "$a" ] && [ -n "$r" ] && [ -n "$p" ]; then
            awk -v c="$ctx" -v rp="$rep" -v a="$a" -v r="$r" -v p="$p" \
                'BEGIN{printf "%s,%s,%.1f,%.1f,%.1f\n", c, rp, a/1000.0, r/1000.0, p/1000.0}' >> "$PO_CSV"
        fi
        sleep 8
    done
done

echo ""
echo "Wrote $MZ_CSV and $PO_CSV"
echo "Plot: python3 scripts/plot/plot_breakdown.py && python3 scripts/plot/plot_overlap.py"
