#!/system/bin/sh

BINARY="/data/local/tmp/mzcache/partial_offload/bin/mz_baseline_swap"
MODEL_NAME="Qwen3-FP16"   # change if needed
LOG_PATH="/data/local/tmp/mzcache/experiment"
LOG_FILE_NAME="20250816_baseline_partial_swap_qwen"

CONTEXT_SIZES="4097 8193 16385 32700"  # execution order is controlled here
CPU_MASK="40"        # toybox taskset: hex mask "40" -> CPU#6
TOTAL_LAYERS=28      # model_total_layer (fixed)

# check binary
if [ ! -f "$BINARY" ] || [ ! -x "$BINARY" ]; then
    echo "ERROR: binary not found or not executable: $BINARY"
    exit 1
fi

# detect taskset
TASKSET_BIN="$(command -v taskset 2>/dev/null || true)"
USE_TASKSET=1   # set to 0 to never use taskset

run_one () {
    layer_num="$1"
    kv_num="$2"
    ctx="$3"

    LOG_FILE_NAME_CTX="${LOG_FILE_NAME}_${ctx}"
    echo "====== Running layer_num: $layer_num, kv_num: $kv_num, model_total_layer: $TOTAL_LAYERS with context_size: $ctx (mmap: true) ======"

    if [ $USE_TASKSET -eq 1 ] && [ -n "$TASKSET_BIN" ]; then
        "$TASKSET_BIN" "$CPU_MASK" "$BINARY" \
            "$layer_num" "$kv_num" "$TOTAL_LAYERS" "$ctx" "$MODEL_NAME" "$LOG_PATH" "$LOG_FILE_NAME_CTX" </dev/null
        rc=$?
        if [ $rc -eq 127 ]; then
            echo "[warn] taskset exec failed (rc=127). Falling back to direct exec."
            "$BINARY" "$layer_num" "$kv_num" "$TOTAL_LAYERS" "$ctx" "$MODEL_NAME" "$LOG_PATH" "$LOG_FILE_NAME_CTX" </dev/null
            rc=$?
        fi
    else
        "$BINARY" "$layer_num" "$kv_num" "$TOTAL_LAYERS" "$ctx" "$MODEL_NAME" "$LOG_PATH" "$LOG_FILE_NAME_CTX" </dev/null
        rc=$?
    fi

    echo "------ Flushing page cache ------"
    sync
    su -c 'echo 3 > /proc/sys/vm/drop_caches' </dev/null

    if [ $rc -ne 0 ]; then
        echo "[warn] binary exited with code $rc (layer_num=$layer_num, kv_num=$kv_num, ctx=$ctx)"
    fi
}

for CONTEXT_SIZE in $CONTEXT_SIZES
do
    case "$CONTEXT_SIZE" in
        32700)
            while IFS=' ' read -r layer_num kv_num; do
                [ -z "$layer_num" ] && continue
                run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
            done <<'PAIRS'
28 28
20 21
14 14
2 8
0 0
PAIRS
        ;;
        16385)
            while IFS=' ' read -r layer_num kv_num; do
                [ -z "$layer_num" ] && continue
                run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
            done <<'PAIRS'
28 28
6 28
1 20
0 10
0 0
PAIRS
        ;;
#         8193)
#             while IFS=' ' read -r layer_num kv_num; do
#                 [ -z "$layer_num" ] && continue
#                 run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
#             done <<'PAIRS'
# 28 28
# 21 21
# 14 14
# 2 12
# 0 0
# PAIRS
#         ;;
#         4097)
#             while IFS=' ' read -r layer_num kv_num; do
#                 [ -z "$layer_num" ] && continue
#                 run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
#             done <<'PAIRS'
# 28 28
# 21 21
# 14 14
# 2 16
# 0 0
# PAIRS
#         ;;
        *)
            echo "ERROR: unknown CONTEXT_SIZE: $CONTEXT_SIZE"
        ;;
    esac
done