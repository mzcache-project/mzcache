#!/system/bin/sh

BINARY="/data/local/tmp/mskim/android-build/bin/mz_baseline_swap"
MODEL_NAME="EXAONE4-1.2B-FP16"   # change if needed
LOG_PATH="/data/local/tmp/mskim/experiment"
LOG_FILE_NAME="20250816_baseline_partial_swap_exaone"

CONTEXT_SIZES="32700 16385 8193 4097"  # execution order
CPU_MASK="40"        # toybox taskset: hex mask "40" -> CPU#6
TOTAL_LAYERS=30      # matches exaone4.block_count=30

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
        65500)
            while IFS=' ' read -r layer_num kv_num; do
                [ -z "$layer_num" ] && continue
                run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
            done <<'PAIRS'
30 30
9 30
6 20
1 11
0 0
PAIRS
        ;;
        32700)
            while IFS=' ' read -r layer_num kv_num; do
                [ -z "$layer_num" ] && continue
                run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
            done <<'PAIRS'
30 30
16 30
6 25
4 11
0 0
PAIRS
        ;;
        16385)
            while IFS=' ' read -r layer_num kv_num; do
                [ -z "$layer_num" ] && continue
                run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
            done <<'PAIRS'
30 30
19 30
15 15
4 15
0 0
PAIRS
        ;;
        8193)
            while IFS=' ' read -r layer_num kv_num; do
                [ -z "$layer_num" ] && continue
                run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
            done <<'PAIRS'
30 30
23 30
15 15
4 23
0 0
PAIRS
        ;;
        4097)
            while IFS=' ' read -r layer_num kv_num; do
                [ -z "$layer_num" ] && continue
                run_one "$layer_num" "$kv_num" "$CONTEXT_SIZE"
            done <<'PAIRS'
30 30
22 28
15 15
5 29
0 0
PAIRS
        ;;
        *)
            echo "ERROR: unknown CONTEXT_SIZE: $CONTEXT_SIZE"
        ;;
    esac
done
