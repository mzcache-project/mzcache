#!/bin/bash
# run_compression.sh — Figure 14 (left two panels): KV-compression space savings and
# TTFT for CacheGen vs FLEXGEN-8bit, on Galaxy at 16k context.
#
# Two compression backends are separate compile-time builds (installed to
# distinct device dirs):
#   flexgen8bit -> $DEVICE_DIR/mzcache_install_flexgen8bit     (MZCACHE_COMPRESSION=FLEXGEN_8BIT)
#   cachegen    -> $DEVICE_DIR/mzcache_install_cachegen (MZCACHE_COMPRESSION=CACHEGEN)
# Build both with:
#   MZCACHE_COMPRESSION=FLEXGEN_8BIT ./mzcache_build.sh mzcache_install_flexgen8bit new
#   MZCACHE_COMPRESSION=CACHEGEN     ./mzcache_build.sh mzcache_install_cachegen
#
# Space savings: mz_prefill at full eviction (fp=0) prints
#   "[SpaceSavings] ... space_savings_pct=<x>" = saved_bytes / raw_bytes over the
#   compressed chunks. FLEXGEN-8bit is a fixed ~46.9%; CacheGen is data-dependent.
# TTFT: same mz_prefill "Prefill time: <x> ms" at remaining fractions {0,.25,.5}.
#
# Usage:
#   ADB_SERIAL=<serial> ./run_compression.sh [space|ttft|all]   (default all)
#     -x 16385                          context size
#     -M "qwen3_0.6B exaone4_1.2B"      models for space savings
#     -T qwen3_0.6B                     model for the TTFT panel
#     -f "0 0.25 0.5"                   TTFT remaining fractions
#     -r 3                              TTFT reps per config
#     -o <dir>                          output (default results/compression)
#
# Output:
#   space_savings.csv  model,algo,space_savings_pct
#   ttft.csv           algo,model,remaining_pct,rep,ttft_ms

source "$(dirname "$0")/../common.sh"

# The mode is an optional FIRST positional arg; flags follow it. Anything else is a
# typo — reject it instead of running neither panel and exiting 0.
MODE=all
case "${1:-}" in
    space|ttft|all) MODE=$1; shift ;;
    ""|-*)          ;;   # no mode given — keep "all" and let getopts read the flags
    *)              echo "ERROR: unknown mode '$1' (expected: space | ttft | all)" >&2; exit 1 ;;
esac
CTX=16385
SPACE_MODELS="qwen3_0.6B exaone4_1.2B"
TTFT_MODEL=qwen3_0.6B
FPS="0 0.25 0.5"
REPS=3
OUT=$RESULTS_ROOT/compression

while getopts "x:M:T:f:r:o:" opt; do
    case $opt in
        x) CTX=$OPTARG ;; M) SPACE_MODELS=$OPTARG ;; T) TTFT_MODEL=$OPTARG ;;
        f) FPS=$OPTARG ;; r) REPS=$OPTARG ;; o) OUT=$OPTARG ;; *) exit 1 ;;
    esac
done

require_device
mkdir -p "$OUT/raw"
echo "mode=$MODE ctx=$CTX space_models=\"$SPACE_MODELS\" ttft_model=$TTFT_MODEL out=$OUT"

# algo -> install dir. Both backends are profile-gated (per-compression profile
# file), so run scripts/setup/run_device_profile.sh once per install dir.
install_dir() { case $1 in cachegen) echo mzcache_install_cachegen ;; flexgen8bit) echo mzcache_install_flexgen8bit ;; esac; }
# algo -> compression name (for the per-compression profile filename)
comp_name() { case $1 in cachegen) echo CACHEGEN ;; flexgen8bit) echo FLEXGEN_8BIT ;; esac; }
# model tag -> on-device gguf path
gguf_path() {
    case $1 in
        qwen3_0.6B)   echo "$GGUF_DIR/Qwen3-0.6B-FP16.gguf" ;;
        exaone4_1.2B) echo "$GGUF_DIR/EXAONE-4.0-1.2B-FP16.gguf" ;;
    esac
}

DEV=$(device_model)

run_one() {  # run_one <algo> <model> <fp> <tag-for-logfile> ; echoes stdout log path
    local algo=$1 model=$2 fp=$3 tag=$4
    local inst; inst=$(install_dir "$algo")
    local gguf; gguf=$(gguf_path "$model")
    local log="$OUT/raw/${algo}_${model}_${CTX}_fp${fp}_${tag}.log"
    require_on_device "$DEVICE_DIR/$inst/bin/mz_prefill" \
        "build $algo: MZCACHE_COMPRESSION=$( [ "$algo" = cachegen ] && echo CACHEGEN || echo FLEXGEN_8BIT ) ./mzcache_build.sh $inst"
    require_on_device "$DEVICE_DIR/mzcache_device_profile_$(comp_name "$algo").txt" \
        "profile the $algo build once: cd $DEVICE_DIR && ./$inst/bin/mz_device_profile"
    MODEL_TAG=$model require_weight_layers
    drop_caches 2>/dev/null; sleep 1
    # feed a newline: mz_prefill blocks on cin.get() before prefill
    timeout 400 adb -s "$ADB_SERIAL" shell \
        "cd $DEVICE_DIR && echo | ./$inst/bin/mz_prefill $gguf $CTX $fp 0" > "$log" 2>&1
    echo "$log"
}

# ---- space savings (fp=0, one run per model x algo) ----
if [ "$MODE" = space -o "$MODE" = all ]; then
    CSV=$OUT/space_savings.csv
    echo "model,algo,space_savings_pct" > "$CSV"
    for model in $SPACE_MODELS; do
        for algo in cachegen flexgen8bit; do
            echo "=== space savings: $algo $model ctx=$CTX ==="
            log=$(run_one "$algo" "$model" 0 space)
            ss=$(grep -m1 '\[SpaceSavings\]' "$log" | sed -nE 's/.*space_savings_pct=([0-9.]+).*/\1/p')
            if [ -n "$ss" ]; then
                echo "$model,$algo,$ss" >> "$CSV"; echo "    space_savings=${ss}%"
            else
                echo "    WARN: no [SpaceSavings] line — see $log"
            fi
            sleep 3
        done
    done
    echo "Wrote $CSV"
fi

# ---- TTFT (remaining {0,0.25,0.5}) ----
if [ "$MODE" = ttft -o "$MODE" = all ]; then
    CSV=$OUT/ttft.csv
    echo "algo,model,remaining_pct,rep,ttft_ms" > "$CSV"
    for algo in cachegen flexgen8bit; do
        for fp in $FPS; do
            for rep in $(seq 1 "$REPS"); do
                echo "=== ttft: $algo $TTFT_MODEL fp=$fp rep=$rep ==="
                log=$(run_one "$algo" "$TTFT_MODEL" "$fp" "ttft_r${rep}")
                ms=$(grep -m1 'Prefill time:' "$log" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
                pct=$(awk -v r="$fp" 'BEGIN{printf "%d", r*100}')
                if [ -n "$ms" ]; then
                    echo "$algo,$TTFT_MODEL,$pct,$rep,$ms" >> "$CSV"; echo "    TTFT=${ms} ms"
                else
                    echo "    WARN: no 'Prefill time:' — see $log"
                fi
                sleep 4
            done
        done
    done
    echo "Wrote $CSV"
fi

echo "Plot: python3 scripts/plot/plot_compression.py"
