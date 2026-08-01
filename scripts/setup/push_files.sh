#!/bin/bash
# push_files.sh — push models and prefill-state files to the device, then verify.
#
# Usage:
#   ADB_SERIAL=<serial> ./push_files.sh [local_artifacts_dir]
#
# With no argument it pushes from the repo root, where get_models.sh leaves the
# *.gguf and get_states.sh / gen_prefill_states.sh leave states/*.kv (see
# EVALUATION.md Section 1.2). Pass a directory to push from elsewhere:
#   <dir>/*.gguf            -> pushed to $GGUF_DIR
#   <dir>/states/*.kv       -> pushed to $DEVICE_DIR/states/
# Either way it then prints a checklist of anything still missing on the device.
#
# State-file naming convention ($DEVICE_DIR/states/):
#   <model_tag>_fa_<ctx>.kv   flash-attn ON  states -> mzCache binaries
#   <model_tag>_<ctx>.kv      flash-attn OFF states -> partial_offload, android_swap
# with <model_tag> in {qwen3_0.6B, exaone4_1.2B} and <ctx> in {8193, 16385, 32700}.

source "$(dirname "$0")/../common.sh"

require_device

# Default to the repo root, where get_models.sh drops *.gguf and get_states.sh /
# gen_prefill_states.sh drop states/*.kv; pass a directory to push from elsewhere.
SRC="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
[ -d "$SRC" ] || { echo "ERROR: $SRC is not a directory" >&2; exit 1; }
adbsh "mkdir -p $GGUF_DIR $DEVICE_DIR/states"
for f in "$SRC"/*.gguf; do
    [ -e "$f" ] || break
    # get_models.sh keeps the downloaded BF16 sources next to the FP16 it produces;
    # only the FP16 models are ever used on the device (~3.7 GB saved).
    case "$(basename "$f")" in
        *BF16*|*bf16*) echo "skip $(basename "$f") (BF16 source, FP16 only on device)"; continue ;;
    esac
    echo "push $f -> $GGUF_DIR/"
    adb -s "$ADB_SERIAL" push "$f" "$GGUF_DIR/"
done
# Push the 12 expected prefill states by name, not every states/*.kv: Section 6.3
# writes its own multi-GB per-document states into a states/ directory too, and
# those must never be shipped to the phone.
for tag in qwen3_0.6B exaone4_1.2B; do
    for ctx in 8193 16385 32700; do
        for f in "$SRC/states/${tag}_fa_${ctx}.kv" "$SRC/states/${tag}_${ctx}.kv"; do
            [ -f "$f" ] || continue
            echo "push $f -> $DEVICE_DIR/states/"
            adb -s "$ADB_SERIAL" push "$f" "$DEVICE_DIR/states/"
        done
    done
done
# The Android app opens the states directory directly as an unprivileged uid.
adbsh "chmod 777 $DEVICE_DIR/states; chmod 666 $DEVICE_DIR/states/*.kv" || true

echo ""
echo "--- device checklist ---"
missing=0
check() {
    if adbsh "test -e $1" >/dev/null 2>&1; then
        echo "  [ok]      $1"
    else
        echo "  [MISSING] $1"
        missing=1
    fi
}

# Check BOTH models, not just $MODEL_TAG: the experiments use Qwen3 by default but
# Sections 3-6 also run EXAONE, and a bundle that lost those files must not pass.
check_model() {   # <model_tag> <fp16-gguf-basename>
    check "$GGUF_DIR/$2"
    for ctx in 8193 16385 32700; do
        check "$DEVICE_DIR/states/${1}_fa_${ctx}.kv"
        check "$DEVICE_DIR/states/${1}_${ctx}.kv"
    done
}
check_model qwen3_0.6B   Qwen3-0.6B-FP16.gguf
check_model exaone4_1.2B EXAONE-4.0-1.2B-FP16.gguf

if [ "$missing" = 1 ]; then
    echo ""
    echo "Some files are missing (EVALUATION.md Section 1.2):"
    echo "  models  -> ./scripts/setup/get_models.sh"
    echo "  states  -> ./scripts/setup/get_states.sh   (or gen_prefill_states.sh on a GPU server)"
    echo "then re-run this script."
    exit 1
fi
echo "All required files present."
