#!/bin/bash
# Phone-side compress+decompress roundtrip for the F1-accuracy experiment
# (docs/reproduce_f1_accuracy.md, step ④).
#
# For each of the 30 TriviaQA documents: push the raw fp16 .kv to the phone,
# run mz_save_state_v2 (<gguf> <mzcache_id> 0 — remaining 0 = fully evict:
# compress + store, then restore + decompress), pull the reconstructed .kv back,
# and delete the on-device copies. The phone only ever holds ~one document.
#
# REPRODUCIBILITY: mzcache schedules swap-out from the measured device profile
# (mzcache_device_profile_<BACKEND>.txt); the profile decides how many chunks are
# lossy-compressed vs stored losslessly, so the resulting F1 depends on it.
# This script therefore PINS the authors' reference profile (from
# scripts/compression/accuracy/profiles/) on the device for the duration of the run, and
# restores the experimenter's own profile on exit — including on Ctrl-C.
#
# Usage:
#   ADB_SERIAL=<serial> ./scripts/compression/accuracy/roundtrip_phone.sh \
#       <qwen3|exaone4> <cachegen|flexgen8bit> <host_raw_dir> <host_out_dir> [install_dir]
#
# Resume-safe: documents whose output .kv already exists are skipped.
set -u

if [ -z "${ADB_SERIAL:-}" ] || [ $# -lt 4 ]; then
    echo "Usage: ADB_SERIAL=<serial> $0 <qwen3|exaone4> <cachegen|flexgen8bit> <host_raw_dir> <host_out_dir> [install_dir]" >&2
    exit 1
fi

MODEL=$1; BACKEND=$2; RAW=$3; OUT=$4
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEV=/data/local/tmp/mzcache
ADB="adb -s $ADB_SERIAL"

# GGUF_NAME is the bare filename (the weight-layer prefix below is derived from
# it); GGUF is the absolute on-device path. The models live in /data/local/tmp/gguf,
# which is NOT $DEV — the directory the binaries are run from — so passing a bare
# filename here makes every compression fail to open the model.
GGUF_DIR=${GGUF_DIR:-/data/local/tmp/gguf}
case "$MODEL" in
    qwen3)   ARCH=qwen3_0.6B_fa;   GGUF_NAME=Qwen3-0.6B-FP16.gguf ;;
    exaone4) ARCH=exaone4_1.2B_fa; GGUF_NAME=EXAONE-4.0-1.2B-FP16.gguf ;;
    *) echo "unknown model '$MODEL' (qwen3|exaone4)" >&2; exit 1 ;;
esac
GGUF=$GGUF_DIR/$GGUF_NAME
case "$BACKEND" in
    cachegen)    COMP=CACHEGEN;     DEVOUT=full_swap_cg;     INSTALL=${5:-mzcache_install_cachegen} ;;
    flexgen8bit) COMP=FLEXGEN_8BIT; DEVOUT=full_swap_fg8bit; INSTALL=${5:-mzcache_install_flexgen8bit} ;;
    *) echo "unknown backend '$BACKEND' (cachegen|flexgen8bit)" >&2; exit 1 ;;
esac

REF_PROF="$SCRIPT_DIR/profiles/mzcache_device_profile_${COMP}.txt"
DEV_PROF="$DEV/mzcache_device_profile_${COMP}.txt"
[ -f "$REF_PROF" ] || { echo "missing reference profile $REF_PROF" >&2; exit 1; }
mkdir -p "$OUT"

# The full swap-out (remaining=0) unloads weight layers and reloads them from
# layers/<model>_layer_<i>.bin — those files must exist first.
LAYER0="$DEV/layers/${GGUF_NAME%-FP16.gguf}_layer_0.bin"
if ! $ADB shell "test -e $LAYER0" >/dev/null 2>&1; then
    echo "ERROR: missing $LAYER0 on the device." >&2
    echo "  generate the per-layer weight files once: ADB_SERIAL=$ADB_SERIAL ./scripts/setup/gen_weight_layers.sh" >&2
    exit 1
fi

# Fail here, with the path, rather than 30 identical "COMPRESS FAILED" lines.
if ! $ADB shell "test -e $GGUF" >/dev/null 2>&1; then
    echo "ERROR: missing $GGUF on the device." >&2
    echo "  push the models once: ADB_SERIAL=$ADB_SERIAL ./scripts/setup/push_files.sh" >&2
    exit 1
fi

# ---- pin the reference profile (backup the experimenter's, if any) ----------
# The backup is only created if absent, so a crashed earlier run (which left the
# reference profile pinned) can never clobber the real experimenter profile.
$ADB shell "[ -f $DEV_PROF ] && [ ! -f $DEV_PROF.experimenter.bak ] && cp $DEV_PROF $DEV_PROF.experimenter.bak" 2>/dev/null
$ADB push "$REF_PROF" "$DEV_PROF" >/dev/null
echo "[profile] pinned reference profile -> $DEV_PROF"

restore_profile() {
    if $ADB shell "[ -f $DEV_PROF.experimenter.bak ]" 2>/dev/null; then
        $ADB shell "mv $DEV_PROF.experimenter.bak $DEV_PROF"
        echo "[profile] restored experimenter profile"
    else
        $ADB shell "rm -f $DEV_PROF"
        echo "[profile] removed pinned profile (no experimenter profile existed)"
    fi
}
trap restore_profile EXIT

# ---- per-document roundtrip -------------------------------------------------
$ADB shell "mkdir -p $DEV/accuracy_test/raw $DEV/accuracy_test/$DEVOUT"
$ADB shell "su -c 'rm -f $DEV/accuracy_test/raw/*.kv $DEV/accuracy_test/$DEVOUT/*.kv'"
ok=0; skip=0; fail=0
for i in $(seq 1 30); do
    f="${ARCH}_${i}.kv"
    if [ ! -f "$RAW/$f" ]; then echo "doc $i: MISSING $RAW/$f"; fail=$((fail+1)); continue; fi
    if [ -f "$OUT/$f" ] && [ "$(stat -c %s "$OUT/$f")" -gt 1000 ]; then
        echo "doc $i: already done, skip"; skip=$((skip+1)); continue
    fi
    $ADB push "$RAW/$f" "$DEV/accuracy_test/raw/" >/dev/null
    # Free the page cache first: the push fills it and starves the compressor
    # (a 3 GB state can OOM when only ~3.5 GB remains available).
    $ADB shell "su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
    # Keep the on-device output: without it a failure is indistinguishable from
    # any other, and the reason (missing model, OOM, no profile) is invisible.
    mkdir -p "$OUT/logs"
    $ADB shell "su -c 'cd $DEV && ./$INSTALL/bin/mz_save_state_v2 $GGUF $i 0'" \
        > "$OUT/logs/doc_${i}.log" 2>&1
    sz=$($ADB shell "stat -c %s $DEV/accuracy_test/$DEVOUT/$f 2>/dev/null" | tr -d '\r')
    if [ -z "$sz" ] || [ "$sz" -lt 1000 ]; then
        echo "doc $i: COMPRESS FAILED (no/empty output) — last lines of $OUT/logs/doc_${i}.log:"
        tail -5 "$OUT/logs/doc_${i}.log" | sed 's/^/    | /'
        fail=$((fail+1))
        $ADB shell "su -c 'rm -f $DEV/accuracy_test/raw/$f $DEV/accuracy_test/$DEVOUT/$f'"
        continue
    fi
    $ADB pull "$DEV/accuracy_test/$DEVOUT/$f" "$OUT/" >/dev/null
    $ADB shell "su -c 'rm -f $DEV/accuracy_test/raw/$f $DEV/accuracy_test/$DEVOUT/$f'"
    echo "doc $i done: $sz bytes -> $OUT/$f"
    ok=$((ok+1))
done
echo "ROUNDTRIP DONE ($MODEL/$BACKEND): ok=$ok skip=$skip fail=$fail"
