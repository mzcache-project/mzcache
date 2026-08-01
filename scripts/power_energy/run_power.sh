#!/bin/bash
# run_power.sh — Figure 13: power sampling during restoration + prefill.
#
# Samples the battery fuel gauge (/sys/class/power_supply/battery/
# {voltage_now,current_now}) every 100 ms around the swap-in and computes
# Power_W = V x |I|. Both runs MUST go through su — the battery sysfs is not
# readable from the adb shell domain.
#
# Paper setting: Galaxy S25+, Qwen3-0.6B, ctx 32700, remaining=0.5.
#
# Measurement note: the fuel gauge updates with ~0.3-1 s latency and the
# absolute wattage offsets with the charge state. Duration and shape are
# reliable; compare the two systems from the SAME session, back to back, and
# keep the device unplugged.
#
# Usage:
#   ADB_SERIAL=<serial> ./run_power.sh [options]
#     -x 32700     context size
#     -R 0.5       remaining-in-memory ratio
#     -o <dir>     output dir (default results/power_energy)
#
# Output:
#   <out>/power_mzcache.csv          (Timestamp_us,Voltage_uV,Current_uA,Power_W)
#   <out>/power_partial_offload.csv  (same schema)

source "$(dirname "$0")/../common.sh"

CTX=32700
REM=0.5
OUT=$RESULTS_ROOT/power_energy

while getopts "x:R:o:" opt; do
    case $opt in
        x) CTX=$OPTARG ;;
        R) REM=$OPTARG ;;
        o) OUT=$OPTARG ;;
        *) exit 1 ;;
    esac
done

require_device
require_su
require_profile_gate
require_weight_layers
require_on_device "$DEVICE_DIR/$MZ_INSTALL/bin/mz_prefill_power" \
    "build and push mzCache first: ADB_SERIAL=$ADB_SERIAL ./mzcache_build.sh $MZ_INSTALL"
require_on_device "$DEVICE_DIR/$PO_INSTALL/bin/exp1_partial_offload" \
    "build and push it first: ADB_SERIAL=$ADB_SERIAL ./exp1_partial_offload_build.sh $PO_INSTALL"

mkdir -p "$OUT/raw"

newest_power_log() {  # newest states/power_log_*.csv on the device
    adbsh "ls -t $DEVICE_DIR/states/power_log_*.csv 2>/dev/null | head -1" | tr -d '\r'
}

echo "=== [1/2] mzCache: mz_prefill_power $CTX $REM ==="
drop_caches; sleep 2
adbsu "cd $DEVICE_DIR && ./$MZ_INSTALL/bin/mz_prefill_power $GGUF $CTX $REM" \
    > "$OUT/raw/mzcache_run.log" 2>&1 || { echo "ERROR: mzCache power run failed — see $OUT/raw/mzcache_run.log" >&2; exit 1; }
f=$(newest_power_log)
[ -n "$f" ] || { echo "ERROR: no power_log_*.csv produced" >&2; exit 1; }
adb -s "$ADB_SERIAL" pull "$f" "$OUT/power_mzcache.csv" >/dev/null
echo "    pulled $(basename "$f") -> power_mzcache.csv"

sleep 10   # settle before the second trace

echo "=== [2/2] partial offload: MZ_POWER_LOG=1 exp1_partial_offload $CTX $REM ==="
drop_caches; sleep 2
adbsu "cd $DEVICE_DIR && MZ_POWER_LOG=1 taskset 80 ./$PO_INSTALL/bin/exp1_partial_offload $GGUF $CTX $REM" \
    > "$OUT/raw/po_run.log" 2>&1 || { echo "ERROR: PO power run failed — see $OUT/raw/po_run.log" >&2; exit 1; }
f=$(newest_power_log)
adb -s "$ADB_SERIAL" pull "$f" "$OUT/power_partial_offload.csv" >/dev/null
echo "    pulled $(basename "$f") -> power_partial_offload.csv"

echo ""
echo "Plot: python3 scripts/plot/plot_power_energy.py"
