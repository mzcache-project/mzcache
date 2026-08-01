#!/bin/bash
# device_setup.sh — one-shot device preparation for all experiments.
#
# Pins every CPU's cpufreq governor to `performance`, disables transparent huge
# pages, and reconfigures zram0 to zstd/8 GiB (the paper's zRAM setting, used by
# the OS-paging experiments). All reset on reboot — re-run after every reboot.
#
# Usage:
#   ADB_SERIAL=<serial> ./device_setup.sh          # apply all settings
#   ADB_SERIAL=<serial> ./device_setup.sh --show   # print current state only

source "$(dirname "$0")/../common.sh"

require_device
require_su

show_state() {
    echo "--- cpufreq governors ---"
    adbsu 'for c in /sys/devices/system/cpu/cpu[0-9]*; do echo -n "$c: "; cat $c/cpufreq/scaling_governor; done'
    echo "--- transparent hugepages ---"
    adbsu 'cat /sys/kernel/mm/transparent_hugepage/enabled'
    echo "--- zram ---"
    adbsu 'cat /sys/block/zram0/comp_algorithm 2>/dev/null; cat /proc/swaps'
    echo "--- memory ---"
    adbsu 'grep -E "MemAvailable|SwapTotal|SwapFree" /proc/meminfo'
}

if [ "${1:-}" = "--show" ]; then
    show_state
    exit 0
fi

echo "[1/2] governor -> performance, THP -> never"
adbsu 'for c in /sys/devices/system/cpu/cpu[0-9]*; do echo performance > $c/cpufreq/scaling_governor; done;
       echo never > /sys/kernel/mm/transparent_hugepage/enabled;
       echo never > /sys/kernel/mm/transparent_hugepage/defrag'

echo "[2/2] zram0 -> zstd, 8 GiB"
adbsu 'swapoff /dev/block/zram0; echo 1 > /sys/block/zram0/reset;
       echo zstd > /sys/block/zram0/comp_algorithm;
       echo 8589934592 > /sys/block/zram0/disksize;
       mkswap /dev/block/zram0; swapon /dev/block/zram0'

show_state
echo "Done. Remember: these settings reset on reboot."
