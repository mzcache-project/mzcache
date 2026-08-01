#!/system/bin/sh
#
# toggle_swap.sh — enable or disable zram swap on Android
#
# Usage:
#   toggle_swap.sh off   # turn swap off and reset zram0
#   toggle_swap.sh on    # turn swap on with ~8 GiB zram0

ZRAM_DEV=/dev/block/zram0
ZRAM_SYS=/sys/block/zram0
# Default zram0 size: 8 GiB
DEFAULT_DISKSIZE=$((8 * 1024 * 1024 * 1024))

case "$1" in
  off)
    echo "Disabling all swap..."
    if [ -e /proc/swaps ]; then
      awk 'NR>1 {print $1}' /proc/swaps | while read dev; do
        echo "Disabling swap on $dev ..."
        swapoff "$dev"
      done
    else
      echo "No swap devices found."
    fi

    # reset the zram device (clears pages & metadata, disksize→0)
    if [ -w "${ZRAM_SYS}/reset" ]; then
      echo 1 > "${ZRAM_SYS}/reset"
      echo "zram0 reset."
    else
      echo "Error: cannot reset zram0 (no write access to ${ZRAM_SYS}/reset)." >&2
      exit 1
    fi
    ;;
  on)
    echo "Enabling zram swap..."

    # if disksize is zero (after reset), set it back to DEFAULT_DISKSIZE
    cur_size=$(cat "${ZRAM_SYS}/disksize" 2>/dev/null || echo 0)
    if [ "$cur_size" -eq 0 ]; then
      echo "Setting zram0 disksize to ${DEFAULT_DISKSIZE} bytes..."
      echo "${DEFAULT_DISKSIZE}" > "${ZRAM_SYS}/disksize"
    else
      echo "zram0 disksize is already ${cur_size} bytes; leaving unchanged."
    fi

    # (re)initialize swap area
    if command -v mkswap >/dev/null 2>&1; then
      mkswap "${ZRAM_DEV}"
    else
      echo "Warning: mkswap not found — assuming zram0 is already formatted." >&2
    fi

    # turn swap on
    swapon "${ZRAM_DEV}" || {
      echo "Error: swapon failed on ${ZRAM_DEV}." >&2
      exit 1
    }

    echo "Swap enabled on ${ZRAM_DEV}."
    ;;
  *)
    cat << EOF
Usage: $0 {on|off}

  off   Disable all swap devices and reset zram0
  on    Set zram0 to 8GiB, (re)format it, and enable swap
EOF
    exit 1
    ;;
esac
