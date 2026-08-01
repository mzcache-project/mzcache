#!/bin/bash
# run_os_paging.sh — Figure 9, OS Paging (zRAM) series.
#
# The baseline is stock CPU llama.cpp (exp1_android_swap): weights are mmapped
# (kernel-reclaimable file pages), the KV cache is anonymous memory (swaps to
# zram/zstd), and only token_embd.weight is mlocked. Two modes:
#
#   resident (default, fully automatic)
#     Per context: start the binary, send one warm-up prompt (faults weights +
#     KV resident), then measure TTFT1 of a second prompt. This is the
#     remaining=100% point. The "first decode" is NOT used: at 32k the KV load
#     evicts weights and re-faults make it unstable (5-20 s).
#
#   evict (operator-guided, root, ramps memory pressure)
#     After the warm-up, repeatedly apply file-cache pressure with mmap_touch
#     (weights are reclaimed first, then KV swaps to zram), showing the resident
#     weight/KV Rss after each step. The operator decides when to measure; each
#     measurement appends one CSV row at the current remaining percentage. Guards
#     abort the ramp when MemAvailable < 700 MB or SwapFree < 400 MB (over-
#     pressure can reboot the device).
#
#   evict-full (fully automatic, root)
#     The FULL-eviction endpoint: drives the resident weight+KV Rss down to <= -T
#     percent of the warm baseline (default 10%) in two stages. First mmap_touch
#     file-cache pressure drops the weights (file pages, reclaimed for free) and
#     most of the KV; that plateaus above the floor because the stress pages are
#     themselves reclaimable. Then anon_hog dirty-anonymous pressure escalates
#     (ramped, capped at -A MiB) — it competes directly with the idle KV for
#     RAM+swap and forces it out to zram, reaching the mlock floor (~6%, just the
#     mlocked embedding). NO MemAvailable/SwapFree guard — this mode deliberately
#     pushes RSS toward 0, accepting the reboot/OOM risk. Then measures TTFT
#     (-r reps, re-applying anonymous pressure back to the target before each rep).
#
# Usage:
#   ADB_SERIAL=<serial> ./run_os_paging.sh [options]
#     -m resident|evict|evict-full   mode (default resident)
#     -c "8193 16385 32700"    context sizes (evict/evict-full: give exactly one)
#     -r 3                     measured repeats per context (resident/evict-full)
#     -o <dir>                 output dir (default results/ttft)
#     -N 5                     evict: initial file count (evict-full default 20)
#     -S 5                     evict: increment per step (evict-full default 10)
#     -T 10                    evict-full: stop when remaining <= this % (default 10)
#     -K 200000                anon Size floor (kB) to count a mapping as KV
#     -A <MiB>                 evict-full: anon_hog cap (default MemTotal - 4 GiB)
#
# Requires: su (mlock exceeds RLIMIT_MEMLOCK), non-fa states, zram set to zstd
# (scripts/setup/device_setup.sh), and for evict/evict-full the memstress tools
# (scripts/ttft/memstress/setup_memstress.sh: mmap_touch, anon_hog, stress files).
#
# Output: <out>/ttft_os_paging_<device>_<model_tag>.csv
#   schema: system,device,model,ctx,remaining_pct,rep,ttft_ms,
#           w_rss_kb,kv_rss_kb,emb_kb,base_w_kb,base_kv_kb,comp
#
# region_rss.awk classifies the process's mappings by backing (file-backed gguf =
# weights, large anon = KV) and sums each region's Rss from smaps. Each row stores
# those components; remaining_pct is computed (here and re-derived in plot_ttft)
# as, EXCLUDING the mlocked embedding (emb = VmLck) and counting the swapped KV's
# compressed zram footprint:
#
#   ((w_rss - emb) + kv_rss + (base_kv - kv_rss)*comp) / ((base_w - emb) + base_kv)
#
# Eviction drives (w_rss - emb) and kv_rss to 0, but comp (the zram residual of
# fp16 KV, ~0.92) is high, so remaining floors well above 0 (~18-75% here).

source "$(dirname "$0")/../common.sh"

MODE=resident
CTXS="8193 16385 32700"
REPS=3
OUT=$RESULTS_ROOT/ttft
N0=5
STEP=5
TARGET=10        # evict-full: stop once the driven-residency control metric <= this %
KV_MIN=200000    # anon Size floor (kB) for a mapping to count as KV
ANON_START=2048  # evict-full: initial anon_hog size (MiB) for the escalation
ANON_STEP=1024   # evict-full: anon_hog growth per escalation step (MiB)
ANON_MAX=0       # evict-full: anon_hog cap (MiB); 0 = derive from MemTotal - 4 GiB
KV_FLOOR=5       # evict-full: stop once KV_Rss <= this % of its baseline (KV out)

# comp = the zram (zstd) compression rate of the swapped fp16 KV: the fraction of
# a swapped KV page that STILL occupies RAM inside zram. The reported remaining-%
# (see calc_remaining / plot_ttft) counts that compressed footprint, so it floors
# above 0 even when the non-embedding weight and KV Rss are both driven to 0.
case $MODEL_TAG in
    exaone4*) COMP=0.907 ;;
    *)        COMP=0.9196 ;;  # qwen3_0.6B
esac

while getopts "m:c:r:o:N:S:T:K:A:" opt; do
    case $opt in
        m) MODE=$OPTARG ;;
        c) CTXS=$OPTARG ;;
        r) REPS=$OPTARG ;;
        o) OUT=$OPTARG ;;
        N) N0=$OPTARG; N0_SET=$OPTARG ;;
        S) STEP=$OPTARG; STEP_SET=$OPTARG ;;
        T) TARGET=$OPTARG ;;
        K) KV_MIN=$OPTARG ;;
        A) ANON_MAX=$OPTARG ;;
        *) exit 1 ;;
    esac
done

require_device
require_su

DEV=$(device_model)
mkdir -p "$OUT/raw"
RUNTAG="os_paging_${DEV}_${MODEL_TAG}"
CSV=$OUT/ttft_${RUNTAG}.csv
if [ ! -e "$CSV" ]; then
    # remaining_pct is computed from the components (w_rss/kv_rss/emb/baselines/
    # comp) by the formula in calc_remaining; the raw components are stored too so
    # plot_ttft recomputes it (and comp can be retuned without re-measuring).
    echo "system,device,model,ctx,remaining_pct,rep,ttft_ms,w_rss_kb,kv_rss_kb,emb_kb,base_w_kb,base_kv_kb,comp" > "$CSV"
fi

BIN=$DEVICE_DIR/$ANDSW_INSTALL/bin/exp1_android_swap
require_on_device "$BIN" \
    "build and push it first: ADB_SERIAL=$ADB_SERIAL ./cpu_build.sh $ANDSW_INSTALL"

# The per-region smaps parser runs on the device; push it once. It is invoked as
# `awk -f` (not inlined) because adbsu wraps its payload in single quotes and an
# inlined awk program cannot contain any.
RSS_AWK_DEV=$DEVICE_DIR/memstress/region_rss.awk
push_region_awk() {
    adbsh "mkdir -p $DEVICE_DIR/memstress" >/dev/null
    adb -s "$ADB_SERIAL" push "$(dirname "$0")/region_rss.awk" "$RSS_AWK_DEV" >/dev/null
}

# --- helpers driving one interactive session over a persistent adb-su shell ---

SESSION_LOG=""

start_session() {  # start_session <ctx>
    local ctx=$1
    local nctx=$((ctx + 57))   # n_ctx headroom over the saved state (8193->8250, ...)
    SESSION_LOG=$OUT/raw/${RUNTAG}_${ctx}_session_$(date +%s).log
    coproc ANDSW {
        adb -s "$ADB_SERIAL" shell \
            "su -c 'cd $DEVICE_DIR && ANDSW_STATE=states/${MODEL_TAG}_${ctx}.kv ./$ANDSW_INSTALL/bin/exp1_android_swap -m $GGUF -t 6 -c $nctx'" 2>&1
    }
    : > "$SESSION_LOG"
}

# read session output until a line matches $1 (regex); echo the matching line.
wait_for() {
    local pat=$1 timeout=${2:-900} line
    while IFS= read -r -t "$timeout" line <&"${ANDSW[0]}"; do
        printf '%s\n' "$line" >> "$SESSION_LOG"
        if [[ $line =~ $pat ]]; then
            printf '%s\n' "$line"
            return 0
        fi
    done
    echo "ERROR: timed out waiting for /$pat/ (see $SESSION_LOG)" >&2
    return 1
}

send_prompt() {
    printf '%s\n' "$1" >&"${ANDSW[1]}"
}

end_session() {
    send_prompt "exit" 2>/dev/null || true
    sleep 1
    [ -n "${ANDSW_PID:-}" ] && kill "$ANDSW_PID" 2>/dev/null
    wait 2>/dev/null
}

parse_ttft_ms() {  # "[Timing] TTFT1 123.45 ms" -> 123.45
    grep -oE '[0-9]+(\.[0-9]+)?' <<< "$1" | head -2 | tail -1
}

pid_on_device() {
    # Match by comm, not -f: the su/sh wrapper's command line also contains the
    # binary path, and -f | head -1 returns that wrapper (whose tiny Pss makes
    # the remaining%% estimate read ~100%). comm is truncated to 15 chars.
    adbsu "pgrep exp1_android" | tr -d '\r' | head -1
}

proc_stat() {  # proc_stat <pid> — prints "VmSwap_kB VmLck_kB PssFile_kB Pss_kB MemAvail_kB SwapFree_kB"
    local pid=$1
    adbsu "grep -E \"VmSwap|VmLck\" /proc/$pid/status; grep -E \"Pss_File|^Pss:\" /proc/$pid/smaps_rollup; grep -E \"MemAvailable|SwapFree\" /proc/meminfo" \
        | tr -d '\r' | awk '
            /VmSwap/       {sw=$2}  /VmLck/     {lk=$2}
            /Pss_File/     {pf=$2}  /^Pss:/     {ps=$2}
            /MemAvailable/ {ma=$2}  /SwapFree/  {sf=$2}
            END {print sw+0, lk+0, pf+0, ps+0, ma+0, sf+0}'
}

# region_rss <pid> — resident Rss (kB) of the weight and KV mappings, classified
# by backing in region_rss.awk. Prints "weight_rss_kB kv_rss_kB".
region_rss() {
    local pid=$1
    adbsu "awk -v MIN=$KV_MIN -f $RSS_AWK_DEV /proc/$pid/smaps" | tr -d '\r'
}

pid_alive() {  # pid_alive <pid> — true while the device process still exists
    [ -n "$(adbsu "ls -d /proc/$1 2>/dev/null" | tr -d '\r')" ]
}

# calc_remaining <w_rss> <kv_rss> <emb> <base_w> <base_kv> <comp> — the reported
# remaining-in-memory %, EXCLUDING the mlocked token embedding:
#
#   ( (w_rss - emb) + kv_rss + (base_kv - kv_rss)*comp )
#   -------------------------------------------------------  x 100
#            ( (base_w - emb) + base_kv )
#
# The non-embedding weight Rss and the KV Rss are what eviction drives to 0; the
# swapped KV still occupies (base_kv - kv_rss)*comp in zram, so this floors above
# 0. All sizes in kB; emb is the mlocked embedding (VmLck).
calc_remaining() {
    awk -v wr="$1" -v kr="$2" -v e="$3" -v bw="$4" -v bk="$5" -v c="$6" 'BEGIN{
        wn = wr - e;  if (wn < 0) wn = 0;
        sw = bk - kr; if (sw < 0) sw = 0;
        d  = (bw - e) + bk; if (d <= 0) { print 0; exit }
        printf "%.0f", 100.0 * (wn + kr + sw*c) / d }'
}

# ------------------------------- resident mode -------------------------------

run_resident() {
    push_region_awk
    for ctx in $CTXS; do
        require_on_device "$DEVICE_DIR/states/${MODEL_TAG}_${ctx}.kv" \
            "push or generate the non-fa state for ctx=$ctx"
        echo "=== OS paging (resident) ctx=$ctx ==="
        drop_caches; sleep 2
        start_session "$ctx"
        wait_for '\[Timing\] Loaded' || { end_session; continue; }

        send_prompt "Please summarize the context so far."   # warm-up: fault everything in
        wait_for 'TTFT1' >/dev/null || { end_session; continue; }

        # Resident IS the baseline: capture the fully-faulted weight/KV Rss and
        # the mlocked embedding (VmLck) so the row carries the same components as
        # the evicted rows (remaining computes to 100%).
        local pid wr kr emb
        pid=$(pid_on_device)
        read -r wr kr <<< "$(region_rss "$pid")"
        read -r _ emb _ _ _ _ <<< "$(proc_stat "$pid")"
        echo "    baseline weight_Rss=${wr}kB KV_Rss=${kr}kB emb(VmLck)=${emb}kB"

        for rep in $(seq 1 "$REPS"); do
            send_prompt "What was mentioned earlier?"
            line=$(wait_for 'TTFT1') || break
            ttft=$(parse_ttft_ms "$line")
            local rem; rem=$(calc_remaining "$wr" "$kr" "$emb" "$wr" "$kr" "$COMP")
            echo "    rep=$rep TTFT1=${ttft} ms (remaining ${rem}%)"
            echo "os_paging,$DEV,$MODEL_TAG,$ctx,$rem,$rep,$ttft,$wr,$kr,$emb,$wr,$kr,$COMP" >> "$CSV"
        done
        end_session
        sleep 5
    done
}

# -------------------------------- evict mode ---------------------------------

run_evict() {
    set -- $CTXS
    if [ $# -ne 1 ]; then
        echo "ERROR: evict mode drives one context at a time, e.g. -m evict -c 16385" >&2
        exit 1
    fi
    local ctx=$1
    require_on_device "$DEVICE_DIR/memstress/mmap_touch" \
        "run scripts/ttft/memstress/setup_memstress.sh first"
    require_on_device "$DEVICE_DIR/states/${MODEL_TAG}_${ctx}.kv" \
        "push or generate the non-fa state for ctx=$ctx"

    echo "=== OS paging (evict) ctx=$ctx — operator-guided ==="
    push_region_awk
    drop_caches; sleep 2
    start_session "$ctx"
    wait_for '\[Timing\] Loaded' || { end_session; exit 1; }
    send_prompt "Please summarize the context so far."       # warm-up
    wait_for 'TTFT1' >/dev/null || { end_session; exit 1; }

    local pid; pid=$(pid_on_device)
    [ -n "$pid" ] || { echo "ERROR: could not find device pid" >&2; end_session; exit 1; }

    local wr kr prem
    read -r wr kr <<< "$(region_rss "$pid")"
    local base_wr=$wr base_kr=$kr
    echo "warm baseline: pid=$pid weight_Rss=${wr}kB KV_Rss=${kr}kB (remaining=100%)"

    local n=$N0 rep=0
    while true; do
        echo ""
        echo "--- pressure step: mmap_touch $n x128MB ---"
        adbsu "cd $DEVICE_DIR/memstress && ./mmap_touch $n 128 3" | sed 's/^/    /'
        sleep 2
        read -r wr kr <<< "$(region_rss "$pid")"
        read -r sw lk _ _ ma sf <<< "$(proc_stat "$pid")"
        # The warm-up can leave some weight/KV pages unfaulted, so the true
        # resident peak may only show up mid-ramp: ratchet each baseline.
        [ "$wr" -gt "$base_wr" ] && base_wr=$wr
        [ "$kr" -gt "$base_kr" ] && base_kr=$kr
        # remaining = resident weight+KV Rss relative to the warm baseline.
        rem=$(awk -v wr="$wr" -v kr="$kr" -v bw="$base_wr" -v bk="$base_kr" \
              'BEGIN{ b=bw+bk; if (b<=0) {print 0; exit} printf "%.0f", 100.0*(wr+kr)/b }')
        echo "    weight_Rss=${wr}kB KV_Rss=${kr}kB VmSwap=${sw}kB VmLck=${lk}kB -> remaining ~${rem}%"
        echo "    system: MemAvailable=${ma}kB SwapFree=${sf}kB"
        if [ "$ma" -lt 716800 ] || [ "$sf" -lt 409600 ]; then
            echo "    GUARD: low MemAvailable/SwapFree — stopping the ramp (reboot risk)."
            break
        fi

        printf "    [m]easure TTFT here / [c]ontinue ramp (+%d) / [q]uit: " "$STEP"
        read -r ans < /dev/tty
        case $ans in
            m|M)
                rep=$((rep + 1))
                # emb = mlocked embedding (VmLck); prem = reported remaining-%.
                prem=$(calc_remaining "$wr" "$kr" "$lk" "$base_wr" "$base_kr" "$COMP")
                send_prompt "What was mentioned earlier?"
                line=$(wait_for 'TTFT1' 1800) || break
                ttft=$(parse_ttft_ms "$line")
                echo "    TTFT1=${ttft} ms at remaining ${prem}%"
                echo "os_paging,$DEV,$MODEL_TAG,$ctx,$prem,$rep,$ttft,$wr,$kr,$lk,$base_wr,$base_kr,$COMP" >> "$CSV"
                # measuring faults data back in: report the re-warmed residency
                read -r wr kr <<< "$(region_rss "$pid")"
                echo "    (post-measure weight_Rss=${wr}kB KV_Rss=${kr}kB)"
                ;;
            q|Q) break ;;
            *)   n=$((n + STEP)) ;;
        esac
    done
    end_session
}

# ------------------------------ evict-full mode ------------------------------

# ramp_evict <start_n> — grow the held working set from <start_n> up to nmax,
# applying mmap_touch pressure each step, until the resident weight+KV Rss has
# fallen to <= TARGET% of the warm baseline (>= (100-TARGET)% evicted) or it
# plateaus at the pool ceiling. NO MemAvailable/SwapFree guard: this endpoint is
# meant to drive Rss to ~0, accepting the reboot risk. Operates on the caller's
# pid/nmax/wr/kr/base_wr/base_kr/rem/n (bash dynamic scope) and leaves the final
# residency in `rem` and the reached file count in `n`.
ramp_evict() {
    local prev_rem=100 plateau=0
    n=$1
    while true; do
        echo "--- pressure: mmap_touch $n x128MB ---"
        adbsu "cd $DEVICE_DIR/memstress && ./mmap_touch $n 128 3" >/dev/null
        sleep 2
        read -r wr kr <<< "$(region_rss "$pid")"
        read -r sw lk _ _ ma sf <<< "$(proc_stat "$pid")"
        # The warm-up can leave some weight/KV pages unfaulted, so the true
        # resident peak may only show up mid-ramp: ratchet each baseline.
        [ "$wr" -gt "$base_wr" ] && base_wr=$wr
        [ "$kr" -gt "$base_kr" ] && base_kr=$kr
        rem=$(awk -v wr="$wr" -v kr="$kr" -v bw="$base_wr" -v bk="$base_kr" \
              'BEGIN{ b=bw+bk; if (b<=0) {print 0; exit} printf "%.0f", 100.0*(wr+kr)/b }')
        echo "    weight_Rss=${wr}kB KV_Rss=${kr}kB VmSwap=${sw}kB VmLck=${lk}kB -> remaining ~${rem}%"
        echo "    system: MemAvailable=${ma}kB SwapFree=${sf}kB"
        if [ "$rem" -le "$TARGET" ]; then
            echo "    reached remaining ${rem}% <= target ${TARGET}% — evicted."
            return
        fi
        if [ "$n" -ge "$nmax" ]; then
            if [ $((prev_rem - rem)) -le 1 ]; then
                plateau=$((plateau + 1))
                [ $plateau -ge 2 ] && { echo "    file-cache pressure plateaued at remaining ${rem}% (pool ceiling $nmax) — handing off to anonymous pressure."; return; }
            else
                plateau=0
            fi
        fi
        prev_rem=$rem
        [ "$n" -lt "$nmax" ] && n=$((n + STEP)) || n=$nmax
    done
}

# ramp_anon — escalate with dirty ANONYMOUS pressure (anon_hog) until the
# resident weight+KV Rss falls to <= TARGET% of the baseline. File-cache pressure
# plateaus above the floor because its clean stress pages are reclaimable; dirty
# anon competes directly with the idle KV for RAM+swap and forces it out to zram,
# reaching the mlock floor. Grows the hog from ANON_LAST/ANON_START by ANON_STEP
# each step, capped at ANON_MAX. Operates on the caller's pid/wr/kr/base_wr/
# base_kr/rem; sets `rem` and remembers the reached size in ANON_LAST. Returns 1
# if the process is OOM-killed (dial ANON_MAX down with -A).
ramp_anon() {
    local mb=${ANON_LAST:-$ANON_START} prev_rem=$rem plateau=0 sw lk ma sf
    echo "=== escalating with anonymous pressure (anon_hog, cap ${ANON_MAX}MB) to reach <=${TARGET}% ==="
    while [ "$rem" -gt "$TARGET" ]; do
        echo "--- anon pressure: anon_hog ${mb}MB ---"
        adbsu "cd $DEVICE_DIR/memstress && ./anon_hog $mb 3" >/dev/null
        sleep 2
        if ! pid_alive "$pid"; then
            echo "    ERROR: process $pid died under anon pressure (OOM at ${mb}MB) — lower -A." >&2
            return 1
        fi
        read -r wr kr <<< "$(region_rss "$pid")"
        read -r sw lk _ _ ma sf <<< "$(proc_stat "$pid")"
        rem=$(awk -v wr="$wr" -v kr="$kr" -v bw="$base_wr" -v bk="$base_kr" \
              'BEGIN{ b=bw+bk; if (b<=0) {print 0; exit} printf "%.0f", 100.0*(wr+kr)/b }')
        local kvp; kvp=$(awk -v kr="$kr" -v bk="$base_kr" \
              'BEGIN{ if (bk<=0) {print 0; exit} printf "%.0f", 100.0*kr/bk }')
        echo "    weight_Rss=${wr}kB KV_Rss=${kr}kB (${kvp}% of KV) VmSwap=${sw}kB -> remaining ~${rem}%   (hog ${mb}MB)"
        echo "    system: MemAvailable=${ma}kB SwapFree=${sf}kB"
        ANON_LAST=$mb
        [ "$rem" -le "$TARGET" ] && { echo "    reached remaining ${rem}% <= target ${TARGET}%."; return 0; }
        # The mlocked embedding cannot be swapped, so on small contexts `rem`
        # floors above TARGET; once the KV itself is essentially out, stop rather
        # than ramping the hog to its cap against un-evictable pinned weight.
        [ "$kvp" -le "$KV_FLOOR" ] && { echo "    KV evicted (${kvp}% of KV Rss left) at remaining ${rem}% — mlocked weight is the floor."; return 0; }
        if [ $((prev_rem - rem)) -le 0 ]; then
            plateau=$((plateau + 1))
            [ $plateau -ge 3 ] && { echo "    anon pressure plateaued at remaining ${rem}% (hog ${mb}MB)."; return 0; }
        else
            plateau=0
        fi
        prev_rem=$rem
        [ "$mb" -ge "$ANON_MAX" ] && { echo "    anon_hog reached cap ${ANON_MAX}MB at remaining ${rem}%."; return 0; }
        mb=$((mb + ANON_STEP)); [ "$mb" -gt "$ANON_MAX" ] && mb=$ANON_MAX
    done
}

# evict_to_target <file_start_n> — file-cache pressure (drops weights first) then
# anonymous escalation (finishes the KV) until remaining <= TARGET%. Returns
# non-zero only if the anon escalation aborts (process OOM-killed); reaching the
# target during the file phase alone is still success.
evict_to_target() {
    ramp_evict "$1"
    if [ "$rem" -gt "$TARGET" ]; then
        ramp_anon || return 1
    fi
    return 0
}

run_evict_full() {
    set -- $CTXS
    if [ $# -ne 1 ]; then
        echo "ERROR: evict-full mode drives one context at a time, e.g. -m evict-full -c 16385" >&2
        exit 1
    fi
    local ctx=$1
    require_on_device "$DEVICE_DIR/memstress/mmap_touch" \
        "run scripts/ttft/memstress/setup_memstress.sh first"
    require_on_device "$DEVICE_DIR/memstress/anon_hog" \
        "run scripts/ttft/memstress/setup_memstress.sh first"
    require_on_device "$DEVICE_DIR/states/${MODEL_TAG}_${ctx}.kv" \
        "push or generate the non-fa state for ctx=$ctx"
    local nmax
    nmax=$(adbsh "ls $DEVICE_DIR/memstress/stress_*.bin 2>/dev/null | wc -l" | tr -d '\r ')
    [ "$nmax" -gt 0 ] || { echo "ERROR: no stress files (run setup_memstress.sh)" >&2; exit 1; }
    # Cap anon pressure below RAM so the hog itself is not OOM-killed before the
    # KV is out; the target-stop keeps it well under this in practice. Leave 4 GiB
    # of headroom: at 2 GiB a 12 GB phone (Galaxy S25+) reboots during the rep-2
    # re-eviction. Dial it down further with -A if your device still reboots.
    if [ "$ANON_MAX" -eq 0 ]; then
        local memtotal_kb; memtotal_kb=$(adbsu "grep MemTotal /proc/meminfo" | awk '{print $2}')
        ANON_MAX=$(( memtotal_kb / 1024 - 4096 ))
    fi

    echo "=== OS paging (evict-full) ctx=$ctx — automatic, pool $nmax x128MB + anon<=${ANON_MAX}MB, target <=${TARGET}% ==="
    push_region_awk
    drop_caches; sleep 2
    start_session "$ctx"
    wait_for '\[Timing\] Loaded' || { end_session; exit 1; }
    send_prompt "Please summarize the context so far."       # warm-up
    wait_for 'TTFT1' >/dev/null || { end_session; exit 1; }

    local pid; pid=$(pid_on_device)
    [ -n "$pid" ] || { echo "ERROR: could not find device pid" >&2; end_session; exit 1; }
    local wr kr sw lk ma sf n rem=100 emb prem
    read -r wr kr <<< "$(region_rss "$pid")"
    read -r _ emb _ _ _ _ <<< "$(proc_stat "$pid")"   # emb = mlocked embedding (VmLck)
    local base_wr=$wr base_kr=$kr
    echo "warm baseline: pid=$pid weight_Rss=${wr}kB KV_Rss=${kr}kB emb(VmLck)=${emb}kB (remaining=100%)"

    evict_to_target "$N0" || { echo "ERROR: eviction aborted (process died)"; end_session; return; }

    echo "=== measuring TTFT (full eviction) ==="
    local rep line ttft
    for rep in $(seq 1 "$REPS"); do
        if [ "$rep" -gt 1 ]; then
            # Measuring rep-1 faulted the KV (and weights) back in, so run the FULL
            # eviction again — file pressure first, then anon. Anon alone plateaus
            # near remaining 100% on a warm page cache, and ramp_anon reports that
            # plateau as success, which used to get measured and recorded as if it
            # were an evicted point.
            echo "  (re-evicting before rep $rep)"
            read -r wr kr <<< "$(region_rss "$pid")"
            rem=$(awk -v wr="$wr" -v kr="$kr" -v bw="$base_wr" -v bk="$base_kr" \
                  'BEGIN{ b=bw+bk; if (b<=0) {print 0; exit} printf "%.0f", 100.0*(wr+kr)/b }')
            evict_to_target "$N0" || { echo "ERROR: re-eviction aborted (process died)"; break; }
        fi
        # wr/kr now hold the evicted-state components (KV faults back in only when
        # the measurement prompt runs, below), so capture remaining here.
        # Never record a rep the eviction did not actually bring down: the ramps
        # give up on a plateau/cap, and a resident-level row here would be pooled
        # into the resident series by plot_ttft.py without any warning.
        rem=$(awk -v wr="$wr" -v kr="$kr" -v bw="$base_wr" -v bk="$base_kr" \
              'BEGIN{ b=bw+bk; if (b<=0) {print 0; exit} printf "%.0f", 100.0*(wr+kr)/b }')
        if [ "$rem" -gt "$TARGET" ]; then
            echo "  rep=$rep SKIPPED: eviction stopped at ${rem}% > target ${TARGET}% (not an evicted point)."
            echo "    weight_Rss=${wr}kB KV_Rss=${kr}kB — raise the anon cap (-A) or the pool, then re-run."
            continue
        fi
        prem=$(calc_remaining "$wr" "$kr" "$emb" "$base_wr" "$base_kr" "$COMP")
        send_prompt "What was mentioned earlier?"
        line=$(wait_for 'TTFT1' 1800) || break
        ttft=$(parse_ttft_ms "$line")
        echo "  rep=$rep TTFT1=${ttft} ms at remaining ${prem}% (weight_Rss=${wr}kB KV_Rss=${kr}kB)"
        echo "os_paging,$DEV,$MODEL_TAG,$ctx,$prem,$rep,$ttft,$wr,$kr,$emb,$base_wr,$base_kr,$COMP" >> "$CSV"
    done
    end_session
}

case $MODE in
    resident)   run_resident ;;
    evict)      run_evict ;;
    evict-full) N0=${N0_SET:-20}; STEP=${STEP_SET:-10}; run_evict_full ;;
    *) echo "ERROR: unknown mode $MODE (resident|evict|evict-full)" >&2; exit 1 ;;
esac

echo ""
echo "CSV: $CSV"
