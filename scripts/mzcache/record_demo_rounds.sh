#!/usr/bin/env bash
# record_demo_rounds.sh — multi-round app-switch demo recorder (paper Fig. 15).
#
# Mirrors the phone with scrcpy (LEFT) next to a live memory trace (RIGHT) and
# screen-records both into one clip while driving a fixed app-switch workload:
# ROUNDS (default 3) rounds over ten apps, each foregrounded for DWELL_S (20s;
# pubg 30s; camera = preview + shutter; roblox = open the Avatar tab; youtube =
# play a video; tiktok/instagram = swipe the feed for the whole dwell).
# Landscape apps like pubg are kept in the portrait frame via SCRCPY_LOCK (below).
# A mid-round `visit` foregrounds the LLM app for VISIT_S (5s) WITHOUT pressing
# Send (just a glance). Each round ends with a return to the LLM app + Send
# (-> swap-in), then is paced to ROUND_PERIOD (275s; 0 = no idle pad). The trace
# shows the memory sawtooth; the verdict is the RESULT line at the end — the
# mzCache app should keep its pid, the OS-paging baseline gets LMK-killed.
#
#   ./record_demo_rounds.sh [mzcache|baseline]     (default: mzcache)
#
# Output: $OUT_DIR/mz_rounds_<variant>_<YYYYmmdd_HHMMSS>.mp4 (~14 min per variant)
#
# Requirements:
#   * host: ffmpeg, scrcpy (SCRCPY_BIN), matplotlib + PyQt5, and a real desktop
#     session ($DISPLAY) at least (scrcpy width + TRACE_W) px wide — not a
#     headless server, not an SSH shell, not the Docker image.
#   * phone: adb + root (su), the app installed with its model + KV state on the
#     device, and ADB_SERIAL exported (or exactly one device attached).
#   * the UI taps below are calibrated for a Galaxy S25+ (1080x2340); on a
#     different screen size they land in the wrong place (the script warns).
#   * mzcache variant: the swapout step is set here from MZ_STEP (default 0.25),
#     overriding whatever debug.mzcache.step held before.
set -u

VARIANT="${1:-mzcache}"

# --- host / device ---------------------------------------------------------
# Prefer $ADB_SERIAL; otherwise fall back to the only attached device, so a
# reviewer with one phone can just run this. Never a hard-coded serial.
SERIAL="${ADB_SERIAL:-}"
if [ -z "$SERIAL" ]; then
  SERIAL=$(adb devices | awk '$2=="device"{print $1}')
  if [ "$(printf '%s\n' "$SERIAL" | grep -c .)" != 1 ]; then
    echo "ERROR: set ADB_SERIAL — 'adb devices' does not show exactly one device:" >&2
    adb devices | tail -n +2 >&2
    exit 1
  fi
fi
SCRCPY_BIN="${SCRCPY_BIN:-scrcpy}"
# Lock the mirror to portrait so landscape apps (pubg) don't rotate the window and
# break the fixed portrait capture region. scrcpy >=3.0: --capture-orientation=@0;
# for scrcpy 2.x set SCRCPY_LOCK=--lock-video-orientation=0; empty ("") to disable.
# (A locked landscape app renders sideways within the portrait frame.)
SCRCPY_LOCK="${SCRCPY_LOCK:---capture-orientation=@0}"
export DISPLAY="${DISPLAY:-:0}"
# Only guess XAUTHORITY when the caller has none and that file really exists —
# hard-coding the GDM path breaks LightDM/SDDM sessions and any uid != 1000.
if [ -z "${XAUTHORITY:-}" ] && [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
  export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-/tmp}"

# --- on-screen layout (px): scrcpy LEFT, trace RIGHT, both TRACE_H tall ------
# 16:6 capture (498 scrcpy + 2382 trace = 2880 wide x 1080 tall) so two of these
# stacked vertically make a 4:3 (2880x2160) side-by-side comparison video.
MAX_SIZE="${MAX_SIZE:-1080}"      # scrcpy --max-size (portrait phone -> 498x1080)
TRACE_W="${TRACE_W:-2382}"        # trace figure width (498 + 2382 = 2880 => 16:6)
TRACE_H="${TRACE_H:-1080}"        # trace figure height (matches scrcpy height)
MARGIN_Y="${MARGIN_Y:-48}"        # top gap so the desktop top bar is NOT recorded
FPS="${FPS:-10}"
SCRCPY_FPS="${SCRCPY_FPS:-10}"   # cap scrcpy capture fps (lower -> less phone GPU load)

# --- workload timing (s) ----------------------------------------------------
ROUNDS="${ROUNDS:-3}"              # app-switch rounds (the paper's run uses 10)
DWELL_S="${DWELL_S:-20}"           # residency per app (default 20s)
PUBG_DWELL_S="${PUBG_DWELL_S:-30}" # pubg residency (a game needs longer to load)
VISIT_S="${VISIT_S:-5}"            # `visit`: brief LLM-foreground glance (no Send)
ROUND_PERIOD="${ROUND_PERIOD:-275}"   # pace each round to this many s (0 = no idle pad); grown
                                      # with DWELL_S so the post-round LLM dwell stays the same
STEADY_S="${STEADY_S:-5}"          # plateau in the loaded app before round 1
RETURN_WAIT="${RETURN_WAIT:-25}"   # dwell in the LLM app after each round's return+Send
TAIL_S="${TAIL_S:-30}"             # idle tail after the last round, then stop
MZ_STEP="${MZ_STEP:-0.25}"         # mzcache swapout eviction per pressure event
TRACE_WINDOW="${TRACE_WINDOW:-900}"   # trace x-axis span (s): fixed 0..900, cumulative

# --- variant -> package, trace tool, y-axis --------------------------------
case "$VARIANT" in
  mzcache)  LLM_PKG=com.example.llama;         TRACE_PY=mem_trace.py;          YMAX="${YMAX:-6}"  ;;
  baseline) LLM_PKG=com.example.llama.cpuswap; TRACE_PY=mem_trace_baseline.py; YMAX="${YMAX:-6}"  ;;
  *) echo "usage: $0 [mzcache|baseline]"; exit 1 ;;
esac

# --- apps + packages -------------------------------------------------------
# All ten must be installed (and signed in, where they need an account): a missing
# package is skipped, which lowers the memory pressure and changes the outcome.
# Region/vendor-specific IDs: pubg is the KR build, tiktok the "trill" build, and
# the camera is Samsung's.
declare -A PKG=(
  [roblox]=com.roblox.client           [pubg]=com.pubg.krmobile
  [camera]=com.sec.android.app.camera  [snow]=com.campmobile.snow
  [chrome]=com.android.chrome          [youtube]=com.google.android.youtube
  [tiktok]=com.ss.android.ugc.trill    [netflix]=com.netflix.mediaclient
  [instagram]=com.instagram.android    [facebook]=com.facebook.katana
)
# Per-round schedules R1..R3 (the paper's first three rounds): a fixed sequence of
# app switches with `visit` (an LLM-foreground glance) interspersed — reproducible
# and identical for mzcache vs baseline. Cycles if ROUNDS > 3 lines.
ROUND_ORDER=(
  "chrome tiktok snow youtube visit pubg facebook instagram camera roblox netflix"
  "instagram tiktok roblox snow youtube visit chrome pubg netflix facebook camera"
  "tiktok chrome netflix facebook visit roblox instagram pubg visit snow camera youtube"
)
LOAD_BTN=(540 2114)      # LLM app "Load Model, KV cache"
SEND_BTN=(990 1935)      # LLM app Send icon
# Per-app interaction points (portrait 1080x2340; tweak from the scrcpy mirror if
# a tap misses). Driving real UI grows each foreground app's working set, so the
# backgrounded LLM app faces more memory pressure (a firmer LMK kill signal).
ROBLOX_AVATAR_BTN=(670 2115)   # Roblox bottom-bar Avatar tab (3rd: Home/Charts/Avatar/Party)
YT_VIDEO_BTN=(540 1585)        # YouTube 2nd feed item (1st real video; top slot is an ad)

_adb()  { command adb -s "$SERIAL" "$@"; }
tap()   { _adb shell input tap "$1" "$2"; }
swipe() { _adb shell input swipe "$1" "$2" "$3" "$4" "${5:-250}"; }
feed_swipe() { swipe 540 1850 540 550 120; }   # quick upward feed scroll (tiktok/insta)
key()   { _adb shell input keyevent "$1"; }
home()  { key KEYCODE_HOME; sleep 1; }
launch(){ _adb shell monkey -p "$1" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1; }
pidof_llm() { _adb shell pidof "$LLM_PKG" | tr -d '\r'; }
log()   { echo "[$(date +%H:%M:%S)] $*"; }

# Foreground the LLM app (a mid-round `visit` — no Send, just a glance).
llm_foreground() { home; launch "$LLM_PKG"; }
# Foreground the LLM app and press Send (end-of-round return -> swapin_generate).
llm_send() { home; launch "$LLM_PKG"; sleep 1.5; tap "${SEND_BTN[@]}"; }

LLM_PID=""
check_alive() {
  local p; p=$(pidof_llm)
  if [ -z "$p" ]; then log "  >> LLM app DEAD (LMK) after: $1"; return 1
  elif [ "$p" != "$LLM_PID" ]; then log "  >> LLM app pid changed ($LLM_PID -> $p) after: $1"; return 1; fi
  return 0
}

# ---------------------------------------------------------------------------
# 0) preflight + layout
# ---------------------------------------------------------------------------
command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found — sudo apt install ffmpeg"; exit 1; }
command -v "$SCRCPY_BIN" >/dev/null 2>&1 || [ -x "$SCRCPY_BIN" ] || {
  echo "ERROR: scrcpy not runnable ('$SCRCPY_BIN') — set SCRCPY_BIN=/path/to/scrcpy"; exit 1; }
[ "$(_adb get-state 2>/dev/null)" = device ] || { echo "ERROR: device $SERIAL not connected"; exit 1; }

SZ=$(_adb shell wm size | tr -d '\r')
DIM=$(echo "$SZ" | awk -F': ' '/Override/{print $2}' | tail -1)
[ -z "$DIM" ] && DIM=$(echo "$SZ" | awk -F': ' '/Physical/{print $2}' | tail -1)
PW=${DIM%x*}; PH=${DIM#*x}
# Every in-app tap below is calibrated for the Galaxy S25+ layout. On a different
# resolution the taps miss, the apps do nothing, and the run still ends in
# "RESULT: SURVIVED" — so say so loudly rather than reporting a false positive.
if [ "$PW" != 1080 ] || [ "$PH" != 2340 ]; then
  log "WARNING: screen is ${PW}x${PH}, taps are calibrated for 1080x2340 (Galaxy S25+)."
  log "         Expect mis-taps; re-measure the *_BTN coordinates before trusting a run."
fi
SCRCPY_W="${SCRCPY_W:-$(awk -v m="$MAX_SIZE" -v pw="$PW" -v ph="$PH" 'BEGIN{ printf "%d", (m*pw/ph)+0.5 }')}"
SCRCPY_X=0; SCRCPY_Y=$MARGIN_Y
TRACE_X="${TRACE_X:-$SCRCPY_W}"; TRACE_Y=$MARGIN_Y
REGION_W=$(( SCRCPY_W + TRACE_W )); REGION_W=$(( REGION_W - REGION_W % 2 ))
REGION_H=$(( TRACE_H - TRACE_H % 2 ))
OUT="$OUT_DIR/mz_rounds_${VARIANT}_$(date +%Y%m%d_%H%M%S).mp4"

log "=== record_demo_rounds: variant=$VARIANT pkg=$LLM_PKG rounds=$ROUNDS ==="
log "record ${REGION_W}x${REGION_H} @(0,${MARGIN_Y}) -> $OUT"

FF_PID=""; SCRCPY_PID=""; TRACE_PID=""
cleanup() {
  [ -n "$FF_PID" ]     && kill -INT "$FF_PID"  2>/dev/null && wait "$FF_PID" 2>/dev/null
  [ -n "$TRACE_PID" ]  && kill      "$TRACE_PID" 2>/dev/null
  [ -n "$SCRCPY_PID" ] && kill      "$SCRCPY_PID" 2>/dev/null
  _adb shell svc power stayon false >/dev/null 2>&1
}
_stop=0
on_int() { [ "$_stop" = 1 ] && exit 130; _stop=1; echo; log "interrupted — stopping"; cleanup; exit 130; }
trap on_int INT TERM

# ---------------------------------------------------------------------------
# 1) clean slate — force-stop every app
# ---------------------------------------------------------------------------
log "force-stopping all apps"
for a in "${!PKG[@]}"; do _adb shell am force-stop "${PKG[$a]}"; done
_adb shell am force-stop com.example.llama
_adb shell am force-stop com.example.llama.cpuswap
_adb shell am kill-all >/dev/null 2>&1
key KEYCODE_WAKEUP
_adb shell svc power stayon true
_adb shell input swipe 540 1900 540 700 200   # dismiss lock if present
home

# ---------------------------------------------------------------------------
# 2) load the LLM app
# ---------------------------------------------------------------------------
if [ "$VARIANT" = mzcache ]; then
  _adb shell setprop debug.mzcache.step "$MZ_STEP"
  log "mzcache swapout step = $MZ_STEP"
fi
log "launch $LLM_PKG"
launch "$LLM_PKG"; sleep 3
BASELINE_GPU=0.0
if [ "$VARIANT" = mzcache ]; then
  BPID=$(pidof_llm)
  BGB=$(_adb shell su -c "cat /sys/devices/virtual/kgsl/kgsl/proc/$BPID/gpumem_mapped" 2>/dev/null | tr -d '\r')
  BASELINE_GPU=$(awk -v b="${BGB:-0}" 'BEGIN{ printf "%.3f", b/1073741824 }')
  log "GPU baseline (pre-load) = $BASELINE_GPU GB"
fi
log "load model + KV (tap Load, ~18s)"
tap "${LOAD_BTN[@]}"; sleep 18
LLM_PID=$(pidof_llm)
[ -z "$LLM_PID" ] && { log "ERROR: LLM app not running after load"; cleanup; exit 1; }
log "loaded, pid=$LLM_PID"   # stays foreground (no home) — plateau shown during steady

# ---------------------------------------------------------------------------
# 3) scrcpy + live trace (no dashed markers), then start recording
# ---------------------------------------------------------------------------
log "open scrcpy + trace windows"
# Cap scrcpy capture at the recording fps (SCRCPY_FPS, default = FPS) — mirroring
# faster than we record just wastes phone display-capture + encoder cycles. Lower
# SCRCPY_FPS (e.g. 15) to shed more phone-side load (the encoder is a separate
# block from the Adreno compute mzcache uses, so this is free, not a speedup).
"$SCRCPY_BIN" -s "$SERIAL" --max-size "$MAX_SIZE" --max-fps "$SCRCPY_FPS" \
  --window-x "$SCRCPY_X" --window-y "$SCRCPY_Y" --window-borderless \
  --window-title "mz-scrcpy" --no-audio --stay-awake $SCRCPY_LOCK \
  >/tmp/mz_scrcpy.log 2>&1 &
SCRCPY_PID=$!

TRACE_ARGS=(--serial "$SERIAL" --pkg "$LLM_PKG" \
  --width "$TRACE_W" --height "$TRACE_H" --window "$TRACE_WINDOW" --ymax "$YMAX" \
  --pos-x "$TRACE_X" --pos-y "$TRACE_Y" --frameless --only-kill)
if [ "$VARIANT" = mzcache ]; then
  TRACE_ARGS+=(--metric gpu --baseline-gpu "$BASELINE_GPU")
fi
MPLBACKEND=QtAgg python3 "$SCRIPT_DIR/$TRACE_PY" "${TRACE_ARGS[@]}" \
  >/tmp/mz_trace.log 2>&1 &
TRACE_PID=$!
sleep 5

# Safety cap on the recording length, derived from the schedule so that raising
# ROUNDS/DWELL_S/ROUND_PERIOD cannot silently truncate the video (the script kills
# ffmpeg at the end anyway; this is only a backstop).
_per_round=$(( ROUND_PERIOD > 0 ? ROUND_PERIOD : 300 ))
REC_S="${REC_S:-$(( STEADY_S + ROUNDS * _per_round + RETURN_WAIT + TAIL_S + 120 ))}"
log "start recording -> $OUT (cap ${REC_S}s)"
ffmpeg -y -f x11grab -framerate "$FPS" -video_size "${REGION_W}x${REGION_H}" \
  -i "${DISPLAY}+0,${MARGIN_Y}" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
  -t "$REC_S" "$OUT" >/tmp/mz_ffmpeg.log 2>&1 &
FF_PID=$!
sleep 1
kill -0 "$FF_PID" 2>/dev/null || { log "ERROR: ffmpeg failed — see /tmp/mz_ffmpeg.log"; cleanup; exit 1; }

log "steady ${STEADY_S}s (loaded LLM app in the foreground)"
sleep "$STEADY_S"

# ---------------------------------------------------------------------------
# 4) ROUNDS rounds of the random app-switch gauntlet
# ---------------------------------------------------------------------------
for r in $(seq 1 "$ROUNDS"); do
  seq_apps=(${ROUND_ORDER[$(( (r - 1) % ${#ROUND_ORDER[@]} ))]})
  log "=== round $r/$ROUNDS: ${seq_apps[*]} ==="
  T_R0=$(date +%s)
  for name in "${seq_apps[@]}"; do
    if [ "$name" = visit ]; then
      log "  r$r: visit — LLM foreground ${VISIT_S}s (no Send)"
      if [ -n "$(pidof_llm)" ]; then
        llm_foreground; sleep "$VISIT_S"; check_alive "visit (r$r)"
      else
        log "     LLM dead — idling ${VISIT_S}s"; sleep "$VISIT_S"
      fi
      continue
    fi
    log "  r$r: $name (${PKG[$name]})"
    if ! _adb shell pm list packages 2>/dev/null | grep -q "${PKG[$name]}"; then
      log "     not installed — skipping"; continue
    fi
    home
    launch "${PKG[$name]}"
    case "$name" in
      camera) sleep 5; key KEYCODE_CAMERA; sleep $(( DWELL_S - 5 )) ;;   # 5s preview + shutter + rest
      pubg)   sleep "$PUBG_DWELL_S" ;;                                   # game — longer load
      roblox) sleep 12; tap "${ROBLOX_AVATAR_BTN[@]}"; sleep 3          # load, open Avatar tab (3D avatar)
              swipe 540 1600 540 800 300; sleep $(( DWELL_S > 15 ? DWELL_S - 15 : 3 )) ;;   # scroll + fill
      youtube) sleep 4; tap "${YT_VIDEO_BTN[@]}"                         # play the 2nd video (top slot is an ad)
               sleep $(( DWELL_S > 4 ? DWELL_S - 4 : 3 )) ;;
      tiktok|instagram)                                                  # feed apps: swipe for the whole dwell
              sleep 2; sw_t=2
              while [ "$sw_t" -lt "$DWELL_S" ]; do feed_swipe; sleep 2; sw_t=$(( sw_t + 2 )); done ;;
      *)      sleep "$DWELL_S" ;;
    esac
    check_alive "$name (r$r)"
  done
  log "  r$r: return to LLM + Send"
  llm_send                        # end-of-round return -> swapin_generate
  sleep "$RETURN_WAIT"
  if [ "$ROUND_PERIOD" -gt 0 ] && [ "$r" -lt "$ROUNDS" ]; then   # pace every round but the last
    ELAPSED=$(( $(date +%s) - T_R0 )); PAD=$(( ROUND_PERIOD - ELAPSED ))
    log "  r$r elapsed=${ELAPSED}s pad=${PAD}s"
    [ "$PAD" -gt 0 ] && sleep "$PAD"
  fi
done

# ---------------------------------------------------------------------------
# 5) tail + stop
# ---------------------------------------------------------------------------
NEWPID=$(pidof_llm)
echo "----------------------------------------------------------------------"
if [ -n "$NEWPID" ] && [ "$NEWPID" = "$LLM_PID" ]; then
  log "RESULT: SURVIVED all $ROUNDS rounds — pid $LLM_PID unchanged"
else
  log "RESULT: LLM app restarted — was pid $LLM_PID, now '${NEWPID:-dead}'"
fi
echo "----------------------------------------------------------------------"
log "tail ${TAIL_S}s"
sleep "$TAIL_S"
log "stopping recording + windows"
cleanup
trap - INT TERM
log "DONE -> $OUT"
