#!/usr/bin/env python3
"""Live memory trace for the CPU-swap baseline app (llamaandroid) demo.

The baseline app (com.example.llama.cpuswap) has no mzcache: weights are
mmapped (file-backed, dropped by the kernel under pressure) and the KV cache is
anonymous memory the kernel swaps to zram (zstd). Its true footprint is the
weight pages that are resident PLUS the KV — resident and the compressed part
still sitting in zram.

Instead of dumpsys TOTAL PSS (which lumps in dalvik / native heap / graphics and
drifts as pages migrate), this isolates exactly the two regions from a single
/proc/<pid>/smaps pass (root), the same way v0.8's mz_trace_sampler.sh did:

  weight = Rss of the *.gguf file-backed VMAs (mmapped weights)
  kv     = Rss + Swap of the large anon KV VMA (>=200MB; the OS names it
           [anon:scudo:secondary] or [anon:libc_malloc] depending on the device)
  value  = weight_rss + kv_rss + kv_swap * ZRAM_RATIO

ZRAM_RATIO is the fraction of the KV that survives zstd in zram — author-
measured 0.9196 for Qwen3-0.6B (default; its KV compresses only ~8%) and
0.907 for EXAONE-4.0-1.2B (--zram-ratio 0.907). Requires root (reads another
app's smaps).

No swapout/swapin markers (nothing is app-triggered) — only a red dashed
"killed" marker when the LMK kills the process; the line drops to 0 and resumes
on relaunch.

  python3 mem_trace_baseline.py                  # --serial defaults to $ADB_SERIAL
  python3 mem_trace_baseline.py --record 90 --out /tmp/t.png
"""
import argparse
import os
import subprocess
import tempfile
import threading
import time
from collections import deque

import matplotlib

C_LINE   = "#2E7D32"   # memory line
C_REQ    = "#8C8C8C"   # "Request arrived" (Send pressed), dashed
C_FIRST  = "#4472C4"   # "First token generated"
C_KILLED = "#D62728"   # process death (red, dashed)
ZRAM_RATIO = 0.9196    # fraction of the KV that survives zstd zram compression
YMAX_DEFAULT = 12.0    # fixed y-axis top (GB)

# (log-substring, label, color, linestyle) — both variants log these. "Request
# arrived" is logged the instant Send is pressed (main thread), so the gap to
# "First token generated" is the TTFT the user is waiting on.
EVENT_MATCH = [
    ("Request arrived: Send pressed", "Request arrived",       C_REQ,   "--"),
    ("chat first-visible token",      "First token generated", C_FIRST, "-"),
]
FONT_BASELINE_PX = 500.0  # fonts/lines scale linearly with figure height vs this
_ONLY_KILL = False        # --only-kill: draw only the "killed" marker (set in main)

DEV_SCRIPT = "/data/local/tmp/mz_smaps_stream.sh"

# On-device sampler: one smaps pass per tick, emits "pid weight_rss kv_rss
# kv_swap total_kb" (kB). pid=0 => process not running. Mirrors v0.8
# mz_trace_sampler.sh but streams to stdout instead of a CSV.
DEV_SCRIPT_BODY = r'''#!/system/bin/sh
PKG=$1; IV=${2:-0.4}; RATIO=${3:-0.9196}
# Bound each smaps read: under peak memory pressure (the demo's Roblox/camera
# tipping point) reading a multi-GB, actively-swapping process's smaps can block
# in the kernel for seconds and freeze the sampler. timeout -s KILL kills a stuck
# read after 1.5s; we then emit a "-1" sentinel so the loop keeps ticking (the
# host skips that tick). Falls back to a plain read if timeout is unavailable.
if command -v timeout >/dev/null 2>&1; then TMO="timeout -s KILL 1.5"; else TMO=""; fi
while true; do
  PID=$(pidof "$PKG"); set -- $PID; PID=$1
  if [ -n "$PID" ] && [ -r "/proc/$PID/smaps" ]; then
    LINE=$($TMO awk -v pid="$PID" -v kvmin=200000 -v ratio="$RATIO" '
      function flush() {
        if (path ~ /\.gguf/) { wr += rss }
        else if (size >= kvmin && (path == "" || path ~ /scudo:secondary/ || path ~ /libc_malloc/)) { kr += rss; ks += swap }
      }
      /^[0-9a-f]+-[0-9a-f]+ / {
        if (seen) flush();
        seen=1; size=0; rss=0; swap=0; path="";
        if (NF>=6) { path=$6; for(i=7;i<=NF;i++) path=path" "$i } next
      }
      $1=="Size:"{size=$2} $1=="Rss:"{rss=$2} $1=="Swap:"{swap=$2}
      END { if (seen) flush(); printf "%d %d %d %d %.0f\n", pid, wr, kr, ks, wr+kr+ks*ratio }
    ' "/proc/$PID/smaps")
    if [ -n "$LINE" ]; then echo "$LINE"; else echo "-1 0 0 0 0"; fi
  else
    echo "0 0 0 0 0"
  fi
  sleep "$IV"
done
'''


class Sampler:
    def __init__(self, serial, pkg, hz, zram_ratio=ZRAM_RATIO):
        self.serial = serial
        self.pkg = pkg
        self.zram_ratio = zram_ratio
        self.dt = 1.0 / hz
        self.lock = threading.Lock()
        self.samples = deque(maxlen=100_000)   # (t, value_gb)
        self.events = []                       # (t, "killed")
        self.t0 = None
        self.alive = False
        self._stop = threading.Event()
        self._push_script()

    def _adb(self, *args):
        return ["adb", "-s", self.serial, *args]

    def _push_script(self):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(DEV_SCRIPT_BODY)
            local = f.name
        subprocess.run(self._adb("push", local, DEV_SCRIPT),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start(self):
        threading.Thread(target=self._poll, daemon=True).start()
        threading.Thread(target=self._poll_log, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _on_value(self, val):
        t = time.time()
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            self.samples.append((t, val))
            self.alive = True

    def _on_dead(self):
        t = time.time()
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            if self.alive:
                self.events.append((t, "killed", C_KILLED, "--"))
                self.alive = False
            self.samples.append((t, 0.0))

    def _poll(self):
        cmd = self._adb("shell", "su", "-c",
                        f"sh {DEV_SCRIPT} {self.pkg} {self.dt} {self.zram_ratio}")
        while not self._stop.is_set():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
                for line in p.stdout:
                    if self._stop.is_set():
                        break
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    pid = parts[0]
                    if pid == "-1":       # smaps read stalled (timed out) — skip this tick
                        continue
                    total_kb = float(parts[4])
                    if pid == "0":
                        self._on_dead()
                    else:
                        self._on_value(total_kb / 1048576.0)
                p.terminate()
            except Exception:
                pass
            if not self._stop.is_set():
                time.sleep(0.5)

    def _poll_log(self):
        while not self._stop.is_set():
            try:
                p = subprocess.Popen(self._adb("logcat", "-T", "1"),
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
                for line in p.stdout:
                    if self._stop.is_set():
                        break
                    for needle, label, color, ls in EVENT_MATCH:
                        if needle in line:
                            t = time.time()
                            with self.lock:
                                self.events.append((t, label, color, ls))
                            break
                p.terminate()
            except Exception:
                pass
            if not self._stop.is_set():
                time.sleep(0.5)

    def snapshot(self):
        with self.lock:
            return list(self.samples), list(self.events), self.t0


def _style(ax, dark, scale=1.0):
    fg = "#EAEAEA" if dark else "#222222"
    grid = "#3A3A3A" if dark else "#DDDDDD"
    ax.set_xlabel("time (s)", fontsize=13 * scale, color=fg)
    ax.set_ylabel("Memory (GB)", fontsize=13 * scale, color=fg)
    ax.set_title("OS paging baseline (weight + KV + comp KV)", fontsize=15 * scale, color=fg)
    ax.grid(True, color=grid, linewidth=0.8 * scale)
    ax.tick_params(colors=fg, labelsize=11 * scale)
    for s in ax.spines.values():
        s.set_color(grid)


def _draw_events(ax, events, t0, seen, scale=1.0, hide_dashed=False):
    for t, label, color, ls in events:
        if _ONLY_KILL and label != "killed":
            continue
        if hide_dashed and ls == "--":
            continue
        key = (round(t, 3), label)
        if key in seen:
            continue
        seen.add(key)
        x = t - t0
        # Dashed event line only — labels intentionally omitted (line, no text).
        ax.axvline(x, color=color, linewidth=1.8 * scale, alpha=0.9, linestyle=ls)


def _place_window(fig, pos_x, pos_y, frameless=False):
    """Best-effort place the live window at (pos_x, pos_y) — and strip its title
    bar when frameless — so it sits flush against the scrcpy mirror with no
    banner offset (no xdotool needed). Works on the Qt and Tk backends (a no-op
    on any other). Call before plt.show()."""
    if pos_x is None and pos_y is None and not frameless:
        return
    try:
        win = fig.canvas.manager.window
    except Exception:
        return
    if frameless:   # drop the "Figure 1" title bar so the plot starts at the top
        try:
            from matplotlib.backends.qt_compat import QtCore
            win.setWindowFlags(win.windowFlags() | QtCore.Qt.FramelessWindowHint)
        except Exception:
            try:
                win.overrideredirect(True)   # Tk
            except Exception:
                pass
    if pos_x is not None or pos_y is not None:
        x, y = int(pos_x or 0), int(pos_y or 0)
        for mover in (lambda: win.move(x, y),                # Qt (QMainWindow)
                      lambda: win.wm_geometry(f"+{x}+{y}")):  # Tk
            try:
                mover()
                return
            except Exception:
                continue


def run_live(sampler, window, dark, ymax, width, height, dpi,
             pos_x=None, pos_y=None, frameless=False, no_dashed=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    if dark:
        plt.style.use("dark_background")
    plt.rcParams["toolbar"] = "None"   # clean window ~= figure size for recording
    scale = height / FONT_BASELINE_PX
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    (line,) = ax.plot([], [], color=C_LINE, linewidth=2.4 * scale)
    _style(ax, dark, scale)
    ax.set_ylim(0, ymax)
    seen = set()

    def update(_):
        samples, events, t0 = sampler.snapshot()
        if not samples or t0 is None:
            return
        xs = [t - t0 for t, _ in samples]
        ys = [g for _, g in samples]
        line.set_data(xs, ys)
        now = xs[-1]
        ax.set_xlim(max(0, now - window), max(window, now))
        ax.set_ylim(0, ymax)
        _draw_events(ax, events, t0, seen, scale, no_dashed)

    _ = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.tight_layout()
    _place_window(fig, pos_x, pos_y, frameless)
    plt.show()


def run_record(sampler, seconds, out, dark, ymax, width, height, dpi,
               no_dashed=False):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if dark:
        plt.style.use("dark_background")
    scale = height / FONT_BASELINE_PX
    print(f"recording {seconds}s ... (drive load / memory pressure / kill on the phone)")
    time.sleep(seconds)
    samples, events, t0 = sampler.snapshot()
    sampler.stop()
    if not samples:
        print("no samples — device connected? app running? root (su) available?")
        return
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    xs = [t - t0 for t, _ in samples]
    ys = [g for _, g in samples]
    ax.plot(xs, ys, color=C_LINE, linewidth=2.4 * scale)
    _style(ax, dark, scale)
    ax.set_xlim(0, max(xs))
    ax.set_ylim(0, ymax)
    _draw_events(ax, events, t0, set(), scale, no_dashed)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    print(f"saved {out}  ({len(samples)} samples, {len(events)} kills, "
          f"{min(ys):.2f}-{max(ys):.2f} GB)")


def default_serial():
    """$ADB_SERIAL, else the only attached device — never a hard-coded serial."""
    env = os.environ.get("ADB_SERIAL")
    if env:
        return env
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None
    devs = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip().endswith("device")]
    return devs[0] if len(devs) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=default_serial(),
                    help="adb device serial (default: $ADB_SERIAL, or the only attached device)")
    ap.add_argument("--pkg", default="com.example.llama.cpuswap", help="baseline app package")
    ap.add_argument("--hz", type=float, default=2.5, help="sample rate (smaps pass ~0.2s)")
    ap.add_argument("--zram-ratio", type=float, default=ZRAM_RATIO,
                    help="zstd residual of swapped KV (Qwen3-0.6B 0.9196, EXAONE-4.0-1.2B 0.907)")
    ap.add_argument("--window", type=float, default=60.0, help="live x-axis width (s)")
    ap.add_argument("--ymax", type=float, default=YMAX_DEFAULT, help="fixed y-axis top (GB)")
    ap.add_argument("--width", type=int, default=1000, help="figure width in px")
    ap.add_argument("--height", type=int, default=500,
                    help="figure height in px (title/axis fonts scale with it)")
    ap.add_argument("--pos-x", type=int, default=None,
                    help="live-window x position in px (place flush beside scrcpy)")
    ap.add_argument("--pos-y", type=int, default=None, help="live-window y position in px")
    ap.add_argument("--frameless", action="store_true",
                    help="strip the window title bar so the plot starts at the top edge")
    ap.add_argument("--no-dashed", action="store_true",
                    help="hide dashed markers (Request arrived / killed); keep solid ones")
    ap.add_argument("--only-kill", action="store_true",
                    help="draw ONLY the killed marker; hide every other event")
    ap.add_argument("--dpi", type=int, default=100, help="pixels per inch")
    ap.add_argument("--dark", action="store_true", help="dark theme")
    ap.add_argument("--record", type=float, metavar="SECONDS",
                    help="headless capture for N seconds, then save --out")
    ap.add_argument("--out", default="/tmp/mem_trace_baseline.png", help="PNG for --record")
    args = ap.parse_args()
    globals()["_ONLY_KILL"] = args.only_kill
    if not args.serial:
        ap.error("no device selected: export ADB_SERIAL or pass --serial (see 'adb devices')")

    sampler = Sampler(args.serial, args.pkg, args.hz, args.zram_ratio)
    sampler.start()
    if args.record:
        run_record(sampler, args.record, args.out, args.dark, args.ymax,
                   args.width, args.height, args.dpi, args.no_dashed)
    else:
        run_live(sampler, args.window, args.dark, args.ymax,
                 args.width, args.height, args.dpi, args.pos_x, args.pos_y,
                 args.frameless, args.no_dashed)


if __name__ == "__main__":
    main()
