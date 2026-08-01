#!/usr/bin/env python3
"""Live memory trace for the mzCache demo.

Streams the llama.android process's memory over adb (no per-tick app change)
and tails its logcat, annotating the trace the instant mzCache swaps out /
swaps in. Run the live window next to a scrcpy mirror and screen-record both.

  # live window (for the demo recording); --serial defaults to $ADB_SERIAL
  python3 mem_trace.py

  # headless capture to a PNG (to verify the pipeline)
  python3 mem_trace.py --record 40 --out /tmp/trace.png

Metric (--metric):
  pss       llama.android TOTAL PSS from dumpsys meminfo (default). Includes the
            SVM KV cache, which lands under "Graphics" and is NOT in /proc RSS.
  memavail  system MemAvailable from /proc/meminfo (device-wide headroom).
  gpu       the process's mapped GPU memory from kgsl (root): weights (GPU) +
            SVM KV cache + always-SVM arenas. Subtract the app's launch-time GPU
            baseline (--measure-baseline -> --baseline-gpu) to get the working
            set; it drops as KV/arenas are offloaded on swapout. y-axis 0..6 GB.
            e.g.: launch app (no load) -> `--measure-baseline`; then
            `--metric gpu --baseline-gpu 0.9`.

Events it marks (already logged by the app):
  swapout   "swapout done: achieved ..."        (KV offloaded -> memory down)
  swapin    "swapin_generate TTFT ..."          (KV restored  -> memory up)
  trim      "onTrimMemory starts: memory: ..."  (the pressure trigger)
"""
import argparse
import os
import re
import subprocess
import threading
import time
from collections import deque

import matplotlib

# Palette (paper green/orange/blue, CVD-checked).
C_LINE    = "#2E7D32"  # memory line
C_SWAPOUT = "#ED7D31"  # swapout events
C_REQ     = "#8C8C8C"  # "Request arrived" (Send pressed), dashed
C_FIRST   = "#4472C4"  # "First token generated"
C_KILLED  = "#D62728"  # process death (red, dashed)

YMAX_DEFAULT = 12.0    # fixed y-axis top (GB) so the scale never rescales
_ONLY_KILL = False     # --only-kill: draw only the "killed" marker (set in main)

# (log-substring, label, color, linestyle). "Request arrived" is logged on the
# main thread the instant Send is pressed (not at the later native prefill), so
# the gap to "First token generated" is the TTFT the user actually waits.
EVENT_MATCH = [
    ("Request arrived: Send pressed", "Request arrived",       C_REQ,     "--"),
    ("chat first-visible token",      "First token generated", C_FIRST,   "-"),
    ("swapout done: achieved",        "evict",               C_SWAPOUT, "-"),
]

YLABEL = {"pss": "Memory (GB)", "memavail": "Available RAM (GB)",
          "gpu": "Memory (GB)"}
TITLE  = {"pss": "mzCache (weight + KV + comp KV)",
          "memavail": "Device available memory — mzCache",
          "gpu": "mzCache (weight + KV + comp KV)"}

# Per-process mapped GPU memory on Adreno/kgsl (bytes). Everything mzcache holds
# lives here: weights (GPU), the SVM KV cache, and the always-SVM arenas — so
# gpumem_mapped minus the app's launch-time baseline is the working set, and it
# drops as the KV/arenas are offloaded to disk on swapout. Needs root.
KGSL_GPUMEM = "/sys/devices/virtual/kgsl/kgsl/proc"


def _needs_root(metric):
    return metric == "gpu"


def _mem_cmd(metric, pkg, dt):
    if metric == "gpu":
        return (f"while true; do P=$(pidof {pkg}); "
                f"F={KGSL_GPUMEM}/$P/gpumem_mapped; "
                f'if [ -n "$P" ] && [ -r "$F" ]; then echo "GPU $(cat $F)"; fi; '
                f"echo ===TICK===; sleep {dt}; done")
    if metric == "pss":
        return (f"while true; do dumpsys meminfo {pkg} 2>/dev/null "
                f"| grep -m1 'TOTAL PSS'; echo ===TICK===; sleep {dt}; done")
    return f"while true; do cat /proc/meminfo; echo ===TICK===; sleep {dt}; done"


def _parse_mem(metric, line):
    """Return the sampled value in GB, or None if this line isn't it."""
    if metric == "gpu":
        return int(line.split()[1]) / 1073741824.0 if line.startswith("GPU ") else None
    if metric == "pss":
        m = re.search(r"TOTAL PSS:\s*(\d+)", line)
        return int(m.group(1)) / 1048576.0 if m else None
    if line.startswith("MemAvailable:"):
        return int(line.split()[1]) / 1048576.0
    return None


class Sampler:
    """Background adb pollers writing into shared, lock-guarded buffers."""

    def __init__(self, serial, pkg, hz, metric, baseline_gpu=0.0):
        self.serial = serial
        self.pkg = pkg
        self.dt = 1.0 / hz
        self.metric = metric
        self.baseline_gpu = baseline_gpu   # GB to subtract for --metric gpu
        self.lock = threading.Lock()
        self.samples = deque(maxlen=100_000)   # (t, value_gb)
        self.events = []                       # (t, label, color)
        self.t0 = None
        self.alive = False                     # is the process currently up?
        self._stop = threading.Event()

    def _adb(self, *args):
        return ["adb", "-s", self.serial, *args]

    def start(self):
        threading.Thread(target=self._poll_mem, daemon=True).start()
        threading.Thread(target=self._poll_log, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _on_value(self, val):
        t = time.time()
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            self.samples.append((t, val))
            self.alive = True   # relaunch just resumes; no marker (user's ask)

    def _on_dead(self):
        # A poll cycle produced no value: the process is not running. Drop the
        # line to 0 and, on the alive->dead edge, drop a red dashed "killed"
        # marker. Ticks keep coming (the adb shell loop outlives the app), so
        # the line stays live at 0 instead of freezing until relaunch.
        t = time.time()
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            if self.alive:
                self.events.append((t, "killed", C_KILLED, "--"))
                self.alive = False
            self.samples.append((t, 0.0))

    def _poll_mem(self):
        # One persistent shell loop keeps per-sample latency low; each iteration
        # ends with a ===TICK=== sentinel, so a tick with no value line means the
        # process is gone. The host timestamps values as they arrive.
        cmd_str = _mem_cmd(self.metric, self.pkg, self.dt)
        # gpu reads kgsl under /sys (root); the loop has no single quotes so it
        # can be single-quoted for `su -c`. pss/memavail run without root.
        cmd = (self._adb("shell", f"su -c '{cmd_str}'") if _needs_root(self.metric)
               else self._adb("shell", cmd_str))
        while not self._stop.is_set():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
                got = False
                for line in p.stdout:
                    if self._stop.is_set():
                        break
                    if "===TICK===" in line:
                        if not got:
                            self._on_dead()
                        got = False
                        continue
                    val = _parse_mem(self.metric, line)
                    if val is not None:
                        if self.metric == "gpu":
                            val = max(0.0, val - self.baseline_gpu)
                        got = True
                        self._on_value(val)
                p.terminate()
            except Exception:
                pass
            if not self._stop.is_set():
                time.sleep(0.5)  # device unplugged? retry

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


# Font/line sizes are tuned for a ~500px-tall figure; everything scales
# linearly with the actual height so a 1080px window gets ~2.2x larger fonts.
FONT_BASELINE_PX = 500.0


def _style(ax, dark, metric, scale=1.0):
    fg = "#EAEAEA" if dark else "#222222"
    grid = "#3A3A3A" if dark else "#DDDDDD"
    ax.set_xlabel("time (s)", fontsize=13 * scale, color=fg)
    ax.set_ylabel(YLABEL[metric], fontsize=13 * scale, color=fg)
    ax.set_title(TITLE[metric], fontsize=15 * scale, color=fg)
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


def run_live(sampler, window, dark, metric, ymax, width, height, dpi,
             pos_x=None, pos_y=None, frameless=False, no_dashed=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    if dark:
        plt.style.use("dark_background")
    plt.rcParams["toolbar"] = "None"   # clean window ~= figure size for recording
    scale = height / FONT_BASELINE_PX
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    (line,) = ax.plot([], [], color=C_LINE, linewidth=2.4 * scale)
    _style(ax, dark, metric, scale)
    ax.set_ylim(0, ymax)   # fixed scale, never rescales
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


def run_record(sampler, seconds, out, dark, metric, ymax, width, height, dpi,
               no_dashed=False):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if dark:
        plt.style.use("dark_background")
    scale = height / FONT_BASELINE_PX
    print(f"recording {seconds}s ... (drive load/swapout/swapin on the phone now)")
    time.sleep(seconds)
    samples, events, t0 = sampler.snapshot()
    sampler.stop()
    if not samples:
        print("no samples — is the device connected and the app running?")
        return
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    xs = [t - t0 for t, _ in samples]
    ys = [g for _, g in samples]
    ax.plot(xs, ys, color=C_LINE, linewidth=2.4 * scale)
    _style(ax, dark, metric, scale)
    ax.set_xlim(0, max(xs))
    ax.set_ylim(0, ymax)   # fixed scale
    _draw_events(ax, events, t0, set(), scale, no_dashed)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    print(f"saved {out}  ({len(samples)} samples, {len(events)} events, "
          f"{min(ys):.2f}-{max(ys):.2f} GB)")


def measure_gpu_baseline(serial, pkg):
    """Read gpumem_mapped once (run with the app just launched, model NOT loaded)
    to get the process's baseline GPU memory to pass as --baseline-gpu."""
    cmd = ["adb", "-s", serial, "shell",
           f"su -c 'P=$(pidof {pkg}); cat {KGSL_GPUMEM}/$P/gpumem_mapped'"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    try:
        gb = int(out) / 1073741824.0
        print(f"gpumem_mapped baseline for {pkg}: {gb:.3f} GB ({out} bytes)")
        print(f"  -> pass  --baseline-gpu {gb:.3f}")
    except ValueError:
        print(f"could not read gpumem_mapped (root? app launched?): {out!r}")


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
    ap.add_argument("--pkg", default="com.example.llama", help="app package")
    ap.add_argument("--metric", choices=["pss", "memavail", "gpu"], default="pss",
                    help="pss = app TOTAL PSS (default); memavail = system free RAM; "
                         "gpu = process kgsl gpumem_mapped (weights+SVM KV+arenas, root)")
    ap.add_argument("--baseline-gpu", type=float, default=0.0,
                    help="GB to subtract for --metric gpu (the app's launch-time GPU "
                         "memory before the model loads; see --measure-baseline)")
    ap.add_argument("--measure-baseline", action="store_true",
                    help="print the current gpumem_mapped (GB) and exit (launch the "
                         "app first, before loading the model), then use --baseline-gpu")
    ap.add_argument("--hz", type=float, default=0.0,
                    help="sample rate (default: 3 pss, 2 gpu, 8 memavail)")
    ap.add_argument("--window", type=float, default=60.0, help="live x-axis width (s)")
    ap.add_argument("--ymax", type=float, default=None,
                    help="fixed y-axis top in GB (default: 6 for gpu, else 12)")
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
    ap.add_argument("--out", default="/tmp/mem_trace.png", help="PNG for --record")
    args = ap.parse_args()
    globals()["_ONLY_KILL"] = args.only_kill
    if not args.serial:
        ap.error("no device selected: export ADB_SERIAL or pass --serial (see 'adb devices')")

    if args.measure_baseline:
        measure_gpu_baseline(args.serial, args.pkg)
        return

    # gpu polls gpumem_mapped via a persistent `su` loop; keep it gentle (2 Hz)
    # so it barely contends with the swapin's kgsl SVM allocs/decompression.
    hz = args.hz or {"pss": 3.0, "gpu": 2.0}.get(args.metric, 8.0)
    ymax = args.ymax if args.ymax is not None else (6.0 if args.metric == "gpu" else YMAX_DEFAULT)
    sampler = Sampler(args.serial, args.pkg, hz, args.metric, args.baseline_gpu)
    sampler.start()
    if args.record:
        run_record(sampler, args.record, args.out, args.dark, args.metric, ymax,
                   args.width, args.height, args.dpi, args.no_dashed)
    else:
        run_live(sampler, args.window, args.dark, args.metric, ymax,
                 args.width, args.height, args.dpi, args.pos_x, args.pos_y,
                 args.frameless, args.no_dashed)


if __name__ == "__main__":
    main()
