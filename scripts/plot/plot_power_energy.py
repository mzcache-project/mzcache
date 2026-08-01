#!/usr/bin/env python3
"""plot_power_energy.py — power consumption during restoration + prefill (paper
Figure 13).

Reads the two battery-gauge traces pulled by scripts/power_energy/run_power.sh
(schema Timestamp_us,Voltage_uV,Current_uA,Power_W) and renders both power
curves over time with the area under each curve shaded (total energy).

Usage:
  python3 plot_power_energy.py [--data-dir results/power_energy] [--out <png>]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ae_style import COLORS, LABELS, MARKERS, apply_style, finish, read_csv_rows


def load_trace(path: Path):
    rows = read_csv_rows(path)
    if not rows:
        return [], []
    t0 = float(rows[0]["Timestamp_us"])
    ts = [(float(r["Timestamp_us"]) - t0) / 1e6 for r in rows]
    ps = [float(r["Power_W"]) for r in rows]
    # pad with zero endpoints so the shaded area closes on the time axis
    return [0.0] + ts + [ts[-1] + 0.1], [0.0] + ps + [0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="results/power_energy", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    out = args.out or args.data_dir / "power_energy.png"

    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    plotted = False
    for system, fname in [("partial_offload", "power_partial_offload.csv"),
                          ("mzcache", "power_mzcache.csv")]:
        path = args.data_dir / fname
        if not path.exists():
            print(f"note: {path} missing, skipping")
            continue
        ts, ps = load_trace(path)
        if not ts:
            continue
        ax.plot(ts, ps, color=COLORS[system], marker=MARKERS[system],
                markersize=5, label=LABELS[system], zorder=3)
        ax.fill_between(ts, ps, color=COLORS[system], alpha=0.40, linewidth=0,
                        zorder=2)
        plotted = True
    if not plotted:
        raise SystemExit(f"no power traces under {args.data_dir} — "
                         "run scripts/power_energy/run_power.sh first")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    finish(fig, out)


if __name__ == "__main__":
    main()
