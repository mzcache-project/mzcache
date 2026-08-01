#!/usr/bin/env python3
"""plot_overlap.py — TTFT with different overlapping strategies (paper
Figure 11a).

Reads results/overlap_breakdown/overlap_mzcache.csv. For every prefill_type the total
swap-in TTFT is alloc + restore + prefill as printed by mz_prefill (the
overlapped portions are folded into the respective columns), medianed across
reps: mz_NO (3) / mz_AR (2) / mz_RP (1) / mz_ARP = mzCache (0).

Usage:
  python3 plot_overlap.py [--data-dir results/overlap_breakdown] [--out <png>]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from ae_style import (OVERLAP_COLORS, OVERLAP_LABELS, OVERLAP_ORDER,
                      apply_style, ctx_label, finish, median, read_csv_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="results/overlap_breakdown", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    out = args.out or args.data_dir / "overlap.png"

    apply_style()
    rows = read_csv_rows(args.data_dir / "overlap_mzcache.csv")
    if not rows:
        raise SystemExit("no rows — run scripts/overlap_breakdown/run_overlap.sh first")

    acc = defaultdict(lambda: defaultdict(list))  # {ctx: {type: [total_ms]}}
    for row in rows:
        acc[int(row["ctx"])][int(row["type"])].append(float(row["total_ms"]))
    ctxs = sorted(acc)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    n = len(OVERLAP_ORDER)
    width = 0.8 / n
    for k, t in enumerate(OVERLAP_ORDER):
        xs = [i + (k - (n - 1) / 2) * (width + 0.008) for i in range(len(ctxs))]
        ys = [median(acc[c].get(t, [])) / 1000.0 if acc[c].get(t) else 0.0
              for c in ctxs]  # ms -> s
        ax.bar(xs, ys, width, color=OVERLAP_COLORS[t], edgecolor="black",
               linewidth=0.8, label=OVERLAP_LABELS[t])
    ax.set_xticks(range(len(ctxs)))
    ax.set_xticklabels([ctx_label(c) for c in ctxs])
    ax.set_xlabel("Context Length")
    ax.set_ylabel("TTFT (s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    finish(fig, out)


if __name__ == "__main__":
    main()
