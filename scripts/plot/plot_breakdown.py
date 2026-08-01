#!/usr/bin/env python3
"""plot_breakdown.py — swap-in latency breakdown, Partial Offload vs mz_NO
(paper Figure 10).

Reads results/overlap_breakdown/overlap_mzcache.csv (prefill_type=3 rows = mz_NO,
no overlap: Alloc/Restore/Prefill measured separately) and breakdown_po.csv
(partial offload's Total Alloc / Total Read / Prefill), medians across reps,
and renders three grouped-bar panels: Allocation, Reload, Prefill.

Usage:
  python3 plot_breakdown.py [--data-dir results/overlap_breakdown] [--out <png>]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from ae_style import COLORS, apply_style, ctx_label, finish, median, read_csv_rows

PANELS = [("Allocation", "alloc"), ("Reload", "reload"), ("Prefill", "prefill")]


def collect(rows, keymap):
    """rows + {panel_key: column} -> {ctx: {panel_key: median}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ctx = int(row["ctx"])
        for key, col in keymap.items():
            acc[ctx][key].append(float(row[col]))
    return {ctx: {k: median(v) for k, v in d.items()} for ctx, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="results/overlap_breakdown", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    out = args.out or args.data_dir / "breakdown.png"

    apply_style()
    mz_rows = [r for r in read_csv_rows(args.data_dir / "overlap_mzcache.csv")
               if int(r["type"]) == 3]
    po_rows = read_csv_rows(args.data_dir / "breakdown_po.csv")
    if not mz_rows or not po_rows:
        raise SystemExit("missing mz_NO (type=3) or PO rows — run scripts/overlap_breakdown/run_overlap.sh first")

    mz = collect(mz_rows, {"alloc": "alloc_ms", "reload": "restore_ms", "prefill": "prefill_ms"})
    po = collect(po_rows, {"alloc": "alloc_ms", "reload": "reload_ms", "prefill": "prefill_ms"})
    ctxs = sorted(set(mz) & set(po))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    x = range(len(ctxs))
    width = 0.34
    for ax, (title, key) in zip(axes, PANELS):
        ax.bar([i - width / 2 - 0.01 for i in x], [po[c][key] for c in ctxs],
               width, color=COLORS["partial_offload"], edgecolor="black",
               linewidth=0.8, label="Partial Offload")
        ax.bar([i + width / 2 + 0.01 for i in x], [mz[c][key] for c in ctxs],
               width, color=COLORS["mzcache"], edgecolor="black",
               linewidth=0.8, label="mz_NO")
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels([ctx_label(c) for c in ctxs])
        ax.set_ylim(bottom=0)
        if ax is axes[0]:
            ax.set_ylabel("Latency (ms)")
    fig.supxlabel("Context Length", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    finish(fig, out)


if __name__ == "__main__":
    main()
