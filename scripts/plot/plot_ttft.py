#!/usr/bin/env python3
"""plot_ttft.py — TTFT across remaining-memory levels (paper Figure 9).

Reads every ttft_*.csv under the data directory (produced by
scripts/ttft/run_*.sh; schema system,device,model,ctx,remaining_pct,rep,ttft_ms),
groups rows by (device, model), and renders one figure per group: a row of
subplots (one per context length) with the three systems as marked lines and
the per-config median across reps. A red dash-dot vertical line marks the
lowest remaining level reached by OS Paging (full eviction still retains the
mlocked token embedding plus the KV working set the file-cache pressure cannot
push out of RAM).

Usage:
  python3 plot_ttft.py [--data-dir results/ttft] [--out-dir results/ttft]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from ae_style import (COLORS, EVICTION_LINE, LABELS, MARKERS, SYSTEMS,
                      apply_style, ctx_label, finish, median, read_csv_rows)


def remaining_pct(row):
    """Remaining-in-memory % for one row.

    OS-paging rows carry the raw components, so recompute here (the formula lives
    at plot time, and `comp` can be retuned without re-measuring). Remaining
    EXCLUDES the mlocked token embedding and counts the swapped KV's compressed
    footprint in zram, so it floors above 0 even when the non-embedding weight and
    KV Rss are fully evicted:

        ((w_rss - emb) + kv_rss + (base_kv - kv_rss)*comp)
        --------------------------------------------------  x 100
                    ((base_w - emb) + base_kv)

    Rows without the component columns (other systems) fall back to remaining_pct.
    """
    try:
        wr = float(row["w_rss_kb"]); kr = float(row["kv_rss_kb"])
        e = float(row["emb_kb"]);    c = float(row["comp"])
        bw = float(row["base_w_kb"]); bk = float(row["base_kv_kb"])
        wn = max(0.0, wr - e); sw = max(0.0, bk - kr)
        den = (bw - e) + bk
        return 100.0 * (wn + kr + sw * c) / den if den > 0 else 0.0
    except (KeyError, ValueError, TypeError):
        return float(row["remaining_pct"])


def load(data_dir: Path):
    """-> {(device, model): {(system, ctx): {remaining_pct: [ttft_ms, ...]}}}"""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    files = sorted(data_dir.glob("ttft_*.csv"))
    for path in files:
        for row in read_csv_rows(path):
            try:
                key = (row["device"], row["model"])
                cfg = (row["system"], int(row["ctx"]))
                # Bin remaining-% to the nearest 5 so the reps of one level share
                # an x-point. OS-paging's evict-full endpoint lands at a slightly
                # different remaining each rep; without binning those become
                # separate near-vertical points. The swept levels of the other
                # systems (0/25/50/75/100) are multiples of 5 and unaffected.
                rem = round(remaining_pct(row) / 5.0) * 5.0
                groups[key][cfg][rem].append(float(row["ttft_ms"]))
            except (KeyError, ValueError):
                continue
    return groups, files


def system_points(system, series):
    """-> (xs, ys_in_seconds) for one (system, ctx) series.

    OS paging is a resident + full-eviction *pair*, not a swept curve: its
    evict-full endpoint lands at a slightly different remaining-% and a noisy
    TTFT each rep, so all evicted (<100%) reps are pooled into a single endpoint
    (median remaining, median TTFT). The swept systems keep every level."""
    if system == "os_paging":
        pts = [(x, median(series[x]) / 1000.0) for x in series if x >= 100]
        evicted = {x: v for x, v in series.items() if x < 100}
        if evicted:
            pooled = [t for v in evicted.values() for t in v]
            pts.append((median(sorted(evicted)), median(pooled) / 1000.0))
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]
    xs = sorted(series)
    return xs, [median(series[x]) / 1000.0 for x in xs]


def plot_group(device, model, data, out_dir: Path):
    ctxs = sorted({ctx for (_sys, ctx) in data})
    fig, axes = plt.subplots(1, len(ctxs), figsize=(3.4 * len(ctxs), 3.1))
    if len(ctxs) == 1:
        axes = [axes]

    for ax, ctx in zip(axes, ctxs):
        for system in SYSTEMS:
            series = data.get((system, ctx))
            if not series:
                continue
            xs, ys = system_points(system, series)
            ax.plot(xs, ys, color=COLORS[system], marker=MARKERS[system],
                    label=LABELS[system], markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3)
        os_series = data.get(("os_paging", ctx))
        if os_series:
            os_xs, _ = system_points("os_paging", os_series)
            if os_xs and min(os_xs) < 100:  # only meaningful with evicted points
                ax.axvline(min(os_xs), color=EVICTION_LINE, linestyle="-.",
                           linewidth=1.5, zorder=2)
        ax.set_title(ctx_label(ctx))
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(bottom=0)
        if ax is axes[0]:
            ax.set_ylabel("TTFT (s)")
    fig.supxlabel("Remaining in Memory (%)", fontweight="bold")

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.12))
    fig.suptitle(f"{device}, {model}", y=-0.08, fontsize=11, fontweight="normal")
    fig.tight_layout()

    out = out_dir / f"ttft_{device}_{model}.png"
    finish(fig, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="results/ttft", type=Path)
    ap.add_argument("--out-dir", default=None, type=Path)
    args = ap.parse_args()
    out_dir = args.out_dir or args.data_dir

    apply_style()
    groups, files = load(args.data_dir)
    if not groups:
        raise SystemExit(
            f"no ttft_*.csv with valid rows under {args.data_dir} — "
            "run scripts/ttft/run_*.sh first")
    print(f"loaded {len(files)} csv file(s)")
    for (device, model), data in sorted(groups.items()):
        plot_group(device, model, data, out_dir)


if __name__ == "__main__":
    main()
