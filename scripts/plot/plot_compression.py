#!/usr/bin/env python3
"""plot_compression.py — Figure 14 (left two panels): compression space savings and
TTFT for CacheGen vs FLEXGEN-8bit.

Left  : grouped bars, x = model, y = space savings (%), one bar per algorithm.
Right : TTFT (s) vs remaining-in-memory (%), one line per algorithm, x in
        [0, 50], y auto-scaled to the data (paper Figure 14 middle panel).

Reads results/compression/space_savings.csv (model,algo,space_savings_pct) and
results/compression/ttft.csv (algo,model,remaining_pct,rep,ttft_ms).

Usage:
  python3 plot_compression.py [--data-dir results/compression] [--out <png>]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from ae_style import apply_style, finish, median, read_csv_rows

# CacheGen = orange, FLEXGEN-8bit = green (paper Figure 14 convention).
ALGOS = ["cachegen", "flexgen8bit"]
COLORS = {"cachegen": "#ED7D31", "flexgen8bit": "#2E7D32"}
LABELS = {"cachegen": "CacheGen", "flexgen8bit": "8bit Quantization"}
MARKERS = {"cachegen": "o", "flexgen8bit": "s"}
MODEL_LABELS = {"qwen3_0.6B": "Qwen3\n0.6B", "exaone4_1.2B": "EXAONE4\n1.2B"}


def plot_space(ax, path: Path):
    rows = read_csv_rows(path)
    models, vals = [], defaultdict(dict)
    for r in rows:
        vals[r["model"]][r["algo"]] = float(r["space_savings_pct"])
    models = list(vals.keys())
    n = len(ALGOS)
    width = 0.8 / n
    for k, algo in enumerate(ALGOS):
        xs = [i + (k - (n - 1) / 2) * (width + 0.02) for i in range(len(models))]
        ys = [vals[m].get(algo, 0.0) for m in models]
        bars = ax.bar(xs, ys, width, color=COLORS[algo], edgecolor="black",
                      linewidth=0.8, label=LABELS[algo])
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.5, f"{y:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_xlabel("Model")
    ax.set_ylabel("Space Savings (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)


def plot_ttft(ax, path: Path):
    rows = read_csv_rows(path)
    series = defaultdict(lambda: defaultdict(list))  # algo -> pct -> [ms]
    for r in rows:
        series[r["algo"]][int(r["remaining_pct"])].append(float(r["ttft_ms"]))
    for algo in ALGOS:
        s = series.get(algo)
        if not s:
            continue
        xs = sorted(s)
        ys = [median(s[x]) / 1000.0 for x in xs]
        ax.plot(xs, ys, color=COLORS[algo], marker=MARKERS[algo],
                label=LABELS[algo], markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel("Remaining in Mem (%)")
    ax.set_ylabel("TTFT (s)")
    ax.set_xlim(-2, 52)
    ax.set_xticks([0, 25, 50])
    ax.legend(loc="upper right", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="results/compression", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    out = args.out or args.data_dir / "compression.png"

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    space_csv = args.data_dir / "space_savings.csv"
    ttft_csv = args.data_dir / "ttft.csv"
    if space_csv.exists():
        plot_space(axes[0], space_csv)
    else:
        axes[0].set_title("(no space_savings.csv)")
    if ttft_csv.exists():
        plot_ttft(axes[1], ttft_csv)
    else:
        axes[1].set_title("(no ttft.csv)")
    fig.tight_layout()
    finish(fig, out)


if __name__ == "__main__":
    main()
