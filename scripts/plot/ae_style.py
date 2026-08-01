"""Shared matplotlib style for the mzCache AE plots.

Approximates the paper's figure conventions (series colors, markers, bold
axis labels, dashed grid). The categorical palettes below were checked for
lightness band, chroma, and color-vision-deficiency separation; series
identity is additionally carried by distinct markers / bar order + legend,
never by color alone.
"""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Figure 9 series (paper order: OS Paging, Partial Offload, mzCache) ------
SYSTEMS = ["os_paging", "partial_offload", "mzcache"]
COLORS = {
    "os_paging":       "#2E7D32",
    "partial_offload": "#ED7D31",
    "mzcache":         "#4472C4",
}
MARKERS = {
    "os_paging":       "^",
    "partial_offload": "o",
    "mzcache":         "s",
}
LABELS = {
    "os_paging":       "OS Paging (zRAM)",
    "partial_offload": "Partial Offload",
    "mzcache":         "mzCache",
}

# --- Figure 11(a) overlap ladder (prefill_type -> label) ---------------------
OVERLAP_ORDER = [3, 2, 1, 0]
OVERLAP_LABELS = {
    3: "mz_NO",
    2: "mz_AR",
    1: "mz_RP",
    0: "mz_ARP (mzCache)",
}
OVERLAP_COLORS = {
    3: "#B7472A",
    2: "#ED7D31",
    1: "#DFA85F",
    0: "#4472C4",
}

EVICTION_LINE = "#D62728"  # red dash-dot marker for full eviction under OS paging


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })


def ctx_label(ctx: int) -> str:
    """32700 -> '32k', 16385 -> '16k', 8193 -> '8k'."""
    return f"{round(int(ctx) / 1024)}k"


def read_csv_rows(path):
    """Read a CSV with a header into a list of dicts (str values)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def median(values):
    return statistics.median(values) if values else None


def finish(fig, out_path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")
