#!/usr/bin/env python3
"""
Publication-quality SSM validation figures for the scapula pipeline.

Produces five separate figures (each print-ready at 300 dpi):
  1. compactness.pdf/png
  2. generalization.pdf/png
  3. specificity.pdf/png
  4. combined_validation.pdf/png   (3-panel, journal figure format)
  5. distance_to_mean.pdf/png

Usage:
  pip install matplotlib pandas numpy
  python3 plot_ssm_metrics.py \
    --plots-dir "/path/to/plots" \
    --out-dir   "/path/to/output"   # default: same as plots-dir
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#CCCCCC",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.6,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.8,
    "lines.markersize":  5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# Colorblind-safe palette (Wong 2011)
PALETTE = {
    1: "#0072B2",   # blue
    2: "#E69F00",   # amber
    3: "#009E73",   # green
    4: "#CC79A7",   # pink
    5: "#D55E00",   # orange
    6: "#56B4E9",   # sky
}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v", 6: "P"}


def find_passes(plots_dir: Path) -> list[int]:
    ns = []
    for f in sorted(plots_dir.glob("compactness_pass*.csv")):
        try:
            ns.append(int(f.stem.split("_pass")[1]))
        except ValueError:
            pass
    return sorted(ns)


def load(plots_dir: Path, prefix: str, passes: list[int]) -> dict[int, pd.DataFrame]:
    out = {}
    for n in passes:
        f = plots_dir / f"{prefix}_pass{n}.csv"
        if f.exists():
            out[n] = pd.read_csv(f)
    return out


def label(n: int) -> str:
    if n == 1:
        return f"SSM{n} (initial reference)"
    return f"SSM{n} (mean of SSM{n-1})"


def modes_at(df: pd.DataFrame, pct: float) -> int:
    cv = df["cumulative_variance_pct"].values
    idx = np.searchsorted(cv, pct)
    return int(df["mode"].iloc[min(idx, len(df)-1)])


# ── individual figure helpers ────────────────────────────────────────────────

def fig_compactness(compact: dict, passes: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for n in passes:
        df = compact[n]
        ax.plot(df["mode"], df["cumulative_variance_pct"],
                color=PALETTE[n], marker=MARKERS[n], markevery=3,
                label=label(n), zorder=3)
    for pct, ls in [(90, (4,2)), (95, (2,2)), (99, (1,2))]:
        ax.axhline(pct, color="#888888", lw=0.8, ls=(0, ls))
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 23,
                pct + 0.8, f"{pct}%", fontsize=7, color="#888888", va="bottom")
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("Compactness")
    ax.set_ylim(0, 102)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_generalization(gen: dict, passes: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for n in passes:
        df = gen[n]
        ax.plot(df["num_modes"], df["mean_error_mm"],
                color=PALETTE[n], marker=MARKERS[n], markevery=3,
                label=label(n), zorder=3)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Mean reconstruction error (mm)")
    ax.set_title("Generalization")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    return fig


def fig_specificity(spec: dict, passes: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for n in passes:
        df = spec[n]
        ax.plot(df["num_modes"], df["mean_specificity_mm"],
                color=PALETTE[n], marker=MARKERS[n], markevery=3,
                label=label(n), zorder=3)
        ax.fill_between(df["num_modes"],
                        df["mean_specificity_mm"] - df["std_specificity_mm"],
                        df["mean_specificity_mm"] + df["std_specificity_mm"],
                        alpha=0.15, color=PALETTE[n])
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Mean distance to nearest training shape (mm)")
    ax.set_title("Specificity")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    return fig


def fig_distance_to_mean(dtm: dict, passes: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    positions = list(range(1, len(passes) + 1))
    bp = ax.boxplot(
        [dtm[n]["mean_mm"].values for n in passes if n in dtm],
        positions=positions[:sum(n in dtm for n in passes)],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", lw=1.5),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        flierprops=dict(marker="x", color="#888888", markersize=4),
    )
    valid_passes = [n for n in passes if n in dtm]
    for patch, n in zip(bp["boxes"], valid_passes):
        patch.set_facecolor(PALETTE[n])
        patch.set_alpha(0.55)
    ax.set_xticks(positions[:len(valid_passes)])
    ax.set_xticklabels([f"SSM{n}" for n in valid_passes])
    ax.set_ylabel("Mean surface distance to SSM mean (mm)")
    ax.set_title("Distance to Mean Shape")
    fig.tight_layout()
    return fig


def fig_combined(compact: dict, gen: dict, spec: dict, passes: list) -> plt.Figure:
    """3-panel journal figure (7 x 2.4 in — fits a double-column layout)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    # — Compactness —
    ax = axes[0]
    for n in passes:
        df = compact[n]
        ax.plot(df["mode"], df["cumulative_variance_pct"],
                color=PALETTE[n], marker=MARKERS[n], markevery=3,
                label=f"SSM{n}", zorder=3)
    for pct, ls in [(90, (4,2)), (95, (2,2)), (99, (1,2))]:
        ax.axhline(pct, color="#AAAAAA", lw=0.7, ls=(0, ls))
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("(a) Compactness")
    ax.set_ylim(0, 102)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7)

    # — Generalization —
    ax = axes[1]
    for n in passes:
        df = gen[n]
        ax.plot(df["num_modes"], df["mean_error_mm"],
                color=PALETTE[n], marker=MARKERS[n], markevery=3,
                label=f"SSM{n}", zorder=3)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Reconstruction error (mm)")
    ax.set_title("(b) Generalization")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7)

    # — Specificity —
    ax = axes[2]
    for n in passes:
        df = spec[n]
        ax.plot(df["num_modes"], df["mean_specificity_mm"],
                color=PALETTE[n], marker=MARKERS[n], markevery=3,
                label=f"SSM{n}", zorder=3)
        ax.fill_between(df["num_modes"],
                        df["mean_specificity_mm"] - df["std_specificity_mm"],
                        df["mean_specificity_mm"] + df["std_specificity_mm"],
                        alpha=0.15, color=PALETTE[n])
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Min dist to training shape (mm)")
    ax.set_title("(c) Specificity")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7)

    fig.tight_layout(pad=0.8)
    return fig


def fig_modes_table(compact: dict, passes: list) -> plt.Figure:
    """Table: modes required for 90 / 95 / 99 % variance — bar chart format."""
    pcts = [90, 95, 99]
    x = np.arange(len(pcts))
    w = 0.8 / max(len(passes), 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for ki, n in enumerate(passes):
        if n not in compact:
            continue
        vals = [modes_at(compact[n], p) for p in pcts]
        offset = (ki - (len(passes)-1)/2) * w
        bars = ax.bar(x + offset, vals, width=w*0.88,
                      color=PALETTE[n], alpha=0.85, label=f"SSM{n}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.15,
                    str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}% variance" for p in pcts])
    ax.set_ylabel("Number of modes required")
    ax.set_title("Modes Needed for Variance Explained")
    ax.legend()
    fig.tight_layout()
    return fig


# ── main ────────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, stem: str, out_dir: Path):
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"  Saved: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-dir", required=True,
                    help="Directory containing *_pass<N>.csv files")
    ap.add_argument("--out-dir",   default=None,
                    help="Output directory for figures (default: same as plots-dir)")
    args = ap.parse_args()

    plots_dir = Path(args.plots_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not plots_dir.is_dir():
        sys.exit(f"ERROR: {plots_dir} not found")

    passes = find_passes(plots_dir)
    if not passes:
        sys.exit(f"ERROR: no compactness_pass*.csv files in {plots_dir}")
    print(f"Passes found: {passes}")

    compact = load(plots_dir, "compactness",      passes)
    gen     = load(plots_dir, "generalization",   passes)
    spec    = load(plots_dir, "specificity",      passes)
    dtm     = load(plots_dir, "distance_to_mean", passes)

    print("\nGenerating figures...")

    save(fig_compactness(compact, passes),      "compactness",      out_dir)
    save(fig_generalization(gen, passes),        "generalization",   out_dir)
    save(fig_specificity(spec, passes),          "specificity",      out_dir)
    save(fig_combined(compact, gen, spec, passes),"combined_validation", out_dir)
    save(fig_modes_table(compact, passes),       "modes_table",      out_dir)
    if dtm:
        save(fig_distance_to_mean(dtm, passes),  "distance_to_mean", out_dir)

    # print summary table
    print("\n" + "="*60)
    print(f"  {'Metric':<30} " + "  ".join(f"SSM{n:>4}" for n in passes))
    print("="*60)
    print(f"  {'PCA rank':<30} " +
          "  ".join(f"{len(compact[n]):>6}" for n in passes if n in compact))
    for pct in (90, 95, 99):
        print(f"  {f'Modes for {pct}% variance':<30} " +
              "  ".join(f"{modes_at(compact[n], pct):>6}"
                        for n in passes if n in compact))
    if dtm:
        print(f"  {'Avg dist-to-mean (mm)':<30} " +
              "  ".join(f"{dtm[n]['mean_mm'].mean():>6.3f}"
                        for n in passes if n in dtm))
    print("="*60)
    print(f"\nAll figures in: {out_dir}")
    plt.show()


if __name__ == "__main__":
    main()
