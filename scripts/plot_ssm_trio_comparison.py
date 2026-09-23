#!/usr/bin/env python3
"""
Plots the standard SSM validation trio -- compactness, specificity, generalization
-- for up to three kernel-configuration run directories, overlaid on one figure
per metric so the configurations can be compared directly.

Reads pca_compactness.csv, pca_specificity.csv, pca_generalization.csv (written by
Stage3PCAModel) from each given directory. A directory missing one of these files
is skipped for that metric only, not the whole run.

Usage:
    python3 scripts/plot_ssm_trio_comparison.py \\
        "Single (sigma=50,s=100)"=<dir1> \\
        "2-kernel"=<dir2> \\
        "3-kernel"=<dir3> \\
        [--out out_dir]

Requires: pandas, matplotlib.
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.linewidth"] = 1.0

INK = "#1a1a1a"
GRID = "#e6e6e6"
# Validated categorical palette (dataviz skill validator: all checks pass)
COLORS = ["#2166ac", "#d6604d", "#762a83"]

METRICS = [
    ("pca_compactness.csv", "num_components", "cumulative_variance_fraction", "Compactness",
     "Number of components", "Cumulative variance explained"),
    ("pca_specificity.csv", "num_components", "mean_distance_to_nearest_real_subject_mm", "Specificity",
     "Number of components used", "Distance to nearest real subject (mm)"),
    ("pca_generalization.csv", "num_components", "mean_reconstruction_error_mm", "Generalization",
     "Number of components used", "Leave-one-out reconstruction error (mm)"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="label=dir pairs")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    labels, dirs = [], []
    for pair in args.configs:
        label, d = pair.rsplit("=", 1)
        labels.append(label)
        dirs.append(os.path.expanduser(d))

    os.makedirs(args.out, exist_ok=True)

    for fname, xcol, ycol, title, xlabel, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        any_plotted = False
        for label, d, color in zip(labels, dirs, COLORS):
            path = os.path.join(d, fname)
            if not os.path.exists(path):
                print(f"(skipping {label} for {title}: {path} not found)")
                continue
            df = pd.read_csv(path)
            y = df[ycol] * 100 if ycol == "cumulative_variance_fraction" and df[ycol].max() <= 1.0 else df[ycol]
            ax.plot(df[xcol], y, marker="o", markersize=4, linewidth=1.6,
                    color=color, label=label, markeredgecolor="white", markeredgewidth=0.4)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            print(f"(no data found for {title} -- skipped)")
            continue

        ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
        ax.set_ylabel(ylabel + (" (%)" if ycol == "cumulative_variance_fraction" else ""), fontsize=9.5, color=INK)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(INK)
        ax.tick_params(colors=INK, labelsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, loc="best")

        fig.tight_layout()
        out_base = os.path.join(args.out, f"ssm_trio_{title.lower()}")
        for ext, dpi in [("png", 600), ("pdf", None)]:
            fig.savefig(f"{out_base}.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_base}.png / .pdf")


if __name__ == "__main__":
    main()
