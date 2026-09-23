#!/usr/bin/env python3
"""
Publication-format figures for the single-Gaussian-kernel (sigma, s) sweep -- reads the same
sigma<S>_s<A> output directories as compare_single_kernel_sweep.py and produces heatmap figures
(the standard format for a 2D hyperparameter sweep) instead of a text table.

Six separate figures, each its own file (matching the convention used for the multi-scale kernel
figures): mean/Chamfer, RMSE, HD95, Hausdorff max, specificity, generalization -- each a sigma (x)
by s (y) heatmap. A grid cell with no completed run (e.g. sigma=50,s=150 and sigma=75,s=150, which
aren't in the 9-point grid) is left blank, not interpolated or guessed.

Conventions: no embedded title (caption goes in the manuscript), sans-serif fonts embedded in the
PDF, 600 dpi PNG + vector PDF, colorblind-safe sequential colormap (viridis -- perceptually uniform,
safe under all common CVD types, standard in the scientific-figure literature).

Usage:
    python3 scripts/plot_single_kernel_sweep.py <sweep_base_dir> [out_dir]

Requires: pandas, matplotlib, numpy.
"""
import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.linewidth"] = 1.0

INK = "#111111"


def read_run_config(run_dir):
    path = os.path.join(run_dir, "run_config.txt")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["key"], df["value"]))


def collect_rows(base_dir):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(base_dir, "sigma*_s*"))):
        if not os.path.isdir(run_dir):
            continue
        cfg = read_run_config(run_dir)
        dist_path = os.path.join(run_dir, "final_fit_quality.csv")
        if not os.path.exists(dist_path):
            print(f"(skipping {run_dir}: no final_fit_quality.csv -- did Stage2 finish?)")
            continue
        dist_df = pd.read_csv(dist_path)

        sigma = cfg.get("sigma_mm")
        scale = cfg.get("scale_mm")
        if sigma is None or scale is None:
            base = os.path.basename(run_dir)
            try:
                sigma = float(base.split("sigma")[1].split("_s")[0])
                scale = float(base.split("_s")[-1])
            except (IndexError, ValueError):
                print(f"(skipping {run_dir}: could not determine sigma/s)")
                continue

        row = {"sigma_mm": float(sigma), "scale_mm": float(scale)}
        row["fit_mean_mm"] = dist_df["fit_mean_mm"].mean()
        row["fit_rms_mm"] = dist_df["fit_rms_mm"].mean()
        row["fit_hd95_mm"] = dist_df["fit_hd95_mm"].mean()
        row["fit_hd_mm"] = dist_df["fit_hd_mm"].mean()

        spec_path = os.path.join(run_dir, "pca_specificity.csv")
        if os.path.exists(spec_path):
            row["specificity_mm"] = pd.read_csv(spec_path)["mean_distance_to_nearest_real_subject_mm"].iloc[-1]
        gen_path = os.path.join(run_dir, "pca_generalization.csv")
        if os.path.exists(gen_path):
            row["generalization_mm"] = pd.read_csv(gen_path)["mean_reconstruction_error_mm"].iloc[-1]

        rows.append(row)
    return pd.DataFrame(rows)


def heatmap_figure(df, value_col, ylabel, out_path):
    sigmas = sorted(df["sigma_mm"].unique())
    scales = sorted(df["scale_mm"].unique())
    grid = np.full((len(scales), len(sigmas)), np.nan)
    for _, r in df.iterrows():
        if value_col not in r or pd.isna(r[value_col]):
            continue
        i = scales.index(r["scale_mm"])
        j = sigmas.index(r["sigma_mm"])
        grid[i, j] = r[value_col]

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")  # missing grid points: neutral gray, not white (would look like a real zero/min)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower")

    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s:.0f}" for s in sigmas], fontsize=9)
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"{s:.0f}" for s in scales], fontsize=9)
    ax.set_xlabel("sigma (mm)", fontsize=9.5)
    ax.set_ylabel("s / scaleFactor (mm)", fontsize=9.5)

    for i in range(len(scales)):
        for j in range(len(sigmas)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5, color="white" if grid[i, j] > np.nanmean(grid) else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(ylabel, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color(INK)
    ax.tick_params(colors=INK)

    fig.tight_layout()
    for ext, dpi in [("png", 600), ("pdf", None)]:
        fig.savefig(f"{out_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    base_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else base_dir

    df = collect_rows(base_dir)
    if df.empty:
        print(f"No completed sigma*_s* runs found under {base_dir}. Nothing to plot.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    figures = [
        ("fig1_mean_chamfer_heatmap", "fit_mean_mm", "Mean surface dist. / Chamfer (mm)"),
        ("fig2_rmse_heatmap", "fit_rms_mm", "RMSE (mm)"),
        ("fig3_hd95_heatmap", "fit_hd95_mm", "HD95 (mm)"),
        ("fig4_hausdorff_max_heatmap", "fit_hd_mm", "Hausdorff, max (mm)"),
        ("fig5_specificity_heatmap", "specificity_mm", "Specificity (mm)"),
        ("fig6_generalization_heatmap", "generalization_mm", "Generalization (mm)"),
    ]
    made = 0
    for name, col, ylabel in figures:
        if col not in df.columns or df[col].isna().all():
            print(f"(skipping {name}: no '{col}' data found -- did Stage3PCAModel run for these points?)")
            continue
        heatmap_figure(df, col, ylabel, os.path.join(out_dir, name))
        made += 1
        print(f"wrote {name}.png / .pdf")

    print(f"\n{made} of 6 figures written to {out_dir}")
    print(f"({len(df)} grid points found; the 9-point grid has 2 combinations not in the design "
          "(sigma=50/75, s=150) -- those cells render blank/gray, not interpolated.)")


if __name__ == "__main__":
    main()
