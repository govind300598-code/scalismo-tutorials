#!/usr/bin/env python3
"""
plot_ssm_metrics.py — Plot SSM validation metrics from CSV files exported by
SSMValidation.scala.

Usage:
    python3 plot_ssm_metrics.py --plots-dir /path/to/outDir/plots

Requirements:
    pip install matplotlib seaborn pandas numpy

Output:
    All PNGs are written into the same --plots-dir directory.

    compactness.png              — cumulative variance explained vs. mode count
    generalization.png           — mean reconstruction error vs. mode count
    reconstruction_heatmap.png   — per-specimen error for every mode count
    specificity.png              — mean min-distance-to-training vs. mode count
    pairwise_distances.png       — N×N symmetric heatmap of pairwise distances
    pairwise_histogram.png       — histogram of all pairwise distances
    distance_to_mean.png         — stacked bar per specimen (mean / RMS / HD95)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("seaborn not found — heatmaps will use matplotlib imshow instead.")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
C_BLUE  = "#1f77b4"
C_GREEN = "#2ca02c"
C_RED   = "#d62728"
C_GRAY  = "#7f7f7f"
C_ORNG  = "#ff7f0e"


# ── Helpers ──────────────────────────────────────────────────────────────────
def load(plots_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(plots_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def savefig(fig, plots_dir: str, name: str) -> None:
    path = os.path.join(plots_dir, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def short(name: str, maxlen: int = 14) -> str:
    return name[-maxlen:] if len(name) > maxlen else name


# ── Plot functions ────────────────────────────────────────────────────────────
def plot_compactness(df: pd.DataFrame, plots_dir: str) -> None:
    """Cumulative variance explained vs. number of modes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["mode"], df["cumulative_variance_pct"],
            color=C_BLUE, marker="o", markersize=3, label="Cumulative variance")

    for thr, col, label in [(90, C_GRAY, "90 %"), (95, C_GREEN, "95 %"), (99, C_RED, "99 %")]:
        ax.axhline(thr, color=col, linestyle="--", linewidth=0.9, label=label)
        hits = df[df["cumulative_variance_pct"] >= thr]
        if not hits.empty:
            k = int(hits["mode"].iloc[0])
            ax.annotate(f"{k}", xy=(k, thr), xytext=(k + 0.4, thr - 5),
                        fontsize=8, color=col, fontweight="bold")

    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Compactness — Cumulative Variance Explained by PCA Modes")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(1, df["mode"].max())
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    savefig(fig, plots_dir, "compactness.png")


def plot_scree(df: pd.DataFrame, plots_dir: str) -> None:
    """Per-mode (not cumulative) variance — scree plot."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["mode"], df["variance_pct"], color=C_BLUE, alpha=0.75, width=0.8)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Variance explained per mode (%)")
    ax.set_title("Scree Plot — Variance per PCA Mode")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0.5, df["mode"].max() + 0.5)
    ax.grid(True, alpha=0.25, axis="y")
    savefig(fig, plots_dir, "scree.png")


def plot_generalization(df: pd.DataFrame, plots_dir: str) -> None:
    """Mean reconstruction error (mm) vs. number of modes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["num_modes"], df["mean_error_mm"],
            color=C_BLUE, marker="o", markersize=3)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Mean reconstruction error (mm)")
    ax.set_title("Generalization — Reconstruction Error vs. Mode Count")
    ax.set_xlim(1, df["num_modes"].max())
    ax.set_ylim(0)
    ax.grid(True, alpha=0.25)
    savefig(fig, plots_dir, "generalization.png")


def plot_reconstruction_heatmap(df: pd.DataFrame, plots_dir: str) -> None:
    """Per-specimen reconstruction error for every mode count (heatmap)."""
    pivot = df.pivot(index="specimen_id", columns="num_modes", values="error_mm")
    # Sort rows by error at max modes (hardest specimens at top)
    pivot = pivot.loc[pivot.iloc[:, -1].sort_values(ascending=False).index]

    fig_h = max(5, len(pivot) * 0.35)
    fig_w = max(8, len(pivot.columns) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    row_labels = [short(s) for s in pivot.index]
    col_labels = list(pivot.columns)

    if HAS_SEABORN:
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd",
                    xticklabels=col_labels, yticklabels=row_labels,
                    cbar_kws={"label": "Mean reconstruction error (mm)"},
                    linewidths=0.2 if len(pivot) <= 30 else 0)
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=7)
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=7)
        fig.colorbar(im, ax=ax, label="Mean reconstruction error (mm)")

    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Specimen")
    ax.set_title("Per-Specimen Reconstruction Error vs. Mode Count (mm)")
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    savefig(fig, plots_dir, "reconstruction_heatmap.png")


def plot_specificity(df: pd.DataFrame, plots_dir: str) -> None:
    """Mean minimum-distance of random samples to training set."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["num_modes"], df["mean_specificity_mm"],
            color=C_RED, marker="o", markersize=3, label="Mean")
    if "std_specificity_mm" in df.columns:
        lo = df["mean_specificity_mm"] - df["std_specificity_mm"]
        hi = df["mean_specificity_mm"] + df["std_specificity_mm"]
        ax.fill_between(df["num_modes"], lo.clip(lower=0), hi,
                        alpha=0.2, color=C_RED, label="± 1 SD")
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Min distance to nearest training shape (mm)")
    ax.set_title("Specificity — Distance of Random Samples to Training Set")
    ax.set_xlim(1, df["num_modes"].max())
    ax.set_ylim(0)
    ax.legend()
    ax.grid(True, alpha=0.25)
    savefig(fig, plots_dir, "specificity.png")


def plot_pairwise_heatmap(df: pd.DataFrame, plots_dir: str) -> None:
    """NxN symmetric matrix of pairwise distances."""
    specs = sorted(set(df["specimen_i"].tolist() + df["specimen_j"].tolist()))
    n     = len(specs)
    idx   = {s: i for i, s in enumerate(specs)}
    mat   = np.zeros((n, n))
    for _, row in df.iterrows():
        i, j = idx[row["specimen_i"]], idx[row["specimen_j"]]
        mat[i, j] = row["mean_mm"]
        mat[j, i] = row["mean_mm"]

    labels   = [short(s) for s in specs]
    cell_px  = 0.5
    fig_size = max(6, n * cell_px)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    annotate = n <= 16
    if HAS_SEABORN:
        sns.heatmap(mat, ax=ax, xticklabels=labels, yticklabels=labels,
                    cmap="viridis", annot=annotate, fmt=".1f",
                    linewidths=0.3 if n <= 24 else 0,
                    cbar_kws={"label": "Mean point-to-point distance (mm)"})
    else:
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, label="Mean point-to-point distance (mm)")

    ax.set_title(f"Pairwise Shape Distances — {n}x{n} matrix (mm)")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    savefig(fig, plots_dir, "pairwise_distances.png")


def plot_pairwise_histogram(df: pd.DataFrame, plots_dir: str) -> None:
    """Distribution of all N*(N-1)/2 pairwise distances."""
    vals = df["mean_mm"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram
    axes[0].hist(vals, bins=min(25, len(vals) // 2 + 1),
                 color=C_BLUE, edgecolor="white", alpha=0.85)
    axes[0].axvline(vals.mean(), color=C_RED, linestyle="--",
                    label=f"Mean = {vals.mean():.2f} mm")
    axes[0].axvline(vals.median(), color=C_ORNG, linestyle=":",
                    label=f"Median = {vals.median():.2f} mm")
    axes[0].set_xlabel("Mean pairwise distance (mm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Pairwise Distances")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25, axis="y")

    # Box plot of all three metrics if available
    metrics = [c for c in ["mean_mm", "rms_mm", "hd95_mm"] if c in df.columns]
    axes[1].boxplot([df[m] for m in metrics], labels=[m.replace("_mm", "") for m in metrics],
                    patch_artist=True,
                    boxprops=dict(facecolor=C_BLUE, alpha=0.6),
                    medianprops=dict(color="black"))
    axes[1].set_ylabel("Distance (mm)")
    axes[1].set_title("Pairwise Distance Metrics Summary")
    axes[1].grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    savefig(fig, plots_dir, "pairwise_histogram.png")


def plot_distance_to_mean(df: pd.DataFrame, plots_dir: str) -> None:
    """Stacked bar: mean / (RMS-mean) / (HD95-RMS) per specimen."""
    df = df.sort_values("mean_mm", ascending=False).reset_index(drop=True)
    x  = np.arange(len(df))
    short_ids = [short(s, 16) for s in df["specimen_id"]]

    fig_w = max(9, len(df) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    ax.bar(x, df["mean_mm"],
           color=C_BLUE, alpha=0.85, label="Mean dist.")
    ax.bar(x, (df["rms_mm"] - df["mean_mm"]).clip(lower=0),
           bottom=df["mean_mm"],
           color=C_GREEN, alpha=0.70, label="RMS – Mean")
    ax.bar(x, (df["hd95_mm"] - df["rms_mm"]).clip(lower=0),
           bottom=df["rms_mm"],
           color=C_RED, alpha=0.60, label="HD95 – RMS")

    # Overall mean line
    grand_mean = df["mean_mm"].mean()
    ax.axhline(grand_mean, color="black", linestyle="--", linewidth=0.9,
               label=f"Dataset mean = {grand_mean:.2f} mm")

    ax.set_xticks(x)
    ax.set_xticklabels(short_ids, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Distance to SSM mean shape (mm)")
    ax.set_title("Per-Specimen Distance to Mean Shape (sorted by mean)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")
    savefig(fig, plots_dir, "distance_to_mean.png")


def plot_all_validation(comp_df, gen_df, spec_df, plots_dir: str) -> None:
    """Three-panel summary: compactness / generalization / specificity."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1 — Compactness
    ax = axes[0]
    ax.plot(comp_df["mode"], comp_df["cumulative_variance_pct"], color=C_BLUE, marker="o", markersize=2)
    for thr, col in [(90, C_GRAY), (95, C_GREEN), (99, C_RED)]:
        ax.axhline(thr, color=col, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Modes"); ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Compactness")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 102); ax.set_xlim(1, comp_df["mode"].max())
    ax.grid(True, alpha=0.2)

    # Panel 2 — Generalization
    ax = axes[1]
    ax.plot(gen_df["num_modes"], gen_df["mean_error_mm"], color=C_BLUE, marker="o", markersize=2)
    ax.set_xlabel("Modes"); ax.set_ylabel("Mean error (mm)")
    ax.set_title("Generalization")
    ax.set_xlim(1, gen_df["num_modes"].max()); ax.set_ylim(0)
    ax.grid(True, alpha=0.2)

    # Panel 3 — Specificity
    ax = axes[2]
    ax.plot(spec_df["num_modes"], spec_df["mean_specificity_mm"], color=C_RED, marker="o", markersize=2)
    if "std_specificity_mm" in spec_df.columns:
        lo = (spec_df["mean_specificity_mm"] - spec_df["std_specificity_mm"]).clip(lower=0)
        hi = spec_df["mean_specificity_mm"] + spec_df["std_specificity_mm"]
        ax.fill_between(spec_df["num_modes"], lo, hi, alpha=0.2, color=C_RED)
    ax.set_xlabel("Modes"); ax.set_ylabel("Min dist. to training (mm)")
    ax.set_title("Specificity")
    ax.set_xlim(1, spec_df["num_modes"].max()); ax.set_ylim(0)
    ax.grid(True, alpha=0.2)

    fig.suptitle("SSM Validation Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, plots_dir, "ssm_validation_summary.png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SSM validation metrics exported by SSMValidation.scala"
    )
    parser.add_argument("--plots-dir", required=True,
                        help="Directory containing the CSV files from SSMValidation")
    args = parser.parse_args()
    plots_dir = args.plots_dir

    if not os.path.isdir(plots_dir):
        print(f"ERROR: directory not found: {plots_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading CSVs from: {plots_dir}\n")

    results = {}
    tasks = [
        ("compactness.csv",             "compactness",    True),
        ("generalization.csv",          "generalization", True),
        ("reconstruction_error_matrix.csv", "recon",     True),
        ("specificity.csv",             "specificity",    True),
        ("pairwise_distances.csv",      "pairwise",       True),
        ("distance_to_mean.csv",        "dtm",            True),
    ]

    for fname, key, required in tasks:
        try:
            results[key] = load(plots_dir, fname)
            print(f"Loaded {fname:50s}  ({len(results[key])} rows)")
        except FileNotFoundError:
            if required:
                print(f"  MISSING: {fname}  — skipping dependent plots")
            results[key] = None

    print()

    if results.get("compactness") is not None:
        print("Plotting compactness..."); plot_compactness(results["compactness"], plots_dir)
        print("Plotting scree...");       plot_scree(results["compactness"], plots_dir)

    if results.get("generalization") is not None:
        print("Plotting generalization..."); plot_generalization(results["generalization"], plots_dir)

    if results.get("recon") is not None:
        print("Plotting reconstruction heatmap..."); plot_reconstruction_heatmap(results["recon"], plots_dir)

    if results.get("specificity") is not None:
        print("Plotting specificity..."); plot_specificity(results["specificity"], plots_dir)

    if results.get("pairwise") is not None:
        print("Plotting pairwise heatmap..."); plot_pairwise_heatmap(results["pairwise"], plots_dir)
        print("Plotting pairwise histogram..."); plot_pairwise_histogram(results["pairwise"], plots_dir)

    if results.get("dtm") is not None:
        print("Plotting distance to mean..."); plot_distance_to_mean(results["dtm"], plots_dir)

    # Three-panel summary (needs all three validation metrics)
    if all(results.get(k) is not None for k in ("compactness", "generalization", "specificity")):
        print("Plotting combined validation summary...")
        plot_all_validation(results["compactness"], results["generalization"],
                            results["specificity"], plots_dir)

    print(f"\nDone. PNGs written to: {plots_dir}")


if __name__ == "__main__":
    main()
