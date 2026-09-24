#!/usr/bin/env python3
"""
Publication-quality SSM validation figures.
Usage:
    python3 scripts/plot_ssm_publication.py /path/to/scapula_ssm_out_n22
"""
import os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#4dac26"
ORANGE = "#f4a582"

def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        sys.exit(f"Missing: {p}  -- run scapula.Stage3PCAModel first.")
    return pd.read_csv(p)

# ── 1. Compactness ─────────────────────────────────────────────────────────
def plot_compactness(df, out_path):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.bar(df["num_components"], df["eigenvalue"],
            color=BLUE, alpha=0.55, label="Eigenvalue", zorder=2)
    ax1.set_xlabel("Number of modes")
    ax1.set_ylabel("Eigenvalue (variance, mm²)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_xticks(df["num_components"])
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(df["num_components"], df["cumulative_variance_fraction"] * 100,
             color=RED, marker="o", markersize=5, linewidth=2,
             label="Cumulative variance (%)", zorder=3)
    ax2.axhline(90, color="gray", linestyle="--", linewidth=1, label="90% threshold")
    ax2.set_ylabel("Cumulative variance explained (%)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, 108)

    # annotate 90% crossing
    n90 = df[df["cumulative_variance_fraction"] >= 0.90]["num_components"].min()
    y90 = float(df[df["num_components"] == n90]["cumulative_variance_fraction"].iloc[0]) * 100
    ax2.annotate(f"{n90} modes\n= 90%",
                 xy=(n90, y90), xytext=(n90 + 1.2, y90 - 12),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=9, color="gray")

    ax1.set_title("Compactness — variance explained vs. number of modes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    print(f"         {n90} of {df['num_components'].max()} modes explain 90% of variance")


# ── 2. Specificity ─────────────────────────────────────────────────────────
def plot_specificity(df, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["num_components"], df["mean_distance_to_nearest_real_subject_mm"],
            color=GREEN, marker="o", markersize=5, linewidth=2)
    ax.fill_between(df["num_components"],
                    df["mean_distance_to_nearest_real_subject_mm"],
                    alpha=0.15, color=GREEN)
    ax.set_xlabel("Number of modes used for sampling")
    ax.set_ylabel("Mean distance to nearest real subject (mm)")
    ax.set_title("Specificity — do random samples look like real anatomy?")
    ax.set_ylim(bottom=0)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.4)

    last = df["mean_distance_to_nearest_real_subject_mm"].iloc[-1]
    ax.annotate(f"Full rank\n{last:.2f} mm",
                xy=(df["num_components"].iloc[-1], last),
                xytext=(df["num_components"].iloc[-1] - 4, last + 0.15),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    print(f"         full-rank specificity: {last:.2f} mm")


# ── 3. Generalisation ──────────────────────────────────────────────────────
def plot_generalization(df, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["num_components"], df["mean_reconstruction_error_mm"],
            color=ORANGE, marker="o", markersize=5, linewidth=2,
            markeredgecolor="#c0392b", markeredgewidth=0.8)
    ax.fill_between(df["num_components"],
                    df["mean_reconstruction_error_mm"],
                    alpha=0.2, color=ORANGE)
    ax.set_xlabel("Number of modes used for reconstruction")
    ax.set_ylabel("Mean leave-one-out reconstruction error (mm)")
    ax.set_title("Generalisation — can the model represent an unseen subject?")
    ax.set_ylim(bottom=0)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.4)

    last = df["mean_reconstruction_error_mm"].iloc[-1]
    ax.annotate(f"Plateau\n{last:.2f} mm",
                xy=(df["num_components"].iloc[-1], last),
                xytext=(df["num_components"].iloc[-1] - 5, last + 0.12),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    print(f"         full-rank generalisation: {last:.2f} mm")


# ── 4. Combined 3-panel figure ─────────────────────────────────────────────
def plot_combined(dc, ds, dg, out_path):
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)

    # --- compactness ---
    ax1 = fig.add_subplot(gs[0])
    ax1b = ax1.twinx()
    ax1.bar(dc["num_components"], dc["eigenvalue"], color=BLUE, alpha=0.55)
    ax1.set_xlabel("Number of modes")
    ax1.set_ylabel("Eigenvalue (mm²)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1b.plot(dc["num_components"], dc["cumulative_variance_fraction"] * 100,
              color=RED, marker="o", markersize=4, linewidth=2)
    ax1b.axhline(90, color="gray", linestyle="--", linewidth=1)
    ax1b.set_ylabel("Cumulative variance (%)", color=RED)
    ax1b.tick_params(axis="y", labelcolor=RED)
    ax1b.set_ylim(0, 108)
    ax1.set_title("(a) Compactness")
    for sp in ["top"]: ax1.spines[sp].set_visible(False)

    # --- specificity ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(ds["num_components"], ds["mean_distance_to_nearest_real_subject_mm"],
             color=GREEN, marker="o", markersize=4, linewidth=2)
    ax2.fill_between(ds["num_components"],
                     ds["mean_distance_to_nearest_real_subject_mm"],
                     alpha=0.15, color=GREEN)
    ax2.set_xlabel("Number of modes")
    ax2.set_ylabel("Mean distance to nearest real subject (mm)")
    ax2.set_ylim(bottom=0)
    ax2.set_title("(b) Specificity")
    ax2.grid(linestyle="--", linewidth=0.5, alpha=0.4)
    for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)

    # --- generalisation ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(dg["num_components"], dg["mean_reconstruction_error_mm"],
             color=ORANGE, marker="o", markersize=4, linewidth=2,
             markeredgecolor="#c0392b", markeredgewidth=0.8)
    ax3.fill_between(dg["num_components"],
                     dg["mean_reconstruction_error_mm"],
                     alpha=0.2, color=ORANGE)
    ax3.set_xlabel("Number of modes")
    ax3.set_ylabel("Mean LOO reconstruction error (mm)")
    ax3.set_ylim(bottom=0)
    ax3.set_title("(c) Generalisation")
    ax3.grid(linestyle="--", linewidth=0.5, alpha=0.4)
    for sp in ["top", "right"]: ax3.spines[sp].set_visible(False)

    fig.suptitle("SSM Validation — Scapula (N=22)", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Documents/database_v1.11/scapula_ssm_out_n22")

    if not os.path.isdir(d):
        sys.exit(f"Directory not found: {d}")

    # output subfolder
    fig_dir = os.path.join(d, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    dc = load(d, "pca_compactness.csv")
    ds = load(d, "pca_specificity.csv")
    dg = load(d, "pca_generalization.csv")

    print("\nGenerating figures ...")
    plot_compactness(dc,  os.path.join(fig_dir, "ssm_compactness.png"))
    plot_specificity(ds,  os.path.join(fig_dir, "ssm_specificity.png"))
    plot_generalization(dg, os.path.join(fig_dir, "ssm_generalization.png"))
    plot_combined(dc, ds, dg, os.path.join(fig_dir, "ssm_validation_combined.png"))

    print(f"\nAll figures saved to: {fig_dir}/")
    print("  ssm_compactness.png")
    print("  ssm_specificity.png")
    print("  ssm_generalization.png")
    print("  ssm_validation_combined.png   <-- single 3-panel figure for slides")

if __name__ == "__main__":
    main()
