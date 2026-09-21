#!/usr/bin/env python3
"""
Plots the three standard SSM validation figures (Styner et al. 2003):
compactness, specificity, and generalization -- from the CSVs
scapula.Stage3PCAModel writes to Config.outDir.

Usage:
    python3 scripts/plot_ssm_validation.py [output_dir]

If output_dir is omitted, uses the same default Config.outDir the Scala
code uses (override with SCAPULA_OUT_DIR to match, if you've set that
env var for the Scala runs too).

Requires: pandas, matplotlib (both ship with a standard conda install).
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_OUT_DIR = os.path.expanduser(
    "~/Documents/database_v1.11/scapula_ssm_out"
)


def out_dir():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.environ.get("SCAPULA_OUT_DIR", DEFAULT_OUT_DIR)


def plot_compactness(csv_path, fig_path):
    df = pd.read_csv(csv_path)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(df["num_components"], df["eigenvalue"], color="#4c72b0", alpha=0.6, label="eigenvalue")
    ax1.set_xlabel("number of components (mode)")
    ax1.set_ylabel("eigenvalue (variance, mm$^2$)", color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")

    ax2 = ax1.twinx()
    ax2.plot(df["num_components"], df["cumulative_variance_fraction"] * 100, color="#c44e52",
             marker="o", markersize=4, label="cumulative variance")
    ax2.axhline(90, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("cumulative variance explained (%)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax2.set_ylim(0, 105)

    plt.title("Compactness: variance explained vs. number of modes")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")

    n90 = df[df["cumulative_variance_fraction"] >= 0.90]["num_components"].min()
    print(f"  -> {n90} of {df['num_components'].max()} modes explain 90% of the variance")


def plot_specificity(csv_path, fig_path):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["num_components"], df["mean_distance_to_nearest_real_subject_mm"],
            color="#55a868", marker="o", markersize=4)
    ax.set_xlabel("number of components used for sampling")
    ax.set_ylabel("mean distance to nearest real subject (mm)")
    ax.set_title("Specificity: do random samples look like real anatomy?")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")
    print(f"  -> at full rank: {df['mean_distance_to_nearest_real_subject_mm'].iloc[-1]:.2f} mm "
          "(lower = more plausible samples)")


def plot_generalization(csv_path, fig_path):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["num_components"], df["mean_reconstruction_error_mm"],
            color="#dd8452", marker="o", markersize=4)
    ax.set_xlabel("number of components used for reconstruction")
    ax.set_ylabel("mean leave-one-out reconstruction error (mm)")
    ax.set_title("Generalization: can the model represent an unseen subject?")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")
    print(f"  -> at full rank: {df['mean_reconstruction_error_mm'].iloc[-1]:.2f} mm mean reconstruction error")


def main():
    d = out_dir()
    if not os.path.isdir(d):
        print(f"!! {d} does not exist. Pass the output dir explicitly: "
              f"python3 {sys.argv[0]} /path/to/scapula_ssm_out")
        sys.exit(1)

    jobs = [
        ("pca_compactness.csv", "compactness.png", plot_compactness),
        ("pca_specificity.csv", "specificity.png", plot_specificity),
        ("pca_generalization.csv", "generalization.png", plot_generalization),
    ]
    any_found = False
    for csv_name, fig_name, fn in jobs:
        csv_path = os.path.join(d, csv_name)
        if os.path.exists(csv_path):
            any_found = True
            fn(csv_path, os.path.join(d, fig_name))
        else:
            print(f"(skipping {fig_name}: {csv_path} not found -- run scapula.Stage3PCAModel first)")

    if not any_found:
        print("No pca_*.csv files found. Run `sbt \"runMain scapula.Stage3PCAModel\"` first.")
        sys.exit(1)

    print(f"\nAll figures written to {d}")


if __name__ == "__main__":
    main()
