#!/usr/bin/env python3
"""
Publication-style validation figure for the scapula SSM pipeline.

Reads the CSVs written by Stage2Registration and Stage3BuildModel and renders one
2x3 report figure:

  A) Compactness            (ssm_validation_metrics.csv)
  B) Variance per mode       (ssm_validation_metrics.csv)
  C) Generalization (LOOCV)  (ssm_validation_metrics.csv)
  D) Specificity              (ssm_validation_metrics.csv)
  E) Registration accuracy    (registration_metrics.csv)      -- per-specimen boxplot
  F) Mode concentration       (mode_concentration_summary.csv) -- correspondence-artifact flag

Usage:
    python3 plot_ssm_report.py [output_dir]

output_dir defaults to $SCAPULA_OUT_DIR, falling back to the pipeline's own default.
Writes <output_dir>/ssm_report_figure.png (300 dpi, white background, for print/paper use).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ---- palette (validated categorical + status steps, light-surface figure) -------------------------------------
BLUE = "#2a78d6"      # series 1: compactness, generalization
ORANGE = "#eb6834"    # series 2: variance bars, specificity
AQUA = "#1baf7a"       # series 3: registration boxplot, concentration (ok)
CRITICAL = "#d03b3b"   # status: concentration (suspicious)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axis(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK_PRIMARY, loc="left")
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def panel_a_compactness(ax, ssm):
    ax.plot(ssm["mode"], ssm["cum_variance_ratio"] * 100, "-o", color=BLUE, lw=2, ms=5, zorder=3)
    ax.axhline(90, color=INK_MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.axhline(95, color=INK_MUTED, linestyle=":", linewidth=1, zorder=2)
    right_edge = ssm["mode"].max() + 0.9
    ax.set_xlim(ssm["mode"].min() - 0.3, right_edge + 0.7)
    ax.set_xticks(ssm["mode"])
    ax.text(right_edge, 88, "90%", va="center", ha="left", fontsize=8, color=INK_MUTED)
    ax.text(right_edge, 97, "95%", va="center", ha="left", fontsize=8, color=INK_MUTED)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_ylim(0, 105)
    style_axis(ax, "A: Compactness")


def panel_b_variance(ax, ssm):
    ax.bar(ssm["mode"], ssm["variance_mm2"] / 1000.0, color=ORANGE, width=0.6, zorder=3)
    ax.set_xlabel("Mode")
    ax.set_ylabel(r"Variance ($\times10^3$ mm$^2$)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    style_axis(ax, "B: Variance per mode")


def panel_c_generalization(ax, ssm):
    ax.plot(ssm["mode"], ssm["generalization_rmse_mm"], "-o", color=BLUE, lw=2, ms=5, zorder=3)
    ax.set_xlabel("Mode")
    ax.set_ylabel("LOOCV RMSE (mm)")
    ax.set_ylim(bottom=0)
    style_axis(ax, "C: Generalization (LOOCV)")


def panel_d_specificity(ax, ssm):
    ax.plot(ssm["mode"], ssm["specificity_rmse_mm"], "-o", color=ORANGE, lw=2, ms=5, zorder=3)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Specificity RMSE (mm)")
    ax.set_ylim(bottom=0)
    style_axis(ax, "D: Specificity")


def panel_e_registration(ax, reg):
    cols = ["mean_mm", "rms_mm", "hd95_mm", "hd_max_mm"]
    labels = ["Mean", "RMS", "95% HD", "Max HD"]
    data = [reg[c].dropna().values for c in cols]
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        boxprops=dict(facecolor=AQUA, alpha=0.35, edgecolor=AQUA, linewidth=1.5),
        medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
        whiskerprops=dict(color=AQUA, linewidth=1.2),
        capprops=dict(color=AQUA, linewidth=1.2),
        flierprops=dict(markeredgecolor=AQUA, markersize=4),
        zorder=3,
    )
    ax.set_ylabel("Distance (mm)")
    ax.set_ylim(bottom=0)
    style_axis(ax, "E: Registration accuracy")


def panel_f_concentration(ax, conc):
    colors = [CRITICAL if s else AQUA for s in conc["suspicious"]]
    bars = ax.bar(conc["mode"], conc["top20_share_pct"], color=colors, width=0.6, zorder=3)
    ax.axhline(60, color=INK_MUTED, linestyle="--", linewidth=1, zorder=2)
    right_edge = conc["mode"].max() + 0.9
    ax.set_xlim(conc["mode"].min() - 0.7, right_edge + 0.7)
    ax.set_xticks(conc["mode"])
    ax.text(right_edge, 60, "60% flag", va="center", ha="left", fontsize=8, color=INK_MUTED)
    for rect, suspicious in zip(bars, conc["suspicious"]):
        if suspicious:
            ax.annotate(
                "SUSPICIOUS",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=CRITICAL,
                fontweight="bold",
            )
    ax.set_xlabel("Mode")
    ax.set_ylabel("Top-20 points' share of mode variance (%)")
    ax.set_ylim(0, 105)
    style_axis(ax, "F: Mode concentration")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "SCAPULA_OUT_DIR", "/home/g25upadh/Documents/database_v1.11/scapula_ssm_out"
    )

    ssm = pd.read_csv(os.path.join(out_dir, "ssm_validation_metrics.csv"))
    reg = pd.read_csv(os.path.join(out_dir, "registration_metrics.csv"))
    conc = pd.read_csv(os.path.join(out_dir, "mode_concentration_summary.csv"))

    fig, axs = plt.subplots(2, 3, figsize=(15, 9), dpi=300, facecolor=SURFACE)
    fig.suptitle("Scapula SSM -- Validation Report", fontsize=14, fontweight="bold", color=INK_PRIMARY, x=0.02, ha="left")

    panel_a_compactness(axs[0, 0], ssm)
    panel_b_variance(axs[0, 1], ssm)
    panel_c_generalization(axs[0, 2], ssm)
    panel_e_registration(axs[1, 0], reg)
    panel_f_concentration(axs[1, 1], conc)
    panel_d_specificity(axs[1, 2], ssm)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, "ssm_report_figure.png")
    fig.savefig(out_path, dpi=300, facecolor=SURFACE)
    print(f"Saved report figure to: {out_path}")

    bad = conc[conc["suspicious"]]
    if not bad.empty:
        print("\nFlagged modes (top-20 points carry >60% of that mode's variance):")
        for _, row in bad.iterrows():
            print(f"  mode {int(row['mode'])}: {row['top20_share_pct']:.1f}% -- see mode_concentration_points.csv")


if __name__ == "__main__":
    main()
