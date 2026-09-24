#!/usr/bin/env python3
"""
Complete SSM metrics table + Gaussian kernel summary.
Usage:
    python3 scripts/plot_ssm_metrics_table.py /path/to/scapula_ssm_out_n22

Reads:
  run_config.txt          -- kernel settings and pipeline config
  final_fit_quality.csv   -- per-subject rigid + GPMM fitting errors
  pca_compactness.csv     -- eigenvalue / cumulative variance per mode
  pca_specificity.csv     -- specificity (random sample distance) per k
  pca_generalization.csv  -- LOO reconstruction error per k
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── palette ────────────────────────────────────────────────────────────────────
BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#4dac26"
ORANGE = "#e08214"
GRAY   = "#555555"
LGRAY  = "#f5f5f5"
DGRAY  = "#dddddd"

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 150,
})


# ── helpers ────────────────────────────────────────────────────────────────────
def load_csv(d, name, required=True):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        if required:
            sys.exit(f"Missing: {p}")
        return None
    return pd.read_csv(p)


def load_config(d):
    p = os.path.join(d, "run_config.txt")
    if not os.path.exists(p):
        return {}
    cfg = {}
    with open(p) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                cfg[parts[0].strip()] = parts[1].strip()
    return cfg


def kernel_description(cfg):
    """
    Returns list of (label, value) rows describing the kernel that was used.
    Handles single-term, dual-term, and the default 3-term mesh-relative kernel.
    """
    s1 = cfg.get("dualKernelSigma1", "unset")
    a1 = cfg.get("dualKernelScale1", "unset")
    s2 = cfg.get("dualKernelSigma2", "unset")
    a2 = cfg.get("dualKernelScale2", "unset")
    if all(v not in ("unset", "") for v in [s1, a1, s2, a2]):
        return [
            ("Kernel type",   "DUAL-term Gaussian (user-specified, absolute mm)"),
            ("Term 1  σ",     f"{s1} mm"),
            ("Term 1  scale", f"{a1} mm  →  variance = {float(a1)**2:.1f} mm²"),
            ("Term 2  σ",     f"{s2} mm"),
            ("Term 2  scale", f"{a2} mm  →  variance = {float(a2)**2:.1f} mm²"),
        ]

    ks = cfg.get("kernelSigma", "unset")
    ka = cfg.get("kernelScale", "unset")
    if ks not in ("unset", "") and ka not in ("unset", ""):
        return [
            ("Kernel type",   "SINGLE-term Gaussian (user-specified, absolute mm)"),
            ("σ (sigma)",     f"{ks} mm"),
            ("scale",         f"{ka} mm  →  variance = {float(ka)**2:.1f} mm²"),
        ]

    kt = cfg.get("kernelTerms", "3")
    terms_label = "3-term (coarse + mid + fine)" if kt == "3" else "2-term (coarse + fine)"
    return [
        ("Kernel type",    f"Multi-scale Gaussian — {terms_label}"),
        ("Parameterisation","Mesh-relative: fractions of bounding-box longest axis L"),
        ("Coarse term σ",  "L / 2   (global shape variation)"),
        ("Coarse scale",   "L × 0.100  →  amplitude ≈ 10% of L"),
        ("Mid term σ",     "L / 5   (medium-scale anatomy)"     if kt == "3" else "—  (disabled)"),
        ("Mid scale",      "L × 0.0667  →  amplitude ≈ 6.7% of L" if kt == "3" else "—  (disabled)"),
        ("Fine term σ",    "L / 10  (local surface detail)"),
        ("Fine scale",     "L × 0.0333  →  amplitude ≈ 3.3% of L"),
        ("Regularisation", "Cholesky low-rank GP  (tol={tol}, max rank={rank})".format(
            tol=cfg.get("gpRelativeTolerance", "0.01"),
            rank=cfg.get("gpMaxRank", "250"))),
    ]


def draw_table(ax, rows, col_headers, col_widths=None, row_colors=None,
               header_color=BLUE, header_text_color="white",
               fontsize=8.5):
    ax.axis("off")
    n_rows = len(rows)
    n_cols = len(col_headers)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    # header
    x = 0
    for j, (h, w) in enumerate(zip(col_headers, col_widths)):
        ax.add_patch(FancyBboxPatch((x, n_rows), w, 0.9,
                                    boxstyle="square,pad=0", linewidth=0,
                                    facecolor=header_color, transform=ax.transData))
        ax.text(x + w / 2, n_rows + 0.45, h, ha="center", va="center",
                fontsize=fontsize, color=header_text_color, fontweight="bold",
                transform=ax.transData)
        x += w

    # rows
    for i, row in enumerate(rows):
        bg = row_colors[i] if row_colors else (LGRAY if i % 2 == 0 else "white")
        x = 0
        for j, (cell, w) in enumerate(zip(row, col_widths)):
            ax.add_patch(FancyBboxPatch((x, n_rows - i - 1), w, 0.9,
                                        boxstyle="square,pad=0", linewidth=0,
                                        facecolor=bg, transform=ax.transData))
            align = "left" if j == 0 else "center"
            xpos = x + 0.01 if j == 0 else x + w / 2
            ax.text(xpos, n_rows - i - 1 + 0.45, str(cell),
                    ha=align, va="center", fontsize=fontsize,
                    transform=ax.transData)
            x += w
        # bottom border
        ax.axhline(n_rows - i - 1, color=DGRAY, linewidth=0.4, xmin=0, xmax=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, n_rows + 1.1)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Documents/database_v1.11/scapula_ssm_out_n22")

    if not os.path.isdir(d):
        sys.exit(f"Directory not found: {d}")

    cfg = load_config(d)
    df_fit = load_csv(d, "final_fit_quality.csv")
    df_c   = load_csv(d, "pca_compactness.csv")
    df_s   = load_csv(d, "pca_specificity.csv")
    df_g   = load_csv(d, "pca_generalization.csv")

    fig_dir = os.path.join(d, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           height_ratios=[0.9, 2.4, 1.0, 1.0])

    fig.suptitle("SSM Pipeline — Scapula N=22\nComplete Distance Error Metrics & Gaussian Kernel Configuration",
                 fontsize=13, fontweight="bold", y=0.995)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1 — Kernel configuration (full width)
    ax_ker = fig.add_subplot(gs[0, :])
    ax_ker.set_title("1.  GPMM Gaussian Kernel — Registration Prior (Stage 2)", loc="left",
                     fontweight="bold", color=BLUE, fontsize=10)

    k_rows = kernel_description(cfg)
    headers = ["Parameter", "Value / Formula"]
    widths  = [0.22, 0.78]
    draw_table(ax_ker, k_rows, headers, widths, header_color=BLUE, fontsize=9)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2 — Per-subject fitting errors (full width)
    ax_fit = fig.add_subplot(gs[1, :])
    ax_fit.set_title("2.  Registration Quality — Per-subject Surface & Landmark Errors (Stage 2, Final Pass)",
                     loc="left", fontweight="bold", color=GRAY, fontsize=10)

    fit_headers = [
        "Subject",
        "Rigid\nMean (mm)", "Rigid\nRMS (mm)", "Rigid\nHD95 (mm)",
        "Rigid\nLm-RMSE (mm)",
        "Fitted\nMean (mm)", "Fitted\nRMS (mm)", "Fitted\nHD95 (mm)",
        "Fitted\nLm-RMSE (mm)",
    ]
    fit_widths = [0.18, 0.092, 0.092, 0.092, 0.112, 0.092, 0.092, 0.092, 0.138]

    fit_rows = []
    for _, r in df_fit.iterrows():
        fit_rows.append([
            r["subject"],
            f"{r['rigid_mean_mm']:.3f}",
            f"{r['rigid_rms_mm']:.3f}",
            f"{r['rigid_hd95_mm']:.2f}",
            f"{r['rigid_landmark_rmse_mm']:.3f}",
            f"{r['fit_mean_mm']:.3f}",
            f"{r['fit_rms_mm']:.3f}",
            f"{r['fit_hd95_mm']:.2f}",
            f"{r['fit_landmark_rmse_mm']:.3f}",
        ])

    # averages row
    def col_mean(col): return df_fit[col].mean()
    fit_rows.append([
        "── AVERAGE ──",
        f"{col_mean('rigid_mean_mm'):.3f}",
        f"{col_mean('rigid_rms_mm'):.3f}",
        f"{col_mean('rigid_hd95_mm'):.2f}",
        f"{col_mean('rigid_landmark_rmse_mm'):.3f}",
        f"{col_mean('fit_mean_mm'):.3f}",
        f"{col_mean('fit_rms_mm'):.3f}",
        f"{col_mean('fit_hd95_mm'):.2f}",
        f"{col_mean('fit_landmark_rmse_mm'):.3f}",
    ])

    row_colors = [LGRAY if i % 2 == 0 else "white" for i in range(len(fit_rows) - 1)]
    row_colors.append("#d0e8ff")   # highlight average row
    draw_table(ax_fit, fit_rows, fit_headers, fit_widths,
               row_colors=row_colors, header_color="#444466", fontsize=8.0)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3 — Compactness summary table  (left)
    ax_comp = fig.add_subplot(gs[2, 0])
    ax_comp.set_title("3a.  Compactness (PCA modes — cumulative variance)",
                      loc="left", fontweight="bold", color=BLUE, fontsize=10)

    comp_rows = []
    for _, r in df_c.iterrows():
        comp_rows.append([
            int(r["num_components"]),
            f"{r['eigenvalue']:.2f}",
            f"{r['cumulative_variance_fraction']*100:.1f}%",
        ])
    n90 = df_c[df_c["cumulative_variance_fraction"] >= 0.90]["num_components"].min()
    comp_row_colors = []
    for r in comp_rows:
        comp_row_colors.append("#fff3b0" if r[0] == n90 else
                               (LGRAY if comp_rows.index(r) % 2 == 0 else "white"))
    draw_table(ax_comp, comp_rows,
               ["Mode", "Eigenvalue (mm²)", "Cumul. Var. (%)"],
               [0.2, 0.42, 0.38],
               row_colors=comp_row_colors,
               header_color=BLUE, fontsize=8.5)
    ax_comp.text(0.5, -0.06,
                 f"★  {n90} modes explain 90% of variance  (of {df_c['num_components'].max()} total)",
                 ha="center", va="top", fontsize=8.5, color=BLUE,
                 transform=ax_comp.transAxes, fontstyle="italic")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 4 — Specificity + Generalisation summary table  (right)
    ax_sg = fig.add_subplot(gs[2, 1])
    ax_sg.set_title("3b.  Specificity & Generalisation (per k modes)",
                    loc="left", fontweight="bold", color=GRAY, fontsize=10)

    sg_rows = []
    max_k = min(len(df_s), len(df_g))
    for k in range(max_k):
        sk = df_s.iloc[k]["mean_distance_to_nearest_real_subject_mm"]
        gk = df_g.iloc[k]["mean_reconstruction_error_mm"]
        sg_rows.append([int(df_s.iloc[k]["num_components"]),
                        f"{sk:.3f}", f"{gk:.3f}"])

    draw_table(ax_sg, sg_rows,
               ["k modes", "Specificity (mm)", "Generalisation (mm)"],
               [0.22, 0.39, 0.39],
               header_color="#446644", fontsize=8.5)
    last_s = df_s["mean_distance_to_nearest_real_subject_mm"].iloc[-1]
    last_g = df_g["mean_reconstruction_error_mm"].iloc[-1]
    ax_sg.text(0.5, -0.06,
               f"Full-rank:  specificity = {last_s:.2f} mm   |   generalisation = {last_g:.2f} mm",
               ha="center", va="top", fontsize=8.5, color=GRAY,
               transform=ax_sg.transAxes, fontstyle="italic")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 5 — Pipeline config summary  (full width, bottom)
    ax_cfg = fig.add_subplot(gs[3, :])
    ax_cfg.set_title("4.  Pipeline Configuration (from run_config.txt)",
                     loc="left", fontweight="bold", color=GRAY, fontsize=10)

    keys_of_interest = [
        ("dataDir",          "Data directory"),
        ("modelResolution",  "Model resolution (vertices)"),
        ("refinePasses",     "Reference refinement passes"),
        ("icpIterations",    "ICP iterations (per pass)"),
        ("landmarkWeight",   "Landmark weight"),
        ("gpRelativeTolerance", "GP Cholesky tolerance"),
        ("gpMaxRank",        "GP max rank"),
        ("kernelTerms",      "Kernel terms"),
        ("seed",             "Random seed"),
        ("buildIndependentModel", "Independent model (one side/subject)"),
    ]
    cfg_rows = [[label, cfg.get(key, "—")] for key, label in keys_of_interest]
    draw_table(ax_cfg, cfg_rows, ["Setting", "Value"],
               [0.35, 0.65], header_color=GRAY, fontsize=8.5)

    # ──────────────────────────────────────────────────────────────────────────
    out = os.path.join(fig_dir, "ssm_metrics_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GAUSSIAN KERNEL (GPMM Registration Prior)")
    print("=" * 70)
    for label, val in kernel_description(cfg):
        print(f"  {label:<28s} {val}")

    print("\n" + "=" * 70)
    print("REGISTRATION QUALITY — AVERAGE OVER ALL SUBJECTS")
    print("=" * 70)
    for col, label in [
        ("rigid_mean_mm",          "Rigid-only   mean surface distance"),
        ("rigid_rms_mm",           "Rigid-only   RMS  surface distance"),
        ("rigid_hd95_mm",          "Rigid-only   HD95 surface distance"),
        ("rigid_landmark_rmse_mm", "Rigid-only   landmark RMSE"),
        ("fit_mean_mm",            "Fitted GPMM  mean surface distance"),
        ("fit_rms_mm",             "Fitted GPMM  RMS  surface distance"),
        ("fit_hd95_mm",            "Fitted GPMM  HD95 surface distance"),
        ("fit_landmark_rmse_mm",   "Fitted GPMM  landmark RMSE"),
    ]:
        print(f"  {label:<40s} {df_fit[col].mean():.3f} mm")

    improvement = (1 - df_fit["fit_mean_mm"].mean() / df_fit["rigid_mean_mm"].mean()) * 100
    print(f"\n  Non-rigid fitting reduced mean surface distance by {improvement:.1f}%")

    print("\n" + "=" * 70)
    print("SSM VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  {n90} of {df_c['num_components'].max()} modes explain 90% of shape variance")
    print(f"  Specificity  (full rank, {len(df_s)} modes): {last_s:.2f} mm")
    print(f"  Generalisation (full rank): {last_g:.2f} mm  (leave-one-out LOO)")


if __name__ == "__main__":
    main()
