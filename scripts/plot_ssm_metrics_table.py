#!/usr/bin/env python3
"""
Complete SSM metrics table + Gaussian kernel summary.

Includes every distance metric written by Stage2ReferenceRefinement:
  Mean (= ASSD = Chamfer distance), RMS, HD95, HD (max Hausdorff), Landmark RMSE
  — for both rigid pre-alignment and GPMM-fitted results, per subject + averages.

Usage:
    python3 scripts/plot_ssm_metrics_table.py /path/to/scapula_ssm_out_n22
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
BLUE    = "#2166ac"
DBLUE   = "#1a4a7a"
RED     = "#d6604d"
GREEN   = "#4dac26"
ORANGE  = "#e08214"
GRAY    = "#444444"
MGRAY   = "#888888"
LGRAY   = "#f4f4f4"
DGRAY   = "#cccccc"
AVGBG   = "#d0e8ff"
AVGFG   = "#003366"
RIGID_H = "#4a4a7a"
FIT_H   = "#2166ac"
WARN    = "#fff3b0"

plt.rcParams.update({
    "font.family": "monospace",
    "font.size":   9,
    "axes.titlesize": 10,
    "figure.dpi":  150,
})


# ── helpers ────────────────────────────────────────────────────────────────────
def load_csv(d, name, required=True):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        if required:
            sys.exit(f"Missing: {p}  — run the corresponding Stage first.")
        return None
    return pd.read_csv(p)


def load_config(d):
    p = os.path.join(d, "run_config.txt")
    if not os.path.exists(p):
        return {}
    cfg = {}
    with open(p) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                cfg[parts[0].strip()] = parts[1].strip()
    return cfg


def kernel_description(cfg):
    s1 = cfg.get("dualKernelSigma1", "unset")
    a1 = cfg.get("dualKernelScale1", "unset")
    s2 = cfg.get("dualKernelSigma2", "unset")
    a2 = cfg.get("dualKernelScale2", "unset")
    if all(v not in ("unset", "") for v in [s1, a1, s2, a2]):
        return [
            ("Kernel type",    "DUAL-term Gaussian  (user-specified, absolute mm)"),
            ("Term 1  σ",      f"{s1} mm"),
            ("Term 1  scale",  f"{a1} mm   →  variance = {float(a1)**2:.1f} mm²"),
            ("Term 2  σ",      f"{s2} mm"),
            ("Term 2  scale",  f"{a2} mm   →  variance = {float(a2)**2:.1f} mm²"),
        ]
    ks = cfg.get("kernelSigma", "unset")
    ka = cfg.get("kernelScale", "unset")
    if ks not in ("unset", "") and ka not in ("unset", ""):
        return [
            ("Kernel type",   "SINGLE-term Gaussian  (user-specified, absolute mm)"),
            ("σ (sigma)",     f"{ks} mm"),
            ("scale",         f"{ka} mm   →  variance = {float(ka)**2:.1f} mm²"),
        ]
    kt = cfg.get("kernelTerms", "3")
    tl = "coarse + mid + fine" if kt == "3" else "coarse + fine only"
    return [
        ("Kernel type",        f"Multi-scale Gaussian — {kt}-term  ({tl})"),
        ("Parameterisation",   "Mesh-relative:  all σ and amplitudes are fractions of L = bounding-box longest axis"),
        ("Coarse  σ",          "L / 2     (global shape variation, ~50% of bone length)"),
        ("Coarse  amplitude",  "L × 0.100    →  std-dev ≈ 10% of L     variance = (0.1L)²"),
        ("Mid     σ",          "L / 5     (medium-scale anatomy, ~20% of bone length)" if kt == "3" else "—  (disabled in 2-term mode)"),
        ("Mid     amplitude",  "L × 0.0667   →  std-dev ≈ 6.7% of L   variance = (0.067L)²"  if kt == "3" else "—  (disabled)"),
        ("Fine    σ",          "L / 10    (local surface detail, ~10% of bone length)"),
        ("Fine    amplitude",  "L × 0.0333   →  std-dev ≈ 3.3% of L   variance = (0.033L)²"),
        ("GP approximation",   "Pivoted-Cholesky low-rank  tol={tol}  max rank={rank}".format(
            tol=cfg.get("gpRelativeTolerance", "0.01"),
            rank=cfg.get("gpMaxRank", "250"))),
    ]


def draw_table(ax, rows, col_headers, col_widths=None,
               row_colors=None, header_color=BLUE,
               header_text_color="white", fontsize=8.2,
               header_height=1.2):
    """Draw a table using matplotlib patches (no dependency on ax.table)."""
    ax.axis("off")
    n_rows = len(rows)
    n_cols = len(col_headers)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    # header row
    x = 0.0
    for h, w in zip(col_headers, col_widths):
        ax.add_patch(FancyBboxPatch(
            (x, n_rows), w, header_height,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=header_color, transform=ax.transData))
        ax.text(x + w / 2, n_rows + header_height / 2, h,
                ha="center", va="center",
                fontsize=fontsize, color=header_text_color,
                fontweight="bold", multialignment="center",
                transform=ax.transData)
        x += w

    # data rows
    for i, row in enumerate(rows):
        bg = row_colors[i] if row_colors else (LGRAY if i % 2 == 0 else "white")
        x = 0.0
        for j, (cell, w) in enumerate(zip(row, col_widths)):
            ax.add_patch(FancyBboxPatch(
                (x, n_rows - i - 1), w, 0.95,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=bg, transform=ax.transData))
            align = "left" if j == 0 else "center"
            xpos  = x + 0.008 if j == 0 else x + w / 2
            ax.text(xpos, n_rows - i - 1 + 0.475, str(cell),
                    ha=align, va="center", fontsize=fontsize,
                    color=(AVGFG if bg == AVGBG else "black"),
                    fontweight=("bold" if bg == AVGBG else "normal"),
                    transform=ax.transData)
            x += w
        ax.axhline(n_rows - i - 1, color=DGRAY, linewidth=0.4)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, n_rows + header_height + 0.1)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Documents/database_v1.11/scapula_ssm_out_n22")

    if not os.path.isdir(d):
        sys.exit(f"Directory not found: {d}")

    cfg    = load_config(d)
    df_fit = load_csv(d, "final_fit_quality.csv")
    df_c   = load_csv(d, "pca_compactness.csv")
    df_s   = load_csv(d, "pca_specificity.csv")
    df_g   = load_csv(d, "pca_generalization.csv")

    fig_dir = os.path.join(d, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    N = len(df_fit)

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 28))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        5, 2, figure=fig,
        hspace=0.60, wspace=0.30,
        height_ratios=[0.75, 1.05, 1.05, 1.05, 0.90],
    )

    fig.suptitle(
        "Statistical Shape Model — Scapula  (N = {n})\n"
        "Complete Distance Error Metrics, Hausdorff Distances & Gaussian Kernel Configuration".format(n=N),
        fontsize=13, fontweight="bold", y=0.998,
    )

    # ── PANEL 1 — Kernel configuration (full width) ─────────────────────────
    ax_ker = fig.add_subplot(gs[0, :])
    ax_ker.set_title(
        "1.  GPMM Gaussian Kernel  —  Registration Prior used in Stage 2",
        loc="left", fontweight="bold", color=BLUE, fontsize=10,
    )
    draw_table(ax_ker, kernel_description(cfg),
               ["Parameter", "Value / Formula"],
               [0.24, 0.76],
               header_color=BLUE, fontsize=9, header_height=1.0)

    # ── PANEL 2 — Rigid pre-alignment errors (full width) ───────────────────
    ax_rig = fig.add_subplot(gs[1, :])
    ax_rig.set_title(
        "2.  Rigid Pre-alignment Errors  (landmark Procrustes + trimmed ICP, similarity transform)",
        loc="left", fontweight="bold", color=RIGID_H, fontsize=10,
    )

    rig_headers = [
        "Subject",
        "Mean / ASSD\n= Chamfer (mm)",
        "RMSE\nsurface (mm)",
        "HD95\n(mm)",
        "HD max\nHausdorff (mm)",
        "Landmark\nRMSE (mm)",
    ]
    rig_widths = [0.22, 0.155, 0.155, 0.135, 0.175, 0.160]

    rig_rows = []
    for _, r in df_fit.iterrows():
        rig_rows.append([
            r["subject"],
            f"{r['rigid_mean_mm']:.3f}",
            f"{r['rigid_rms_mm']:.3f}",
            f"{r['rigid_hd95_mm']:.2f}",
            f"{r['rigid_hd_mm']:.2f}",
            f"{r['rigid_landmark_rmse_mm']:.3f}",
        ])

    avg_rig = [
        "─── AVERAGE  (N={})".format(N),
        f"{df_fit['rigid_mean_mm'].mean():.3f}",
        f"{df_fit['rigid_rms_mm'].mean():.3f}",
        f"{df_fit['rigid_hd95_mm'].mean():.2f}",
        f"{df_fit['rigid_hd_mm'].mean():.2f}",
        f"{df_fit['rigid_landmark_rmse_mm'].mean():.3f}",
    ]
    rig_colors = [LGRAY if i % 2 == 0 else "white" for i in range(len(rig_rows))]
    rig_rows.append(avg_rig)
    rig_colors.append(AVGBG)

    draw_table(ax_rig, rig_rows, rig_headers, rig_widths,
               row_colors=rig_colors, header_color=RIGID_H, fontsize=8.5)

    # ── PANEL 3 — GPMM fitted errors (full width) ───────────────────────────
    ax_fit_ax = fig.add_subplot(gs[2, :])
    ax_fit_ax.set_title(
        "3.  GPMM Non-rigid Fitting Errors  (landmark-informed GPMM, coarse-to-fine registration)",
        loc="left", fontweight="bold", color=FIT_H, fontsize=10,
    )

    fit_headers = [
        "Subject",
        "Mean / ASSD\n= Chamfer (mm)",
        "RMSE\nsurface (mm)",
        "HD95\n(mm)",
        "HD max\nHausdorff (mm)",
        "Landmark\nRMSE (mm)",
    ]
    fit_widths = [0.22, 0.155, 0.155, 0.135, 0.175, 0.160]

    fit_rows = []
    for _, r in df_fit.iterrows():
        fit_rows.append([
            r["subject"],
            f"{r['fit_mean_mm']:.3f}",
            f"{r['fit_rms_mm']:.3f}",
            f"{r['fit_hd95_mm']:.2f}",
            f"{r['fit_hd_mm']:.2f}",
            f"{r['fit_landmark_rmse_mm']:.3f}",
        ])

    avg_fit = [
        "─── AVERAGE  (N={})".format(N),
        f"{df_fit['fit_mean_mm'].mean():.3f}",
        f"{df_fit['fit_rms_mm'].mean():.3f}",
        f"{df_fit['fit_hd95_mm'].mean():.2f}",
        f"{df_fit['fit_hd_mm'].mean():.2f}",
        f"{df_fit['fit_landmark_rmse_mm'].mean():.3f}",
    ]
    fit_colors = [LGRAY if i % 2 == 0 else "white" for i in range(len(fit_rows))]
    fit_rows.append(avg_fit)
    fit_colors.append(AVGBG)

    draw_table(ax_fit_ax, fit_rows, fit_headers, fit_widths,
               row_colors=fit_colors, header_color=FIT_H, fontsize=8.5)

    # ── PANEL 4 — Improvement table + notation key ──────────────────────────
    ax_imp  = fig.add_subplot(gs[3, 0])
    ax_key  = fig.add_subplot(gs[3, 1])

    # 4a — improvement
    ax_imp.set_title("4a.  Rigid → Fitted Improvement per Metric",
                     loc="left", fontweight="bold", color=GRAY, fontsize=10)

    metrics = [
        ("Mean / ASSD  (Chamfer)",    "rigid_mean_mm",          "fit_mean_mm"),
        ("RMSE surface",              "rigid_rms_mm",           "fit_rms_mm"),
        ("HD95  (Hausdorff 95th %)",  "rigid_hd95_mm",          "fit_hd95_mm"),
        ("HD max  (Hausdorff)",       "rigid_hd_mm",            "fit_hd_mm"),
        ("Landmark RMSE",             "rigid_landmark_rmse_mm", "fit_landmark_rmse_mm"),
    ]
    imp_rows = []
    for label, rc, fc in metrics:
        rv = df_fit[rc].mean()
        fv = df_fit[fc].mean()
        pct = (1 - fv / rv) * 100
        imp_rows.append([label,
                         f"{rv:.3f}",
                         f"{fv:.3f}",
                         f"{pct:+.1f}%"])

    imp_colors = [LGRAY if i % 2 == 0 else "white" for i in range(len(imp_rows))]
    draw_table(ax_imp, imp_rows,
               ["Metric", "Rigid avg (mm)", "Fitted avg (mm)", "Reduction"],
               [0.42, 0.19, 0.20, 0.19],
               row_colors=imp_colors, header_color=GRAY, fontsize=8.5)

    # 4b — notation key
    ax_key.set_title("4b.  Distance Metric Definitions",
                     loc="left", fontweight="bold", color=GRAY, fontsize=10)

    key_rows = [
        ["Mean / ASSD",  "Average Symmetric Surface Distance (= Chamfer distance).\nBidirectional mean of closest-point distances."],
        ["RMSE surface", "Root Mean Square of all closest-point surface distances.\nMore sensitive to large outliers than ASSD."],
        ["HD95",         "95th-percentile Hausdorff distance.\nRobust to surface noise spikes; standard in med-imaging."],
        ["HD max",       "Maximum Hausdorff distance (worst-case point error).\nSensitive to outliers / artefacts."],
        ["Landmark RMSE","RMSE of the 5 named anatomical landmarks (GC, TS, IA,\nPLA, AC) after non-rigid fitting."],
        ["Chamfer dist", "Same quantity as ASSD/Mean above.\n'Chamfer' = point-cloud CV term; 'ASSD/Mean' = medical-imaging term."],
    ]
    key_colors = [LGRAY if i % 2 == 0 else "white" for i in range(len(key_rows))]
    draw_table(ax_key, key_rows,
               ["Term", "Definition"],
               [0.28, 0.72],
               row_colors=key_colors, header_color=GRAY, fontsize=8.0,
               header_height=1.0)

    # ── PANEL 5 — SSM validation (compactness, specificity, generalisation) ──
    ax_ssm = fig.add_subplot(gs[4, :])
    ax_ssm.set_title(
        "5.  SSM Validation Metrics  (Styner et al. 2003)",
        loc="left", fontweight="bold", color=BLUE, fontsize=10,
    )

    n90  = int(df_c[df_c["cumulative_variance_fraction"] >= 0.90]["num_components"].min())
    last_s = df_s["mean_distance_to_nearest_real_subject_mm"].iloc[-1]
    last_g = df_g["mean_reconstruction_error_mm"].iloc[-1]

    # merge all three CSVs into one wide table row-by-row
    max_k = min(len(df_c), len(df_s), len(df_g))
    ssm_rows = []
    for i in range(max_k):
        k   = int(df_c.iloc[i]["num_components"])
        ev  = df_c.iloc[i]["eigenvalue"]
        cv  = df_c.iloc[i]["cumulative_variance_fraction"] * 100
        sp  = df_s.iloc[i]["mean_distance_to_nearest_real_subject_mm"]
        gen = df_g.iloc[i]["mean_reconstruction_error_mm"]
        row = [k, f"{ev:.2f}", f"{cv:.1f}%", f"{sp:.3f}", f"{gen:.3f}"]
        ssm_rows.append(row)

    ssm_colors = []
    for i, row in enumerate(ssm_rows):
        if row[0] == n90:
            ssm_colors.append(WARN)
        else:
            ssm_colors.append(LGRAY if i % 2 == 0 else "white")

    draw_table(ax_ssm, ssm_rows,
               ["k modes",
                "Eigenvalue (mm²)",
                "Cumul. variance (%)",
                "Specificity\nmean dist to nearest real (mm)",
                "Generalisation\nLOO recon. error (mm)"],
               [0.10, 0.18, 0.20, 0.26, 0.26],
               row_colors=ssm_colors, header_color=BLUE, fontsize=8.5,
               header_height=1.3)

    ax_ssm.text(
        0.5, -0.04,
        f"★  {n90} modes explain 90% of shape variance  (highlighted row)   |   "
        f"Full-rank specificity = {last_s:.2f} mm   |   "
        f"Full-rank generalisation (LOO) = {last_g:.2f} mm",
        ha="center", va="top", fontsize=9, color=BLUE, fontstyle="italic",
        transform=ax_ssm.transAxes,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    out = os.path.join(fig_dir, "ssm_metrics_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("GAUSSIAN KERNEL (GPMM — Stage 2 registration prior)")
    print("=" * 72)
    for label, val in kernel_description(cfg):
        print(f"  {label:<30s}  {val}")

    print("\n" + "=" * 72)
    print("RIGID PRE-ALIGNMENT  — averages over {} subjects".format(N))
    print("=" * 72)
    for col, label in [
        ("rigid_mean_mm",          "Mean / ASSD = Chamfer"),
        ("rigid_rms_mm",           "RMSE surface"),
        ("rigid_hd95_mm",          "HD95  (Hausdorff 95th %)"),
        ("rigid_hd_mm",            "HD    (max Hausdorff)"),
        ("rigid_landmark_rmse_mm", "Landmark RMSE"),
    ]:
        print(f"  {label:<35s}  {df_fit[col].mean():.3f} mm")

    print("\n" + "=" * 72)
    print("GPMM FITTED  — averages over {} subjects".format(N))
    print("=" * 72)
    for col, label in [
        ("fit_mean_mm",          "Mean / ASSD = Chamfer"),
        ("fit_rms_mm",           "RMSE surface"),
        ("fit_hd95_mm",          "HD95  (Hausdorff 95th %)"),
        ("fit_hd_mm",            "HD    (max Hausdorff)"),
        ("fit_landmark_rmse_mm", "Landmark RMSE"),
    ]:
        print(f"  {label:<35s}  {df_fit[col].mean():.3f} mm")

    print("\n" + "=" * 72)
    print("IMPROVEMENT  (rigid → GPMM fitted)")
    print("=" * 72)
    for label, rc, fc in metrics:
        rv  = df_fit[rc].mean()
        fv  = df_fit[fc].mean()
        pct = (1 - fv / rv) * 100
        print(f"  {label:<35s}  {rv:.3f} → {fv:.3f} mm  ({pct:+.1f}%)")

    print("\n" + "=" * 72)
    print("SSM VALIDATION")
    print("=" * 72)
    print(f"  Modes to 90% variance          {n90} of {int(df_c['num_components'].max())}")
    print(f"  Specificity  (full rank)        {last_s:.3f} mm")
    print(f"  Generalisation (full rank LOO)  {last_g:.3f} mm")


if __name__ == "__main__":
    main()
