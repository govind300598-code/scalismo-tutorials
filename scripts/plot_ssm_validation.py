#!/usr/bin/env python3
"""
SSM Validation Plots
====================
Reads the CSV files produced by Stage3SSMValidation and generates one
publication-ready figure (PNG + SVG) with three panels:
  1. Compactness  – cumulative variance explained vs number of modes
  2. Generalization – mean RMS reconstruction error vs number of modes
  3. Specificity   – mean minimum RMS distance to training set vs number of modes

Usage
-----
  python3 scripts/plot_ssm_validation.py          # uses $SCAPULA_OUT_DIR or ~/Documents/combined_scapula_data/scapula_ssm_out_fast_no_outliers
  SCAPULA_OUT_DIR=/path/to/out python3 scripts/plot_ssm_validation.py
"""

import os
import sys
import csv
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate output directory
# ---------------------------------------------------------------------------
DEFAULT_OUT = os.path.expanduser(
    "~/Documents/combined_scapula_data/scapula_ssm_out_fast_no_outliers"
)
out_dir = Path(os.environ.get("SCAPULA_OUT_DIR", DEFAULT_OUT))
if not out_dir.exists():
    sys.exit(f"Output directory not found: {out_dir}\n"
             "Set SCAPULA_OUT_DIR or adjust DEFAULT_OUT in this script.")

print(f"Reading CSVs from: {out_dir}")

# ---------------------------------------------------------------------------
# Import matplotlib (with Agg backend for headless environments)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
STYLE = {
    "hillsachs": dict(color="#2171b5", ls="-",  lw=2.0, marker="o", ms=4,
                      label="Hill-Sachs (N=49)"),
    "paired":    dict(color="#cb181d", ls="--", lw=2.0, marker="s", ms=4,
                      label="Paired controls (N=50)"),
}
ANNOT_KW = dict(fontsize=8, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points")

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def read_csv(path: Path):
    """Return header list and list-of-dicts (values as float)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [{k: float(v) for k, v in row.items()} for row in reader]
    return rows

def load_metric(group: str, metric: str):
    """Load one validation CSV or return None if not found."""
    fname = f"ssm_validation_{group}_{metric}.csv"
    p = out_dir / fname
    if not p.exists():
        print(f"  [warn] not found: {fname}")
        return None
    rows = read_csv(p)
    print(f"  loaded: {fname}  ({len(rows)} rows)")
    return rows

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 11))
fig.suptitle("SSM Validation  –  Scapula Dataset (99 specimens, outliers excluded)",
             fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.08, right=0.97, top=0.93, bottom=0.07)

# ----- helper: annotate a horizontal threshold line -----
def annotate_thresh(ax, rows, col_x, col_y, thresh, fmt_y, color, prefix=""):
    xs = [r[col_x] for r in rows]
    ys = [r[col_y] for r in rows]
    # find first x where y crosses threshold
    for x, y in zip(xs, ys):
        if y >= thresh:
            ax.axhline(thresh, color=color, lw=0.8, ls=":", alpha=0.6)
            ax.annotate(f"{prefix}{thresh:.0f}% → {x:.0f} modes",
                        xy=(x, thresh), color=color, **ANNOT_KW)
            break

def annotate_val(ax, rows, col_x, col_y, color, fmt=".2f"):
    last = rows[-1]
    ax.annotate(f"{last[col_y]:{fmt}} mm",
                xy=(last[col_x], last[col_y]),
                color=color, **ANNOT_KW)

# ---------------------------------------------------------------------------
# Panel 1 – COMPACTNESS  (left column = hillsachs, right = both combined)
# ---------------------------------------------------------------------------
ax_comp_hs  = fig.add_subplot(gs[0, 0])
ax_comp_pa  = fig.add_subplot(gs[0, 1], sharey=ax_comp_hs)

for ax, group in [(ax_comp_hs, "hillsachs"), (ax_comp_pa, "paired")]:
    rows = load_metric(group, "compactness")
    if rows is None:
        continue
    xs = [r["modes"] for r in rows]
    ys = [r["cumulative_variance_pct"] for r in rows]
    sty = STYLE[group]
    ax.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
            marker=sty["marker"], ms=sty["ms"], markevery=max(1, len(xs)//12),
            label=sty["label"])
    annotate_thresh(ax, rows, "modes", "cumulative_variance_pct", 90,
                    ".1f", sty["color"])
    annotate_thresh(ax, rows, "modes", "cumulative_variance_pct", 95,
                    ".1f", sty["color"])
    ax.set_xlabel("Number of modes", fontsize=9)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=9)
    title_suffix = "Hill-Sachs" if group == "hillsachs" else "Paired controls"
    ax.set_title(f"Compactness – {title_suffix}", fontsize=10)
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------------------
# Panel 2 – GENERALIZATION
# ---------------------------------------------------------------------------
ax_gen_hs = fig.add_subplot(gs[1, 0])
ax_gen_pa = fig.add_subplot(gs[1, 1])

for ax, group in [(ax_gen_hs, "hillsachs"), (ax_gen_pa, "paired")]:
    rows = load_metric(group, "generalization")
    if rows is None:
        continue
    xs = [r["modes"] for r in rows]
    ys = [r["mean_rms_error_mm"] for r in rows]
    sty = STYLE[group]
    ax.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
            marker=sty["marker"], ms=sty["ms"], markevery=max(1, len(xs)//12),
            label=sty["label"])
    annotate_val(ax, rows, "modes", "mean_rms_error_mm", sty["color"])
    ax.set_xlabel("Number of modes", fontsize=9)
    ax.set_ylabel("Mean RMS error (mm)", fontsize=9)
    title_suffix = "Hill-Sachs" if group == "hillsachs" else "Paired controls"
    ax.set_title(f"Generalization – {title_suffix}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------------------
# Panel 3 – SPECIFICITY
# ---------------------------------------------------------------------------
ax_spec_hs = fig.add_subplot(gs[2, 0])
ax_spec_pa = fig.add_subplot(gs[2, 1])

for ax, group in [(ax_spec_hs, "hillsachs"), (ax_spec_pa, "paired")]:
    rows = load_metric(group, "specificity")
    if rows is None:
        continue
    xs = [r["modes"] for r in rows]
    ys = [r["mean_min_rms_mm"] for r in rows]
    sty = STYLE[group]
    ax.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
            marker=sty["marker"], ms=sty["ms"],
            label=sty["label"])
    annotate_val(ax, rows, "modes", "mean_min_rms_mm", sty["color"])
    ax.set_xlabel("Number of modes used for sampling", fontsize=9)
    ax.set_ylabel("Mean min RMS dist. to training (mm)", fontsize=9)
    title_suffix = "Hill-Sachs" if group == "hillsachs" else "Paired controls"
    ax.set_title(f"Specificity (100 K samples) – {title_suffix}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------------------
# Save 6-panel figure
# ---------------------------------------------------------------------------
for ext in ("png", "svg"):
    save_path = out_dir / f"ssm_validation_plots.{ext}"
    fig.savefig(save_path, dpi=150 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {save_path}")

# ---------------------------------------------------------------------------
# Second figure – combined overlay (1 col × 3 rows, both groups per panel)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(3, 1, figsize=(7, 14))
fig2.suptitle("SSM Validation – Hillsachs vs Paired Controls",
              fontsize=12, fontweight="bold", y=0.99)
fig2.subplots_adjust(hspace=0.4, top=0.96, bottom=0.06)

METRIC_META = [
    ("compactness",    "cumulative_variance_pct",
     "Compactness",    "Cumulative variance explained (%)", (0, 102)),
    ("generalization", "mean_rms_error_mm",
     "Generalization", "Mean RMS reconstruction error (mm)", None),
    ("specificity",    "mean_min_rms_mm",
     "Specificity (100 K samples)", "Mean min RMS dist. to training (mm)", None),
]

for ax, (metric, col_y, title, ylabel, ylim) in zip(axes2, METRIC_META):
    for group in ("hillsachs", "paired"):
        rows = load_metric(group, metric)
        if rows is None:
            continue
        xs = [r["modes"] for r in rows]
        ys = [r[col_y] for r in rows]
        sty = STYLE[group]
        ax.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                marker=sty["marker"], ms=sty["ms"],
                markevery=max(1, len(xs)//12),
                label=sty["label"])
        if metric == "compactness":
            for thresh in (90, 95):
                for x, y in zip(xs, ys):
                    if y >= thresh:
                        ax.axvline(x, color=sty["color"], lw=0.7, ls=":", alpha=0.5)
                        ax.text(x + 0.3, thresh - 4,
                                f"{thresh}%→{x:.0f}", fontsize=7, color=sty["color"])
                        break

    ax.set_xlabel("Number of modes", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

for ext in ("png", "svg"):
    save_path = out_dir / f"ssm_validation_combined.{ext}"
    fig2.savefig(save_path, dpi=150 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {save_path}")

# ---------------------------------------------------------------------------
# Variance scree plots
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
fig3.suptitle("Scree Plots – Variance per Mode", fontsize=11, fontweight="bold")
fig3.subplots_adjust(wspace=0.3)

for ax, group in zip(axes3, ("hillsachs", "paired")):
    rows = load_metric(group, "variance")
    if rows is None:
        continue
    modes = [r["mode"] for r in rows]
    pcts  = [r["variance_pct"] for r in rows]
    cums  = [r["cumulative_pct"] for r in rows]
    sty   = STYLE[group]

    ax.bar(modes, pcts, color=sty["color"], alpha=0.6, label="Per-mode %")
    ax2 = ax.twinx()
    ax2.plot(modes, cums, color=sty["color"], lw=2, ls="-", label="Cumulative")
    ax2.set_ylabel("Cumulative variance (%)", fontsize=9, color=sty["color"])
    ax2.set_ylim(0, 102)
    ax2.axhline(90, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(95, color="grey", lw=0.8, ls=":",  alpha=0.6)

    title_suffix = "Hill-Sachs" if group == "hillsachs" else "Paired controls"
    ax.set_title(f"{title_suffix}  ({len(modes)} modes)", fontsize=10)
    ax.set_xlabel("Mode index", fontsize=9)
    ax.set_ylabel("Variance explained (%)", fontsize=9)
    ax.grid(True, alpha=0.2)

for ext in ("png",):
    save_path = out_dir / f"ssm_validation_scree.{ext}"
    fig3.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

print("Done.")
