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
    "combined":  dict(color="#238b45", ls="-",  lw=2.2, marker="D", ms=4,
                      label="Combined (N=99)"),
    "hillsachs": dict(color="#2171b5", ls="-",  lw=2.0, marker="o", ms=4,
                      label="Hill-Sachs (N=49)"),
    "paired":    dict(color="#cb181d", ls="--", lw=2.0, marker="s", ms=4,
                      label="Paired controls (N=50)"),
}
GROUPS_PER_GROUP   = ("hillsachs", "paired")
GROUPS_COMBINED    = ("combined",)
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
# Build figure A – combined model only (3 panels × 1 col)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(7, 14))
fig.suptitle("SSM Validation  –  Combined (N=99, outliers excluded)",
             fontsize=12, fontweight="bold", y=0.99)
fig.subplots_adjust(hspace=0.4, top=0.96, bottom=0.06)

# ----- helpers -----
def annotate_thresh(ax, rows, col_x, col_y, thresh, color):
    for r in rows:
        if r[col_y] >= thresh:
            x = r[col_x]
            ax.axhline(thresh, color=color, lw=0.8, ls=":", alpha=0.5)
            ax.annotate(f"{thresh:.0f}% → {x:.0f} modes",
                        xy=(x, thresh), color=color, **ANNOT_KW)
            break

def annotate_val(ax, rows, col_x, col_y, color):
    last = rows[-1]
    ax.annotate(f"{last[col_y]:.2f} mm",
                xy=(last[col_x], last[col_y]),
                color=color, **ANNOT_KW)

ax_comp  = fig.add_subplot(3, 1, 1)
ax_gen   = fig.add_subplot(3, 1, 2)
ax_spec  = fig.add_subplot(3, 1, 3)

for group in ("combined",):
    sty = STYLE[group]

    rows = load_metric(group, "compactness")
    if rows:
        xs = [r["modes"] for r in rows]
        ys = [r["cumulative_variance_pct"] for r in rows]
        ax_comp.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                     marker=sty["marker"], ms=sty["ms"],
                     markevery=max(1, len(xs)//15), label=sty["label"])
        annotate_thresh(ax_comp, rows, "modes", "cumulative_variance_pct", 90, sty["color"])
        annotate_thresh(ax_comp, rows, "modes", "cumulative_variance_pct", 95, sty["color"])

    rows = load_metric(group, "generalization")
    if rows:
        xs = [r["modes"] for r in rows]
        ys = [r["mean_rms_error_mm"] for r in rows]
        ax_gen.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                    marker=sty["marker"], ms=sty["ms"],
                    markevery=max(1, len(xs)//15), label=sty["label"])
        annotate_val(ax_gen, rows, "modes", "mean_rms_error_mm", sty["color"])

    rows = load_metric(group, "specificity")
    if rows:
        xs = [r["modes"] for r in rows]
        ys = [r["mean_min_rms_mm"] for r in rows]
        ax_spec.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                     marker=sty["marker"], ms=sty["ms"], label=sty["label"])
        annotate_val(ax_spec, rows, "modes", "mean_min_rms_mm", sty["color"])
        n_spec = 100_000
        ax_spec.set_title(f"Specificity ({n_spec:,} samples)", fontsize=10, fontweight="bold")

ax_comp.set_ylabel("Cumulative variance (%)", fontsize=9)
ax_comp.set_xlabel("Number of modes", fontsize=9)
ax_comp.set_title("Compactness", fontsize=10, fontweight="bold")
ax_comp.set_ylim(0, 102)
ax_comp.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax_comp.legend(fontsize=8, loc="lower right"); ax_comp.grid(True, alpha=0.25)

ax_gen.set_ylabel("Mean RMS reconstruction error (mm)", fontsize=9)
ax_gen.set_xlabel("Number of modes", fontsize=9)
ax_gen.set_title("Generalization", fontsize=10, fontweight="bold")
ax_gen.legend(fontsize=8, loc="upper right"); ax_gen.grid(True, alpha=0.25)

ax_spec.set_ylabel("Mean min RMS dist. to training set (mm)", fontsize=9)
ax_spec.set_xlabel("Number of modes used for sampling", fontsize=9)
ax_spec.legend(fontsize=8, loc="upper left"); ax_spec.grid(True, alpha=0.25)

# ---------------------------------------------------------------------------
# Build figure B – group comparison (3 rows × 2 cols)
# ---------------------------------------------------------------------------
gs = gridspec.GridSpec(3, 2, figure=plt.figure(figsize=(13, 11)),
                       hspace=0.45, wspace=0.35,
                       left=0.08, right=0.97, top=0.93, bottom=0.07)
fig_grp = gs.figure
fig_grp.suptitle("SSM Validation  –  Hill-Sachs vs Paired Controls",
                 fontsize=13, fontweight="bold", y=0.98)

PANEL_GROUPS = [("hillsachs", "Hill-Sachs"), ("paired", "Paired controls")]

for col, (group, gtitle) in enumerate(PANEL_GROUPS):
    sty = STYLE[group]

    ax_c = fig_grp.add_subplot(gs[0, col])
    rows = load_metric(group, "compactness")
    if rows:
        xs = [r["modes"] for r in rows]
        ys = [r["cumulative_variance_pct"] for r in rows]
        ax_c.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                  marker=sty["marker"], ms=sty["ms"],
                  markevery=max(1, len(xs)//12), label=sty["label"])
        annotate_thresh(ax_c, rows, "modes", "cumulative_variance_pct", 90, sty["color"])
        annotate_thresh(ax_c, rows, "modes", "cumulative_variance_pct", 95, sty["color"])
    ax_c.set_xlabel("Number of modes", fontsize=9)
    ax_c.set_ylabel("Cumulative variance (%)", fontsize=9)
    ax_c.set_title(f"Compactness – {gtitle}", fontsize=10)
    ax_c.set_ylim(0, 102)
    ax_c.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_c.legend(fontsize=8, loc="lower right"); ax_c.grid(True, alpha=0.25)

    ax_g = fig_grp.add_subplot(gs[1, col])
    rows = load_metric(group, "generalization")
    if rows:
        xs = [r["modes"] for r in rows]
        ys = [r["mean_rms_error_mm"] for r in rows]
        ax_g.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                  marker=sty["marker"], ms=sty["ms"],
                  markevery=max(1, len(xs)//12), label=sty["label"])
        annotate_val(ax_g, rows, "modes", "mean_rms_error_mm", sty["color"])
    ax_g.set_xlabel("Number of modes", fontsize=9)
    ax_g.set_ylabel("Mean RMS error (mm)", fontsize=9)
    ax_g.set_title(f"Generalization – {gtitle}", fontsize=10)
    ax_g.legend(fontsize=8, loc="upper right"); ax_g.grid(True, alpha=0.25)

    ax_s = fig_grp.add_subplot(gs[2, col])
    rows = load_metric(group, "specificity")
    if rows:
        xs = [r["modes"] for r in rows]
        ys = [r["mean_min_rms_mm"] for r in rows]
        ax_s.plot(xs, ys, color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                  marker=sty["marker"], ms=sty["ms"], label=sty["label"])
        annotate_val(ax_s, rows, "modes", "mean_min_rms_mm", sty["color"])
    ax_s.set_xlabel("Number of modes used for sampling", fontsize=9)
    ax_s.set_ylabel("Mean min RMS dist. (mm)", fontsize=9)
    ax_s.set_title(f"Specificity (100 K) – {gtitle}", fontsize=10)
    ax_s.legend(fontsize=8, loc="upper left"); ax_s.grid(True, alpha=0.25)

# ---------------------------------------------------------------------------
# Save figures
# ---------------------------------------------------------------------------
for ext in ("png", "svg"):
    p = out_dir / f"ssm_validation_combined99.{ext}"
    fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {p}")

for ext in ("png", "svg"):
    p = out_dir / f"ssm_validation_groups.{ext}"
    fig_grp.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {p}")


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
