#!/usr/bin/env python3
"""
SSM validation metric plots for the scapula pipeline.

Usage:
  pip install matplotlib seaborn pandas numpy
  python3 plot_ssm_metrics.py --plots-dir "/path/to/plots"

Reads:
  compactness_pass<N>.csv
  generalization_pass<N>.csv
  specificity_pass<N>.csv
  distance_to_mean_pass<N>.csv
  pairwise_distances_pass<N>.csv
  stability.csv

Writes:  <plots-dir>/ssm_validation.png  (and shows the window)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── palette ────────────────────────────────────────────────────────────────────
COLORS = ["#3B82F6", "#8B5CF6", "#F59E0B", "#34D399", "#F87171", "#22D3EE"]
STYLE  = {
    "axes.facecolor":   "#0F1929",
    "figure.facecolor": "#070D1A",
    "axes.edgecolor":   "#1E3050",
    "grid.color":       "#1E3050",
    "text.color":       "#E2EAF8",
    "axes.labelcolor":  "#94A3B8",
    "xtick.color":      "#6A85A8",
    "ytick.color":      "#6A85A8",
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
}


def load_passes(plots_dir: Path, prefix: str, passes: list[int]) -> dict[int, pd.DataFrame]:
    out = {}
    for n in passes:
        f = plots_dir / f"{prefix}_pass{n}.csv"
        if f.exists():
            out[n] = pd.read_csv(f)
    return out


def find_passes(plots_dir: Path) -> list[int]:
    ns = []
    for f in sorted(plots_dir.glob("compactness_pass*.csv")):
        try:
            ns.append(int(f.stem.split("_pass")[1]))
        except ValueError:
            pass
    return sorted(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-dir", required=True)
    ap.add_argument("--out", default=None, help="output PNG path (default: <plots-dir>/ssm_validation.png)")
    args = ap.parse_args()

    plots_dir = Path(args.plots_dir)
    if not plots_dir.is_dir():
        sys.exit(f"ERROR: {plots_dir} is not a directory")

    passes = find_passes(plots_dir)
    if not passes:
        sys.exit(f"ERROR: no compactness_pass*.csv files found in {plots_dir}")
    print(f"Found passes: {passes}")

    out_path = Path(args.out) if args.out else plots_dir / "ssm_validation.png"

    # ── load data ──────────────────────────────────────────────────────────────
    compact  = load_passes(plots_dir, "compactness",         passes)
    gen      = load_passes(plots_dir, "generalization",      passes)
    spec     = load_passes(plots_dir, "specificity",         passes)
    dtm      = load_passes(plots_dir, "distance_to_mean",    passes)
    stab_f   = plots_dir / "stability.csv"
    stab     = pd.read_csv(stab_f) if stab_f.exists() else None

    matplotlib.rcParams.update(STYLE)
    nP = len(passes)

    # ── layout ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle("Scapula SSM — Validation Metrics", fontsize=14,
                 color="#F1F7FF", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.38,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    ax_comp = fig.add_subplot(gs[0, 0])
    ax_gen  = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[0, 2])
    ax_dtm  = fig.add_subplot(gs[1, 0:2])
    ax_pair = fig.add_subplot(gs[1, 2])
    ax_stab = fig.add_subplot(gs[2, 0])
    ax_cum  = fig.add_subplot(gs[2, 1])
    ax_info = fig.add_subplot(gs[2, 2])

    # ── 1. Compactness ─────────────────────────────────────────────────────────
    for n, df in compact.items():
        c = COLORS[passes.index(n)]
        ax_comp.plot(df["mode"], df["cumulative_variance_pct"],
                     color=c, lw=2, label=f"SSM{n}")
    ax_comp.axhline(90, color="#6A85A8", lw=0.8, ls=":")
    ax_comp.axhline(95, color="#6A85A8", lw=0.8, ls=":")
    ax_comp.axhline(99, color="#6A85A8", lw=0.8, ls=":")
    ax_comp.set_xlabel("# modes")
    ax_comp.set_ylabel("cumulative variance %")
    ax_comp.set_title("Compactness", color="#F1F7FF")
    ax_comp.legend(fontsize=8)
    ax_comp.set_ylim(0, 101)

    # ── 2. Generalization ──────────────────────────────────────────────────────
    for n, df in gen.items():
        c = COLORS[passes.index(n)]
        ax_gen.plot(df["num_modes"], df["mean_error_mm"],
                    color=c, lw=2, label=f"SSM{n}")
    ax_gen.set_xlabel("# modes")
    ax_gen.set_ylabel("mean error (mm)")
    ax_gen.set_title("Generalization", color="#F1F7FF")
    ax_gen.legend(fontsize=8)

    # ── 3. Specificity ─────────────────────────────────────────────────────────
    for n, df in spec.items():
        c = COLORS[passes.index(n)]
        ax_spec.plot(df["num_modes"], df["mean_specificity_mm"],
                     color=c, lw=2, label=f"SSM{n}")
        ax_spec.fill_between(
            df["num_modes"],
            df["mean_specificity_mm"] - df["std_specificity_mm"],
            df["mean_specificity_mm"] + df["std_specificity_mm"],
            alpha=0.15, color=c)
    ax_spec.set_xlabel("# modes")
    ax_spec.set_ylabel("min dist to training (mm)")
    ax_spec.set_title("Specificity", color="#F1F7FF")
    ax_spec.legend(fontsize=8)

    # ── 4. Distance to mean (box plot per SSM) ─────────────────────────────────
    box_data, box_labels, box_colors = [], [], []
    for n, df in dtm.items():
        box_data.append(df["mean_mm"].values)
        box_labels.append(f"SSM{n}")
        box_colors.append(COLORS[passes.index(n)])

    bp = ax_dtm.boxplot(box_data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="#F1F7FF", lw=2))
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.4)
    for element in ["whiskers","caps","fliers"]:
        for item in bp[element]:
            item.set_color("#6A85A8")
    ax_dtm.set_xticklabels(box_labels)
    ax_dtm.set_ylabel("mean surface dist to SSM mean (mm)")
    ax_dtm.set_title("Distance to Mean Shape", color="#F1F7FF")

    # ── 5. Pairwise distances (heatmap of mean column) ─────────────────────────
    pw_files = list(plots_dir.glob("pairwise_distances_pass*.csv"))
    if pw_files:
        # show mean pairwise distance per specimen from last pass
        last_n = max(passes)
        pw_f = plots_dir / f"pairwise_distances_pass{last_n}.csv"
        if pw_f.exists():
            pw = pd.read_csv(pw_f)
            # aggregate: average mean_mm per specimen_i
            per_spec = pw.groupby("specimen_i")["mean_mm"].mean().sort_values()
            ax_pair.barh(range(len(per_spec)), per_spec.values,
                         color=COLORS[passes.index(last_n)], alpha=0.7)
            ax_pair.set_yticks(range(len(per_spec)))
            ax_pair.set_yticklabels(
                [s[-8:] for s in per_spec.index], fontsize=7)
            ax_pair.set_xlabel("avg pairwise dist (mm)")
            ax_pair.set_title(f"Pairwise Dist (SSM{last_n})", color="#F1F7FF")
    else:
        ax_pair.set_visible(False)

    # ── 6. Stability (consecutive mean shift) ──────────────────────────────────
    if stab is not None and len(stab) > 0:
        labels = [f"SSM{int(r.pass_from)}→{int(r.pass_to)}" for _, r in stab.iterrows()]
        vals   = stab["mean_mm"].values
        bars = ax_stab.bar(labels, vals,
                           color=[COLORS[i] for i in range(len(vals))], alpha=0.8)
        ax_stab.axhline(1.0, color="#F87171", lw=1, ls="--", label="1 mm threshold")
        for bar, v in zip(bars, vals):
            ax_stab.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                         color="#F1F7FF")
        ax_stab.set_ylabel("mean surface dist (mm)")
        ax_stab.set_title("Mean-Shape Stability", color="#F1F7FF")
        ax_stab.legend(fontsize=8)
    else:
        ax_stab.set_visible(False)

    # ── 7. Cumulative variance comparison table ─────────────────────────────────
    pct_targets = [90, 95, 99]
    x = np.arange(len(pct_targets))
    w = 0.8 / max(nP, 1)
    for ki, (n, df) in enumerate(compact.items()):
        cv = df["cumulative_variance_pct"].values
        modes_at = []
        for pct in pct_targets:
            idx = np.searchsorted(cv, pct)
            modes_at.append(int(df["mode"].iloc[min(idx, len(df)-1)]))
        offset = (ki - (nP-1)/2) * w
        bars2 = ax_cum.bar(x + offset, modes_at, width=w*0.85,
                           color=COLORS[ki], alpha=0.8, label=f"SSM{n}")
        for bar, v in zip(bars2, modes_at):
            ax_cum.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                        str(v), ha="center", va="bottom", fontsize=8,
                        color="#F1F7FF")
    ax_cum.set_xticks(x)
    ax_cum.set_xticklabels([f"{p}% var" for p in pct_targets])
    ax_cum.set_ylabel("# modes required")
    ax_cum.set_title("Modes for Variance Explained", color="#F1F7FF")
    ax_cum.legend(fontsize=8)

    # ── 8. Summary text panel ──────────────────────────────────────────────────
    ax_info.axis("off")
    lines = ["SSM SUMMARY\n"]
    for n, df in compact.items():
        cv = df["cumulative_variance_pct"].values
        modes_list = df["mode"].values
        def m_at(pct):
            idx = np.searchsorted(cv, pct)
            return int(modes_list[min(idx, len(df)-1)])
        lines.append(f"SSM{n}  rank={len(df)}")
        lines.append(f"  90%: {m_at(90)} modes")
        lines.append(f"  95%: {m_at(95)} modes")
        lines.append(f"  99%: {m_at(99)} modes")
        if n in dtm:
            avg = dtm[n]["mean_mm"].mean()
            lines.append(f"  avg dist-to-mean: {avg:.3f} mm")
        lines.append("")
    if stab is not None and len(stab) > 0:
        lines.append("CONVERGENCE")
        for _, r in stab.iterrows():
            tag = "✓" if r.mean_mm < 1.0 else "!"
            lines.append(f"  {tag} SSM{int(r.pass_from)}→{int(r.pass_to)}: {r.mean_mm:.3f} mm")

    ax_info.text(0.05, 0.95, "\n".join(lines), transform=ax_info.transAxes,
                 va="top", ha="left", fontsize=8.5,
                 color="#E2EAF8", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#0F1929",
                           edgecolor="#1E3050", alpha=0.9))

    # ── save ───────────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
