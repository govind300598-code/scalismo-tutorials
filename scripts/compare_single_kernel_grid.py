#!/usr/bin/env python3
"""
Compares N single-Gaussian-kernel runs (SCAPULA_KERNEL_SIGMA / SCAPULA_KERNEL_SCALE grid
search) and prints/writes ONE ranked table across all of them.

Each directory must already contain a completed Stage2ReferenceRefinement run for that
(sigma, scaleFactor) point (SCAPULA_OUT_DIR set to that directory when it ran).

Usage:
    python3 scripts/compare_single_kernel_grid.py <label1>=<dir1> <label2>=<dir2> ... [--out report.md]

Example (the 9-point grid from this session):
    python3 scripts/compare_single_kernel_grid.py \\
        "50:100"=~/Documents/database_v1.11/grid_s50_a100 \\
        "50:200"=~/Documents/database_v1.11/grid_s50_a200 \\
        "75:100"=~/Documents/database_v1.11/grid_s75_a100 \\
        "75:200"=~/Documents/database_v1.11/grid_s75_a200 \\
        "100:100"=~/Documents/database_v1.11/grid_s100_a100 \\
        "100:150"=~/Documents/database_v1.11/grid_s100_a150 \\
        "100:200"=~/Documents/database_v1.11/grid_s100_a200 \\
        "150:100"=~/Documents/database_v1.11/grid_s150_a100 \\
        "150:200"=~/Documents/database_v1.11/grid_s150_a200

Ranks by fit_mean_mm (= Chamfer distance / ASSD) ascending, ties broken by fit_hd95_mm.
Requires: pandas.
"""
import os
import sys

import pandas as pd

METRIC_COLS = [
    ("fit_mean_mm", "Mean / Chamfer (mm)"),
    ("fit_rms_mm", "RMSE (mm)"),
    ("fit_hd95_mm", "HD95 (mm)"),
    ("fit_hd_mm", "Hausdorff, max (mm)"),
    ("fit_landmark_rmse_mm", "Landmark RMSE (mm)"),
]


def load_one(label, path):
    path = os.path.expanduser(path)
    csv_path = os.path.join(path, "final_fit_quality.csv")
    if not os.path.exists(csv_path):
        print(f"!! {label}: {csv_path} not found -- skipping (did Stage2ReferenceRefinement finish there?)")
        return None
    df = pd.read_csv(csv_path)
    row = {"label": label, "n_subjects": len(df)}
    for col, _ in METRIC_COLS:
        row[col] = df[col].mean()

    cfg_path = os.path.join(path, "run_config.txt")
    if os.path.exists(cfg_path):
        cfg = dict(zip(pd.read_csv(cfg_path)["key"], pd.read_csv(cfg_path)["value"].astype(str)))
        row["kernelSigma"] = cfg.get("kernelSigma", "?")
        row["kernelScale"] = cfg.get("kernelScale", "?")
    return row


def main():
    args = sys.argv[1:]
    out_path = "single_kernel_grid_report.md"
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        args = args[:i] + args[i + 2:]

    if not args:
        print(__doc__)
        sys.exit(1)

    rows = []
    for pair in args:
        if "=" not in pair:
            print(f"!! skipping malformed argument (expected label=dir): {pair}")
            continue
        label, path = pair.split("=", 1)
        r = load_one(label, path)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No completed runs found among the given directories.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("fit_mean_mm").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    lines = []
    lines.append("# Single-Gaussian-kernel grid search -- ranked results")
    lines.append("")
    lines.append(f"{len(df)} of {len(args)} requested configurations completed and loaded.")
    lines.append("")
    header = ["Rank", "sigma:scale"] + [name for _, name in METRIC_COLS] + ["N"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for _, r in df.iterrows():
        cells = [str(r["rank"]), r["label"]] + [f"{r[c]:.3f}" for c, _ in METRIC_COLS] + [str(int(r["n_subjects"]))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    best = df.iloc[0]
    lines.append(f"**Best by mean surface distance / Chamfer:** `{best['label']}` "
                 f"(mean={best['fit_mean_mm']:.3f} mm, HD95={best['fit_hd95_mm']:.3f} mm, "
                 f"HD={best['fit_hd_mm']:.3f} mm)")
    lines.append("")
    lines.append("Standard metrics used here, and their names in the literature (Styner et al. 2003; "
                  "general point-cloud/registration literature):")
    lines.append("- **Mean surface distance** == **Chamfer distance** (point-cloud/CV term) == "
                  "**ASSD**, average symmetric surface distance (medical-imaging term) -- identical computation.")
    lines.append("- **RMSE** -- root-mean-square of the same per-point distances.")
    lines.append("- **HD95** -- 95th-percentile Hausdorff distance, the outlier-robust variant used in most "
                  "medical image segmentation/registration papers instead of raw Hausdorff.")
    lines.append("- **Hausdorff (max)** -- the true worst-case point-to-surface distance; sensitive to a single "
                  "outlier point, reported alongside HD95 for that reason.")

    report = "\n".join(lines)
    print(report)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
