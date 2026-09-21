#!/usr/bin/env python3
"""
Reads every sigma<S>_s<A> output directory produced by scripts/run_single_kernel_sweep.sh (one per
(sigma, s) pair in the single-Gaussian-kernel selection grid) and builds one ranked comparison table
covering both the standard distance-error metrics (final_fit_quality.csv) and the standard SSM
validation metrics (pca_compactness.csv, pca_specificity.csv, pca_generalization.csv) -- the same two
families of numbers scripts/compare_multiscale_kernel_experiment.py already uses to compare the 2-term vs. 3-term
multi-kernel pipeline, generalized here from a 2-way comparison to an N-way sweep over (sigma, s).

Each subdirectory must already contain the output of running, in order, with SCAPULA_OUT_DIR set to it:
    sbt "runMain scapula.singlekernel.Stage2SingleKernelReferenceRefinement"
    sbt "runMain scapula.singlekernel.Stage3PCAModel"
(run_single_kernel_sweep.sh does exactly this for every grid point.)

Usage:
    python3 scripts/compare_single_kernel_sweep.py <sweep_base_dir> [report_path]

Requires: pandas (ships with a standard conda install).
"""
import glob
import os
import sys

import pandas as pd

DIST_COLUMNS = [
    ("fit_mean_mm", "Mean surface dist. (Chamfer/ASSD)"),
    ("fit_rms_mm", "RMSE"),
    ("fit_hd95_mm", "HD95"),
    ("fit_hd_mm", "Hausdorff (max)"),
    ("fit_landmark_rmse_mm", "Landmark RMSE"),
]

# Metrics used for the composite "best value" ranking, and whether lower is better for each. This is a
# convenience summary, not a substitute for reading the full table: "best" is genuinely multi-criteria, and a
# config that wins the composite but is a clear outlier on one metric (e.g. HD95) is worth a second look.
RANKING_METRICS = [
    ("fit_mean_mm", True),
    ("fit_hd95_mm", True),
    ("fit_landmark_rmse_mm", True),
    ("generalization_full_rank_mm", True),
    ("specificity_full_rank_mm", True),
]


def read_run_config(run_dir):
    path = os.path.join(run_dir, "run_config.txt")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["key"], df["value"]))


def distance_summary(run_dir):
    path = os.path.join(run_dir, "final_fit_quality.csv")
    if not os.path.exists(path):
        return None, 0
    df = pd.read_csv(path)
    return {col: df[col].mean() for col, _ in DIST_COLUMNS}, len(df)


def ssm_summary(run_dir):
    """Full-rank compactness / specificity / generalization numbers (last row of each curve)."""
    result = {}
    compact = os.path.join(run_dir, "pca_compactness.csv")
    if os.path.exists(compact):
        df = pd.read_csv(compact)
        result["rank"] = int(df["num_components"].max())
        n90 = df[df["cumulative_variance_fraction"] >= 0.90]["num_components"].min()
        result["modes_for_90pct_variance"] = int(n90) if pd.notna(n90) else None
    specificity = os.path.join(run_dir, "pca_specificity.csv")
    if os.path.exists(specificity):
        df = pd.read_csv(specificity)
        result["specificity_full_rank_mm"] = df["mean_distance_to_nearest_real_subject_mm"].iloc[-1]
    generalization = os.path.join(run_dir, "pca_generalization.csv")
    if os.path.exists(generalization):
        df = pd.read_csv(generalization)
        result["generalization_full_rank_mm"] = df["mean_reconstruction_error_mm"].iloc[-1]
    return result


def collect_rows(base_dir):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(base_dir, "sigma*_s*"))):
        if not os.path.isdir(run_dir):
            continue
        cfg = read_run_config(run_dir)
        dist, n_subjects = distance_summary(run_dir)
        if dist is None:
            print(f"(skipping {run_dir}: no final_fit_quality.csv -- did Stage2 finish?)")
            continue
        ssm = ssm_summary(run_dir)

        sigma = cfg.get("sigma_mm")
        scale = cfg.get("scale_mm")
        if sigma is None or scale is None:
            # Fall back to parsing the directory name if run_config.txt is missing/older.
            base = os.path.basename(run_dir)
            try:
                sigma = float(base.split("sigma")[1].split("_s")[0])
                scale = float(base.split("_s")[-1])
            except (IndexError, ValueError):
                print(f"(skipping {run_dir}: could not determine sigma/s)")
                continue

        row = {"run_dir": run_dir, "sigma_mm": float(sigma), "scale_mm": float(scale), "n_subjects": n_subjects}
        row.update(dist)
        row.update(ssm)
        rows.append(row)
    return pd.DataFrame(rows)


def add_composite_rank(df):
    """Lower composite_score = better. Averages each metric's min-max-normalized value (0=best, 1=worst) over
    every RANKING_METRICS column that is actually present for every row; missing columns are skipped."""
    usable = [(col, lower_is_better) for col, lower_is_better in RANKING_METRICS if col in df.columns and df[col].notna().all()]
    if not usable:
        df["composite_score"] = float("nan")
        return df, []
    norm = pd.DataFrame(index=df.index)
    for col, lower_is_better in usable:
        span = df[col].max() - df[col].min()
        if span == 0:
            norm[col] = 0.0
        else:
            n = (df[col] - df[col].min()) / span
            norm[col] = n if lower_is_better else (1 - n)
    df["composite_score"] = norm.mean(axis=1)
    return df, [c for c, _ in usable]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    base_dir = sys.argv[1]
    report_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base_dir, "single_kernel_sweep_comparison.md")

    df = collect_rows(base_dir)
    if df.empty:
        print(f"No completed sigma*_s* runs found under {base_dir}.")
        sys.exit(1)

    df, used_for_ranking = add_composite_rank(df)
    df = df.sort_values("composite_score")
    best = df.iloc[0]

    lines = []
    lines.append("# Single-Gaussian-kernel selection: (sigma, s) sweep comparison")
    lines.append("")
    lines.append(f"- Base directory: `{base_dir}`")
    lines.append(f"- Grid points compared: {len(df)}")
    lines.append("")
    lines.append("## Distance error metrics (mean across all subjects, final model)")
    lines.append("")
    header = ["sigma (mm)", "s (mm)", "N"] + [label for _, label in DIST_COLUMNS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for _, r in df.iterrows():
        cells = [f"{r['sigma_mm']:.0f}", f"{r['scale_mm']:.0f}", f"{int(r['n_subjects'])}"]
        cells += [f"{r[col]:.3f}" if col in r and pd.notna(r[col]) else "n/a" for col, _ in DIST_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## SSM validation (Styner et al. 2003), at full model rank")
    lines.append("")
    lines.append("| sigma (mm) | s (mm) | Model rank | Modes for 90% variance | Specificity (mm) | Generalization (mm) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        spec = f"{r['specificity_full_rank_mm']:.3f}" if pd.notna(r.get("specificity_full_rank_mm")) else "n/a"
        gen = f"{r['generalization_full_rank_mm']:.3f}" if pd.notna(r.get("generalization_full_rank_mm")) else "n/a"
        rank = r.get("rank", "n/a")
        modes90 = r.get("modes_for_90pct_variance", "n/a")
        lines.append(f"| {r['sigma_mm']:.0f} | {r['scale_mm']:.0f} | {rank} | {modes90} | {spec} | {gen} |")
    lines.append("")
    lines.append("## Ranked by composite score (lower = better)")
    lines.append("")
    lines.append(f"Composite score averages the min-max-normalized value of: {', '.join(used_for_ranking) if used_for_ranking else 'n/a'} "
                 "(each contributes equally; this is a convenience summary, not a single ground truth -- read the full "
                 "tables above too, especially if the winner is only marginally ahead or is an outlier on one metric).")
    lines.append("")
    lines.append("| Rank | sigma (mm) | s (mm) | Composite score |")
    lines.append("|---|---|---|---|")
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        marker = "  <-- recommended" if i == 1 else ""
        lines.append(f"| {i} | {r['sigma_mm']:.0f} | {r['scale_mm']:.0f} | {r['composite_score']:.4f}{marker} |")
    lines.append("")
    lines.append(f"**Recommendation: sigma={best['sigma_mm']:.0f} mm, s={best['scale_mm']:.0f} mm** "
                 f"(run: `{best['run_dir']}`) -- lowest composite score across distance-error and SSM validation metrics.")

    report = "\n".join(lines)
    print(report)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote {report_path}")
    print(f"Also inspect the meshes visually: SCAPULA_OUT_DIR='{best['run_dir']}' sbt \"runMain scapula.singlekernel.ViewResults\"")


if __name__ == "__main__":
    main()
