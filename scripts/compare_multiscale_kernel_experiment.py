#!/usr/bin/env python3
"""
Compares two Stage2ReferenceRefinement + Stage3PCAModel runs made with different
SCAPULA_KERNEL_TERMS settings (3-term coarse+mid+fine vs. 2-term coarse+fine), and
prints/writes a single markdown table covering both the standard distance-error
metrics (final_fit_quality.csv) and the standard SSM validation metrics
(pca_compactness.csv, pca_specificity.csv, pca_generalization.csv).

This is the direct ablation of Dennis Madsen's mailing-list fix: his ORIGINAL post used
2 kernel terms (coarse+fine, no mid), his FIXED reply added the mid term. This answers
"did adding it actually help, on this dataset" with numbers instead of intuition.

Usage:
    python3 scripts/compare_multiscale_kernel_experiment.py <3kernel_out_dir> <2kernel_out_dir> [report_path]

Each <out_dir> must already contain the output of running, in order, on THAT SAME
directory (set via SCAPULA_OUT_DIR):
    sbt "runMain scapula.Stage2ReferenceRefinement"
    sbt "runMain scapula.Stage3PCAModel"

Requires: pandas (ships with a standard conda install).
"""
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


def read_run_config(out_dir):
    path = os.path.join(out_dir, "run_config.txt")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["key"], df["value"]))


def distance_summary(out_dir):
    path = os.path.join(out_dir, "final_fit_quality.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run Stage2ReferenceRefinement with SCAPULA_OUT_DIR={out_dir}")
    df = pd.read_csv(path)
    return {col: df[col].mean() for col, _ in DIST_COLUMNS}, len(df)


def ssm_summary(out_dir):
    """Full-rank compactness / specificity / generalization numbers (last row of each curve)."""
    result = {}
    compact = os.path.join(out_dir, "pca_compactness.csv")
    if os.path.exists(compact):
        df = pd.read_csv(compact)
        result["rank"] = int(df["num_components"].max())
        n90 = df[df["cumulative_variance_fraction"] >= 0.90]["num_components"].min()
        result["modes_for_90pct_variance"] = int(n90) if pd.notna(n90) else None
    specificity = os.path.join(out_dir, "pca_specificity.csv")
    if os.path.exists(specificity):
        df = pd.read_csv(specificity)
        result["specificity_full_rank_mm"] = df["mean_distance_to_nearest_real_subject_mm"].iloc[-1]
    generalization = os.path.join(out_dir, "pca_generalization.csv")
    if os.path.exists(generalization):
        df = pd.read_csv(generalization)
        result["generalization_full_rank_mm"] = df["mean_reconstruction_error_mm"].iloc[-1]
    return result


def pct_change(a, b):
    """b relative to a, as a signed percentage; positive = b is worse (larger)."""
    if a == 0:
        return float("nan")
    return (b - a) / a * 100.0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    dir3, dir2 = sys.argv[1], sys.argv[2]
    report_path = sys.argv[3] if len(sys.argv) > 3 else "multiscale_kernel_comparison.md"

    cfg3, cfg2 = read_run_config(dir3), read_run_config(dir2)
    for label, cfg, expected in (("3-kernel dir", cfg3, "3"), ("2-kernel dir", cfg2, "2")):
        actual = str(cfg.get("kernelTerms", "?"))
        if actual != expected:
            print(f"!! WARNING: {label} ({sys.argv[1 if expected=='3' else 2]}) has run_config.txt "
                  f"kernelTerms={actual}, expected {expected}. Did you set SCAPULA_KERNEL_TERMS correctly "
                  "for that run? Continuing anyway.")

    dist3, n3 = distance_summary(dir3)
    dist2, n2 = distance_summary(dir2)
    if n3 != n2:
        print(f"!! WARNING: {n3} subjects in the 3-kernel run vs {n2} in the 2-kernel run -- "
              "not an apples-to-apples comparison unless these match (check SCAPULA_SUBJECT_LIMIT / SCAPULA_TOP_K "
              "were identical for both runs).")

    ssm3, ssm2 = ssm_summary(dir3), ssm_summary(dir2)

    lines = []
    lines.append("# Kernel ablation: 3-term (coarse+mid+fine) vs. 2-term (coarse+fine)")
    lines.append("")
    lines.append(f"- 3-kernel run: `{dir3}` (N={n3} subjects)")
    lines.append(f"- 2-kernel run: `{dir2}` (N={n2} subjects)")
    lines.append("")
    lines.append("## Distance error metrics (mean across all subjects, final model)")
    lines.append("")
    lines.append("| Metric | 3-kernel | 2-kernel | Change (2-kernel vs 3-kernel) |")
    lines.append("|---|---:|---:|---:|")
    for col, label in DIST_COLUMNS:
        v3, v2 = dist3[col], dist2[col]
        change = pct_change(v3, v2)
        arrow = "worse" if change > 0 else ("better" if change < 0 else "same")
        lines.append(f"| {label} | {v3:.3f} mm | {v2:.3f} mm | {change:+.1f}% ({arrow}) |")
    lines.append("")
    lines.append("## SSM validation (Styner et al. 2003), at full model rank")
    lines.append("")
    lines.append("| Metric | 3-kernel | 2-kernel |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Model rank (num. modes) | {ssm3.get('rank', 'n/a')} | {ssm2.get('rank', 'n/a')} |")
    lines.append(f"| Modes needed for 90% variance (compactness) | {ssm3.get('modes_for_90pct_variance', 'n/a')} | "
                  f"{ssm2.get('modes_for_90pct_variance', 'n/a')} |")
    s3 = ssm3.get("specificity_full_rank_mm")
    s2 = ssm2.get("specificity_full_rank_mm")
    lines.append("| Specificity (mean dist. to nearest real subject) | " +
                 (f"{s3:.3f} mm" if s3 is not None else "n/a") + " | " +
                 (f"{s2:.3f} mm" if s2 is not None else "n/a") + " |")
    g3 = ssm3.get("generalization_full_rank_mm")
    g2 = ssm2.get("generalization_full_rank_mm")
    lines.append(f"| Generalization (leave-one-out recon. error) | " +
                 (f"{g3:.3f} mm" if g3 is not None else "n/a") + " | " +
                 (f"{g2:.3f} mm" if g2 is not None else "n/a") + " |")
    lines.append("")
    lines.append("**Reading this table:** lower is better everywhere. If the 2-kernel numbers are close to the "
                  "3-kernel numbers, the mid-scale term Dennis Madsen added isn't earning its extra model complexity "
                  "on this population/resolution -- report that as a finding, not a failure. If 2-kernel is "
                  "meaningfully worse (especially at the glenoid / other small-radius features, which is exactly "
                  "what a mid/fine-scale term is supposed to capture), that's direct quantitative support for "
                  "keeping his fix.")

    report = "\n".join(lines)
    print(report)
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
