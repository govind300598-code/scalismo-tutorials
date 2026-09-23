#!/usr/bin/env python3
"""
Computes mean +/- SD (sample, ddof=1) for every fit-quality metric in one run
directory's final_fit_quality.csv -- the same computation already used for the
2-kernel vs 3-kernel comparison, applied here to a single run (e.g. the
sigma=50,s=100 single-kernel point) so all kernel configs can be reported with
the same statistical rigor.

Usage:
    python3 scripts/mean_sd_from_run.py <run_dir>

Example (single-kernel sigma=50,s=100 grid point):
    python3 scripts/mean_sd_from_run.py ~/Documents/"100 plus scapula data"/single_kernel_grid/grid_s50_a100

Requires: pandas, numpy.
"""
import os
import sys

import numpy as np
import pandas as pd

METRICS = [
    ("fit_mean_mm", "Mean distance"),
    ("fit_rms_mm", "RMS"),
    ("fit_hd95_mm", "HD95"),
    ("fit_hd_mm", "Hausdorff max"),
    ("fit_landmark_rmse_mm", "Landmark RMSE"),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    run_dir = os.path.expanduser(sys.argv[1])
    csv_path = os.path.join(run_dir, "final_fit_quality.csv")
    if not os.path.exists(csv_path):
        print(f"!! {csv_path} not found -- did Stage2ReferenceRefinement finish here?")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"{run_dir}\n({n} subjects)\n")
    print(f"{'Metric':16s} {'mean':>8s}   {'SD':>8s}   {'median':>8s}")
    for col, label in METRICS:
        if col not in df.columns or df[col].isna().all():
            print(f"{label:16s} -- column '{col}' not found in this CSV")
            continue
        vals = df[col].dropna().to_numpy()
        mean = vals.mean()
        sd = vals.std(ddof=1) if len(vals) > 1 else float("nan")
        median = float(np.median(vals))
        print(f"{label:16s} {mean:8.3f} ± {sd:6.3f}   {median:8.3f}")


if __name__ == "__main__":
    main()
