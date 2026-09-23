#!/usr/bin/env bash
# Runs the 9-point single-Gaussian-kernel grid search (sigma:scaleFactor, absolute mm) sequentially
# against scapula.singlekernel.Stage2SingleKernelReferenceRefinement (branch claude/clever-davinci-cmoyz3,
# ~/scalismo-tutorials-singlekernel) -- NOT the override on this branch, since that pipeline is more
# complete. Each point writes to its own directory, then all 9 are ranked at the end.
#
# Uses the FAST N=11 paired_scapulae dataset, not the larger N=38 one -- 9 full runs at N=38 would take
# many hours; at N=11 each run is roughly 15-20 minutes, ~2.5-3 hours total for all 9. Once you know the
# winning (sigma, scale), rerun that ONE config against whichever dataset you need the final model from.
#
# EDIT THESE THREE PATHS FOR YOUR MACHINE, then run:
#   bash scripts/run_single_kernel_grid.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="$SCRIPT_DIR/compare_single_kernel_grid.py"

REPO_DIR="$HOME/scalismo-tutorials-singlekernel"
DATA_DIR="$HOME/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"
OUT_BASE="$HOME/Documents/100 plus scapula data/single_kernel_grid"

# sigma:scale pairs, exactly the 9-point grid
GRID=(
  "50:100" "50:200"
  "75:100" "75:200"
  "100:100" "100:150" "100:200"
  "150:100" "150:200"
)

cd "$REPO_DIR" || { echo "!! $REPO_DIR not found -- edit REPO_DIR at the top of this script"; exit 1; }
git pull origin claude/clever-davinci-cmoyz3 || { echo "!! git pull failed -- stopping"; exit 1; }

COMPARE_ARGS=()

for point in "${GRID[@]}"; do
  sigma="${point%%:*}"
  scale="${point##*:}"
  out_dir="$OUT_BASE/grid_s${sigma}_a${scale}"

  echo ""
  echo "================================================================"
  echo "Grid point: sigma=${sigma} mm, scale=${scale} mm  ->  $out_dir"
  echo "================================================================"

  export SCAPULA_DATA_DIR="$DATA_DIR"
  export SCAPULA_OUT_DIR="$out_dir"
  export SCAPULA_SK_SIGMA_MM="$sigma"
  export SCAPULA_SK_SCALE_MM="$scale"

  sbt "runMain scapula.singlekernel.Stage2SingleKernelReferenceRefinement"
  status=$?

  if [ $status -ne 0 ]; then
    echo "!! Grid point ${sigma}:${scale} FAILED (exit $status, see log above) -- continuing to the next point"
    continue
  fi

  COMPARE_ARGS+=("${sigma}:${scale}=${out_dir}")
done

unset SCAPULA_SK_SIGMA_MM SCAPULA_SK_SCALE_MM

echo ""
echo "================================================================"
echo "All grid points attempted. Ranking results..."
echo "================================================================"

python3 "$COMPARE_SCRIPT" "${COMPARE_ARGS[@]}" --out "$OUT_BASE/single_kernel_grid_report.md"
