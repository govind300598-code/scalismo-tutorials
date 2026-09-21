#!/usr/bin/env bash
#
# Runs the single-Gaussian-kernel pipeline (scapula.singlekernel.*) once per (sigma, s) pair in the
# kernel-selection grid, each into its own output subdirectory, then builds the ranked comparison report.
#
# Every (sigma, s) run gets its OWN sbt/JVM process (not one long-lived sweep inside a single JVM): the
# multi-kernel pipeline's Stage2ReferenceRefinement docstring documents that keeping a live registration/UI
# scene graph across many passes x subjects is what previously ran a long batch out of memory -- separate
# processes bound that per-run instead of letting it accumulate across the whole 9-point sweep.
#
# Usage:
#   SCAPULA_DATA_DIR=/path/to/paired_scapulae_STLs scripts/run_single_kernel_sweep.sh [base_out_dir]
#
# All the other shared pipeline knobs (SCAPULA_MODEL_RES, SCAPULA_ICP_ITERS, SCAPULA_REFINE_PASSES,
# SCAPULA_GP_TOL, SCAPULA_GP_MAX_RANK, SCAPULA_INDEPENDENT_MODEL, SCAPULA_SEED, SCAPULA_LANDMARK_WEIGHT,
# SCAPULA_SUBJECT_LIMIT, SCAPULA_TOP_K -- see scapula.Config) are inherited from the calling shell exactly as
# for a single manual run, so every grid point is fit under identical conditions except sigma/s themselves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE_OUT="${1:-${SCAPULA_SK_SWEEP_OUT:-$HOME/Documents/100 plus scapula data/single_kernel_sweep}}"
mkdir -p "$BASE_OUT"

if ! command -v sbt >/dev/null 2>&1; then
  echo "!! sbt not found on PATH -- install sbt first." >&2
  exit 1
fi

# The 9-point (sigma, s) grid from the kernel-selection experiment design: sigma spans the lower/default/upper
# correlation length (50/75/100/150 mm), s spans the tight/default/wide amplitude (100/150/200 mm), covering
# both bounds plus Madsen's/Lüthi's/Alemneh's literature defaults (see the grid's own "why included" column).
# Override by setting SCAPULA_SK_GRID to a space-separated "sigma:s" list before calling this script.
GRID="${SCAPULA_SK_GRID:-50:100 50:200 75:100 75:200 100:100 100:150 100:200 150:100 150:200}"

echo "Single-Gaussian-kernel sweep"
echo "  data dir : ${SCAPULA_DATA_DIR:-<scapula.Config default>}"
echo "  base out : $BASE_OUT"
echo "  grid     : $GRID"
echo

for pair in $GRID; do
  sigma="${pair%%:*}"
  scale="${pair##*:}"
  run_dir="$BASE_OUT/sigma${sigma}_s${scale}"
  echo "=================================================================================================="
  echo "sigma=$sigma mm, s=$scale mm  ->  $run_dir"
  echo "=================================================================================================="

  export SCAPULA_SK_SIGMA_MM="$sigma"
  export SCAPULA_SK_SCALE_MM="$scale"
  export SCAPULA_OUT_DIR="$run_dir"

  sbt -batch "runMain scapula.singlekernel.Stage2SingleKernelReferenceRefinement"
  sbt -batch "runMain scapula.singlekernel.Stage3PCAModel"
done

unset SCAPULA_SK_SIGMA_MM SCAPULA_SK_SCALE_MM SCAPULA_OUT_DIR

echo
echo "All grid points finished. Building the ranked comparison report..."
python3 scripts/compare_single_kernel_sweep.py "$BASE_OUT"
