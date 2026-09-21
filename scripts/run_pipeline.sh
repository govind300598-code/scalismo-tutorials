#!/usr/bin/env bash
#
# Runs the ENTIRE single-Gaussian-kernel pipeline in one go, in the correct order, against ONE (sigma, s)
# configuration: compile -> Stage 1 diagnostics -> Stage 2 (reference refinement + non-rigid GPMM fitting) ->
# Stage 3 (PCA model + SSM validation) -> validation figures. Stops immediately (set -e) if any step fails, so
# you never run an expensive later stage against a broken earlier one.
#
# Usage:
#   scripts/run_pipeline.sh <data_dir> <out_dir> [sigma_mm] [scale_mm] [subject_limit]
#
# sigma_mm / scale_mm default to 100/100 (scapula.singlekernel.SingleKernelConfig's own defaults).
# subject_limit is optional: leave it off for a real run over the whole folder; pass a small number (e.g. 4) for
# a fast smoke test that exercises every stage end-to-end without waiting on the full population.
#
# To pool SEVERAL dataset folders into one combined population instead of a single <data_dir>, export
# SCAPULA_DATA_DIRS (":"-separated) yourself before calling this script -- it takes priority over <data_dir>,
# which you can then set to any one of those folders (informational only in that case). See README.md.
#
# Examples:
#   # fast smoke test on the smallest folder, 4 subjects, default kernel
#   scripts/run_pipeline.sh "$HOME/Documents/100 plus scapula data/paired_scapulae_STLs" \
#                            "$HOME/Documents/100 plus scapula data/scapula_single_gaussian_kernel_out_smoketest" \
#                            100 100 4
#
#   # real run, full folder, sigma=75mm s=150mm
#   scripts/run_pipeline.sh "$HOME/Documents/100 plus scapula data/paired_shoulder_STLs_scapula" \
#                            "$HOME/Documents/100 plus scapula data/scapula_single_gaussian_kernel_out" 75 150
#
#   # combined pool across all three folders
#   export SCAPULA_DATA_DIRS="$HOME/Documents/100 plus scapula data/hill_sachs_STLs_scapula:$HOME/Documents/100 plus scapula data/paired_shoulder_STLs_scapula:$HOME/Documents/100 plus scapula data/paired_scapulae_STLs"
#   scripts/run_pipeline.sh "$HOME/Documents/100 plus scapula data/paired_scapulae_STLs" \
#                            "$HOME/Documents/100 plus scapula data/scapula_single_gaussian_kernel_out_combined" 100 100
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ $# -lt 2 ]; then
  echo "Usage: $0 <data_dir> <out_dir> [sigma_mm=100] [scale_mm=100] [subject_limit]" >&2
  exit 1
fi

export SCAPULA_DATA_DIR="$1"
export SCAPULA_OUT_DIR="$2"
export SCAPULA_SK_SIGMA_MM="${3:-100}"
export SCAPULA_SK_SCALE_MM="${4:-100}"
if [ -n "${5:-}" ]; then
  export SCAPULA_SUBJECT_LIMIT="$5"
fi

if ! command -v sbt >/dev/null 2>&1; then
  echo "!! sbt not found on PATH -- install sbt first." >&2
  exit 1
fi

echo "=================================================================================================="
echo "Single-Gaussian-kernel pipeline -- full run"
if [ -n "${SCAPULA_DATA_DIRS:-}" ]; then
  echo "  data dirs (combined, SCAPULA_DATA_DIRS takes priority over <data_dir>):"
  echo "    ${SCAPULA_DATA_DIRS//:/$'\n    '}"
else
  echo "  data dir        : $SCAPULA_DATA_DIR"
fi
echo "  out dir         : $SCAPULA_OUT_DIR"
echo "  sigma / s (mm)  : $SCAPULA_SK_SIGMA_MM / $SCAPULA_SK_SCALE_MM"
echo "  subject limit   : ${SCAPULA_SUBJECT_LIMIT:-<none -- full population>}"
echo "=================================================================================================="
echo

echo "[1/5] sbt compile"
sbt -batch compile

echo
echo "[2/5] Stage 1 -- dataset diagnostics"
sbt -batch "runMain scapula.singlekernel.Stage1Diagnostics"

echo
echo "[3/5] Stage 2 -- reference refinement + landmark-informed non-rigid GPMM fitting"
sbt -batch "runMain scapula.singlekernel.Stage2SingleKernelReferenceRefinement"

echo
echo "[4/5] Stage 3 -- PCA model + SSM validation (compactness / specificity / generalization)"
sbt -batch "runMain scapula.singlekernel.Stage3PCAModel"

echo
echo "[5/5] Validation figures"
python3 scripts/plot_scapula_ssm_validation.py "$SCAPULA_OUT_DIR"

echo
echo "=================================================================================================="
echo "Done. Results in: $SCAPULA_OUT_DIR"
echo "=================================================================================================="
ls "$SCAPULA_OUT_DIR"
echo
echo "Next: inspect visually with"
echo "  SCAPULA_OUT_DIR=\"$SCAPULA_OUT_DIR\" sbt \"runMain scapula.singlekernel.ViewResults\""
echo "  SCAPULA_OUT_DIR=\"$SCAPULA_OUT_DIR\" sbt \"runMain scapula.singlekernel.ErrorHeatmap\""
