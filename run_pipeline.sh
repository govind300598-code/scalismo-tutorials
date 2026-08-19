#!/usr/bin/env bash
#
# Full scapula SSM pipeline, terminal-only, no UI.
#
#   Stage 1  diagnostics          -- sanity check: mirror/landmark/rigid-alignment correctness
#   Stage 2  registration         -- rigid landmark alignment + automatic ICP + GP non-rigid registration
#   Stage 3  build model          -- PCA shape model + compactness / generalization / specificity
#
# Usage:
#   ./run_pipeline.sh                      # run all three stages
#   ./run_pipeline.sh stage2 stage3        # run only the stages named
#   SCAPULA_DATA_DIR=/path SCAPULA_OUT_DIR=/path ./run_pipeline.sh
#
set -euo pipefail

# ---- dataset location: override by exporting these before running, or edit the defaults here -------------------
export SCAPULA_DATA_DIR="${SCAPULA_DATA_DIR:-/home/g25upadh/Documents/database_v1.11/paired_scapulae_STLs}"
export SCAPULA_OUT_DIR="${SCAPULA_OUT_DIR:-/home/g25upadh/Documents/database_v1.11/scapula_ssm_out}"

STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(stage1 stage2 stage3)
fi

cd "$(dirname "$0")"

echo "================================================================================"
echo " SCAPULA SSM PIPELINE"
echo "   SCAPULA_DATA_DIR = $SCAPULA_DATA_DIR"
echo "   SCAPULA_OUT_DIR  = $SCAPULA_OUT_DIR"
echo "   stages to run    = ${STAGES[*]}"
echo "================================================================================"

for stage in "${STAGES[@]}"; do
  case "$stage" in
    stage1)
      echo; echo ">>> Stage 1: diagnostics"
      sbt -Djava.awt.headless=true "runMain scapula.Stage1Diagnostics"
      ;;
    stage2)
      echo; echo ">>> Stage 2: rigid alignment + automatic ICP + GP non-rigid registration"
      sbt -Djava.awt.headless=true "runMain scapula.Stage2Registration"
      ;;
    stage3)
      echo; echo ">>> Stage 3: build statistical shape model"
      sbt -Djava.awt.headless=true "runMain scapula.Stage3BuildModel"
      ;;
    *)
      echo "Unknown stage '$stage'. Valid stages: stage1 stage2 stage3" >&2
      exit 1
      ;;
  esac
done

echo
echo "================================================================================"
echo " DONE. Outputs in: $SCAPULA_OUT_DIR"
echo "   reference.stl                 (Stage 2 reference mesh)"
echo "   corresponded/*.stl            (Stage 2 per-specimen corresponded meshes)"
echo "   scapula_ssm.h5                (Stage 3 statistical shape model)"
echo "   ssm_validation_metrics.csv    (Stage 3 compactness / generalization / specificity)"
echo "================================================================================"
