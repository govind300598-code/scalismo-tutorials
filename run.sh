#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  EDIT THESE TWO LINES — everything else is automatic.
# ════════════════════════════════════════════════════════════════
DATA_DIR="/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"
OUT_DIR="/home/g25upadh/Documents/100 plus scapula data/ssm_pipeline_output"
# ════════════════════════════════════════════════════════════════

export SCAPULA_DATA_DIR="$DATA_DIR"
export SCAPULA_OUT_DIR="$OUT_DIR"
export SCAPULA_MODEL_RES=5000    # vertices in the reference mesh
export SCAPULA_ICP_ITERS=40      # GP-ICP iterations per target
export SCAPULA_GP_TOL=0.01       # Cholesky rank tolerance (smaller = higher rank)
export SCAPULA_SEED=42

echo "============================================================"
echo "  Scapula SSM Pipeline"
echo "  Data   : $DATA_DIR"
echo "  Output : $OUT_DIR"
echo "============================================================"
echo ""
echo "Running Stage 1 diagnostics (landmark + alignment checks)..."
sbt -J-Xmx8g "runMain scapula.Stage1Diagnostics"

echo ""
echo "Running SSM pipeline (SSM1 → SSM2 → Evaluation → Comparison)..."
sbt -J-Xmx8g "runMain scapula.SsmPipeline"

echo ""
echo "All done. Results are in: $OUT_DIR"
