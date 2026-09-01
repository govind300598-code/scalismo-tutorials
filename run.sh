#!/usr/bin/env bash
# ============================================================================
# Scapula SSM Pipeline – easy launcher
# ============================================================================
# Edit the three variables below, then run:   bash run.sh
# ============================================================================

# ── 1. Path to the folder that contains the .stl files and the landmark CSV ─
DATA_DIR="/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"

# ── 2. Where outputs (registered meshes, SSM .h5 files, CSV metrics) go ─────
OUT_DIR="/home/g25upadh/Documents/100 plus scapula data/ssm_pipeline_output"

# ── 3. (Optional) Fix a reference specimen by its model-id (without .stl).
#       Leave empty ("") to use the first alphabetically sorted specimen.
REF_ID=""

# ── Advanced tuning (safe to leave as-is for a first run) ───────────────────
MODEL_RES=5000        # vertices on the decimated reference mesh
ICP_ITERS=40          # GP-ICP iterations per registration
GP_TOL=0.01           # pivoted-Cholesky relative tolerance (smaller => higher rank)
GP_MAX_RANK=250       # hard cap on GP prior rank
SEED=42               # random seed for reproducible evaluation

# ============================================================================
set -euo pipefail

export SCAPULA_DATA_DIR="$DATA_DIR"
export SCAPULA_OUT_DIR="$OUT_DIR"
export SCAPULA_MODEL_RES="$MODEL_RES"
export SCAPULA_ICP_ITERS="$ICP_ITERS"
export SCAPULA_GP_TOL="$GP_TOL"
export SCAPULA_GP_MAX_RANK="$GP_MAX_RANK"
export SCAPULA_SEED="$SEED"
export SCAPULA_UI="false"   # disable Scalismo UI window in headless mode

if [ -n "$REF_ID" ]; then
  export SCAPULA_REF_ID="$REF_ID"
fi

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Scapula SSM Pipeline"
echo "  Data   : $DATA_DIR"
echo "  Output : $OUT_DIR"
echo "============================================================"

# Check that sbt is available
if ! command -v sbt &>/dev/null; then
  echo ""
  echo "ERROR: 'sbt' not found on PATH."
  echo "Install it from https://www.scala-sbt.org/download  or via:"
  echo "  sudo apt install sbt          # Debian/Ubuntu"
  echo "  brew install sbt              # macOS"
  echo ""
  exit 1
fi

# ── Run Stage 1 diagnostics first (optional but recommended) ────────────────
echo ""
echo "Running Stage 1 diagnostics (landmark + alignment checks)..."
sbt --no-colors "runMain scapula.Stage1Diagnostics" 2>&1 | tee "$OUT_DIR/stage1_diagnostics.log"

# ── Run the full SSM pipeline ────────────────────────────────────────────────
echo ""
echo "Running full SSM pipeline (Stages 2-6)..."
sbt --no-colors "runMain scapula.SsmPipeline" 2>&1 | tee "$OUT_DIR/pipeline.log"

echo ""
echo "Done.  Outputs are in: $OUT_DIR"
