#!/bin/bash
# One-shot script: convert NRRD volumes to NIfTI, then run the density pipeline.
# Usage (from the scalismo-tutorials directory):
#   bash run_density_pipeline.sh
# Or with custom paths:
#   SCAPULA_NRRD_DIR="/your/nrrd/dir" SCAPULA_STL_DIR="/your/stl/dir" bash run_density_pipeline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NRRD_DIR="${SCAPULA_NRRD_DIR:-/home/g25upadh/Documents/all ct from hoel 3d}"
STL_DIR="${SCAPULA_STL_DIR:-${NRRD_DIR}/stl/both stl/combined 1/Combined}"
OUT_DIR="${SCAPULA_DENSITY_OUT:-/home/g25upadh/Documents/scapula_density_sdm_$(date +%Y-%m-%d)}"

echo "========================================================================"
echo "  Scapula Statistical Density Pipeline — Full Run"
echo "========================================================================"
echo "  NRRD dir : ${NRRD_DIR}"
echo "  STL dir  : ${STL_DIR}"
echo "  Output   : ${OUT_DIR}"
echo ""

# ── Step 1: ensure SimpleITK is available ────────────────────────────────────
echo "=== [Step 1/3] Checking Python / SimpleITK ==="
if ! python3 -c "import SimpleITK" 2>/dev/null; then
  echo "  SimpleITK not found. Installing..."
  pip install --quiet SimpleITK
  echo "  SimpleITK installed."
else
  echo "  SimpleITK already available."
fi

# ── Step 2: convert *_volume.nrrd → *_volume.nii.gz ─────────────────────────
echo ""
echo "=== [Step 2/3] Converting NRRD volumes to NIfTI ==="
python3 "${SCRIPT_DIR}/convert_nrrd_to_nii.py" "${NRRD_DIR}"

# ── Step 3: run the Scala pipeline ──────────────────────────────────────────
echo ""
echo "=== [Step 3/3] Running Statistical Density Pipeline ==="
SCAPULA_NRRD_DIR="${NRRD_DIR}" \
SCAPULA_STL_DIR="${STL_DIR}" \
SCAPULA_DENSITY_OUT="${OUT_DIR}" \
sbt "runMain scapula.DensityPipeline"

echo ""
echo "All done. Results in: ${OUT_DIR}"
