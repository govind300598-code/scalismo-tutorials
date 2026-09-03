# ScapulaAtlasRefinement

Iterative Statistical Shape Model (SSM) pipeline for scapula anatomy using Scalismo.

## Pipeline overview

```
Original STLs (unchanged in data/original/)
       │
       ▼ stride-based decimation (~8,000 vertices)
data/preprocessing/8k/
       │
       ▼ for each SSM iteration (1–4):
       │
       ├─ Landmark-based rigid registration
       │    landmarks (GC, TS, IA, PLA, AC) → rigid3DLandmarkRegistration → trimmed ICP
       │
       ├─ GPMM non-rigid registration
       │    Gaussian kernel: σ=30 mm, scaleFactor=10 mm (fixed across SSM1–SSM4)
       │    NearestNeighborInterpolator3D (see NonRigidReg.scala for full documentation)
       │    GP-ICP: 40 iterations × 500 correspondences per iteration
       │
       ├─ Dense correspondence established (each output mesh ≡ reference topology)
       │
       ├─ PCA → SSM{n}  (saved as model/SSM{n}.h5)
       │
       └─ Mean{n} mesh  (saved as mean/SSM{n}_mean.stl)
              │
              └─→ used as reference for SSM{n+1}

SSM1 → Mean1 → SSM2 → Mean2 → SSM3 → Mean3 → SSM4 → Mean4

comparison/
  Mean1_vs_Mean2/   ─ surface distances, VTK maps
  Mean2_vs_Mean3/
  Mean3_vs_Mean4/
  convergence_summary.csv
  SSM_comparison/ssm_comparison.md
```

## Requirements

- **JDK 11+**
- **sbt 1.x**
- Data: `paired_scapulae_STLs_scapula/` folder containing
  - `paired_scapula_*.stl` files
  - `paired_scapulae_model_data_v1.1.csv` landmark file (columns: GC_x/y/z, TS_x/y/z, IA_x/y/z, PLA_x/y/z, AC_x/y/z)

## Running

```bash
# Run the full pipeline (SSM1 → SSM4)
SCAPULA_DATA_DIR="/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula" \
SCAPULA_OUT_DIR="/home/g25upadh/Documents/100 plus scapula data/scapula_atlas_out" \
sbt "runMain scapula.Main"

# Launch the Scalismo UI viewer (after pipeline has run)
sbt "runMain scapula.VisualizationApp"

# Run stage 1 diagnostics only
sbt "runMain scapula.Stage1Diagnostics"
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `SCAPULA_DATA_DIR` | `/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula` | Folder with STLs + CSV |
| `SCAPULA_OUT_DIR` | `…/scapula_atlas_out` | Root output folder |
| `SCAPULA_MODEL_RES` | `8000` | Target vertex count for working meshes |
| `SCAPULA_ICP_ITERS` | `40` | GP-ICP iterations per registration |
| `SCAPULA_GP_TOL` | `0.01` | Relative tolerance for GP rank (Cholesky) |
| `SCAPULA_GP_MAX_RANK` | `250` | Hard cap on GP rank |
| `SCAPULA_UI` | `false` | Launch UI after pipeline (`true`/`false`) |

## Output structure

```
scapula_atlas_out/
├── data/
│   └── preprocessing/8k/          ← ~8k decimated working meshes (generated once)
├── results/
│   ├── SSM1/
│   │   ├── reference/             ← reference mesh used for SSM1
│   │   ├── rigid_registered/      ← landmark-rigid aligned meshes
│   │   ├── nonrigid_registered/   ← GPMM registered meshes
│   │   ├── correspondences/
│   │   ├── mean/SSM1_mean.stl
│   │   ├── model/SSM1.h5
│   │   ├── PCA_modes/             ← mode 1-3 at -3σ/mean/+3σ
│   │   ├── surface_distance/      ← per-subject registration error CSV
│   │   ├── metrics/               ← variance report, gen, spec
│   │   └── logs/reproducibility.txt
│   ├── SSM2/ … SSM4/              ← same layout
└── comparison/
    ├── Mean1_vs_Mean2/
    │   ├── Mean1_vs_Mean2_metrics.csv
    │   └── figures/*.vtk           ← colour-mapped distance maps
    ├── Mean2_vs_Mean3/
    ├── Mean3_vs_Mean4/
    ├── convergence_summary.csv
    └── SSM_comparison/ssm_comparison.md
```

## NearestNeighborInterpolator3D — where and why

The `NearestNeighborInterpolator3D` is used **once**: inside
`NonRigidReg.buildPrior` when calling
`LowRankGaussianProcess.approximateGPCholesky`.

**What it interpolates**: The continuous Gaussian Process (displacement field)
is discretised at the reference mesh vertices. When the posterior is queried at
a point not in that discrete set, the interpolator returns the GP value from
the nearest reference vertex.

**Why it is appropriate**: The kernel σ=30 mm is much larger than the typical
inter-vertex spacing at 8k resolution (~2–3 mm), so the displacement field
is smooth at that scale and nearest-neighbour interpolation introduces no
perceptible artefact in the registered surface.

If visibly non-smooth mode deformations appear, the cause is kernel/rank
parameters, **not** the nearest-neighbour interpolator.

## Landmarks

Five anatomical landmarks per scapula (Steinmann labelling):

| ID  | Description |
|-----|-------------|
| GC  | Glenoid Centre |
| TS  | Trigone of Spine |
| IA  | Inferior Angle |
| PLA | Posterolateral Angle |
| AC  | Acromioclavicular Joint |

## Convergence interpretation

| Comparison | Meaning |
|---|---|
| Mean1 ↔ Mean2 | Template change from initial reference to first refined mean |
| Mean2 ↔ Mean3 | Second refinement step |
| Mean3 ↔ Mean4 | Third refinement – should be smallest if converging |

Convergence criterion: `dist(Mean3↔Mean4) < dist(Mean2↔Mean3) < dist(Mean1↔Mean2)`

## Not committed

- Raw STL patient data
- Generated result files (`.stl`, `.h5`, `.vtk`, `.csv` in results/)
- sbt/Ivy caches
