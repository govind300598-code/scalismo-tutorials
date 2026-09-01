# Scapula Statistical Shape Model Pipeline

Builds and compares two SSMs (SSM1 and SSM2) from paired scapula STL meshes using
Scalismo / Scala.

---

## What this does

| Stage | Description |
|-------|-------------|
| 1 | Diagnostics – checks landmark CSV, mirror validity, within- vs between-subject distances |
| 2 | Rigid + non-rigid registration of every mesh to an initial reference → SSM1 training set |
| 3 | PCA on the SSM1 training set → **SSM1** |
| 4 | Re-register every mesh to the SSM1 mean shape → SSM2 training set |
| 4b | PCA on the SSM2 training set → **SSM2** |
| 5 | Evaluate both models: compactness, generalization (LOO), specificity |
| 6 | Compare SSM1 and SSM2 mean shapes (surface distance + RMSE) |

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Java (JDK) | 11 or 17 | `sudo apt install openjdk-17-jdk` or from https://adoptium.net |
| sbt | 1.x | https://www.scala-sbt.org/download (Ubuntu: `sudo apt install sbt`) |

No other installation is needed – sbt downloads Scala and Scalismo automatically on first run.

---

## Quick start (3 steps)

### Step 1 – Clone / download the repository

```bash
git clone <repo-url>
cd scalismo-tutorials
```

Or download as a ZIP from GitHub and extract it.

### Step 2 – Edit `run.sh`

Open `run.sh` in any text editor and set the two required paths at the top:

```bash
# Path to the folder containing the .stl files and the landmark CSV
DATA_DIR="/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"

# Where outputs (registered meshes, SSM files, metrics CSVs) are written
OUT_DIR="/home/g25upadh/Documents/100 plus scapula data/ssm_pipeline_output"
```

Everything else can stay as-is for a first run.

### Step 3 – Run

```bash
bash run.sh
```

The first run downloads dependencies and compiles (~5 min).
The actual pipeline time depends on mesh count and your machine (allow 1–4 hours for 24 meshes).

---

## Input format

Your data folder must contain:

- **STL files** named like `paired_scapula_001_M_64_L.stl` / `paired_scapula_001_M_64_R.stl`
  (the pipeline detects left/right from the trailing `_L` / `_R`).
- **One landmark CSV** whose filename contains both `scapula` and `model_data`
  (e.g. `paired_scapulae_model_data_v1.1.csv`).

The CSV must have columns for landmarks **GC, TS, IA, PLA, AC**, each with x/y/z sub-columns.
Column names are detected automatically from the header row.

---

## Output files

```
ssm_pipeline_output/
├── ssm1.h5                        # SSM1 model (Scalismo HDF5 format)
├── ssm1_mean.stl                  # SSM1 mean shape
├── ssm2.h5                        # SSM2 model
├── ssm2_mean.stl                  # SSM2 mean shape
├── ssm1_registered/               # Meshes registered to original reference
│   └── paired_scapula_*_registered.stl
├── ssm2_registered/               # Meshes registered to SSM1 mean
│   └── paired_scapula_*_registered.stl
├── ssm1_compactness.csv           # mode, cumulative_variance (SSM1)
├── ssm2_compactness.csv           # mode, cumulative_variance (SSM2)
├── evaluation.csv                 # generalization + specificity for both
├── mean_shape_comparison.csv      # SSM1↔SSM2 mean shape distances
├── stage1_diagnostics.log         # Stage 1 console output
└── pipeline.log                   # Full pipeline console output
```

---

## Configuration (environment variables)

You can set these before `bash run.sh`, or edit them inside `run.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCAPULA_DATA_DIR` | *(required)* | Folder with STL files and landmark CSV |
| `SCAPULA_OUT_DIR` | *(required)* | Output folder |
| `SCAPULA_REF_ID` | first mesh | Model-id of the reference specimen (without `.stl`) |
| `SCAPULA_MODEL_RES` | `5000` | Target vertex count of the decimated reference |
| `SCAPULA_ICP_ITERS` | `40` | GP-ICP iterations per registration |
| `SCAPULA_GP_TOL` | `0.01` | Pivoted-Cholesky relative tolerance (smaller = higher rank) |
| `SCAPULA_GP_MAX_RANK` | `250` | Maximum GP prior rank |
| `SCAPULA_SEED` | `42` | Random seed for reproducibility |

---

## Running individual stages

```bash
# Stage 1 only (quick sanity check, ~2 min)
sbt "runMain scapula.Stage1Diagnostics"

# Full pipeline (Stages 2-6)
sbt "runMain scapula.SsmPipeline"
```

Set environment variables before the `sbt` command:

```bash
export SCAPULA_DATA_DIR="/path/to/your/data"
export SCAPULA_OUT_DIR="/path/to/output"
sbt "runMain scapula.SsmPipeline"
```

---

## Registration is cached

If a registered mesh already exists in the output folder it is loaded directly.
This means you can safely interrupt and resume the pipeline – already-done
registrations are not repeated.

---

## Interpreting the final table

| Metric | Better value | What it means |
|--------|-------------|---------------|
| Compactness | Higher % at N modes | The model captures more variance with fewer modes |
| Generalization | Lower mm | The model accurately reconstructs unseen shapes |
| Specificity | Lower mm | Random model samples look like real scapulae |
| Mean shape distance | Lower mm | SSM1 and SSM2 converged to a similar mean |
| RMSE between means | Lower mm | Pointwise stability between the two iterations |

SSM2 typically shows slightly better generalization because its reference (the SSM1 mean) is
already unbiased toward any single specimen.
