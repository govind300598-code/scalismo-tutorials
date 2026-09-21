# Scapula statistical shape model (SSM) pipeline

Builds a statistical shape model of the scapula from a population of STL scans + landmark CSVs. There are two
sibling pipelines, sharing every methodological step except the GPMM prior's kernel:

- **`scapula.*`** -- the original multi-scale pipeline. The GPMM prior sums 2 or 3 Gaussian kernel terms at
  different length scales (`SCAPULA_KERNEL_TERMS`), following Dennis Madsen's mailing-list recipe.
- **`scapula.singlekernel.*`** -- a separate pipeline whose GPMM prior is exactly ONE Gaussian kernel term, with
  its correlation length (`sigma`) and amplitude (`s`) given directly in millimetres (`SCAPULA_SK_SIGMA_MM` /
  `SCAPULA_SK_SCALE_MM`) instead of derived from the reference mesh's bounding box. Everything else -- dataset
  loading, mirroring, medoid-based unbiased-template bootstrap, rigid(+scale) pre-alignment, landmark-informed
  coarse-to-fine non-rigid GPMM fitting, PCA/SSM validation -- is identical code, reused unchanged.

Both pipelines are driven by the same `scapula.Config` object (data directory, output directory, model
resolution, ICP iterations, refinement passes, GP rank/tolerance, seed, landmark weight, subject limit, etc.),
so a multi-kernel run and a single-kernel run can be compared on genuinely equal footing.

## Methodology (both pipelines, in order)

1. **Stage 1 -- diagnostics** (`scapula.Stage1Diagnostics` / `scapula.singlekernel.Stage1Diagnostics`, same code).
   Sanity-checks the dataset before any modelling: landmark-column resolution, mirror-orientation validity, and a
   within-subject-vs-between-subject surface-distance comparison that tells you whether the rigid pipeline is
   working (within-subject distance should be clearly smaller than between-subject distance) before you spend
   time on non-rigid registration. **Always run this first on a new dataset.**
2. **Reference bootstrap** (`scapula.ReferenceSelection`). Picks the pool medoid (the specimen whose rigidly
   +scale-aligned shape is, on average, closest to every other specimen) as the starting registration template.
3. **Stage 2 -- unbiased reference + landmark-informed non-rigid GPMM fitting**
   (`scapula.Stage2ReferenceRefinement` / `scapula.singlekernel.Stage2SingleKernelReferenceRefinement`). Builds
   the GPMM on the current reference, rigidly (+scale) pre-aligns every subject via landmarks then trimmed ICP,
   non-rigidly fits the GPMM to each subject (surface term + named-landmark term), and averages the
   now-corresponded fits into the next pass's reference. Writes the distance-error and landmark-RMSE report
   (`final_fit_quality.csv`, `fit_quality_by_pass.csv`) plus every registered mesh.
4. **Stage 3 -- the actual PCA statistical shape model** (`scapula.Stage3PCAModel`, shared -- it only reads what
   Stage 2 wrote, so both pipelines call the identical implementation). Builds PCA over the registered shapes and
   reports the three standard SSM validation metrics (Styner et al. 2003): compactness, specificity, and
   leave-one-out generalization.
5. **Visual inspection** (`scapula.ViewResults`, `scapula.ErrorHeatmap`, shared). Opens the reference, the PCA
   model, and per-subject target/fit pairs in the Scalismo viewer; `ErrorHeatmap` colors each fit by its
   per-vertex distance error to the real target surface.

No step is skipped between the two pipelines -- `scapula.singlekernel.*` re-exposes the shared, kernel-agnostic
stages (1, Stage 3, viewers) under its own namespace purely so it's runnable end-to-end on its own, without
duplicating their logic (see `scapula/singlekernel/Delegates.scala`).

## Dataset directory

`SCAPULA_DATA_DIR` must point at a folder containing the STL meshes plus their landmark CSV, e.g. any of:

- `.../100 plus scapula data/paired_scapulae_STLs`
- `.../100 plus scapula data/paired_shoulder_STLs_scapula`
- `.../100 plus scapula data/hill_sachs_STLs_scapula`

Some of these folders (notably `paired_scapulae_STLs`) hold every OTHER dataset's `*_model_data*.csv` alongside
their own; `ScapulaData.csvFile` picks the right one automatically by matching the directory's own name against
the candidate filenames (see its doc comment), after excluding `single_*`-prefixed CSVs (a different, unpaired
dataset). It also strips a trailing `_Scapula` from STL filenames before matching them to CSV rows, since some
of these datasets' STLs carry that suffix while the landmark CSV's subject-id column does not.

## Configurable parameters

Shared by both pipelines (`scapula.Config`):

| Parameter | Default | Env var |
|---|---|---|
| Data directory | `paired_scapulae_STLs` | `SCAPULA_DATA_DIR` |
| Output directory | `scapula_ssm_out` | `SCAPULA_OUT_DIR` |
| Model resolution (vertices) | 5000 | `SCAPULA_MODEL_RES` |
| Rigid ICP iterations | 40 | `SCAPULA_ICP_ITERS` |
| GPA refinement passes | 2 | `SCAPULA_REFINE_PASSES` |
| GP relative tolerance | 0.01 | `SCAPULA_GP_TOL` |
| GP max rank | 250 | `SCAPULA_GP_MAX_RANK` |
| Build independent (one-side) model | true | `SCAPULA_INDEPENDENT_MODEL` |
| Show UI | true | `SCAPULA_UI` |
| Random seed | 42 | `SCAPULA_SEED` |
| Landmark weight | 10.0 | `SCAPULA_LANDMARK_WEIGHT` |
| Subject limit | -1 (off) | `SCAPULA_SUBJECT_LIMIT` |
| Top-K most-average filter | -1 (off) | `SCAPULA_TOP_K` |
| ICP trim fraction (hardcoded, not env) | 0.15 | -- |
| ICP sample points (hardcoded default) | 2000 | -- |

Multi-kernel pipeline only (`scapula.*`):

| Parameter | Default | Env var |
|---|---|---|
| Kernel terms (2 or 3) | 3 | `SCAPULA_KERNEL_TERMS` |

Single-Gaussian-kernel pipeline only (`scapula.singlekernel.*`):

| Parameter | Default | Env var |
|---|---|---|
| Kernel correlation length, sigma (mm) | 100.0 | `SCAPULA_SK_SIGMA_MM` |
| Kernel amplitude, s (mm) | 100.0 | `SCAPULA_SK_SCALE_MM` |

## Running the single-Gaussian-kernel pipeline

One (sigma, s) configuration:

```bash
export SCAPULA_DATA_DIR="/path/to/100 plus scapula data/paired_scapulae_STLs"
export SCAPULA_OUT_DIR="/path/to/100 plus scapula data/scapula_ssm_out_single"
export SCAPULA_SK_SIGMA_MM=100
export SCAPULA_SK_SCALE_MM=100

sbt "runMain scapula.singlekernel.Stage1Diagnostics"                          # run once per dataset, first
sbt "runMain scapula.singlekernel.Stage2SingleKernelReferenceRefinement"
sbt "runMain scapula.singlekernel.Stage3PCAModel"
sbt "runMain scapula.singlekernel.ViewResults"                                # visual inspection
sbt "runMain scapula.singlekernel.ErrorHeatmap"                               # per-vertex error heatmap
python3 scripts/plot_ssm_validation.py "$SCAPULA_OUT_DIR"                     # compactness/specificity/generalization figures
```

### Selecting sigma and s: sweep + comparison

To answer "which (sigma, s) is best on this dataset" with the distance-error and SSM-validation metrics instead
of a guess, run the full 9-point grid (lower/default/upper sigma x tight/default/wide s) and get a ranked
comparison:

```bash
export SCAPULA_DATA_DIR="/path/to/100 plus scapula data/paired_scapulae_STLs"
scripts/run_single_kernel_sweep.sh "/path/to/100 plus scapula data/single_kernel_sweep"
```

This runs Stage 2 + Stage 3 once per grid point (each its own `sbt` process, to bound memory the same way the
existing multi-kernel comparison workflow already does), then calls
`scripts/compare_single_kernel_sweep.py` to build `single_kernel_sweep_comparison.md`: a table of every
grid point's mean/RMS/HD95/Hausdorff/landmark-RMSE, compactness/specificity/generalization, and a composite
"lowest is best" ranking. Override the grid with `SCAPULA_SK_GRID="sigma1:s1 sigma2:s2 ..."`, or run
`scripts/compare_single_kernel_sweep.py <base_dir>` again standalone against any already-completed sweep
directory (e.g. after adding more grid points by hand).

## Comparing the two pipelines against each other

`scripts/compare_kernel_experiment.py <3term_out_dir> <2term_out_dir>` compares two multi-kernel runs. To compare
a multi-kernel run against a single-kernel run on the same metrics, point it at both output directories the same
way -- it only reads `run_config.txt`, `final_fit_quality.csv` and the `pca_*.csv` files, all of which both
pipelines write in the same format.
