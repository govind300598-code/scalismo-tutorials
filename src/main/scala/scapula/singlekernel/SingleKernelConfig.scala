package scapula.singlekernel

/**
 * Configuration for the SINGLE-Gaussian-kernel pipeline. This is a genuinely separate pipeline from
 * `scapula.Stage2ReferenceRefinement` (which sums 2 or 3 Gaussian terms, controlled by `SCAPULA_KERNEL_TERMS` --
 * see `scapula.Config.kernelTerms`): here the GPMM prior is built from exactly ONE Gaussian kernel term,
 * `GaussianKernel3D(sigmaMm, scaleMm)`, with `sigmaMm` (correlation length) and `scaleMm` (amplitude "s") given
 * directly in millimetres rather than derived as a fraction of the reference mesh's bounding box. That matches
 * how the kernel-selection experiment (comparing candidate (sigma, s) pairs by their resulting distance-error
 * and SSM-validation metrics) is meant to sweep them: as absolute, directly comparable physical quantities.
 *
 * Every OTHER configurable parameter of the pipeline (data directory, output directory, model resolution, ICP
 * iterations, refinement passes, GP rank/tolerance, independent-model flag, seed, landmark weight, subject
 * limit, top-K filter) is shared verbatim with the multi-kernel pipeline via `scapula.Config`, so both pipelines
 * are run and compared under identical experimental conditions except for the one thing being tested: the kernel.
 */
object SingleKernelConfig {
  private def env(key: String, default: String): String = sys.env.getOrElse(key, default)

  /** Correlation length (mm) of the single Gaussian kernel term -- "sigma" in the kernel-selection grid. */
  val sigmaMm: Double = env("SCAPULA_SK_SIGMA_MM", "100.0").toDouble

  /** Amplitude (mm) of the single Gaussian kernel term -- "s" in the kernel-selection grid. Variance = s^2. */
  val scaleMm: Double = env("SCAPULA_SK_SCALE_MM", "100.0").toDouble

  require(sigmaMm > 0, s"SCAPULA_SK_SIGMA_MM must be positive, got $sigmaMm")
  require(scaleMm > 0, s"SCAPULA_SK_SCALE_MM must be positive, got $scaleMm")
}
