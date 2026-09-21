package scapula.singlekernel

/**
 * These stages of the methodology (dataset diagnostics, PCA/SSM validation, and visual inspection) are entirely
 * kernel-agnostic in the original pipeline -- they only read/write `Config.outDir` and never construct a GPMM
 * kernel themselves, so there is nothing single-kernel-specific to reimplement. They're re-exported here under
 * `scapula.singlekernel` purely so this pipeline is runnable end-to-end by a consistent, self-contained set of
 * `runMain scapula.singlekernel.*` commands, without silently skipping a step of the methodology or duplicating
 * its logic. See `scapula.Stage1Diagnostics` / `scapula.Stage3PCAModel` / `scapula.ViewResults` /
 * `scapula.ErrorHeatmap` for the actual implementation.
 */
object Stage1Diagnostics {
  def main(args: Array[String]): Unit = scapula.Stage1Diagnostics.main(args)
}

object Stage3PCAModel {
  def main(args: Array[String]): Unit = scapula.Stage3PCAModel.main(args)
}

object ViewResults {
  def main(args: Array[String]): Unit = scapula.ViewResults.main(args)
}

object ErrorHeatmap {
  def main(args: Array[String]): Unit = scapula.ErrorHeatmap.main(args)
}
