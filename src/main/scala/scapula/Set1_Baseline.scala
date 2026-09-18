package scapula

/**
 * Single-Gaussian-kernel registration — Set 1: Baseline
 *
 * σ = 100 mm  (Gaussian bandwidth — reach of the deformation)
 * s = 100     (scale factor — amplitude of the deformation)
 *
 * The GPMM paper default.  Use this as the starting point; compare its
 * registered surface distances against the other sets to decide whether
 * more reach (σ) or more amplitude (s) helps on your dataset.
 *
 * Run:
 *   sbt "runMain scapula.Set1_Baseline"
 *
 * Output folder:  $SCAPULA_OUT_DIR/set1-baseline/
 */
object Set1_Baseline {
  def main(args: Array[String]): Unit =
    Stage2NonRigidReg.runWithGpParams(Stage2NonRigidReg.gpParamSets(0))
}
