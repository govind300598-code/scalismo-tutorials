package scapula

/**
 * Single-Gaussian-kernel registration — Set 5: TightFlex
 *
 * σ =  75 mm  (shorter reach — same as Set 4)
 * s = 200     (maximum amplitude in the σ=75 family)
 *
 * Combines Madsen's tighter reach with Alemneh's maximum amplitude.
 * Useful when the dataset has both large inter-subject shape variability
 * (needs high s) and problematic thin structures (needs low σ).  Compare
 * HD95 from this set against Set 3 to decide which trade-off wins on your
 * specific cohort.
 *
 * Run:
 *   sbt "runMain scapula.Set5_TightFlex"
 *
 * Output folder:  $SCAPULA_OUT_DIR/set5-tight-flex/
 */
object Set5_TightFlex {
  def main(args: Array[String]): Unit =
    Stage2NonRigidReg.runWithGpParams(Stage2NonRigidReg.gpParamSets(4))
}
