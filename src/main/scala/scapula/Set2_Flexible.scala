package scapula

/**
 * Single-Gaussian-kernel registration — Set 2: Flexible
 *
 * σ = 100 mm  (same reach as baseline)
 * s = 150     (50% more amplitude → larger deformations allowed)
 *
 * Try this when Set 1 underestimates scapula shape variability but the
 * overall correspondence pattern already looks correct.
 *
 * Run:
 *   sbt "runMain scapula.Set2_Flexible"
 *
 * Output folder:  $SCAPULA_OUT_DIR/set2-flexible/
 */
object Set2_Flexible {
  def main(args: Array[String]): Unit =
    Stage2NonRigidReg.runWithGpParams(Stage2NonRigidReg.gpParamSets(1))
}
