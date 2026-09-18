package scapula

/**
 * Single-Gaussian-kernel registration — Set 4: Madsen (tighter reach)
 *
 * σ =  75 mm  (shorter reach — localises deformation more than σ=100)
 * s = 150     (moderate amplitude)
 *
 * Motivated by Madsen et al.'s observation that a narrower kernel reduces
 * fold-overs on the acromion and coracoid process by preventing far-field
 * compensation between thin structures.  Use this when Set 3 produces
 * self-intersections or implausible deformations near thin processes.
 *
 * Run:
 *   sbt "runMain scapula.Set4_Madsen"
 *
 * Output folder:  $SCAPULA_OUT_DIR/set4-madsen/
 */
object Set4_Madsen {
  def main(args: Array[String]): Unit =
    Stage2NonRigidReg.runWithGpParams(Stage2NonRigidReg.gpParamSets(3))
}
