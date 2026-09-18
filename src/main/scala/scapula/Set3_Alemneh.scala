package scapula

/**
 * Single-Gaussian-kernel registration — Set 3: Alemneh (default)
 *
 * σ = 100 mm  (broad reach — the whole blade deforms coherently)
 * s = 200     (maximum amplitude in the σ=100 family)
 *
 * Tuned for scapulae in Alemneh et al.  This is the recommended starting
 * point for new datasets: broad kernel keeps the deformation smooth while
 * the high amplitude allows the registration to close large inter-subject
 * shape differences.
 *
 * Run:
 *   sbt "runMain scapula.Set3_Alemneh"
 *
 * Output folder:  $SCAPULA_OUT_DIR/set3-alemneh/
 */
object Set3_Alemneh {
  def main(args: Array[String]): Unit =
    Stage2NonRigidReg.runWithGpParams(Stage2NonRigidReg.gpParamSets(2))
}
