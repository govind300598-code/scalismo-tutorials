package scapula

import scalismo.io.StatismoIO
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Interactive SSM viewer with PCA sliders.
 *
 * Loads the SSM from pass2/ssm.h5 (the bias-corrected model) and opens the
 * scalismo-ui window.  Use the coefficient sliders to explore each shape mode.
 */
object ViewSSM {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir   = if (args.nonEmpty) new File(args(0)) else Config.outDir
    val pass2Dir = new File(outDir, "pass2")

    require(pass2Dir.exists(),
      s"pass2/ directory not found: ${pass2Dir.getPath}\n  Set SCAPULA_OUT_DIR or pass the output directory as the first argument.")

    val ssmFile = new File(pass2Dir, "ssm.h5")
    require(ssmFile.exists(),
      s"SSM not found: ${ssmFile.getPath} — run RebuildSSM first")

    println(s"Loading SSM from ${ssmFile.getPath}")
    val ssm = StatismoIO.readStatismoMeshModel(ssmFile).get
    println(s"  ${ssm.rank} components, ${ssm.mean.pointSet.numberOfPoints} reference points")

    val ui    = ScalismoUI()
    val group = ui.createGroup("SSM2 (pass2)")
    ui.show(group, ssm, "SSM")
    println("Viewer open — use the coefficient sliders to explore shape modes.")
  }
}
