package scapula

import breeze.linalg.DenseVector
import scalismo.io.StatisticalModelIO
import scalismo.ui.api.ScalismoUI

import java.awt.Color
import java.io.File

/**
 * JOR-2024-style transparent overlay visualization of the first three PCA modes.
 *
 * For each mode (PC1, PC2, PC3) a SINGLE ScalismoUI group contains all three states stacked:
 *   - −3σ  blue  (solid, opacity 1.0)
 *   - Mean  light grey  (semi-transparent, opacity 0.35)
 *   - +3σ  red   (solid, opacity 1.0)
 *
 * The mean and the two extremes overlap in one view, so anatomical displacement between the states
 * is immediately visible without switching panels -- exactly the convention used in published SSM
 * mode-shape figures (JOR 2024, Halloran JSES 2018 supplement).
 *
 * Groups for PC2 and PC3 start hidden; unhide them in the ScalismoUI tree to compare modes without
 * visual clutter. The mean-only group ("Mean shape") is always visible as a clean reference.
 *
 * Requires: scapula_pca_model.json built by Stage3PCAModel.
 * Run: sbt "runMain scapula.ViewModeShapes"
 */
object ViewModeShapes {

  // How many modes to visualise (creates one overlay group per mode)
  private val numModes = 3
  // Extremes in units of standard deviations
  private val sdExtreme = 3.0

  private val colorNeg  = new Color( 52, 152, 219) // blue   – negative extreme  (−3σ)
  private val colorMean = new Color(200, 200, 200) // light grey – mean shape
  private val colorPos  = new Color(231,  76,  60) // red    – positive extreme  (+3σ)

  private val opacityExtreme = 1.0f
  private val opacityMean    = 0.35f

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dir = Config.outDir
    val pcaModelFile = new File(dir, "scapula_pca_model.json")
    require(pcaModelFile.exists(),
      s"$pcaModelFile not found -- run `sbt \"runMain scapula.Stage3PCAModel\"` first.")

    val pcaModel = StatisticalModelIO.readStatisticalTriangleMeshModel3D(pcaModelFile).get
    println(s"Loaded PCA model  rank=${pcaModel.rank}")

    val ui = ScalismoUI()

    // Always-visible reference: the population mean shape
    val meanGroup = ui.createGroup("Mean shape")
    val meanMesh  = pcaModel.mean
    val meanView  = ui.show(meanGroup, meanMesh, "mean")
    meanView.color   = colorMean
    meanView.opacity = 1.0f
    println("Group: Mean shape (always visible)")

    // One overlay group per mode
    (0 until math.min(numModes, pcaModel.rank)).foreach { k =>
      val groupName = s"PC${k + 1} overlay (−${sdExtreme.toInt}σ / Mean / +${sdExtreme.toInt}σ)"
      val group = ui.createGroup(groupName)

      def instance(sigma: Double) = {
        val c = DenseVector.zeros[Double](pcaModel.rank)
        c(k) = sigma
        pcaModel.instance(c)
      }

      // −3σ (blue, solid)
      val negView = ui.show(group, instance(-sdExtreme), s"PC${k + 1}_neg${sdExtreme.toInt}sd")
      negView.color   = colorNeg
      negView.opacity = opacityExtreme

      // Mean (grey, semi-transparent)
      val mView = ui.show(group, meanMesh, s"PC${k + 1}_mean")
      mView.color   = colorMean
      mView.opacity = opacityMean

      // +3σ (red, solid)
      val posView = ui.show(group, instance(sdExtreme), s"PC${k + 1}_pos${sdExtreme.toInt}sd")
      posView.color   = colorPos
      posView.opacity = opacityExtreme

      // Hide PC2 and PC3 groups at start so the viewer isn't cluttered
      if (k > 0) {
        negView.opacity = 0.0f
        mView.opacity   = 0.0f
        posView.opacity = 0.0f
      }

      val variance = pcaModel.gp.variance.toArray
      val totalVar = variance.sum
      val cumVar   = variance.take(k + 1).sum / totalVar * 100.0
      println(f"Group: $groupName  (PC${k+1} explains ${variance(k)/totalVar*100}%.1f%% variance; " +
        f"cumulative ${cumVar}%.1f%%)${if (k > 0) "  [hidden -- toggle in tree]" else ""}")
    }

    println(
      s"""
         |Legend:
         |  Blue  = −${sdExtreme.toInt}σ  (solid)
         |  Grey  = Mean  (${(opacityMean * 100).toInt}% transparent)
         |  Red   = +${sdExtreme.toInt}σ  (solid)
         |
         |PC1 overlay is visible by default.
         |Unhide PC2 / PC3 groups in the ScalismoUI tree to compare modes.
         |Use the Appearance > Opacity slider to adjust transparency per mesh.
         |""".stripMargin)
  }
}
