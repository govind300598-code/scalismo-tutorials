package scapula

import scalismo.io.StatisticalModelIO
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Opens SSM1 (Hill-Sachs) and SSM2 (paired scapula + paired shoulder) side by side in scalismo-ui, each as its
 * own interactive model with mean + mode-of-variation sliders -- same as ViewSSMResult.scala, just for both
 * group-specific models from Stage3PCAModelByGroup instead of one pooled model.
 *
 * Run after Stage3PCAModelByGroup.
 */
object ViewGroupSSMs {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val outDir = Config.outDir
    val fileA = new File(outDir, "scapula_pca_model_hillsachs.json")
    val fileB = new File(outDir, "scapula_pca_model_paired.json")
    require(fileA.exists(), s"$fileA not found -- run Stage3PCAModelByGroup first.")
    require(fileB.exists(), s"$fileB not found -- run Stage3PCAModelByGroup first.")

    val modelA = StatisticalModelIO.readStatisticalTriangleMeshModel3D(fileA).get
    val modelB = StatisticalModelIO.readStatisticalTriangleMeshModel3D(fileB).get
    println(s"SSM1 (hillsachs): rank ${modelA.rank}")
    println(s"SSM2 (paired):    rank ${modelB.rank}")

    val ui = ScalismoUI()

    val groupA = ui.createGroup("SSM1 - Hill-Sachs (n=49)")
    ui.show(groupA, modelA, "hillsachs SSM")

    val groupB = ui.createGroup("SSM2 - Paired scapula+shoulder (n=50)")
    ui.show(groupB, modelB, "paired SSM")

    println("\nBoth models are now in the scene tree as separate groups:")
    println("  'SSM1 - Hill-Sachs (n=49)'  -- select 'hillsachs SSM' for its mean + mode sliders")
    println("  'SSM2 - Paired scapula+shoulder (n=50)' -- select 'paired SSM' for its mean + mode sliders")
    println("\nBoth are shown overlapping at the same position by default -- hide one group's checkbox in the " +
      "scene tree to look at the other alone, or leave both visible (semi-transparent) to eyeball how far apart " +
      "their means sit. Drag either model's PC1 slider independently to compare how each group's dominant mode " +
      "of variation looks.")
  }
}
