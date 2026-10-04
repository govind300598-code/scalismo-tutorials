package scapula

import scalismo.common.ScalarMeshField
import scalismo.io.StatisticalModelIO
import scalismo.ui.api.ScalismoUI

import java.io.File
import scala.io.Source
import scala.util.Using

/**
 * Opens the per-point difference between the Hill-Sachs and paired mean shapes (written by CompareGroupSSMs as
 * group_mean_shape_difference.csv) as a color-mapped scalar field in scalismo-ui -- same Blue-Red heatmap
 * convention as the registration-error heatmap, so the sliders/opacity/screenshot workflow is identical.
 *
 * Run after CompareGroupSSMs.
 */
object ViewGroupMeanShapeDifference {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val outDir = Config.outDir
    val modelFile = new File(outDir, "scapula_pca_model_hillsachs.json")
    require(modelFile.exists(), s"$modelFile not found -- run Stage3PCAModelByGroup first.")
    val mesh = StatisticalModelIO.readStatisticalTriangleMeshModel3D(modelFile).get.mean

    val csv = new File(outDir, "group_mean_shape_difference.csv")
    require(csv.exists(), s"$csv not found -- run CompareGroupSSMs first.")
    val diffs = Using.resource(Source.fromFile(csv)) { src =>
      src.getLines().drop(1).filter(_.trim.nonEmpty).map { line =>
        val cols = line.split(",")
        cols(0).toInt -> cols(4).toDouble
      }.toIndexedSeq.sortBy(_._1).map(_._2)
    }
    require(diffs.length == mesh.pointSet.numberOfPoints,
      s"CSV has ${diffs.length} rows but the mean mesh has ${mesh.pointSet.numberOfPoints} points -- " +
        "they must be out of sync (re-run CompareGroupSSMs).")

    val scalarField = ScalarMeshField(mesh, diffs)

    val ui = ScalismoUI()
    val group = ui.createGroup("Hill-Sachs vs paired -- mean shape difference (mm)")
    ui.show(group, scalarField, "diff_mm")

    println(f"Loaded ${diffs.length} per-point differences (mean=${diffs.sum / diffs.length}%.3f mm, " +
      f"max=${diffs.max}%.3f mm).")
    println("\nOpened in scalismo-ui as a scalar field colored by diff_mm (Blue=low, Red=high), same convention as " +
      "your registration-error heatmap:")
    println(f"  1. Fix the color sliders to 0.00 - ${diffs.max}%.2f for the full range (or 0-3 for comparability " +
      "across figures)")
    println("  2. Opacity to 100%")
    println("  3. Orient anterior, screenshot; rotate 180 deg, screenshot posterior")
    println("  4. Check where the hottest (red) region sits relative to your landmarks (especially GC, the " +
      "glenoid centre) -- that location is what tells you whether this lines up with the glenoid/CSA finding " +
      "from the literature, or sits somewhere else on the bone.")
  }
}
