package scapula

import scalismo.common.ScalarMeshField3D
import scalismo.io.MeshIO
import scalismo.ui.api.ScalismoUI
import scalismo.ui.model.properties.ScalarRange

import java.io.File

/**
 * Colors each subject's FITTED mesh by its own per-vertex distance error to that subject's real target surface --
 * the standard registration-quality heatmap: white/blue = the fit landed almost exactly on the real surface at
 * that point, red = it's still off by a visible amount there. This is a much more informative way to judge a fit
 * than a single averaged number, because it shows WHERE the error is (e.g. concentrated at the acromion tip or
 * the glenoid rim, vs. spread evenly), which a mean/RMS/HD95 table can't show at all.
 *
 * The color scale is capped at `maxErrorMm` for every subject (not auto-ranged per mesh), so subjects are
 * visually comparable to each other -- a subject whose worst point is 2mm should look very different from one
 * whose worst point is 15mm, not both "maximally red".
 *
 * Reads what Stage2ReferenceRefinement wrote (final_<subject>_fit.stl / _target.stl); does not recompute anything.
 */
object ErrorHeatmap {

  private val maxErrorMm = 5.0f

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dir = Config.outDir
    require(dir.exists(), s"${dir.getAbsolutePath} does not exist -- run Stage2ReferenceRefinement first.")

    val fitFiles = Option(dir.listFiles((_, name) => name.startsWith("final_") && name.endsWith("_fit.stl")))
      .getOrElse(Array.empty[File])
      .sortBy(_.getName)
    require(fitFiles.nonEmpty, s"No final_*_fit.stl files found in ${dir.getAbsolutePath}.")

    val ui = ScalismoUI()
    val group = ui.createGroup(f"error heatmap (white/blue = 0mm, red = ${maxErrorMm}%.0f mm+)")

    fitFiles.zipWithIndex.foreach { case (fitFile, i) =>
      val subjectId = fitFile.getName.stripPrefix("final_").stripSuffix("_fit.stl")
      val targetFile = new File(dir, s"final_${subjectId}_target.stl")
      if (targetFile.exists()) {
        val fittedMesh = MeshIO.readMesh(fitFile).get
        val targetMesh = MeshIO.readMesh(targetFile).get

        // Per-vertex distance from the fit to the real target surface -- exactly the same quantity
        // Metrics.symmetric averages into the "fitted mean" you see in the console output, but per-point instead
        // of collapsed into one number.
        val perVertexError = Metrics.surfaceDistances(fittedMesh, targetMesh)
        val field = ScalarMeshField3D(fittedMesh, perVertexError)

        val view = ui.show(group, field, subjectId)
        view.scalarRange = ScalarRange(0f, maxErrorMm)
        view.opacity = if (i == 0) 1.0 else 0.0 // only the first subject visible by default, same reasoning as ViewResults

        val worst = perVertexError.max
        val mean = perVertexError.sum / perVertexError.length
        println(f"  $subjectId%-24s mean=$mean%5.2f mm  worst-point=$worst%6.2f mm" +
          (if (i == 0) "  (visible)" else "  (hidden -- toggle Opacity to inspect)"))
      } else {
        println(s"  !! no matching $targetFile for $fitFile, skipped")
      }
    }

    println(s"\nLoaded ${fitFiles.length} error heatmaps. Only the first is visible by default -- use the")
    println("Opacity slider in the Appearance panel to bring another subject into view one at a time.")
    println("Red patches are exactly the parts of that subject's anatomy the model fit worst -- if they cluster")
    println("in one region (e.g. the acromion tip, the glenoid rim) across several subjects, that's a real,")
    println("localized weakness in the reference/kernel, not just optimizer noise.")
  }
}
