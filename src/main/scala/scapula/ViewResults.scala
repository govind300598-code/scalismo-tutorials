package scapula

import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Opens Stage2ReferenceRefinement's saved output (Config.outDir) in the Scalismo viewer. Kept separate from the
 * actual registration so that looking at the result doesn't require re-running (or holding in memory) the whole
 * batch fitting pass -- see the note at the top of Stage2ReferenceRefinement for why that combination is what
 * ran the process out of memory the first time.
 *
 * Run Stage2ReferenceRefinement first; this only reads what it wrote.
 */
object ViewResults {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dir = Config.outDir
    require(dir.exists(), s"${dir.getAbsolutePath} does not exist -- run Stage2ReferenceRefinement first.")

    val ui = ScalismoUI()

    val modelFile = new File(dir, "scapula_gpmm.h5")
    if (modelFile.exists()) {
      val gpmm = StatisticalModelIO.readStatisticalTriangleMeshModel3D(modelFile).get
      val modelGroup = ui.createGroup("final GPMM")
      ui.show(modelGroup, gpmm, "scapula_gpmm")
      println(s"Loaded $modelFile (rank ${gpmm.rank})")
    } else {
      println(s"!! $modelFile not found -- did Stage2ReferenceRefinement finish?")
    }

    val referenceFile = new File(dir, "reference_mean_shape.stl")
    if (referenceFile.exists()) {
      val referenceGroup = ui.createGroup("reference")
      ui.show(referenceGroup, MeshIO.readMesh(referenceFile).get, "reference_mean_shape")
    }

    // Per-pass references, so you can see the template converge (pass1_reference.stl, pass2_reference.stl, ...).
    val passFiles = Option(dir.listFiles((_, name) => name.matches("pass\\d+_reference\\.stl"))).getOrElse(Array.empty[File])
    if (passFiles.nonEmpty) {
      val passGroup = ui.createGroup("reference by pass")
      passFiles.sortBy(_.getName).foreach { f =>
        ui.show(passGroup, MeshIO.readMesh(f).get, f.getName.stripSuffix(".stl"))
      }
    }

    // Per-subject target vs. fit, so you can visually compare each subject to what the GPMM produced for them.
    val targetFiles = Option(dir.listFiles((_, name) => name.startsWith("final_") && name.endsWith("_target.stl")))
      .getOrElse(Array.empty[File])
      .sortBy(_.getName)
    if (targetFiles.nonEmpty) {
      val fitsGroup = ui.createGroup("subjects: target vs fit")
      targetFiles.foreach { targetFile =>
        val subjectId = targetFile.getName.stripPrefix("final_").stripSuffix("_target.stl")
        val fitFile = new File(dir, s"final_${subjectId}_fit.stl")
        ui.show(fitsGroup, MeshIO.readMesh(targetFile).get, s"${subjectId}_target")
        if (fitFile.exists()) ui.show(fitsGroup, MeshIO.readMesh(fitFile).get, s"${subjectId}_fit")
      }
      println(s"Loaded ${targetFiles.length} subject target/fit pairs")
    } else {
      println(s"!! No final_*_target.stl files found in ${dir.getAbsolutePath}")
    }

    println(s"\nError-metric CSVs are plain text/spreadsheet files in ${dir.getAbsolutePath}:")
    println("  pairwise_distance_matrix.csv, medoid_ranking.csv, fit_quality_by_pass.csv, final_fit_quality.csv")
  }
}
