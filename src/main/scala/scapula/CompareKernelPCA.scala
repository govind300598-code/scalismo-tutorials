package scapula

import scalismo.io.StatisticalModelIO
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Loads two (or more) run directories' real, data-driven PCA models (scapula_pca_model.json, written by
 * Stage3PCAModel) into ONE ScalismoUI window, each in its own group -- so their mode sliders can be compared
 * directly on screen instead of juggling separate windows.
 *
 * Directories are given as a comma-separated list via SCAPULA_COMPARE_DIRS, e.g.:
 *   export SCAPULA_COMPARE_DIRS="2-Kernel=$HOME/Documents/database_v1.11/scapula_ssm_out_2kernel,3-Kernel=$HOME/Documents/database_v1.11/scapula_ssm_out"
 *   sbt "runMain scapula.CompareKernelPCA"
 *
 * Each entry is label=path; the label is just what shows up in the scene tree.
 */
object CompareKernelPCA {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val raw = sys.env.getOrElse("SCAPULA_COMPARE_DIRS",
      throw new RuntimeException(
        "Set SCAPULA_COMPARE_DIRS to a comma-separated list of label=dir pairs, e.g.\n" +
        "  export SCAPULA_COMPARE_DIRS=\"2-Kernel=/path/to/scapula_ssm_out_2kernel,3-Kernel=/path/to/scapula_ssm_out\""
      ))
    val entries = raw.split(",").map(_.trim).filter(_.nonEmpty).map { pair =>
      val idx = pair.lastIndexOf('=')
      require(idx > 0, s"malformed entry (expected label=dir): $pair")
      (pair.substring(0, idx), new File(pair.substring(idx + 1)))
    }
    require(entries.nonEmpty, "SCAPULA_COMPARE_DIRS had no valid label=dir entries")

    val ui = ScalismoUI()

    entries.foreach { case (label, dir) =>
      val modelFile = new File(dir, "scapula_pca_model.json")
      if (!modelFile.exists()) {
        println(s"!! $label: $modelFile not found -- run Stage3PCAModel there first. Skipping.")
      } else {
        val model = StatisticalModelIO.readStatisticalTriangleMeshModel3D(modelFile).get
        val group = ui.createGroup(s"$label (rank ${model.rank})")
        ui.show(group, model, label)
        println(s"Loaded $label from $modelFile (rank ${model.rank})")
      }
    }

    println("\nAll requested models loaded. Adjust each group's mode sliders in the Appearance panel to compare.")
  }
}
