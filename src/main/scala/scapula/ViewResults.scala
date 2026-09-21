package scapula

import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.ui.api.ScalismoUI

import java.awt.Color
import java.io.File
import java.nio.file.{Files, StandardCopyOption}

/**
 * Opens Stage2ReferenceRefinement's saved output (Config.outDir) in the Scalismo viewer. Kept separate from the
 * actual registration so that looking at the result doesn't require re-running (or holding in memory) the whole
 * batch fitting pass -- see the note at the top of Stage2ReferenceRefinement for why that combination is what
 * ran the process out of memory the first time.
 *
 * IMPORTANT: this used to add every single mesh (all subjects' target+fit pairs, both models, every pass
 * reference -- easily 25+ meshes for an 11-subject run) to the scene at full opacity with no visibility control,
 * so every screenshot of it looked like a tangled mess of overlapping, unrelated meshes -- not because anything
 * was misaligned, but because you were always looking at every subject stacked on top of every other subject
 * simultaneously. Everything is still loaded (so it's all there in the tree to inspect), but only a clean default
 * subset -- the reference, the PCA model, and ONE subject's target+fit pair -- starts at visible opacity. Use the
 * Appearance panel's Opacity slider on any other item to bring it into view, one at a time.
 *
 * Run Stage2ReferenceRefinement first; this only reads what it wrote.
 */
object ViewResults {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dir = Config.outDir
    require(dir.exists(), s"${dir.getAbsolutePath} does not exist -- run Stage2ReferenceRefinement first.")

    val ui = ScalismoUI()

    // scalismo 0.92.1 always writes its own JSON-based format regardless of file name, but its READER dispatches
    // purely on extension: ".h5" is parsed as real binary HDF5 (and fails on this content), ".json" is parsed
    // correctly. Stage2ReferenceRefinement now writes "scapula_gpmm.json" directly; this also self-heals a
    // model saved as "scapula_gpmm.h5" by an older run, by reading it through a renamed temp copy instead.
    val jsonModelFile = new File(dir, "scapula_gpmm.json")
    val legacyModelFile = new File(dir, "scapula_gpmm.h5")
    val modelFileToRead =
      if (jsonModelFile.exists()) Some(jsonModelFile)
      else if (legacyModelFile.exists()) {
        val tmp = Files.createTempFile("scapula_gpmm", ".json")
        Files.copy(legacyModelFile.toPath, tmp, StandardCopyOption.REPLACE_EXISTING)
        println(s"(found legacy $legacyModelFile -- its content is actually JSON despite the .h5 name; " +
          s"reading it via a renamed temp copy)")
        Some(tmp.toFile)
      } else None

    modelFileToRead match {
      case Some(modelFile) =>
        val gpmm = StatisticalModelIO.readStatisticalTriangleMeshModel3D(modelFile).get
        val modelGroup = ui.createGroup("analytic GPMM (registration prior, NOT population-trained)")
        val gpmmView = ui.show(modelGroup, gpmm, "scapula_gpmm")
        gpmmView.referenceView.opacity = 0.0 // hidden by default -- toggle on to compare against the PCA model below
        println(s"Loaded analytic GPMM (rank ${gpmm.rank}, hidden by default) -- its 'Random' samples reflect " +
          "the hand-picked kernel, not real anatomy")
      case None =>
        println(s"!! Neither $jsonModelFile nor $legacyModelFile found -- did Stage2ReferenceRefinement finish?")
    }

    // The actual, data-driven statistical shape model (Stage3PCAModel), if it's been built. Comparing its
    // "Random" samples against the analytic GPMM above is the direct visual answer to "why do samples look wrong".
    val pcaModelFile = new File(dir, "scapula_pca_model.json")
    if (pcaModelFile.exists()) {
      val pcaModel = StatisticalModelIO.readStatisticalTriangleMeshModel3D(pcaModelFile).get
      val pcaGroup = ui.createGroup("PCA model (data-driven, real SSM)")
      ui.show(pcaGroup, pcaModel, "scapula_pca_model") // visible by default
      println(s"Loaded PCA model (rank ${pcaModel.rank}, visible) -- its 'Random' samples are drawn from the " +
        "actual population's learned variation")
    } else {
      println(s"(no $pcaModelFile yet -- run `sbt \"runMain scapula.Stage3PCAModel\"` to build the real SSM)")
    }

    val referenceFile = new File(dir, "reference_mean_shape.stl")
    if (referenceFile.exists()) {
      val referenceGroup = ui.createGroup("reference")
      ui.show(referenceGroup, MeshIO.readMesh(referenceFile).get, "reference_mean_shape") // visible by default
    }

    // Per-pass references, so you can see the template converge (pass1_reference.stl, pass2_reference.stl, ...).
    // Hidden by default: same reason as everything else below.
    val passFiles = Option(dir.listFiles((_, name) => name.matches("pass\\d+_reference\\.stl"))).getOrElse(Array.empty[File])
    if (passFiles.nonEmpty) {
      val passGroup = ui.createGroup("reference by pass")
      passFiles.sortBy(_.getName).foreach { f =>
        val view = ui.show(passGroup, MeshIO.readMesh(f).get, f.getName.stripSuffix(".stl"))
        view.opacity = 0.0
      }
    }

    // Per-subject target vs. fit, so you can visually compare each subject to what the GPMM produced for them.
    // Only the FIRST subject starts visible (target in default color, fit in orange at 50% opacity so both show
    // through each other) -- everything else loads hidden. Bring another subject into view with its Opacity
    // slider in the Appearance panel; leave the rest at 0 rather than stacking subjects on top of each other.
    val targetFiles = Option(dir.listFiles((_, name) => name.startsWith("final_") && name.endsWith("_target.stl")))
      .getOrElse(Array.empty[File])
      .sortBy(_.getName)
    if (targetFiles.nonEmpty) {
      val fitsGroup = ui.createGroup("subjects: target vs fit")
      targetFiles.zipWithIndex.foreach { case (targetFile, i) =>
        val subjectId = targetFile.getName.stripPrefix("final_").stripSuffix("_target.stl")
        val fitFile = new File(dir, s"final_${subjectId}_fit.stl")
        val targetView = ui.show(fitsGroup, MeshIO.readMesh(targetFile).get, s"${subjectId}_target")
        val fitViewOpt = if (fitFile.exists()) Some(ui.show(fitsGroup, MeshIO.readMesh(fitFile).get, s"${subjectId}_fit")) else None
        if (i == 0) {
          fitViewOpt.foreach { v =>
            v.color = new Color(230, 126, 34) // orange, distinct from the target's default color
            v.opacity = 0.5
          }
          println(s"  -> $subjectId is the one visible by default (target=default color, fit=orange @ 50%% opacity)")
        } else {
          targetView.opacity = 0.0
          fitViewOpt.foreach(_.opacity = 0.0)
        }
      }
      println(s"Loaded ${targetFiles.length} subject target/fit pairs (only the first is visible by default)")
    } else {
      println(s"!! No final_*_target.stl files found in ${dir.getAbsolutePath}")
    }

    println(s"\nError-metric CSVs are plain text/spreadsheet files in ${dir.getAbsolutePath}:")
    println("  pairwise_distance_matrix.csv, medoid_ranking.csv, fit_quality_by_pass.csv, final_fit_quality.csv")
  }
}
