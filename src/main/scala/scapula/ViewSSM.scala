package scapula

import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Opens the final SSM and all Pass-2 registered shapes in Scalismo UI.
 * Run with:  sbt "runMain scapula.ViewSSM"
 *
 * Override the output directory via the same env var the pipeline uses:
 *   SCAPULA_OUT_DIR=/your/path sbt "runMain scapula.ViewSSM"
 */
object ViewSSM {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    require(outDir.exists(), s"Output directory not found: ${outDir.getAbsolutePath}")

    val modelFile = new File(outDir, "ssm_final.h5")
    require(modelFile.exists(), s"SSM file not found: ${modelFile.getAbsolutePath}\nRun FullPipeline first.")

    println(s"Loading SSM from: ${modelFile.getAbsolutePath}")
    val model = StatisticalModelIO.readStatisticalMeshModel(modelFile) match {
      case scala.util.Success(m) => m
      case scala.util.Failure(e) =>
        e.printStackTrace()
        throw new RuntimeException(s"readStatisticalMeshModel failed: ${e.getMessage}", e)
    }

    println(s"  ${model.rank} modes  |  ${model.referenceMesh.pointSet.numberOfPoints} vertices")

    val ui = ScalismoUI("Scapula SSM Viewer")

    // ── SSM group ──────────────────────────────────────────────────────────────
    val ssmGroup = ui.createGroup("SSM (drag sliders to explore modes)")
    ui.show(ssmGroup, model, "ssm_final")

    // ── Mean shape ─────────────────────────────────────────────────────────────
    val meanFile = new File(outDir, "mean_pass2.stl")
    if (meanFile.exists()) {
      val meanGroup = ui.createGroup("Mean shape (Pass 2)")
      val mean = MeshIO.readMesh(meanFile).getOrElse(throw new RuntimeException("Cannot read mean_pass2.stl"))
      ui.show(meanGroup, mean, "mean_pass2")
      println(s"Mean shape loaded: ${mean.pointSet.numberOfPoints} vertices")
    } else {
      println("mean_pass2.stl not found — skipping mean shape")
    }

    // ── Registered shapes ──────────────────────────────────────────────────────
    val pass2Dir = new File(outDir, "pass2")
    if (pass2Dir.exists()) {
      val regFiles = Option(pass2Dir.listFiles())
        .getOrElse(Array.empty[File])
        .filter(f => f.getName.startsWith("reg_") && f.getName.endsWith(".stl"))
        .sortBy(_.getName)

      if (regFiles.nonEmpty) {
        val regGroup = ui.createGroup(s"Pass-2 registrations (${regFiles.length} specimens)")
        regFiles.foreach { f =>
          val mesh = MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read ${f.getName}"))
          val id = f.getName.stripPrefix("reg_").stripSuffix(".stl")
          ui.show(regGroup, mesh, id)
        }
        println(s"Loaded ${regFiles.length} registered shapes from pass2/")
      }
    } else {
      println("pass2/ directory not found — registered shapes not loaded")
    }

    println()
    println("Scalismo UI open. Close the window to exit.")
    println("Tip: in the SSM group, drag the mode sliders to explore shape variation.")
    println("     Mode 1 = 29.7% variance (dominant shape axis)")
  }
}
