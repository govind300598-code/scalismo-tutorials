package scapula

import scalismo.geometry._3D
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Builds the SSM in memory from pass2/reg_*.stl and opens Scalismo UI.
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

    val pass2Dir = new File(outDir, "pass2")
    require(pass2Dir.exists(), s"pass2/ not found — run FullPipeline first")

    val meanFile = new File(outDir, "mean_pass2.stl")
    require(meanFile.exists(), s"mean_pass2.stl not found — run FullPipeline first")

    // ── Load reference (mean shape) ────────────────────────────────────────────
    println(s"Loading reference from mean_pass2.stl ...")
    val reference: TriangleMesh[_3D] = MeshIO.readMesh(meanFile)
      .getOrElse(throw new RuntimeException("Cannot read mean_pass2.stl"))
    println(s"  ${reference.pointSet.numberOfPoints} vertices")

    // ── Load registered shapes ─────────────────────────────────────────────────
    val regFiles = Option(pass2Dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(f => f.getName.startsWith("reg_") && f.getName.endsWith(".stl"))
      .sortBy(_.getName)

    require(regFiles.nonEmpty, s"No reg_*.stl files found in ${pass2Dir.getAbsolutePath}")
    println(s"Loading ${regFiles.length} registered meshes from pass2/ ...")

    val meshes: IndexedSeq[TriangleMesh[_3D]] = regFiles.map { f =>
      MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read ${f.getName}"))
    }.toIndexedSeq

    // ── Build SSM in memory ────────────────────────────────────────────────────
    println("Building SSM via PCA ...")
    val dc    = DataCollection.fromTriangleMesh3DSequence(reference, meshes)
    val model = PointDistributionModel.createUsingPCA[_3D, TriangleMesh](dc)
    println(s"  ${model.rank} modes  |  ${model.reference.pointSet.numberOfPoints} vertices")

    // ── Open UI ───────────────────────────────────────────────────────────────
    val ui = ScalismoUI("Scapula SSM Viewer")

    val ssmGroup = ui.createGroup("SSM (drag sliders to explore modes)")
    ui.show(ssmGroup, model, "ssm_final")

    val meanGroup = ui.createGroup("Mean shape (Pass 2)")
    ui.show(meanGroup, reference, "mean_pass2")

    val regGroup = ui.createGroup(s"Pass-2 registrations (${regFiles.length} specimens)")
    regFiles.zip(meshes).foreach { case (f, mesh) =>
      val id = f.getName.stripPrefix("reg_").stripSuffix(".stl")
      ui.show(regGroup, mesh, id)
    }

    println()
    println("Scalismo UI open. Close the window to exit.")
    println("Tip: in the SSM group, drag the mode sliders to explore shape variation.")
    println("     Mode 1 = 29.7% variance (dominant shape axis)")
  }
}
