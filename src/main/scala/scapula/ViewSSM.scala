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
 * View the scapula SSM in the Scalismo UI — no HDF5 file required.
 *
 * Reads all pass2/reg_*.stl files from the output directory, builds the SSM
 * in memory via PCA (~26 s for 24 meshes), then opens the Scalismo UI with:
 *   - "SSM (sliders)"  : the statistical model with interactive sliders
 *   - "Mean shape"     : the PCA mean mesh
 *   - "Specimens"      : all 24 registered shapes
 *
 * Usage (override paths with env vars or pass the output dir as the first arg):
 *   sbt "runMain scapula.ViewSSM [/path/to/output/dir]"
 *   SCAPULA_OUT_DIR=/your/path sbt "runMain scapula.ViewSSM"
 */
object ViewSSM {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val baseDir  = if (args.nonEmpty) new File(args(0)) else Config.outDir
    val pass2Dir = new File(baseDir, "pass2")

    require(
      pass2Dir.isDirectory,
      s"pass2/ directory not found: ${pass2Dir.getAbsolutePath}\n" +
      s"  Set SCAPULA_OUT_DIR or pass the output directory as the first argument."
    )

    val regFiles = Option(pass2Dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith("reg_"))
      .sortBy(_.getName)

    require(regFiles.nonEmpty, s"No reg_*.stl files found in ${pass2Dir.getAbsolutePath}")

    println(s"Loading ${regFiles.length} registered meshes from ${pass2Dir.getAbsolutePath}")
    val meshes = regFiles.map(ScapulaData.loadMesh).toIndexedSeq
    val nPts   = meshes.head.pointSet.numberOfPoints
    println(s"  ${meshes.length} meshes, $nPts vertices each")

    println("Building SSM in memory via PCA...")
    val t0    = System.currentTimeMillis()
    val dc    = DataCollection.fromTriangleMesh3DSequence(meshes.head, meshes)
    val model = PointDistributionModel.createUsingPCA(dc)
    println(f"  Done in ${(System.currentTimeMillis() - t0) / 1000.0}%.1f s  " +
            f"| rank = ${model.rank}  | vertices = ${model.reference.pointSet.numberOfPoints}")

    // Quick variance summary
    val eigenvalues = model.gp.eigenvalues.toArray
    val totalVar    = eigenvalues.sum
    val cumVar      = eigenvalues.scanLeft(0.0)(_ + _).tail.map(_ / totalVar * 100.0)
    def modesFor(pct: Double): Int = { val i = cumVar.indexWhere(_ >= pct); if (i < 0) model.rank else i + 1 }
    println(f"  90%% variance: ${modesFor(90)} modes  |  95%%: ${modesFor(95)}  |  99%%: ${modesFor(99)}")

    println("Opening Scalismo UI...")
    val ui        = ScalismoUI()
    val ssmGroup  = ui.createGroup("SSM (sliders)")
    val meanGroup = ui.createGroup("Mean shape")
    val specGroup = ui.createGroup("Specimens")

    ui.show(ssmGroup, model, "SSM")
    ui.show(meanGroup, model.mean, "mean")
    meshes.zip(regFiles).foreach { case (m, f) =>
      ui.show(specGroup, m, f.getName.stripSuffix(".stl"))
    }

    println("UI open. Close the window to exit.")
  }
}
