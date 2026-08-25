package scapula

import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.StatisticalModelIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

object ScapulaGPPipeline {

  // ── tuneable parameters ────────────────────────────────────────────────────
  // All three can be overridden without touching code:
  //   SCAPULA_SIGMA   – Gaussian kernel bandwidth in mm (default 50)
  //   SCAPULA_SCALE   – kernel amplitude in mm          (default 100)
  //   SCAPULA_RANK    – number of modes to keep         (default 5)
  //   SCAPULA_REF_ID  – model-id of the reference specimen
  //   SCAPULA_TGT_ID  – model-id of the target  specimen
  val sigma: Double = sys.env.getOrElse("SCAPULA_SIGMA", "50.0").toDouble
  val scale: Double = sys.env.getOrElse("SCAPULA_SCALE", "100.0").toDouble
  val rank:  Int    = sys.env.getOrElse("SCAPULA_RANK",  "5").toInt
  // ──────────────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // ── Step 1: Load meshes and landmarks ──────────────────────────────────
    println("Step 1 – Loading meshes …")
    val dir       = Config.dataDir
    val specimens = ScapulaData.specimens(dir)
    require(specimens.nonEmpty,
      s"No STL files found in ${dir.getAbsolutePath}. Set SCAPULA_DATA_DIR to your folder.")

    // Default to the first two left scapulae (alphabetical order).
    val lefts = specimens.filter(!_.isRight)
    require(lefts.size >= 2,
      s"Need at least 2 left scapulae in the data dir; found ${lefts.size}.")

    val refId    = sys.env.getOrElse("SCAPULA_REF_ID", lefts(0).modelId)
    val targetId = sys.env.getOrElse("SCAPULA_TGT_ID", lefts(1).modelId)

    val refSpec = specimens.find(_.modelId == refId).getOrElse(
      throw new RuntimeException(
        s"Reference '$refId' not found. Available: ${specimens.map(_.modelId).mkString(", ")}"))
    val tgtSpec = specimens.find(_.modelId == targetId).getOrElse(
      throw new RuntimeException(s"Target '$targetId' not found."))

    val refMesh = ScapulaData.loadMesh(refSpec.file)
    val tgtMesh = ScapulaData.loadMesh(tgtSpec.file)
    println(s"  Reference : ${refMesh.pointSet.numberOfPoints} vertices  (${refSpec.modelId})")
    println(s"  Target    : ${tgtMesh.pointSet.numberOfPoints} vertices  (${tgtSpec.modelId})")

    println("  Loading landmark CSV …")
    val csv = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val refLms = landmarks.getOrElse(refId,
      throw new RuntimeException(s"No landmark row for reference '$refId' in ${csv.getName}"))
    val tgtLms = landmarks.getOrElse(targetId,
      throw new RuntimeException(s"No landmark row for target '$targetId' in ${csv.getName}"))
    println(s"  Loaded ${refLms.length} landmarks for reference  (${refLms.map(_.id).mkString(", ")})")

    // ── Step 2: Rigid landmark registration ───────────────────────────────
    println("Step 2 – Rigid landmark registration …")
    val (alignedRef, _) = RigidAlign.landmarkThenIcp(refMesh, refLms, tgtMesh, tgtLms)
    println("  Reference mesh rigidly aligned into target coordinate frame")

    // ── Step 3: Build GP model ─────────────────────────────────────────────
    println(f"Step 3 – Building GP model  [sigma=$sigma%.1f mm | scale=$scale%.1f | rank=$rank] …")

    // Isotropic Gaussian kernel → diagonal matrix kernel (same deformation in x, y, z).
    val scalarKernel = GaussianKernel[_3D](sigma) * scale
    val matrixKernel = DiagonalKernel(scalarKernel, 3)
    val gp           = GaussianProcess[_3D, EuclideanVector[_3D]](matrixKernel)

    // Pivoted-Cholesky low-rank approximation evaluated on the reference mesh vertices.
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      alignedRef.pointSet,
      gp,
      relativeTolerance = 0.01,
      interpolator = NearestNeighborInterpolator3D()
    )

    // Keep only the requested number of modes (top eigen-directions by variance).
    val truncated = lowRankGP.truncate(rank)
    val model     = PointDistributionModel[_3D, TriangleMesh](alignedRef, truncated)
    println(s"  GP model built — rank = ${model.rank}")

    // ── Step 4: Save model to HDF5 ─────────────────────────────────────────
    val modelFile = new File("/tmp/scapula_gp_model.h5")
    println(s"Step 4 – Saving model to ${modelFile.getAbsolutePath} …")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, modelFile)
      .getOrElse(throw new RuntimeException(s"Failed to save model to $modelFile"))
    println("  Saved")

    // ── Step 5: Open Scalismo UI ───────────────────────────────────────────
    if (Config.showUi) {
      println("Step 5 – Opening Scalismo UI …")
      println(s"  ${model.rank} modes will appear.  Drag each slider to explore each mode.")
      println("  Close the viewer window to exit the programme.")
      val ui  = ScalismoUI()
      val grp = ui.createGroup("GP model")
      ui.show(grp, model,   "scapula_gp")
      ui.show(grp, tgtMesh, "target")
    }
  }
}
