package scapula

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, MultivariateNormalDistribution, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File
import scala.io.Source
import scala.util.Using

/**
 * Scapula GP pipeline — 7 steps.
 *
 * Run:  sbt "runMain scapula.ScapulaGpPipeline"
 *
 * Edit the Parameters block below. Nothing else needs touching.
 */
object ScapulaGpPipeline {

  // ── Parameters ─────────────────────────────────────────────────────────────

  // Reference scapula mesh (.stl or .vtk) — the fixed template shape
  val referenceMeshPath: String      = "data/reference.stl"
  // Landmarks for the reference (.json = Scalismo format  |  .csv = name,x,y,z with header)
  val referenceLandmarksPath: String = "data/reference_landmarks.json"

  // Target scapula mesh (.stl or .vtk) — the shape to register onto
  val targetMeshPath: String         = "data/target.stl"
  // Landmarks for the target (same format as reference)
  val targetLandmarksPath: String    = "data/target_landmarks.json"

  // Where the fitted GP model is written (must end in .h5)
  val modelOutputPath: String        = "data/scapula_gp_model.h5"

  // Gaussian kernel width in mm — LARGER sigma → smoother, longer-range deformations
  //   Too small → high-frequency ripples / jagged artifacts on thin structures
  //   Too large → deformations cannot capture local shape detail
  val kernelSigma: Double = 50.0

  // Kernel amplitude — LARGER scale → bigger overall deformation magnitudes allowed
  //   Too small → model cannot reach the target; too large → overfitting noise
  val kernelScale: Double = 100.0

  // Number of deformation modes kept (rank of the low-rank GP approximation)
  val gpRank: Int = 5

  // Random seed — change to get a different Nyström basis (results vary slightly)
  val seed: Long = 42L

  // Reference points sampled per ICP iteration — trade-off: speed vs. coverage
  val numIcpPoints: Int = 2000

  // GP-ICP iterations — more iterations = tighter fit, but diminishing returns after ~20
  val numIcpIterations: Int = 20

  // Correspondence noise std-dev in mm — smaller = fit trusts correspondences more
  //   Increase if the target is noisy or has missing regions
  val noiseStd: Double = 1.0

  // ── Main ───────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(seed)

    // ── Step 1: Load reference mesh + landmarks ───────────────────────────────
    println("=== Step 1: Reference mesh + landmarks ===")
    val refMesh = loadMesh(referenceMeshPath)
    val refLms  = loadLandmarks(referenceLandmarksPath)
    println(s"  ${refMesh.pointSet.numberOfPoints} vertices, ${refLms.size} landmarks")

    // ── Step 2: Load target mesh + landmarks ──────────────────────────────────
    println("=== Step 2: Target mesh + landmarks ===")
    val tgtMesh = loadMesh(targetMeshPath)
    val tgtLms  = loadLandmarks(targetLandmarksPath)
    println(s"  ${tgtMesh.pointSet.numberOfPoints} vertices, ${tgtLms.size} landmarks")

    // ── Step 3: Rigid registration — align target into reference frame ────────
    // Landmarks pair the two coordinate systems.  We compute the rigid transform
    // that moves the TARGET onto the REFERENCE, then apply it to the target mesh.
    // After this the two bones occupy roughly the same space, which is required
    // before the non-rigid step can make meaningful correspondences.
    println("=== Step 3: Rigid registration (target → reference frame) ===")
    val pairs = matchedPairs(moving = tgtLms, fixed = refLms)
    require(pairs.nonEmpty,
      "No shared landmark IDs found — make sure both files use identical landmark names.")
    val rigidTransform = LandmarkRegistration.rigid3DLandmarkRegistration(pairs, Point3D(0, 0, 0))
    val alignedTarget  = tgtMesh.transform(rigidTransform)
    println(s"  Aligned target using ${pairs.size} landmark pairs.")

    // ── Step 4: GP deformation model — single Gaussian kernel, rank 5 ─────────
    // The GP model defines the SPACE of deformations the reference is allowed to
    // undergo.  Only deformations that the kernel considers plausible are reachable:
    //
    //   • sigma  controls the spatial REACH of each basis deformation.
    //            Large sigma → smooth, correlated, global bends.
    //            Small sigma → local, oscillatory deformations → jagged artifacts
    //            on thin bony structures (acromion, coracoid, glenoid rim).
    //
    //   • scale  controls the MAGNITUDE of allowed deformation.
    //            Larger scale → the model can move points further.
    //
    // This is exactly what your professors mean when they discuss sigma/scale producing
    // jagged artifacts: a Gaussian kernel with insufficient sigma cannot represent
    // the smooth bending a scapula actually undergoes — it fragments it into
    // high-frequency wiggles instead.
    println("=== Step 4: Building GP deformation model ===")
    val scalarKernel = GaussianKernel[_3D](kernelSigma) * kernelScale
    val matrixKernel = DiagonalKernel[_3D](scalarKernel, 3)  // independent x,y,z channels
    val gp           = GaussianProcess[_3D, EuclideanVector[_3D]](matrixKernel)
    val lowRankGp    = LowRankGaussianProcess.approximateGPNystrom(
      refMesh.pointSet,
      gp,
      numBasisFunctions = gpRank,           // exactly 5 modes
      interpolator      = NearestNeighborInterpolator3D()
    )
    val model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGp)
    println(s"  Rank = ${model.rank}  sigma=$kernelSigma mm  scale=$kernelScale")

    // ── Step 5: Non-rigid registration (free-form deformation via GP posterior ICP) ──
    // The GP model defines WHAT deformations are plausible (Step 4).
    // This step finds the BEST deformation within that space that makes the
    // reference match the rigidly-aligned target.
    //
    // Algorithm — GP posterior ICP:
    //   1. For each sampled reference point, find the nearest point on the target surface.
    //   2. Treat each correspondence as a noisy observation of where that point should move.
    //   3. Compute the GP posterior: the MAP deformation field consistent with all observations.
    //   4. The posterior mean is the new "current" mesh.  Repeat.
    //
    // The posterior is computed from the ORIGINAL prior each iteration, so the rank
    // (number of free parameters) stays fixed at gpRank throughout.
    println("=== Step 5: Non-rigid registration (GP posterior ICP) ===")
    val noiseModel = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * (noiseStd * noiseStd)
    )
    val sampledIds: IndexedSeq[PointId] =
      UniformMeshSampler3D(model.reference, numIcpPoints)
        .sample()
        .map { case (pt, _) => model.reference.pointSet.findClosestPoint(pt).id }
        .distinct

    var currentMesh = model.reference
    for (iter <- 1 to numIcpIterations) {
      val observations: IndexedSeq[(PointId, Point[_3D], MultivariateNormalDistribution)] =
        sampledIds.map { id =>
          val movingPt = currentMesh.pointSet.point(id)
          val tgtPt    = alignedTarget.operations.closestPointOnSurface(movingPt).point
          (id, tgtPt, noiseModel)
        }
      currentMesh = model.posterior(observations).mean
      if (iter % 5 == 0 || iter == 1) println(s"  Iteration $iter / $numIcpIterations")
    }
    val fittedMesh = currentMesh

    val dists   = fittedMesh.pointSet.points.map { p =>
      (p - alignedTarget.operations.closestPointOnSurface(p).point).norm
    }.toIndexedSeq
    println(f"  Mean surface error = ${dists.sum / dists.size}%.2f mm   Max = ${dists.max}%.2f mm")

    // ── Step 6: Save GP model as .h5 ──────────────────────────────────────────
    println(s"=== Step 6: Saving model → $modelOutputPath ===")
    new File(modelOutputPath).getParentFile.mkdirs()
    StatisticalModelIO
      .writeStatisticalTriangleMeshModel3D(model, new File(modelOutputPath))
      .fold(
        err => sys.error(s"Save failed: ${err.getMessage}"),
        _   => println("  Saved OK.")
      )

    // ── Step 7: Visualise 5 deformation modes + fitting result ────────────────
    // The GP model slider lets you dial each mode from −3σ to +3σ.
    // Mode i shows one smooth deformation pattern the kernel allows.
    // With large sigma you see global bending; with small sigma you see local ripples.
    println("=== Step 7: Opening Scalismo viewer ===")
    val ui = ScalismoUI()

    val modelGroup = ui.createGroup(s"GP modes  σ=$kernelSigma  scale=$kernelScale  rank=$gpRank")
    ui.show(modelGroup, model, "scapula_gp")
    println(s"  ${model.rank} coefficient sliders are live.")
    println( "  Drag slider i from −3 to +3 to see what mode i allows the bone to do.")
    println( "  Change kernelSigma and rerun to compare smooth vs. jagged deformations.")

    ui.show(ui.createGroup("Reference"), refMesh, "reference")
    ui.show(ui.createGroup("Target (rigidly aligned)"), alignedTarget, "target")
    ui.show(ui.createGroup("Fitted (non-rigid)"), fittedMesh, "fitted")

    println("Done. Close the viewer window to exit.")
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  def loadMesh(path: String): TriangleMesh[_3D] =
    MeshIO.readMesh(new File(path))
          .getOrElse(sys.error(s"Cannot read mesh: $path"))

  def loadLandmarks(path: String): IndexedSeq[Landmark[_3D]] =
    if (path.endsWith(".json"))
      LandmarkIO.readLandmarksJson[_3D](new File(path))
               .getOrElse(sys.error(s"Cannot read landmarks: $path"))
    else
      parseLandmarkCsv(path)

  // CSV format: one header row (skipped), then rows of:  name , x , y , z
  def parseLandmarkCsv(path: String): IndexedSeq[Landmark[_3D]] =
    Using.resource(Source.fromFile(path)) { src =>
      src.getLines().drop(1).filter(_.trim.nonEmpty).map { line =>
        val c = line.split(",").map(_.trim)
        Landmark[_3D](c(0), Point3D(c(1).toDouble, c(2).toDouble, c(3).toDouble))
      }.toIndexedSeq
    }

  // Returns (movingPoint, fixedPoint) pairs for landmarks with matching ids.
  def matchedPairs(
    moving: IndexedSeq[Landmark[_3D]],
    fixed:  IndexedSeq[Landmark[_3D]]
  ): IndexedSeq[(Point[_3D], Point[_3D])] = {
    val fixedById = fixed.map(l => l.id -> l.point).toMap
    moving.flatMap(lm => fixedById.get(lm.id).map(fp => lm.point -> fp))
  }
}
