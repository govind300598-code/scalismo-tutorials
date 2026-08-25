package scapula

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, MultivariateNormalDistribution, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Scapula GP Pipeline — five self-contained steps.
 *
 * Run with:   sbt "runMain scapula.ScapulaGpPipeline"
 * Swap the paths and kernel parameters at the top; nothing else needs editing.
 */
object ScapulaGpPipeline {

  // ── Parameters ─────────────────────────────────────────────────────────────
  // Reference scapula mesh, assumed already rigidly aligned (.stl or .vtk)
  val referenceMeshPath: String = "data/reference.stl"
  // Target scapula mesh the model will be fitted to in Step 5
  val targetMeshPath: String    = "data/target.stl"
  // Output path for the saved GP model
  val modelOutputPath: String   = "data/scapula_gp_model.h5"

  // Gaussian kernel width in mm — larger sigma → smoother, longer-range deformations
  val kernelSigma: Double = 50.0
  // Kernel amplitude — larger scale → bigger magnitude deformations overall
  val kernelScale: Double = 100.0
  // Number of deformation modes (rank of the low-rank GP approximation)
  val gpRank: Int = 5
  // Random seed for the Nyström approximation (any integer; change to get different bases)
  val seed: Long = 42L

  // Number of reference vertices sampled per ICP iteration for correspondence search
  val numIcpPoints: Int = 2000
  // Total GP-ICP iterations for non-rigid fitting in Step 5
  val numIcpIterations: Int = 20
  // Assumed correspondence noise std-dev in mm — smaller = tighter fit, more sensitive to outliers
  val noiseStd: Double = 1.0

  // ── Main ───────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(seed)

    // -------------------------------------------------------------------------
    // Step 1 — Load reference mesh
    // -------------------------------------------------------------------------
    println("=== Step 1: Loading reference mesh ===")
    val refMesh = MeshIO.readMesh(new File(referenceMeshPath))
      .getOrElse(sys.error(s"Cannot read mesh: $referenceMeshPath"))
    println(s"  ${refMesh.pointSet.numberOfPoints} vertices, " +
            s"${refMesh.triangulation.triangles.size} triangles")

    // -------------------------------------------------------------------------
    // Step 2 — Single Gaussian kernel GP model (rank 5)
    // -------------------------------------------------------------------------
    println("=== Step 2: Building GP deformation model ===")
    // One scalar Gaussian kernel, turned into a 3-D matrix kernel for (dx, dy, dz)
    val scalarKernel = GaussianKernel[_3D](kernelSigma) * kernelScale
    val matrixKernel = DiagonalKernel[_3D](scalarKernel, 3)
    // Zero-mean GP — we are modelling deformations away from the reference
    val gp           = GaussianProcess[_3D, EuclideanVector[_3D]](matrixKernel)
    // Nyström approximation retains exactly `gpRank` basis functions
    val lowRankGp    = LowRankGaussianProcess.approximateGPNystrom(
      refMesh.pointSet,
      gp,
      numBasisFunctions = gpRank,
      interpolator      = NearestNeighborInterpolator3D()
    )
    val model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGp)
    println(s"  GP model ready. Rank = ${model.rank}  " +
            s"(sigma=$kernelSigma mm, scale=$kernelScale)")

    // -------------------------------------------------------------------------
    // Step 3 — Save as .h5
    // -------------------------------------------------------------------------
    println(s"=== Step 3: Saving model → $modelOutputPath ===")
    new File(modelOutputPath).getParentFile.mkdirs()
    StatisticalModelIO
      .writeStatisticalTriangleMeshModel3D(model, new File(modelOutputPath))
      .fold(
        err => sys.error(s"Model save failed: ${err.getMessage}"),
        _   => println("  Saved OK.")
      )

    // -------------------------------------------------------------------------
    // Step 4 — Visualize 5 deformation modes interactively
    // -------------------------------------------------------------------------
    println("=== Step 4: Opening Scalismo viewer ===")
    val ui = ScalismoUI()

    // Showing a PointDistributionModel creates one slider per mode automatically.
    // Drag slider i to ±3 to see the i-th deformation shape.
    val modelGroup = ui.createGroup(s"GP (σ=$kernelSigma s=$kernelScale rank=$gpRank)")
    ui.show(modelGroup, model, "scapula_gp")
    println(s"  ${model.rank} coefficient sliders are live in the viewer.")
    println("  Drag each slider left/right to see how that mode deforms the bone.")

    // -------------------------------------------------------------------------
    // Step 5 — Non-rigid fitting to a target mesh (GP posterior ICP)
    // -------------------------------------------------------------------------
    println("=== Step 5: Non-rigid fitting to target mesh ===")
    val tgtMesh = MeshIO.readMesh(new File(targetMeshPath))
      .getOrElse(sys.error(s"Cannot read mesh: $targetMeshPath"))
    println(s"  Target: ${tgtMesh.pointSet.numberOfPoints} vertices")

    // Noise model — isotropic Gaussian, controls how tightly we trust each correspondence
    val noiseModel = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * (noiseStd * noiseStd)
    )

    // Uniformly distributed vertex ids on the reference used for correspondence queries
    val sampledIds: IndexedSeq[PointId] =
      UniformMeshSampler3D(model.reference, numIcpPoints)
        .sample()
        .map { case (pt, _) => model.reference.pointSet.findClosestPoint(pt).id }
        .distinct

    // GP-ICP: for each iteration, find closest-point correspondences on the target,
    // compute the GP posterior given those observations, take its mean as the new mesh.
    var currentMesh = model.reference
    for (iter <- 1 to numIcpIterations) {
      val observations: IndexedSeq[(PointId, Point[_3D], MultivariateNormalDistribution)] =
        sampledIds.map { id =>
          val movingPt = currentMesh.pointSet.point(id)
          val tgtPt    = tgtMesh.operations.closestPointOnSurface(movingPt).point
          (id, tgtPt, noiseModel)
        }
      // Posterior is always computed from the original prior so rank stays fixed at gpRank
      val posteriorModel = model.posterior(observations)
      currentMesh        = posteriorModel.mean
      if (iter % 5 == 0 || iter == 1)
        println(s"  Iteration $iter / $numIcpIterations")
    }

    // Surface fitting error: mean and max distance from fitted mesh to target surface
    val dists   = currentMesh.pointSet.points.map { p =>
      (p - tgtMesh.operations.closestPointOnSurface(p).point).norm
    }.toIndexedSeq
    val meanErr = dists.sum / dists.size
    val maxErr  = dists.max
    println(f"  Fitting complete.  Mean error = $meanErr%.2f mm   Max error = $maxErr%.2f mm")

    // Show fitted result and target in the already-open viewer
    val fitGroup = ui.createGroup("Fitted (Step 5)")
    ui.show(fitGroup, currentMesh, "fitted_mesh")
    val tgtGroup = ui.createGroup("Target")
    ui.show(tgtGroup, tgtMesh, "target_mesh")

    println("Done. Close the viewer window to exit.")
  }
}
