package scapula

import breeze.linalg.DenseVector
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

/**
 * Gaussian Process Morphable Model (GPMM) non-rigid registration.
 *
 * Kernel: single Gaussian
 *   σ  = 130 mm  (spatial correlation range — controls how far a deformation reaches)
 *   scaleFactor = 30 mm  (amplitude; variance = scaleFactor²)
 *
 * The NearestNeighborInterpolator3D is used in the low-rank GP discretisation step.
 * It returns the GP value at the nearest reference vertex when queried at an arbitrary
 * point — appropriate because the reference mesh is dense (~5–8k vertices) and the
 * kernel σ is much larger than inter-vertex spacing, so NN introduces no visible artefact.
 *
 * If visibly non-smooth deformation appears, the cause is kernel parameters or GP rank,
 * NOT the nearest-neighbour interpolator.
 */
object NonRigidReg {

  // Single Gaussian kernel parameters (user spec: sigma=130, scale=30, iters=10)
  val gpSigma: Double       = Config.gpSigma
  val gpScaleFactor: Double = Config.gpScale

  /**
   * Build the GPMM prior (LowRankGaussianProcess) on `reference`.
   *
   * kernel: DiagonalKernel(GaussianKernel(σ) × scale², 3)
   * The Nystrom/Cholesky approximation is built on the ORIGINAL-resolution reference
   * for accuracy; it is then evaluated at the decimated mesh vertices via the NN interpolator.
   */
  def buildPrior(
    reference: TriangleMesh[_3D],
    relativeTolerance: Double = 0.01,
    maxRank: Int = 250
  )(implicit rng: Random): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {

    val scalarKernel = GaussianKernel[_3D](gpSigma) * (gpScaleFactor * gpScaleFactor)
    val matKernel    = DiagonalKernel(scalarKernel, 3)
    val gp           = GaussianProcess(matKernel)

    // Discretise at reference vertices using NearestNeighborInterpolator3D.
    // Pass the TriangleMesh directly (it implements DiscreteDomain[_3D]).
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = relativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )

    println(f"  GP prior rank = ${lowRankGP.rank} (σ=$gpSigma%.0f mm, scale=$gpScaleFactor%.0f mm, tol=$relativeTolerance)")
    if (lowRankGP.rank > maxRank)
      println(s"  WARNING: rank ${lowRankGP.rank} exceeds maxRank $maxRank; consider raising SCAPULA_GP_MAX_RANK")
    lowRankGP
  }

  /**
   * GP-ICP non-rigid registration of `target` onto `reference`.
   *
   * Each iteration:
   *   1. Uniformly sample `numCorrespondences` vertices from the current mean mesh.
   *   2. Find closest point on `target` surface for each sampled vertex.
   *   3. Compute GP posterior given those (sourceId → targetPoint) observations.
   *   4. The posterior mean is the updated estimate.
   *
   * The ORIGINAL prior model is conditioned fresh each iteration (no rank accumulation).
   * Returns the registered mesh in dense correspondence with `reference`.
   */
  def register(
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D],
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    nIter: Int = 10,
    sigma2: Double = 1.0,
    numCorrespondences: Int = 500
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val model0    = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
    val targetOps = target.operations
    var currentMesh = model0.mean

    for (iter <- 0 until nIter) {
      val sampleIds: IndexedSeq[PointId] = RigidAlign.uniformIds(currentMesh, numCorrespondences)

      val observations: IndexedSeq[(PointId, Point[_3D])] = sampleIds.map { ptId =>
        val pt        = currentMesh.pointSet.point(ptId)
        val closestPt = targetOps.closestPointOnSurface(pt).point
        (ptId, closestPt)
      }

      val posterior = model0.posterior(observations, sigma2)
      currentMesh   = posterior.mean

      if (iter % 5 == 4) {
        val dists = Metrics.surfaceDistances(currentMesh, target)
        println(f"    iter ${iter + 1}%3d  mean=${dists.sum / dists.length}%5.2f mm")
      }
    }

    currentMesh
  }

  /** Compute the vertex-wise mean across a set of registered meshes (all must share the same topology). */
  def meanMesh(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    require(meshes.nonEmpty)
    val n        = meshes.head.pointSet.numberOfPoints
    val sumCoords = Array.fill(n)((0.0, 0.0, 0.0))
    meshes.foreach { m =>
      m.pointSet.pointIds.foreach { id =>
        val p = m.pointSet.point(id)
        val (sx, sy, sz) = sumCoords(id.id)
        sumCoords(id.id) = (sx + p.x, sy + p.y, sz + p.z)
      }
    }
    val k = meshes.length.toDouble
    import scalismo.geometry.Point3D
    val avgPts = sumCoords.map { case (x, y, z) => Point3D(x / k, y / k, z / k) }.toIndexedSeq
    scalismo.mesh.TriangleMesh3D(avgPts, meshes.head.triangulation)
  }
}
