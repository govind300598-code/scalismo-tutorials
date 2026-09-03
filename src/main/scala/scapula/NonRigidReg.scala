package scapula

import breeze.linalg.DenseVector
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.common.{PointId, Vectorizer}
import scalismo.geometry.{EuclideanVector, EuclideanVector3D, Point, _3D}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

/**
 * Gaussian Process Morphable Model (GPMM) non-rigid registration.
 *
 * Kernel fixed throughout SSM1–SSM4:
 *   σ  = 30 mm  (spatial correlation range of the Gaussian kernel)
 *   scaleFactor = 10 mm  (amplitude; variance = scaleFactor²)
 *
 * The NearestNeighborInterpolator3D is used explicitly in the low-rank GP
 * discretisation step (LowRankGaussianProcess.approximateGPCholesky).
 *
 * WHAT the interpolator does:
 *   approximateGPCholesky discretises the continuous GP at the reference mesh's
 *   point set. When the posterior is queried at a point NOT in that discrete set
 *   (e.g., during ICP when the current mean has shifted), the
 *   NearestNeighborInterpolator3D returns the GP value from the nearest reference
 *   vertex. This is appropriate here because:
 *     1. The reference mesh is dense enough (~8 k vertices) that the nearest vertex
 *        is always within a few mm of any query point on the surface.
 *     2. The kernel σ = 30 mm is much larger than the typical inter-vertex spacing,
 *        so the deformation field is smooth and a nearest-neighbour interpolation
 *        introduces no perceptible artefact in the final registered shape.
 *     3. Smooth interpolation (e.g., trilinear) would require volumetric embedding
 *        that is unnecessary for a surface GP.
 *
 * If visibly non-smooth deformation appears in the PCA modes, the cause is the
 * kernel parameters or GP rank, NOT the nearest-neighbour interpolator.
 */
object NonRigidReg {

  val gpSigma: Double      = 30.0
  val gpScaleFactor: Double = 10.0

  /**
   * Build the GPMM prior (LowRankGaussianProcess) on `reference`.
   *
   * The kernel is a diagonal (isotropic) Gaussian: each spatial direction is
   * independent, with the same Gaussian covariance function.
   *
   * The NearestNeighborInterpolator3D is passed here — this is the ONE place
   * where it is used and is documented above.
   */
  def buildPrior(
    reference: TriangleMesh[_3D],
    relativeTolerance: Double = 0.01,
    maxRank: Int = 250
  )(implicit rng: Random): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {

    // Scalar Gaussian kernel scaled by scaleFactor² to get the variance
    val scalarKernel = GaussianKernel[_3D](gpSigma) * (gpScaleFactor * gpScaleFactor)
    // Diagonal matrix-valued kernel: same kernel applied to x, y, z independently
    val matKernel = DiagonalKernel(scalarKernel, 3)

    // Explicit vectorizer resolves ambiguous given instances (Short/Int vectorizers
    // that scalismo exposes for internal use can confuse inference at this call site).
    given ev: Vectorizer[EuclideanVector[_3D]] = EuclideanVector3D.vectorizer

    // Zero-mean Gaussian Process over 3D displacement fields
    val gp: GaussianProcess[_3D, EuclideanVector[_3D]] = GaussianProcess(matKernel)

    // Discretise at reference mesh vertices using NearestNeighborInterpolator3D
    // (see class-level scaladoc for why this interpolator is appropriate).
    // TriangleMesh[_3D] implements DiscreteDomain[_3D] directly; do NOT pass .pointSet.
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = relativeTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )

    println(f"  GP prior rank = ${lowRankGP.rank} (σ=$gpSigma%.0f mm, scale=$gpScaleFactor%.0f mm, tol=$relativeTolerance)")
    if (lowRankGP.rank > maxRank) {
      println(s"  WARNING: rank ${lowRankGP.rank} exceeds maxRank $maxRank; truncating")
    }
    lowRankGP
  }

  /**
   * GP-ICP non-rigid registration.
   *
   * Iterates:
   *   1. Find the closest point on `target` for a uniformly sampled subset of the
   *      current mean mesh's vertices.
   *   2. Compute the GP posterior given those (sourceId → targetPoint) observations.
   *   3. The posterior mean is the new best estimate of the registered shape.
   *
   * The prior model is reset each iteration (we always condition the ORIGINAL prior
   * on the latest correspondences) to avoid rank explosion from chained posteriors.
   *
   * Returns the registered mesh in dense correspondence with `reference`.
   */
  def register(
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D],
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    nIter: Int = 40,
    sigma2: Double = 1.0,
    numCorrespondences: Int = 500
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val model0  = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
    val targetOps = target.operations

    var currentMesh = model0.mean

    for (iter <- 0 until nIter) {
      val sampleIds: IndexedSeq[PointId] = RigidAlign.uniformIds(currentMesh, numCorrespondences)

      val observations: IndexedSeq[(PointId, Point[_3D])] = sampleIds.map { ptId =>
        val pt        = currentMesh.pointSet.point(ptId)
        val closestPt = targetOps.closestPointOnSurface(pt).point
        (ptId, closestPt)
      }

      // Condition the ORIGINAL prior on the current correspondences (no rank accumulation)
      val posterior = model0.posterior(observations, sigma2)
      currentMesh = posterior.mean

      if (iter % 10 == 9) {
        val dists = Metrics.surfaceDistances(currentMesh, target)
        println(f"    iter ${iter + 1}%3d  mean=${dists.sum / dists.length}%5.2f mm")
      }
    }

    currentMesh
  }

  /** Wrap a LowRankGP + reference mesh into a PointDistributionModel for PCA later. */
  def priorModel(
    reference: TriangleMesh[_3D],
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]]
  ): PointDistributionModel[_3D, TriangleMesh] =
    PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
}
