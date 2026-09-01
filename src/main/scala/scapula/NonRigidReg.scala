package scapula

import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.common.{Field, PointId}
import scalismo.geometry.*
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}

/**
 * GP-based non-rigid registration.
 *
 * A multi-scale Gaussian kernel is approximated on the reference mesh via pivoted Cholesky.
 * GP-ICP then iterates: find closest-surface correspondences → compute posterior mean → repeat.
 * The result is a deformed copy of the reference that matches the target in shape while
 * preserving the reference topology and point count exactly.
 */
object NonRigidReg {

  /**
   * Sigma/scale pairs for a multi-scale Gaussian kernel suitable for scapula-sized bones (units: mm).
   * Large sigmas handle global shape; small sigmas refine local details.
   */
  val defaultScales: Seq[(Double, Double)] = Seq(
    (80.0, 10.0),
    (40.0,  5.0),
    (20.0,  2.0),
    (10.0,  1.0)
  )

  /** Build a low-rank GP prior over the reference mesh vertex set. */
  def buildGpPrior(
    reference : TriangleMesh[_3D],
    scales    : Seq[(Double, Double)] = defaultScales
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {

    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])

    // Build as a collection typed to the supertype so `+` resolves correctly
    val kernelList: Seq[scalismo.kernels.MatrixValuedPDKernel[_3D]] =
      scales.map { case (sigma, s) => DiagonalKernel[_3D](GaussianKernel[_3D](sigma) * s, 3) }
    val kernel = kernelList.reduce(_ + _)

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)

    LowRankGaussianProcess.approximateGPCholesky(
      reference.pointSet,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D[EuclideanVector[_3D]]()
    )
  }

  /**
   * GP-ICP (non-rigid iterative closest point).
   *
   * Each iteration:
   *   1. For every vertex of the current deformed reference, find the closest surface point on the target.
   *   2. Reject correspondences whose distance exceeds `outlierThreshMm` (thin structures on one mesh only).
   *   3. Compute the posterior GP mean conditioned on those correspondences.
   *   4. The posterior mean is the next current mesh.
   *
   * Returns the registered mesh: same topology and vertex count as `reference`, fitted to `target`.
   */
  def gpIcp(
    reference       : TriangleMesh[_3D],
    target          : TriangleMesh[_3D],
    lowRankGP       : LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    iterations      : Int,
    sigma2          : Double = 2.0,
    outlierThreshMm : Double = 25.0
  ): TriangleMesh[_3D] = {

    val pdm = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
    var currentMesh = pdm.mean

    for (i <- 0 until iterations) {
      val correspondences =
        currentMesh.pointSet.points.toIndexedSeq.zipWithIndex.flatMap { case (pt, idx) =>
          val cp   = target.operations.closestPointOnSurface(pt).point
          val dist = (pt - cp).norm
          if (dist < outlierThreshMm) Some(PointId(idx) -> cp) else None
        }

      if (correspondences.size >= 50) {
        val posterior = pdm.posterior(correspondences, sigma2)
        currentMesh = posterior.mean
      }

      if ((i + 1) % 10 == 0)
        println(f"      GP-ICP iter ${i + 1}%3d/$iterations  correspondences=${correspondences.size}")
    }

    currentMesh
  }
}
