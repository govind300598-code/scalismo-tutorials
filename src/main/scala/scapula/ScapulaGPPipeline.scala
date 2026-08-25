package scapula

import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

object ScapulaGPPipeline {

  def defaultGP: GaussianProcess[_3D, EuclideanVector[_3D]] =
    GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](75.0) * 12.0, 3))

  /**
   * GP-ICP non-rigid registration.
   *
   * alignedRef must already be rigidly aligned to the target before calling this.
   *
   * The first argument to approximateGPCholesky must be a DiscreteDomain[_3D].
   * TriangleMesh[_3D] satisfies that — pass alignedRef directly.
   * alignedRef.pointSet returns UnstructuredPoints[_3D], which is NOT a DiscreteDomain
   * and causes a compile-time type-mismatch error.
   */
  def register(
    alignedRef: TriangleMesh[_3D],
    target: TriangleMesh[_3D],
    gp: GaussianProcess[_3D, EuclideanVector[_3D]] = defaultGP,
    gpTolerance: Double = 0.005,
    iterations: Int = 10,
    maxCorrDist: Double = 15.0,
    sigma2: Double = 1.0
  )(implicit rng: Random): (PointDistributionModel[_3D, TriangleMesh], TriangleMesh[_3D]) = {

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      alignedRef,         // TriangleMesh[_3D] is a DiscreteDomain[_3D] — correct
      gp,
      relativeTolerance = gpTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )

    val step  = math.max(1, alignedRef.pointSet.numberOfPoints / 2500)
    val ptIds = (0 until alignedRef.pointSet.numberOfPoints by step).map(PointId(_))

    var model = PointDistributionModel[_3D, TriangleMesh](alignedRef, lowRankGP)

    for (_ <- 0 until iterations) {
      val mean = model.mean
      val correspondences = ptIds.flatMap { pid =>
        val pt      = mean.pointSet.point(pid)
        val nearest = target.operations.closestPointOnSurface(pt).point
        if ((nearest - pt).norm < maxCorrDist) Some((pid, nearest)) else None
      }
      if (correspondences.nonEmpty)
        model = model.posterior(correspondences, sigma2 = sigma2)
    }

    (model, model.mean)
  }
}
