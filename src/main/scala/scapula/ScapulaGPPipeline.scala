package scapula

import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

/** Reusable GP-ICP non-rigid registration pipeline for a single target. */
object ScapulaGPPipeline {

  /** Default GP prior: isotropic Gaussian kernel, σ = 75 mm, amplitude = 12 mm, rank capped via Cholesky. */
  def defaultGP: GaussianProcess[_3D, EuclideanVector[_3D]] =
    GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](75.0) * 12.0, 3))

  /**
   * Register alignedRef to target using GP-ICP.
   *
   * alignedRef must already be rigidly aligned to target before calling this.
   *
   * Type note — the first arg to approximateGPCholesky must be a DiscreteDomain[_3D]:
   *   TriangleMesh[_3D]       implements DiscreteDomain[_3D]  → pass alignedRef ✓
   *   alignedRef.pointSet     returns UnstructuredPoints[_3D]
   *                           which does NOT implement DiscreteDomain in Scalismo 0.92 → E007 ✗
   *
   * @return (posterior model, model.mean — the registered surface)
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

    // approximateGPCholesky[D, DDomain[D] <: DiscreteDomain[D], Value]
    // NearestNeighborInterpolator3D is FieldInterpolator[_3D, DiscreteDomain, EuclideanVector[_3D]],
    // contravariant-compatible with FieldInterpolator[_3D, TriangleMesh, EuclideanVector[_3D]].
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      alignedRef,
      gp,
      relativeTolerance = gpTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )

    val step  = math.max(1, alignedRef.pointSet.numberOfPoints / 2500)
    val ptIds = (0 until alignedRef.pointSet.numberOfPoints by step).map(PointId(_))

    var model = PointDistributionModel[_3D, TriangleMesh](alignedRef, lowRankGP)

    for (_ <- 0 until iterations) {
      val mean = model.mean
      val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
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
