package scapula.singlekernel

import scapula.Config
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.{EuclideanVector, _3D}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess3D, LowRankGaussianProcess, PointDistributionModel, PointDistributionModel3D}

/**
 * GPMM prior built from exactly ONE isotropic Gaussian kernel term, `GaussianKernel3D(sigmaMm, scaleMm)` -- the
 * single-kernel counterpart of `scapula.GpmmFitting.buildGpmm` (which sums 2 or 3 such terms at different
 * length scales). Everything downstream of the kernel -- coarse-to-fine LBFGS registration, landmark-informed
 * fitting, low-rank truncation -- is identical and reused unchanged from `scapula.GpmmFitting`.
 */
object SingleGaussianGpmm {

  /**
   * `sigmaMm`: correlation length of the kernel, in mm -- how far a deformation at one point "pulls" nearby
   * points. `scaleMm`: amplitude of the kernel, in mm -- the standard deviation of the deformation magnitude
   * (variance = scaleMm^2). Both are absolute physical quantities here (unlike `GpmmFitting.buildGpmm`, which
   * derives them as ratios of the reference mesh's own bounding box), so that a kernel-selection sweep over
   * candidate (sigma, s) pairs is directly comparable across runs and across datasets.
   */
  def buildGpmm(referenceMesh: TriangleMesh[_3D],
               sigmaMm: Double = SingleKernelConfig.sigmaMm,
               scaleMm: Double = SingleKernelConfig.scaleMm,
               relativeTolerance: Double = Config.gpRelativeTolerance,
               maxRank: Int = Config.gpMaxRank
  ): (LowRankGaussianProcess[_3D, EuclideanVector[_3D]], PointDistributionModel[_3D, TriangleMesh]) = {
    require(sigmaMm > 0, s"sigmaMm must be positive, got $sigmaMm")
    require(scaleMm > 0, s"scaleMm must be positive, got $scaleMm")

    val box = referenceMesh.pointSet.boundingBox
    val extent = box.oppositeCorner - box.origin
    println(f"  [SingleGaussianGpmm] reference bounding box: ${extent.x}%.1f x ${extent.y}%.1f x ${extent.z}%.1f mm")
    println(f"  [SingleGaussianGpmm] kernel: single Gaussian term, sigma=$sigmaMm%.1f mm, s=$scaleMm%.1f mm")

    val kernel = DiagonalKernel3D(GaussianKernel3D(sigmaMm, scaleMm), outputDim = 3)
    val gp = GaussianProcess3D[EuclideanVector[_3D]](kernel)
    val fullRankGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh,
      gp,
      relativeTolerance = relativeTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )
    val lowRankGP = if (fullRankGP.rank > maxRank) fullRankGP.truncate(maxRank) else fullRankGP
    (lowRankGP, PointDistributionModel3D(referenceMesh, lowRankGP))
  }
}
