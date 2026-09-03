package ssmpipeline

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.*
import scalismo.mesh.*
import scalismo.statisticalmodel.*
import scalismo.utils.Random

/**
 * Non-rigid (GP-ICP) registration.
 *
 * All registered meshes share the topology and point ids of the reference, giving an explicit, consistent
 * point-to-point correspondence across the whole dataset. That correspondence is the prerequisite for GPA and PCA.
 */
object NonRigidReg {

  /**
   * Multi-scale Gaussian kernel prior.
   *
   * Three length scales: 75 mm captures global bend/twist residuals that rigid alignment leaves behind; 25 mm handles
   * regional shape differences (e.g. fossa depth); 8 mm resolves fine surface detail such as the coracoid tip.
   * The amplitude ratios are tuned so the prior is loose enough to match any scapula shape but tight enough to prevent
   * unphysical deformations in regions with few ICP correspondences.
   */
  def buildGpModel(reference: TriangleMesh[_3D]): PointDistributionModel[_3D, TriangleMesh] = {
    val k = DiagonalKernel[_3D](
      GaussianKernel[_3D](75.0) * 150.0
        + GaussianKernel[_3D](25.0) * 40.0
        + GaussianKernel[_3D](8.0) * 8.0,
      3
    )
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](k)
    val lowRankGp = LowRankGaussianProcess.approximateGPCholesky(
      reference.pointSet,
      gp,
      Config.gpRelativeTolerance,
      interpolator = TriangleMeshInterpolator3D()
    )
    PointDistributionModel[_3D, TriangleMesh](reference, lowRankGp)
  }

  /**
   * GP-ICP: find surface correspondences between the current model mean and `target`, condition the GP posterior on
   * those correspondences, and repeat. The posterior mean of the final iteration is returned as the registered mesh --
   * a copy of the reference topology deformed to match the target.
   *
   * `noiseSigma` sets how rigidly the correspondences are enforced; 1 mm is appropriate when the surface is
   * well-sampled and both meshes are clean.  Increase it (2-3 mm) for noisy or poorly triangulated surfaces.
   */
  def register(model: PointDistributionModel[_3D, TriangleMesh],
               target: TriangleMesh[_3D],
               iterations: Int,
               numPoints: Int = 2000,
               noiseSigma: Double = 1.0
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val targetOps = target.operations
    val noiseCov  = DenseMatrix.eye[Double](3) * (noiseSigma * noiseSigma)
    var currentModel = model

    for (_ <- 0 until iterations) {
      val mean = currentModel.mean
      val ids  = RigidAlign.uniformIds(mean, numPoints)
      val obs  = ids.map { id =>
        val pt      = mean.pointSet.point(id)
        val closest = targetOps.closestPointOnSurface(pt).point
        (id, closest, MultivariateNormalDistribution(DenseVector.zeros[Double](3), noiseCov))
      }
      currentModel = currentModel.posterior(obs)
    }
    currentModel.mean
  }
}
