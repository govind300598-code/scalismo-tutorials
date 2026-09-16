package scapula

import scalismo.common.{Field, RealSpace}
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, LBFGSOptimizer}
import scalismo.registration.{GaussianProcessTransformationSpace, L2Regularizer, MeanSquaresMetric, Registration}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random
import breeze.linalg.DenseVector

/**
 * Non-rigid registration following Dennis Madsen's approach (Scalismo forum, Jun 2024):
 *   - Multi-scale Gaussian kernel (coarse + mid + fine), sized to the scapula (~150 mm)
 *   - Cholesky low-rank GP approximation with NearestNeighborInterpolator3D
 *   - LBFGS optimizer on a MeanSquares distance-image metric
 *   - Multi-stage coarse-to-fine regularisation schedule
 */
object NonRigidRegLBFGS {

  case class RegistrationParameters(
    regularizationWeight: Double,
    numberOfIterations: Int,
    numberOfSampledPoints: Int
  )

  /**
   * Coarse-to-fine LBFGS schedule.
   * Stage 1: strong regularisation, few points — fast global pose
   * Stage 2: medium regularisation, more points — regional shape
   * Stage 3: light regularisation, more points — fine local details
   * Stage 4: very light regularisation, many points — sub-mm surface detail
   */
  val defaultSchedule: Seq[RegistrationParameters] = Seq(
    RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 50,  numberOfSampledPoints = 500),
    RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 50,  numberOfSampledPoints = 1000),
    RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 100, numberOfSampledPoints = 2000),
    RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 100, numberOfSampledPoints = 5000)
  )

  /**
   * Build the low-rank GP using a multi-scale Gaussian kernel.
   *
   * Sigma values are chosen relative to the scapula's longest dimension (~150 mm):
   *   coarse  σ = 75 mm  — global/whole-bone deformations
   *   mid     σ = 30 mm  — regional shape (glenoid, acromion)
   *   fine    σ = 15 mm  — local surface detail
   *
   * Scale (amplitude) values follow Dennis Madsen's recommendation: 15, 10, 5.
   */
  def buildLowRankGP(
    reference: TriangleMesh[_3D]
  )(implicit rng: Random): (LowRankGaussianProcess[_3D, EuclideanVector[_3D]], PointDistributionModel[_3D, TriangleMesh]) = {

    val kernelCoarse = GaussianKernel[_3D](75.0) * 15.0
    val kernelMid    = GaussianKernel[_3D](30.0) * 10.0
    val kernelFine   = GaussianKernel[_3D](15.0) *  5.0
    val kernelTotal  = kernelCoarse + kernelMid + kernelFine
    val kernel       = DiagonalKernel3D(kernelTotal, 3)

    val zeroMean = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp       = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = 0.01,
      interpolator = NearestNeighborInterpolator3D[EuclideanVector[_3D]]()
    )

    val gpmm = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
    (lowRankGP, gpmm)
  }

  /**
   * Run one LBFGS stage.
   */
  private def doRegistration(
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    referenceMesh: TriangleMesh[_3D],
    targetMesh: TriangleMesh[_3D],
    initialCoefficients: DenseVector[Double],
    params: RegistrationParameters
  ): DenseVector[Double] = {

    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
    val fixedImage  = referenceMesh.operations.toDistanceImage
    val movingImage = targetMesh.operations.toDistanceImage
    val sampler     = FixedPointsUniformMeshSampler3D(referenceMesh, params.numberOfSampledPoints)
    val metric      = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
    val optimizer   = LBFGSOptimizer(maxNumberOfIterations = params.numberOfIterations)
    val regularizer = L2Regularizer(transformationSpace)
    val registration = Registration(metric, regularizer, params.regularizationWeight, optimizer)

    registration.iterator(initialCoefficients).toSeq.last.parameters
  }

  /**
   * Register `target` to `reference` using the coarse-to-fine LBFGS schedule.
   * Returns the fitted mesh (same topology as `reference`).
   */
  def register(
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    gpmm: PointDistributionModel[_3D, TriangleMesh],
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D],
    schedule: Seq[RegistrationParameters] = defaultSchedule
  ): TriangleMesh[_3D] = {

    val initialCoefficients = DenseVector.zeros[Double](lowRankGP.rank)
    val finalCoefficients = schedule.foldLeft(initialCoefficients) { (coeffs, params) =>
      doRegistration(lowRankGP, reference, target, coeffs, params)
    }
    gpmm.instance(finalCoefficients)
  }
}
