package scapula

import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, LBFGSOptimizer}
import scalismo.registration.{GaussianProcessTransformationSpace, L2Regularizer, MeanSquaresMetric, Registration, RegistrationMetric}
import scalismo.statisticalmodel.{GaussianProcess3D, LowRankGaussianProcess, PointDistributionModel, PointDistributionModel3D}
import scalismo.utils.Random as ScalismoRandom

import breeze.linalg.DenseVector

/**
 * GPMM construction and the coarse-to-fine registration loop, shared by NonRigidFittingExample (fits to a
 * synthetic GPMM sample, following Dennis Madsen's mailing-list example) and Stage2ReferenceRefinement (fits to
 * the real scapula dataset). Kept in one place so both build/fit the model exactly the same way.
 */
object GpmmFitting {

  case class RegistrationParameters(regularizationWeight: Double, numberOfIterations: Int, numberOfSampledPoints: Int)

  /** Coarse-to-fine schedule from Dennis Madsen's mailing-list example. */
  val defaultSchedule: Seq[RegistrationParameters] = Seq(
    RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 50, numberOfSampledPoints = 100),
    RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 50, numberOfSampledPoints = 500),
    RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 50, numberOfSampledPoints = 100),
    RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 50, numberOfSampledPoints = 5000)
  )

  /**
   * Coarse + mid + fine Gaussian kernel (same recipe as the mailing-list example), approximated to a low-rank GP
   * and capped at `maxRank` (via truncation) so a fine kernel on a dense reference cannot blow up the model size.
   */
  def buildGpmm(referenceMesh: TriangleMesh[_3D],
               longestSideMm: Double = 150,
               relativeTolerance: Double = Config.gpRelativeTolerance,
               maxRank: Int = Config.gpMaxRank
  ): (LowRankGaussianProcess[_3D, EuclideanVector[_3D]], PointDistributionModel[_3D, TriangleMesh]) = {
    val kernelCoarse = GaussianKernel3D(longestSideMm / 2, 15)
    val kernelMid = GaussianKernel3D(longestSideMm / 5, 10)
    val kernelFine = GaussianKernel3D(longestSideMm / 10, 5)
    val kernel = DiagonalKernel3D(kernelCoarse + kernelMid + kernelFine, outputDim = 3)
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

  /**
   * `landmarkCorrespondences`: optional (referencePoint, targetPoint) pairs -- when non-empty, the surface
   * (mean-squares distance-image) term is joined with a LandmarkMetric term, weighted by `landmarkWeight`, so the
   * non-rigid fit is pulled by the same named anatomical landmarks used for rigid pre-alignment, not just the
   * anonymous closest-surface-point data term.
   */
  def doRegistration(lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
                     referenceMesh: TriangleMesh[_3D],
                     targetMesh: TriangleMesh[_3D],
                     initialCoefficients: DenseVector[Double],
                     registrationParameters: RegistrationParameters,
                     onIteration: DenseVector[Double] => Unit = _ => (),
                     landmarkCorrespondences: IndexedSeq[(Point[_3D], Point[_3D])] = IndexedSeq.empty,
                     landmarkWeight: Double = Config.landmarkWeight
  )(implicit rng: ScalismoRandom): DenseVector[Double] = {
    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
    val fixedImage = referenceMesh.operations.toDistanceImage
    val movingImage = targetMesh.operations.toDistanceImage
    val sampler = FixedPointsUniformMeshSampler3D(referenceMesh, registrationParameters.numberOfSampledPoints)
    val surfaceMetric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
    val metric: RegistrationMetric[_3D] =
      if (landmarkCorrespondences.isEmpty) surfaceMetric
      else SumMetric(Seq(surfaceMetric -> 1.0, LandmarkMetric(landmarkCorrespondences, transformationSpace) -> landmarkWeight))
    val optimizer = LBFGSOptimizer(registrationParameters.numberOfIterations)
    val regularizer = L2Regularizer(transformationSpace)
    val registration = Registration(metric, regularizer, registrationParameters.regularizationWeight, optimizer)
    val registrationIterator = registration.iterator(initialCoefficients)
    val visualizingRegistrationIterator = for (it <- registrationIterator) yield {
      onIteration(it.parameters)
      it
    }
    visualizingRegistrationIterator.toSeq.last.parameters
  }

  /** Runs the full coarse-to-fine `schedule` starting from zero coefficients. */
  def fit(lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
         referenceMesh: TriangleMesh[_3D],
         targetMesh: TriangleMesh[_3D],
         schedule: Seq[RegistrationParameters] = defaultSchedule,
         onIteration: DenseVector[Double] => Unit = _ => (),
         landmarkCorrespondences: IndexedSeq[(Point[_3D], Point[_3D])] = IndexedSeq.empty,
         landmarkWeight: Double = Config.landmarkWeight
  )(implicit rng: ScalismoRandom): DenseVector[Double] = {
    val initial = DenseVector.zeros[Double](lowRankGP.rank)
    schedule.foldLeft(initial) { (coefficients, params) =>
      println(s"    $params")
      doRegistration(lowRankGP, referenceMesh, targetMesh, coefficients, params, onIteration,
        landmarkCorrespondences, landmarkWeight)
    }
  }
}
