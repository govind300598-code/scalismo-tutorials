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
   * Coarse + mid + fine Gaussian kernel, sized to the ACTUAL reference mesh instead of a hard-coded constant.
   *
   * `sigma` (correlation length, mm) and `scaleFactor` (amplitude, mm -- variance is scaleFactor^2) are both
   * lengths in the mesh's own units. Dennis Madsen's original mailing-list recipe picked them as fractions of
   * HIS mesh's ~150mm longest side (sigma = size/2, size/5, size/10; amplitude = 15, 10, 5, i.e. 10%, 6.7%, 3.3%
   * of that size) -- but hard-coded the 150mm rather than measuring it. If `longestSideMm` is left unset, this
   * now measures the actual `referenceMesh` bounding box and keeps those same ratios, so the kernel's scale
   * always matches whatever mesh it's actually being built on.
   *
   * The RATIOS themselves are still someone's manual judgment call, not derived from data -- how much a real
   * population of scapulae varies by, at each spatial scale, is exactly what building an empirical (PCA-based)
   * model from the registered population would tell you instead of guessing.
   */
  def buildGpmm(referenceMesh: TriangleMesh[_3D],
               longestSideMm: Option[Double] = None,
               relativeTolerance: Double = Config.gpRelativeTolerance,
               maxRank: Int = Config.gpMaxRank
  ): (LowRankGaussianProcess[_3D, EuclideanVector[_3D]], PointDistributionModel[_3D, TriangleMesh]) = {
    val box = referenceMesh.pointSet.boundingBox
    val extent = box.oppositeCorner - box.origin
    val measuredLongestSide = Seq(extent.x, extent.y, extent.z).max
    val size = longestSideMm.getOrElse(measuredLongestSide)
    println(f"  [GpmmFitting] reference bounding box: ${extent.x}%.1f x ${extent.y}%.1f x ${extent.z}%.1f mm " +
      f"(longest axis ${measuredLongestSide}%.1f mm)${if (longestSideMm.isDefined) f" -- OVERRIDDEN to ${size}%.1f mm" else ""}")

    val kernelCoarse = GaussianKernel3D(size / 2, size * 0.10)
    val kernelMid = GaussianKernel3D(size / 5, size * 0.0667)
    val kernelFine = GaussianKernel3D(size / 10, size * 0.0333)
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
