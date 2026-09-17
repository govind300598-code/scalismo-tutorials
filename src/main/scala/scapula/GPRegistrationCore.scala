package scapula

import breeze.linalg.DenseVector
import scalismo.common.{EuclideanSpace3D, Field}
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, LBFGSOptimizer}
import scalismo.registration.{GaussianProcessTransformationSpace, L2Regularizer, MeanSquaresMetric, Registration}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess}
import scalismo.utils.Random

/**
 * STAGE 2 -- NON-RIGID (GP) REGISTRATION.
 *
 * This is the algorithm from the scalismo mailing-list thread "parametric, non-rigid registration" (Loane Le Gall /
 * Dennis Madsen, June 2024), kept as close to the original as a batch pipeline over a whole population allows:
 *   - the deformation prior is a low-rank Gaussian process over the reference, built from a sum of Gaussian kernels
 *     (`doRegistration`'s transformation space, metric, regularizer and optimizer are the same four objects, wired
 *     together the same way, as in the original `doRegistration` function);
 *   - fitting is a cascade of `RegistrationParameters` with decreasing regularization weight and increasing sampled
 *     points, each stage warm-started from the previous stage's coefficients via `foldLeft` -- exactly the pattern
 *     in both the original poster's code and Dennis Madsen's reply.
 *
 * Two changes from the literal mailing-list code, both requested by Dennis Madsen in the same thread:
 *   (1) kernel length scales are derived from the reference's own bounding-box diagonal instead of hardcoded
 *       millimetre constants, so the same three-scale design applies regardless of specimen size;
 *   (2) `NearestNeighborInterpolator3D` is used for the low-rank approximation instead of the triangle-mesh
 *       interpolator, per Dennis Madsen's "I find ... NearestNeighbourInterpolator worked better for me".
 * The interactive UI callback (`gpView.coefficients = ...`) is dropped because this runs headlessly over 24
 * specimens x multiple refinement passes; nothing about the optimization itself is changed.
 */
object GPRegistrationCore {

  final case class RegistrationParameters(regularizationWeight: Double, numberOfIterations: Int, numberOfSampledPoints: Int)

  /** The four-stage cascade from Config, scaled to how many points the reference actually has. */
  def defaultCascade(reference: TriangleMesh[_3D]): Seq[RegistrationParameters] = {
    val n = reference.pointSet.numberOfPoints
    require(
      Config.registrationSampleFractions.length == Config.registrationRegWeights.length &&
        Config.registrationRegWeights.length == Config.registrationIterations.length,
      "SCAPULA_REG_SAMPLE_FRACTIONS / SCAPULA_REG_WEIGHTS / SCAPULA_REG_ITERS must have the same length"
    )
    Config.registrationSampleFractions.lazyZip(Config.registrationRegWeights).lazyZip(Config.registrationIterations).map {
      case (frac, weight, iters) =>
        RegistrationParameters(
          regularizationWeight = weight,
          numberOfIterations = iters,
          numberOfSampledPoints = math.max(50, math.min(n, math.round(frac * n).toInt))
        )
    }
  }

  /**
   * Sum of four Gaussian kernels (coarse blade / mid body / fine glenoid detail / ultra-fine glenoid rim & coracoid),
   * diagonal over the 3 output dimensions, matching `kernelCoarse + kernelFine` (extended to four scales) and
   * `DiagonalKernel3D(..., outputDim = 3)` from the original code. The 4th scale directly targets the small
   * structures Loane Le Gall's original mailing-list post wanted more precision on.
   */
  def buildMultiscaleGP(reference: TriangleMesh[_3D])(implicit rng: Random): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    val points = reference.pointSet.points.toIndexedSeq
    require(points.nonEmpty, "reference mesh has no points")
    val (minP, maxP) = points.foldLeft((points.head, points.head)) { case ((mn, mx), p) =>
      (Point3D(math.min(mn.x, p.x), math.min(mn.y, p.y), math.min(mn.z, p.z)),
       Point3D(math.max(mx.x, p.x), math.max(mx.y, p.y), math.max(mx.z, p.z)))
    }
    val dx = maxP.x - minP.x
    val dy = maxP.y - minP.y
    val dz = maxP.z - minP.z
    val diag = math.sqrt(dx * dx + dy * dy + dz * dz)

    val kernelCoarse = GaussianKernel3D(diag / Config.kernelCoarseDivisor, Config.kernelCoarseScale)
    val kernelMid = GaussianKernel3D(diag / Config.kernelMidDivisor, Config.kernelMidScale)
    val kernelFine = GaussianKernel3D(diag / Config.kernelFineDivisor, Config.kernelFineScale)
    val kernelUltraFine = GaussianKernel3D(diag / Config.kernelUltraFineDivisor, Config.kernelUltraFineScale)
    val kernelFinal = kernelCoarse + kernelMid + kernelFine + kernelUltraFine
    val kernel = DiagonalKernel3D(kernelFinal, outputDim = 3)

    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp = GaussianProcess(zeroMean, kernel)

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )
    if (lowRankGP.rank > Config.gpMaxRank) lowRankGP.truncate(Config.gpMaxRank) else lowRankGP
  }

  /**
   * One `doRegistration` call from the original code, generalized to fold over an arbitrary cascade. Identical
   * wiring: `GaussianProcessTransformationSpace` on the GP prior, `MeanSquaresMetric` on the two distance images,
   * `FixedPointsUniformMeshSampler3D` on the reference, `LBFGSOptimizer`, `L2Regularizer`, `Registration`, take the
   * last iterate's parameters, warm-start the next stage with them.
   */
  def registerToTarget(
      lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
      referenceMesh: TriangleMesh[_3D],
      target: TriangleMesh[_3D],
      initialCoefficients: DenseVector[Double],
      cascade: Seq[RegistrationParameters]
  )(implicit rng: Random): DenseVector[Double] = {
    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
    val fixedImage = referenceMesh.operations.toDistanceImage
    val movingImage = target.operations.toDistanceImage

    def doRegistration(coefficients: DenseVector[Double], params: RegistrationParameters): DenseVector[Double] = {
      val sampler = FixedPointsUniformMeshSampler3D(referenceMesh, numberOfPoints = params.numberOfSampledPoints)
      val metric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
      val optimizer = LBFGSOptimizer(maxNumberOfIterations = params.numberOfIterations)
      val regularizer = L2Regularizer(transformationSpace)
      val registration = Registration(metric, regularizer, regularizationWeight = params.regularizationWeight, optimizer)
      registration.iterator(coefficients).toSeq.last.parameters
    }

    cascade.foldLeft(initialCoefficients)(doRegistration)
  }

  /** Warp the reference by the fitted coefficients -- the actual registered mesh, in exact reference correspondence. */
  def warpedMesh(
      lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
      referenceMesh: TriangleMesh[_3D],
      coefficients: DenseVector[Double]
  ): TriangleMesh[_3D] = {
    val transformationSpace = GaussianProcessTransformationSpace(lowRankGP)
    referenceMesh.transform(transformationSpace.transformationForParameters(coefficients))
  }
}
