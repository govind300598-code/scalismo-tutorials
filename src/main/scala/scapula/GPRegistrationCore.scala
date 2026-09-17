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

  /**
   * Iterative closest-point Gaussian process regression (ICP-GPR) registration -- an alternative to
   * [[registerToTarget]]'s single continuous LBFGS fit, for the case that fit structurally struggles with: a
   * target with one large, spatially localized deformation from the reference (as opposed to many small ones
   * spread across the whole surface). Root cause: MeanSquaresMetric averages residuals over ALL sampled points,
   * so a small badly-fit region contributes little to the gradient and the optimizer settles for a smoothed-out
   * partial fit -- confirmed empirically (see SyntheticParamExperiment): six different settings of the existing
   * cascade (more iterations, less regularization, larger kernel amplitude, higher GP rank) left the worst-case
   * error essentially unchanged on a synthetic "one big local bump" test case.
   *
   * This instead does what ICP does for RIGID alignment, but non-rigidly: at each iteration, find the closest
   * point on the target surface for a sample of reference points (a direct, per-point pull toward wherever the
   * true surface actually is -- no averaging-away of a small badly-fit region), then take the closed-form
   * POSTERIOR of the ORIGINAL prior GP given those correspondences as noisy observations
   * (`LowRankGaussianProcess.posterior`, exact Gaussian process regression, no gradient descent / no local-minimum
   * risk from an optimizer trajectory). The observation noise (`sigma2Schedule`) is annealed from loose to tight
   * across iterations, exactly as in Amberg et al.'s "Optimal Step Non-Rigid ICP" and the "ICP-GPR" instance of
   * GiNGR (Madsen et al. 2022, the same author's own successor to the mailing-list `doRegistration` approach this
   * pipeline otherwise replicates) -- early iterations trust the (likely wrong) correspondences loosely, later
   * iterations trust the (by-then-refined) correspondences tightly.
   *
   * The posterior is always computed from the ORIGINAL `priorGP`, never a posterior-of-a-posterior, which is both
   * simpler and numerically safer. `warpedMesh(posteriorGP, referenceMesh, zeros)` extracts each iteration's
   * posterior MEAN shape (deterministic part, no sampling) to warp forward into the next iteration.
   */
  def icpGprRegister(
      priorGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
      referenceMesh: TriangleMesh[_3D],
      target: TriangleMesh[_3D],
      iterations: Int,
      sigma2Schedule: IndexedSeq[Double],
      numPoints: Int,
      trimFraction: Double
  )(implicit rng: Random): TriangleMesh[_3D] = {
    require(iterations > 0, "icpGprRegister needs at least one iteration")
    val sampleIds = FixedPointsUniformMeshSampler3D(referenceMesh, numPoints)
      .sample()
      .map { case (pt, _) => referenceMesh.pointSet.findClosestPoint(pt).id }
      .distinct

    var currentMesh = referenceMesh
    for (iter <- 0 until iterations) {
      val sigma2 = sigma2Schedule(math.min(iter, sigma2Schedule.length - 1))
      val targetOps = target.operations

      val pairs = sampleIds.map { id =>
        val refPt = referenceMesh.pointSet.point(id)
        val curPt = currentMesh.pointSet.point(id)
        val closest = targetOps.closestPointOnSurface(curPt).point
        (refPt, curPt, closest)
      }
      // Trim the worst correspondences before regressing, same rationale as RigidAlign.rigidIcp: a handful of
      // bad matches (e.g. a point currently far from any true correspondence) would otherwise pull the whole
      // posterior toward a wrong displacement.
      val keep = math.max(10, (pairs.length * (1.0 - trimFraction)).toInt)
      val trimmed = pairs.sortBy { case (_, cur, closest) => (cur - closest).norm }.take(keep)

      val trainingData = trimmed.map { case (refPt, _, closest) => (refPt, closest - refPt) }
      val posteriorGP = priorGP.posterior(trainingData, sigma2)
      currentMesh = warpedMesh(posteriorGP, referenceMesh, DenseVector.zeros[Double](posteriorGP.rank))
    }
    currentMesh
  }
}
