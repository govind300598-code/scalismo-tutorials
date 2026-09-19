package nonrigid

import breeze.linalg.DenseVector
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.*
import scalismo.mesh.*
import scalismo.numerics.*
import scalismo.registration.*
import scalismo.statisticalmodel.*
import scalismo.ui.api.*
import scalismo.utils.Random
import scapula.{Config, Metrics, RigidAlign, ScapulaData}

/**
 * Stage 2 – Non-rigid registration pipeline.
 *
 * Pipeline per specimen:
 *   1. Landmark-based rigid alignment (Procrustes)
 *   2. Trimmed rigid ICP (pose refinement)
 *   3. GP conditioned on landmark correspondences (shape prior)
 *   4. Multi-resolution GP / LBFGS non-rigid ICP
 *
 * After each specimen the Scalismo viewer shows:
 *   - before-nonrigid group: rigid-only result vs reference
 *   - result group         : fitted mesh, target, and error-distance colour map
 *
 * Error metrics (mean, RMS, HD95, HD) are printed to stdout.
 */
object Stage2NonRigidReg {

  // -------------------------------------------------------------------------
  // Registration schedule (coarse → fine)
  // -------------------------------------------------------------------------
  private case class RegParams(regWeight: Double, iterations: Int, nSamples: Int)

  private val regSchedule: Seq[RegParams] = Seq(
    RegParams(1e-1, 50,  100),
    RegParams(1e-2, 50,  500),
    RegParams(1e-4, 50, 1000),
    RegParams(1e-6, 50, 5000)
  )

  // -------------------------------------------------------------------------
  // Build Dennis-Madsen-style GP prior
  // -------------------------------------------------------------------------
  private def buildPriorGP(
    reference: TriangleMesh[_3D]
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    // Scapula ~150 mm on longest side; kernels span coarse / mid / fine scales
    val kCoarse = GaussianKernel3D(150.0 / 2,  15)
    val kMid    = GaussianKernel3D(150.0 / 5,  10)
    val kFine   = GaussianKernel3D(150.0 / 10,  5)
    val kernel  = DiagonalKernel3D(kCoarse + kMid + kFine, outputDim = 3)

    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp       = GaussianProcess(zeroMean, kernel)

    LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )
  }

  // -------------------------------------------------------------------------
  // Condition the GP on landmark correspondences
  // -------------------------------------------------------------------------
  private def conditionOnLandmarks(
    gp: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    refLms: IndexedSeq[Landmark[_3D]],
    tgtLms: IndexedSeq[Landmark[_3D]],
    sigma2: Double = 1.0
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    val trainingData: IndexedSeq[(Point[_3D], EuclideanVector[_3D])] =
      refLms.zip(tgtLms).map { case (r, t) => (r.point, t.point - r.point) }
    gp.posterior(trainingData, sigma2)
  }

  // -------------------------------------------------------------------------
  // One LBFGS non-rigid registration pass
  // -------------------------------------------------------------------------
  private def doRegistration(
    gp: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D],
    initCoeffs: DenseVector[Double],
    params: RegParams,
    gpView: TransformationFieldView
  ): DenseVector[Double] = {
    val tSpace      = GaussianProcessTransformationSpace(gp)
    val fixedImg    = reference.operations.toDistanceImage
    val movingImg   = target.operations.toDistanceImage
    val sampler     = FixedPointsUniformMeshSampler3D(reference, params.nSamples)
    val metric      = MeanSquaresMetric(fixedImg, movingImg, tSpace, sampler)
    val optimizer   = LBFGSOptimizer(maxNumberOfIterations = params.iterations)
    val regularizer = L2Regularizer(tSpace)
    val reg         = Registration(metric, regularizer, params.regWeight, optimizer)

    val iter = reg.iterator(initCoeffs)
    val visualizing = for ((state, i) <- iter.zipWithIndex) yield {
      gpView.coefficients = state.parameters
      state
    }
    visualizing.toSeq.last.parameters
  }

  // -------------------------------------------------------------------------
  // Main
  // -------------------------------------------------------------------------
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens         = ScapulaData.specimens(dir)

    val leftSpecimens = specimens.filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(leftSpecimens.nonEmpty, "No left scapulae with landmarks found")

    // Use first left scapula as the reference
    val refSpec   = leftSpecimens.head
    val rawRef    = ScapulaData.loadMesh(refSpec.file)
    val reference = rawRef.operations.decimate(Config.modelResolution)
    val refLms    = landmarks(refSpec.modelId)

    println(s"Reference : ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices)")

    val priorGP = buildPriorGP(reference)
    println(s"GP prior rank: ${priorGP.rank}")

    val ui = ScalismoUI()

    // Show the reference once, globally
    val refGroup = ui.createGroup("reference")
    ui.show(refGroup, reference, "reference")
    ui.show(refGroup, refLms.toList, "refLandmarks")

    // Register the remaining left scapulae (first 3 for a quick demo)
    val targets = leftSpecimens.drop(1).take(3)

    targets.foreach { spec =>
      println(s"\n=== ${spec.modelId} ===")

      val rawTarget = ScapulaData.loadMesh(spec.file)
      val tgtLms    = landmarks(spec.modelId)

      // ------------------------------------------------------------------
      // Step 1 + 2: Landmark rigid + trimmed ICP
      // ------------------------------------------------------------------
      val (rigidAligned, alignedLms) =
        RigidAlign.landmarkThenIcp(rawTarget, tgtLms, reference, refLms)

      // Show the non-registered (rigid-only) surfaces side by side
      val preGroup = ui.createGroup(s"before-nonrigid-${spec.modelId}")
      ui.show(preGroup, reference,    "reference")
      ui.show(preGroup, rigidAligned, "rigidAligned")

      val preStats = Metrics.symmetric(rigidAligned, reference)
      println(s"  After rigid   : ${preStats.render}")

      // ------------------------------------------------------------------
      // Step 3: Condition the GP on landmark correspondences
      // ------------------------------------------------------------------
      val condGP = conditionOnLandmarks(priorGP, refLms, alignedLms)
      println(s"  Conditioned GP rank: ${condGP.rank}")

      // ------------------------------------------------------------------
      // Step 4: GP non-rigid ICP optimisation
      // ------------------------------------------------------------------
      val regGroup = ui.createGroup(s"nonrigid-${spec.modelId}")
      ui.show(regGroup, reference,    "reference")
      ui.show(regGroup, rigidAligned, "target")
      val gpView   = ui.addTransformation(regGroup, condGP, "gp")

      val initCoeffs = DenseVector.zeros[Double](condGP.rank)
      val finalCoeffs = regSchedule.foldLeft(initCoeffs) { (coeffs, params) =>
        println(s"    w=${params.regWeight}  iters=${params.iterations}  n=${params.nSamples}")
        doRegistration(condGP, reference, rigidAligned, coeffs, params, gpView)
      }

      // Materialise the fitted mesh
      val deformField = condGP.instance(finalCoeffs)
      val fitted      = reference.transform(pt => pt + deformField(pt))

      // ------------------------------------------------------------------
      // Show result + error map in viewer
      // ------------------------------------------------------------------
      val resultGroup = ui.createGroup(s"result-${spec.modelId}")
      ui.show(resultGroup, fitted,       "fitted")
      ui.show(resultGroup, rigidAligned, "target")

      val errPerVertex = Metrics.surfaceDistances(fitted, rigidAligned).toArray
      val scalarField  = ScalarMeshField(fitted, errPerVertex)
      ui.show(resultGroup, scalarField, "errorMap")

      val postStats = Metrics.symmetric(fitted, rigidAligned)
      println(s"  After non-rigid: ${postStats.render}")
      println(f"  Error-map stats: min=${errPerVertex.min}%.2f  max=${errPerVertex.max}%.2f  " +
        f"mean=${errPerVertex.sum / errPerVertex.length}%.2f mm")
    }

    println("\nDone — close the Scalismo viewer when finished.")
  }
}
