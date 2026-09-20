package nonrigid

import breeze.linalg.DenseVector
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.common.{Field, ScalarMeshField3D}
import scalismo.geometry.{EuclideanSpace3D, EuclideanVector, Landmark, Point, _3D}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, LBFGSOptimizer}
import scalismo.registration.{GaussianProcessTransformationSpace, L2Regularizer, MeanSquaresMetric, Registration}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess}
import scalismo.utils.Random
import scapula.{Config, RigidAlign, ScapulaData}

/**
 * Stage 2 – Non-rigid registration pipeline (3-specimen pilot).
 *
 * Per specimen the pipeline runs:
 *   1. Landmark-based rigid Procrustes alignment
 *   2. Trimmed rigid ICP (pose refinement)
 *   3. GP posterior conditioned on landmark correspondences
 *   4. Multi-resolution LBFGS non-rigid registration (coarse → fine)
 *
 * Scalismo viewer groups:
 *   reference/           – decimated reference mesh + landmarks
 *   before-nonrigid-<id> – rigid-only result vs reference
 *   nonrigid-<id>        – live optimisation preview
 *   result-<id>          – fitted mesh, target, scalar error-distance map
 *
 * Console output: before/after metrics (directed mean/RMSE, symmetric mean/RMSE,
 * HD95, Hausdorff, Chamfer, landmark residual).
 */
object Stage2NonRigidReg {

  // -------------------------------------------------------------------------
  // Coarse-to-fine registration schedule
  // -------------------------------------------------------------------------
  private case class RegParams(regWeight: Double, iterations: Int, nSamples: Int)

  private val schedule: Seq[RegParams] = Seq(
    RegParams(regWeight = 1e-1, iterations = 50, nSamples =   100),
    RegParams(regWeight = 1e-2, iterations = 50, nSamples =   500),
    RegParams(regWeight = 1e-4, iterations = 50, nSamples =  1000),
    RegParams(regWeight = 1e-6, iterations = 50, nSamples =  5000)
  )

  // -------------------------------------------------------------------------
  // GP prior  (Dennis-Madsen-style three-kernel design)
  // -------------------------------------------------------------------------
  private def buildPriorGP(
    reference: TriangleMesh[_3D]
  )(implicit rng: Random): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    // Three Gaussian kernels that span coarse / mid / fine deformation scales.
    // Sigma values are fractions of the ~150 mm scapula diameter; scaling factors
    // set relative amplitudes so finer kernels do not dominate.
    val kCoarse = GaussianKernel3D(150.0 / 2,  15.0)  // 75 mm, amplitude 15
    val kMid    = GaussianKernel3D(150.0 / 5,  10.0)  // 30 mm, amplitude 10
    val kFine   = GaussianKernel3D(150.0 / 10,  5.0)  // 15 mm, amplitude  5
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
  // Condition the prior GP on landmark correspondences
  // -------------------------------------------------------------------------
  private def conditionOnLandmarks(
    gp:     LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    refLms: IndexedSeq[Landmark[_3D]],
    tgtLms: IndexedSeq[Landmark[_3D]],
    sigma2: Double = 1.0
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    // Training data: at each reference landmark the observed displacement
    // is (aligned-target landmark position) - (reference landmark position).
    val trainingData: IndexedSeq[(Point[_3D], EuclideanVector[_3D])] =
      refLms.zip(tgtLms).map { case (r, t) => (r.point, t.point - r.point) }
    gp.posterior(trainingData, sigma2)
  }

  // -------------------------------------------------------------------------
  // One LBFGS optimisation pass
  // -------------------------------------------------------------------------
  // The viewer callback is typed as DenseVector[Double] => Unit so the
  // caller controls how (and whether) the UI is updated — avoids binding
  // this function to a specific scalismo-ui view type.
  private def optimise(
    gp:         LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference:  TriangleMesh[_3D],
    target:     TriangleMesh[_3D],
    initCoeffs: DenseVector[Double],
    params:     RegParams,
    onUpdate:   DenseVector[Double] => Unit = _ => ()
  )(implicit rng: Random): DenseVector[Double] = {

    val tSpace      = GaussianProcessTransformationSpace(gp)
    val fixedImg    = reference.operations.toDistanceImage
    val movingImg   = target.operations.toDistanceImage
    // FixedPointsUniformMeshSampler3D pre-samples a fixed point set once and
    // reuses it across iterations — faster than UniformMeshSampler3D.
    val sampler     = FixedPointsUniformMeshSampler3D(reference, params.nSamples)
    val metric      = MeanSquaresMetric(fixedImg, movingImg, tSpace, sampler)
    val optimizer   = LBFGSOptimizer(maxNumberOfIterations = params.iterations)
    val regularizer = L2Regularizer(tSpace)
    val reg         = Registration(metric, regularizer, params.regWeight, optimizer)

    val iter = reg.iterator(initCoeffs)
    val visualizing = for ((state, _) <- iter.zipWithIndex) yield {
      onUpdate(state.parameters)
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

    val dir  = Config.dataDir
    val csv  = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens         = ScapulaData.specimens(dir)

    val leftSpecimens = specimens.filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(leftSpecimens.nonEmpty, "No left scapulae with landmarks found")

    // ------------------------------------------------------------------
    // Reference: first left scapula, decimated to model resolution
    // ------------------------------------------------------------------
    val refSpec   = leftSpecimens.head
    val rawRef    = ScapulaData.loadMesh(refSpec.file)
    val reference = rawRef.operations.decimate(Config.modelResolution)
    val refLms    = landmarks(refSpec.modelId)

    println(s"Reference : ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices)")

    // Build prior GP (shared across all targets)
    val priorGP = buildPriorGP(reference)
    println(s"GP prior rank : ${priorGP.rank}")

    // ------------------------------------------------------------------
    // Scalismo viewer
    // ------------------------------------------------------------------
    import scalismo.ui.api.ScalismoUI
    val ui = ScalismoUI()

    val refGroup = ui.createGroup("reference")
    ui.show(refGroup, reference, "reference")
    ui.show(refGroup, refLms.toList, "refLandmarks")

    // ------------------------------------------------------------------
    // Process the first 3 targets (pilot run)
    // ------------------------------------------------------------------
    val targets = leftSpecimens.drop(1).take(3)

    targets.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n=== [${idx + 1}/3]  ${spec.modelId} ===")

      val rawTarget = ScapulaData.loadMesh(spec.file)
      val tgtLms    = landmarks(spec.modelId)

      // ----------------------------------------------------------------
      // Step 1+2: Landmark rigid + trimmed ICP
      // ----------------------------------------------------------------
      val (rigidAligned, alignedLms) =
        RigidAlign.landmarkThenIcp(rawTarget, tgtLms, reference, refLms)

      // Show surfaces BEFORE non-rigid registration
      val preGroup = ui.createGroup(s"before-nonrigid-${spec.modelId}")
      ui.show(preGroup, reference,    "reference")
      ui.show(preGroup, rigidAligned, "rigidAligned")

      // ----------------------------------------------------------------
      // Step 3: Condition GP on landmark correspondences
      // ----------------------------------------------------------------
      val condGP = conditionOnLandmarks(priorGP, refLms, alignedLms)
      println(s"  Conditioned GP rank : ${condGP.rank}")

      // ----------------------------------------------------------------
      // Step 4: Multi-resolution LBFGS non-rigid registration
      // ----------------------------------------------------------------
      val regGroup = ui.createGroup(s"nonrigid-${spec.modelId}")
      ui.show(regGroup, reference,    "reference")
      ui.show(regGroup, rigidAligned, "target")
      // addTransformation attaches the GP to the group; setting .coefficients
      // on the returned view animates the deforming reference in the viewer.
      val gpView = ui.addTransformation(regGroup, condGP, "gp")

      val initCoeffs  = DenseVector.zeros[Double](condGP.rank)
      val finalCoeffs = schedule.foldLeft(initCoeffs) { (coeffs, params) =>
        println(s"    w=${params.regWeight}  iters=${params.iterations}  n=${params.nSamples}")
        optimise(condGP, reference, rigidAligned, coeffs, params,
          onUpdate = c => gpView.coefficients = c)
      }

      // Materialise fitted mesh: apply final deformation field to reference
      val deformField = condGP.instance(finalCoeffs)
      val fitted      = reference.transform(pt => pt + deformField(pt))

      // Deform reference landmarks through the same field (for landmark residual)
      val fittedLms = refLms.map(lm => lm.copy(point = lm.point + deformField(lm.point)))

      // ----------------------------------------------------------------
      // Show result + per-vertex error-distance colour map
      // ----------------------------------------------------------------
      val resultGroup = ui.createGroup(s"result-${spec.modelId}")
      ui.show(resultGroup, fitted,       "fitted")
      ui.show(resultGroup, rigidAligned, "target")

      // Per-vertex distance: fitted vertex → closest point on rigidAligned surface
      val errPerVertex: IndexedSeq[Double] = fitted.pointSet.points
        .map(p => (p - rigidAligned.operations.closestPointOnSurface(p).point).norm)
        .toIndexedSeq
      // ScalarMeshField3D is the non-deprecated factory in scalismo.common
      val errorMap = ScalarMeshField3D(fitted, errPerVertex)
      ui.show(resultGroup, errorMap, "errorMap")

      // ----------------------------------------------------------------
      // All error metrics
      // ----------------------------------------------------------------
      // "before" = rigid-only result compared to reference (shape mismatch floor)
      val before = RegistrationMetrics.compute(rigidAligned, reference)
      // "after"  = fitted result compared to rigidly aligned target (registration residual)
      val after  = RegistrationMetrics.compute(
        fitted, rigidAligned,
        fittedLms  = Some(fittedLms),
        targetLms  = Some(alignedLms)
      )

      RegistrationMetrics.printComparison(spec.modelId, before, after)
    }

    println("\nDone — close the Scalismo viewer when finished.")
  }
}
