package scapula

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.mesh.*
import scalismo.registration.*
import scalismo.numerics.*
import scalismo.statisticalmodel.*
import scalismo.kernels.*
import scalismo.io.MeshIO
import scalismo.utils.Random
import breeze.linalg.DenseVector

import java.io.{File, PrintWriter}

/**
 * STAGE 2 -- NON-RIGID REGISTRATION.
 *
 * This is the scalismo.org tutorial pipeline, unchanged, wired to Config/ScapulaData/RigidAlign instead of the
 * tutorial's own toy bunny/face dataset:
 *
 *   step 1  Rigid alignment          -- scalismo tutorial "Rigid alignment": landmark-based Procrustes
 *                                       (LandmarkRegistration.rigid3DLandmarkRegistration), pose only.
 *           Automatic ICP            -- iterative closest point refinement of that rigid pose.
 *                                       (Both already implemented in RigidAlign.scala: landmarkThenIcp.)
 *
 *   step 2  Gaussian process model   -- scalismo tutorial "Gaussian processes, kernels and PCA": a hand-designed,
 *                                       multi-scale sum-of-Gaussian-kernels prior on deformations of the reference,
 *                                       approximated by a low-rank basis (pivoted Cholesky).
 *
 *   step 3  Non-rigid registration   -- scalismo tutorial "Parametric, non-rigid registration": fit the GP model's
 *                                       coefficients to each target by gradient-based optimisation (LBFGS) of a
 *                                       distance-image metric plus an L2 regularizer on the coefficients, run
 *                                       coarse-to-fine (high regularization/few points first, low regularization/many
 *                                       points last).
 *
 * Output of this stage is a set of meshes that all have the SAME number of points and the SAME point ids as the
 * reference -- i.e. point k on every mesh is "the same" anatomical location. That correspondence is the only thing
 * Stage 3 (building the SSM) needs.
 */
object Stage2Registration {

  // ---- step 3 registration schedule, coarse-to-fine (scalismo tutorial "Parametric, non-rigid registration") ----
  final case class Schedule(regularizationWeight: Double, iterations: Int, numberOfSampledPoints: Int)

  val schedule: Seq[Schedule] = Seq(
    Schedule(regularizationWeight = 1e-1, iterations = 25, numberOfSampledPoints = 800),
    Schedule(regularizationWeight = 1e-2, iterations = 30, numberOfSampledPoints = 1500),
    Schedule(regularizationWeight = 1e-3, iterations = 35, numberOfSampledPoints = 3000),
    Schedule(regularizationWeight = 5e-4, iterations = 35, numberOfSampledPoints = 4500)
  )

  final case class GpModel(
    pdm: PointDistributionModel[_3D, TriangleMesh],
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]]
  )

  /**
   * Per-specimen registration quality, written to registration_metrics.csv. This is what lets you find WHICH
   * specimen produced a bad fit (e.g. a stretched acromion/coracoid) instead of only seeing the symptom later,
   * amplified, in a PCA mode of the finished SSM.
   *
   *   meanMm / rmsMm / hd95Mm / hdMaxMm  -- symmetric surface distance (both directions pooled, Metrics.symmetric)
   *                                         between the non-rigidly fitted mesh and the rigidly-aligned target.
   *                                         hd95Mm/hdMaxMm are always >= meanMm by construction (same pooled
   *                                         distance distribution, just a higher percentile / the maximum).
   *   volumeBeforeCm3 / volumeAfterCm3   -- signed volume (ScapulaData.signedVolume) of the rigid target and of the
   *                                         fitted mesh. A large swing, or a sign flip, means the non-rigid step
   *                                         folded or self-intersected the surface -- exactly the failure mode a
   *                                         thin structure (acromion, coracoid, glenoid rim) is prone to.
   *   orientationOk                      -- false if the fitted mesh's volume went negative or flipped sign.
   */
  final case class RegistrationRow(
    specimen: String,
    meanMm: Double,
    rmsMm: Double,
    hd95Mm: Double,
    hdMaxMm: Double,
    volumeBeforeCm3: Double,
    volumeAfterCm3: Double,
    volumeChangePct: Double,
    orientationOk: Boolean
  )

  def evaluateRegistration(specimen: String, target: TriangleMesh[_3D], fitted: TriangleMesh[_3D]): RegistrationRow = {
    val stats = Metrics.symmetric(fitted, target)
    val volBefore = ScapulaData.signedVolume(target) / 1000.0
    val volAfter = ScapulaData.signedVolume(fitted) / 1000.0
    val volChangePct = (volAfter - volBefore) / volBefore * 100.0
    val orientationOk = volAfter > 0 && math.signum(volBefore) == math.signum(volAfter)
    RegistrationRow(specimen, stats.mean, stats.rms, stats.hd95, stats.hd, volBefore, volAfter, volChangePct, orientationOk)
  }

  /** step 2: scalismo tutorial "Gaussian processes, kernels and PCA" -- multi-scale sum-of-Gaussians prior. */
  def buildGpModel(referenceMesh: TriangleMesh[_3D]): GpModel = {
    val kernel = DiagonalKernel3D(
      GaussianKernel3D(75.0, 15.0) +
        GaussianKernel3D(30.0, 10.0) +
        GaussianKernel3D(15.0, 5.0) +
        GaussianKernel3D(5.0, 1.5),
      outputDim = 3
    )
    val gp = GaussianProcess3D[EuclideanVector[_3D]](kernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )
    GpModel(PointDistributionModel3D(referenceMesh, lowRankGP), lowRankGP)
  }

  /** step 3: scalismo tutorial "Parametric, non-rigid registration" -- run coarse-to-fine, exactly as taught. */
  def registerNonRigidly(
    model: GpModel,
    referenceMesh: TriangleMesh[_3D],
    target: TriangleMesh[_3D]
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val transformationSpace = GaussianProcessTransformationSpace(model.lowRankGP)
    val fixedImage = referenceMesh.operations.toDistanceImage
    val movingImage = target.operations.toDistanceImage

    val finalCoefficients = schedule.foldLeft(DenseVector.zeros[Double](model.pdm.rank)) { (coeffs, step) =>
      val sampler = FixedPointsUniformMeshSampler3D(referenceMesh, numberOfPoints = step.numberOfSampledPoints)
      val metric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler)
      val optimizer = LBFGSOptimizer(maxNumberOfIterations = step.iterations)
      val regularizer = L2Regularizer(transformationSpace)
      val registration = Registration(metric, regularizer, regularizationWeight = step.regularizationWeight, optimizer)
      registration.iterator(coeffs).toSeq.last.parameters
    }
    model.pdm.instance(finalCoefficients)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val outDir = Config.outDir
    outDir.mkdirs()
    val corrDir = new File(outDir, "corresponded")
    corrDir.mkdirs()

    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dir))
    val all = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    require(all.nonEmpty, s"no STL/CSV matches found in ${dir.getPath}")

    // Left and mirrored-right scapulae from the SAME person are not independent shape samples (see
    // ScapulaData.Config.buildIndependentModel). Respect that flag here already, at cohort-selection time, rather
    // than fixing it after the fact in Stage 3.
    val cohort = if (Config.buildIndependentModel) all.filter(!_.isRight) else all
    require(cohort.nonEmpty, "no specimens selected for the training cohort")
    println(s"[Stage2] Cohort: ${cohort.length} specimens (buildIndependentModel=${Config.buildIndependentModel})")

    // Every specimen is brought into one canonical ("left") shape space, mirroring right-side bones.
    // scalismo tutorial: mirroring is a reflection (determinant -1); triangle winding must be flipped to keep
    // outward-pointing normals -- see ScapulaData.mirrorMesh / Stage1Diagnostics [A2] for the correctness check.
    def canonical(s: ScapulaData.Specimen): (TriangleMesh[_3D], IndexedSeq[Landmark[_3D]]) = {
      val mesh = ScapulaData.loadMesh(s.file)
      val lms = landmarks(s.modelId)
      if (s.isRight) (ScapulaData.mirrorMesh(mesh), ScapulaData.mirrorLandmarks(lms)) else (mesh, lms)
    }

    // ---- reference mesh: scalismo tutorial picks an arbitrary specimen as reference; we do the same ------------
    val referenceSpecimen = cohort.head
    val (refRaw, refLms) = canonical(referenceSpecimen)
    val referenceMesh =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution) refRaw.operations.decimate(Config.modelResolution)
      else refRaw
    println(s"[Stage2] Reference: ${referenceSpecimen.modelId} (${referenceMesh.pointSet.numberOfPoints} points)")
    MeshIO.writeMesh(referenceMesh, new File(outDir, "reference.stl")).get

    val gpModel = buildGpModel(referenceMesh)
    println(s"[Stage2] Gaussian process model rank: ${gpModel.pdm.rank}")

    val targets = cohort.filter(_.modelId != referenceSpecimen.modelId)
    val registrationRows = targets.zipWithIndex.map { case (s, i) =>
      print(s"[Stage2] [${i + 1}/${targets.length}] registering ${s.modelId} ... ")
      val (rawMesh, rawLms) = canonical(s)

      // step 1a: landmark Procrustes (pose only).
      // step 1b: automatic (trimmed) rigid ICP refinement.
      val (rigidMesh, _) = RigidAlign.landmarkThenIcp(rawMesh, rawLms, referenceMesh, refLms, Config.icpIterations)

      // step 2+3: GP-based non-rigid registration onto the reference topology.
      val fitted = registerNonRigidly(gpModel, referenceMesh, rigidMesh)
      MeshIO.writeMesh(fitted, new File(corrDir, s"${s.modelId}.stl")).get

      val row = evaluateRegistration(s.modelId, rigidMesh, fitted)
      val flag = if (!row.orientationOk) "  <-- ORIENTATION/VOLUME PROBLEM" else ""
      println(f"mean=${row.meanMm}%.2fmm hd95=${row.hd95Mm}%.2fmm hdMax=${row.hdMaxMm}%.2fmm dVol=${row.volumeChangePct}%+.1f%%$flag")
      row
    }

    // The reference itself is also a member of the training cohort, already in correspondence with itself.
    MeshIO.writeMesh(referenceMesh, new File(corrDir, s"${referenceSpecimen.modelId}.stl")).get

    // ---- registration quality CSV: which specimen(s), if any, produced a bad non-rigid fit ------------------------
    val metricsCsv = new File(outDir, "registration_metrics.csv")
    val pw = new PrintWriter(metricsCsv)
    try {
      pw.println("specimen,mean_mm,rms_mm,hd95_mm,hd_max_mm,volume_before_cm3,volume_after_cm3,volume_change_pct,orientation_ok")
      registrationRows.foreach { r =>
        pw.println(
          s"${r.specimen},${r.meanMm},${r.rmsMm},${r.hd95Mm},${r.hdMaxMm},${r.volumeBeforeCm3},${r.volumeAfterCm3},${r.volumeChangePct},${r.orientationOk}"
        )
      }
    } finally pw.close()
    println(s"[Stage2] Registration metrics written to ${metricsCsv.getAbsolutePath}")

    val badOrientation = registrationRows.filterNot(_.orientationOk)
    val worstByHd95 = registrationRows.sortBy(-_.hd95Mm).take(3)
    if (badOrientation.nonEmpty) {
      println(s"[Stage2] !! ${badOrientation.length} specimen(s) failed the orientation/volume check: " +
        badOrientation.map(_.specimen).mkString(", "))
      println("[Stage2]    The fitted mesh folded or self-intersected -- inspect these first, they are the most")
      println("[Stage2]    likely cause of an implausible PCA mode (e.g. a stretched acromion/coracoid at +/-3 sigma).")
    }
    println(s"[Stage2] Worst 3 specimens by HD95: " + worstByHd95.map(r => f"${r.specimen} (${r.hd95Mm}%.2fmm)").mkString(", "))

    println(s"[Stage2] Done. ${cohort.length} corresponded meshes written to ${corrDir.getAbsolutePath}")
    println("[Stage2] Next: run Stage3BuildModel to build the statistical shape model.")
  }
}
