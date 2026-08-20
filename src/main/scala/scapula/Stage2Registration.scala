package scapula

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.mesh.*
import scalismo.registration.*
import scalismo.numerics.*
import scalismo.statisticalmodel.*
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.kernels.*
import scalismo.io.MeshIO
import scalismo.utils.Random
import breeze.linalg.DenseVector

import java.io.{File, PrintWriter}

/**
 * STAGE 2 -- NON-RIGID REGISTRATION.
 *
 * Implements the scalismo.org tutorial pipeline wired to Config/ScapulaData/RigidAlign:
 *
 *   step 1  Rigid alignment          -- landmark-based Procrustes + trimmed ICP (RigidAlign.landmarkThenIcp).
 *
 *   step 2  Gaussian process model   -- multi-scale sum-of-Gaussian-kernels prior on deformations of the
 *                                       reference, approximated by a low-rank pivoted-Cholesky basis, capped
 *                                       at Config.gpMaxRank to keep memory and iteration cost bounded.
 *
 *   step 3  Non-rigid registration   -- coarse-to-fine LBFGS (distance-image metric + L2 regularizer).
 *
 *   step 4  Reference bias removal   -- Config.refinePasses controls how many registration passes run.
 *                                       After each intermediate pass the mean of the fitted meshes replaces
 *                                       the reference, so the final SSM is centred on the population mean
 *                                       rather than the shape of an arbitrary first specimen.
 *
 * Output: corresponded meshes (all same topology) in <outDir>/corresponded/ and registration_metrics.csv.
 */
object Stage2Registration {

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
   * Per-specimen registration quality written to registration_metrics.csv.
   *
   *   meanMm / rmsMm / hd95Mm / hdMaxMm  -- symmetric surface distance between the non-rigidly fitted mesh
   *                                         and the rigidly-aligned target.
   *   volumeBeforeCm3 / volumeAfterCm3   -- signed volume before and after non-rigid fitting.
   *                                         A sign flip means the mesh folded / self-intersected.
   *   orientationOk                      -- false if volume went negative or changed sign.
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

  /** step 2: multi-scale sum-of-Gaussians GP prior, capped at Config.gpMaxRank. */
  def buildGpModel(referenceMesh: TriangleMesh[_3D]): GpModel = {
    val kernel = DiagonalKernel3D(
      GaussianKernel3D(75.0, 15.0) +
        GaussianKernel3D(30.0, 10.0) +
        GaussianKernel3D(15.0, 5.0) +
        GaussianKernel3D(5.0, 1.5),
      outputDim = 3
    )
    val gp = GaussianProcess3D[EuclideanVector[_3D]](kernel)
    val rawGP = LowRankGaussianProcess.approximateGPCholesky(
      referenceMesh,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator = NearestNeighborInterpolator3D()
    )
    val lowRankGP = if (rawGP.rank > Config.gpMaxRank) rawGP.truncate(Config.gpMaxRank) else rawGP
    println(s"[Stage2]   GP rank: ${lowRankGP.rank} (raw Cholesky=${rawGP.rank}, cap=${Config.gpMaxRank})")
    GpModel(PointDistributionModel3D(referenceMesh, lowRankGP), lowRankGP)
  }

  /** step 3: coarse-to-fine LBFGS registration, exactly as in the scalismo tutorial. */
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

    val cohort = if (Config.buildIndependentModel) all.filter(!_.isRight) else all
    require(cohort.nonEmpty, "no specimens selected for the training cohort")
    println(s"[Stage2] Cohort: ${cohort.length} specimens (buildIndependentModel=${Config.buildIndependentModel})")
    println(s"[Stage2] Registration passes: ${Config.refinePasses} (pass 1 = arbitrary reference; pass 2+ = mean-shape reference)")

    def canonical(s: ScapulaData.Specimen): (TriangleMesh[_3D], IndexedSeq[Landmark[_3D]]) = {
      val mesh = ScapulaData.loadMesh(s.file)
      val lms  = landmarks(s.modelId)
      if (s.isRight) (ScapulaData.mirrorMesh(mesh), ScapulaData.mirrorLandmarks(lms)) else (mesh, lms)
    }

    // ---- initial reference: first specimen alphabetically, decimated to model resolution ----
    val referenceSpecimen = cohort.head
    val (refRaw, refLms) = canonical(referenceSpecimen)
    val initialRef =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution) refRaw.operations.decimate(Config.modelResolution)
      else refRaw
    println(s"[Stage2] Initial reference: ${referenceSpecimen.modelId} (${initialRef.pointSet.numberOfPoints} pts)")

    // ---- load all canonical meshes once ----
    println("[Stage2] Loading canonical meshes...")
    val canonicalMeshes: IndexedSeq[(TriangleMesh[_3D], IndexedSeq[Landmark[_3D]])] = cohort.map(canonical)

    // ---- pass-0 rigid alignment: landmark Procrustes + trimmed ICP to initial reference ----
    println("[Stage2] Pass 0: rigid alignment (landmark Procrustes + ICP) to initial reference...")
    val pass0Rigid: IndexedSeq[TriangleMesh[_3D]] = canonicalMeshes.zipWithIndex.map { case ((mesh, lms), i) =>
      print(s"  [${i + 1}/${cohort.length}] ${cohort(i).modelId} ... ")
      val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, initialRef, refLms, Config.icpIterations)
      println("done")
      aligned
    }

    // ---- multi-pass non-rigid registration + mean-shape reference update ----
    var currentRef   = initialRef
    var gpModel      = buildGpModel(currentRef)
    var currentRigid = pass0Rigid  // rigid-aligned meshes used as input to non-rigid registration

    // (specimen, rigidMesh, fittedMesh, qualityRow) collected from the FINAL pass only
    var finalPassResults: IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D], TriangleMesh[_3D], RegistrationRow)] =
      IndexedSeq.empty

    for (pass <- 0 until Config.refinePasses) {
      val isFinal = pass == Config.refinePasses - 1
      println(s"\n[Stage2] === Non-rigid registration pass ${pass + 1}/${Config.refinePasses}${if (isFinal) " (FINAL)" else ""} ===")

      val passResults = cohort.zip(currentRigid).zipWithIndex.map { case ((s, rigidMesh), i) =>
        print(s"  [${i + 1}/${cohort.length}] ${s.modelId} ... ")
        val fitted = registerNonRigidly(gpModel, currentRef, rigidMesh)
        val row    = evaluateRegistration(s.modelId, rigidMesh, fitted)
        val flag   = if (!row.orientationOk) "  <-- ORIENTATION/VOLUME PROBLEM" else ""
        println(f"mean=${row.meanMm}%.2fmm hd95=${row.hd95Mm}%.2fmm hdMax=${row.hdMaxMm}%.2fmm dVol=${row.volumeChangePct}%+.1f%%$flag")
        (s, rigidMesh, fitted, row)
      }

      if (isFinal) {
        finalPassResults = passResults
      } else {
        // Compute mean of this pass's fitted meshes → use as new reference for the next pass.
        // The mean eliminates reference bias: instead of pulling everything toward cohort.head's shape,
        // the next pass pulls toward the population centroid.
        val fittedMeshes = passResults.map(_._3)
        val dc = DataCollection.fromTriangleMesh3DSequence(currentRef, fittedMeshes)
        currentRef = PointDistributionModel.createUsingPCA(dc).mean
        println(s"[Stage2] Mean-shape reference computed for pass ${pass + 2}. Rebuilding GP model...")
        gpModel = buildGpModel(currentRef)

        // Re-align canonical meshes to new mean reference via ICP.
        // Start from pass-0-rigid positions (already roughly aligned) so ICP converges quickly.
        println(s"[Stage2] Re-aligning all specimens to mean reference via ICP...")
        currentRigid = pass0Rigid.zipWithIndex.map { case (mesh, i) =>
          print(s"  [${i + 1}/${pass0Rigid.length}] ${cohort(i).modelId} ... ")
          val result = RigidAlign.rigidIcp(mesh, currentRef, Config.icpIterations)
          println("done")
          result
        }
      }
    }

    // ---- write final reference and all corresponded meshes ----
    MeshIO.writeMesh(currentRef, new File(outDir, "reference.stl")).get
    println(s"\n[Stage2] Reference mesh written: " +
      s"${if (Config.refinePasses > 1) "population mean shape" else "initial (arbitrary) specimen"}" +
      s" (${currentRef.pointSet.numberOfPoints} pts)")

    val registrationRows = finalPassResults.map { case (s, rigidMesh, fitted, row) =>
      MeshIO.writeMesh(fitted, new File(corrDir, s"${s.modelId}.stl")).get
      row
    }

    // ---- registration quality CSV ----
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
    val worstByHd95   = registrationRows.sortBy(-_.hd95Mm).take(3)
    if (badOrientation.nonEmpty) {
      println(s"[Stage2] !! ${badOrientation.length} specimen(s) failed the orientation/volume check: " +
        badOrientation.map(_.specimen).mkString(", "))
      println("[Stage2]    Fitted mesh folded or self-intersected -- inspect these first.")
    }
    println(s"[Stage2] Worst 3 by HD95: " +
      worstByHd95.map(r => f"${r.specimen} (${r.hd95Mm}%.2fmm)").mkString(", "))
    println(s"[Stage2] Done. ${cohort.length} corresponded meshes in ${corrDir.getAbsolutePath}")
    println("[Stage2] Next: run Stage3BuildModel to build the statistical shape model.")
  }
}
