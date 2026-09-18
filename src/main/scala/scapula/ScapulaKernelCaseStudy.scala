package scapula

import scalismo.common.{EuclideanSpace3D, Field}
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D, PDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.io.MeshIO
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Dedicated, self-contained kernel case-study pipeline -- distinct from Stage2GPNonRigidRegistration so its output
 * can never be confused with that pipeline's. Fixed to one specific 6-specimen set (1 reference + 5 real targets),
 * comparing exactly two multi-scale GP kernel designs (3-scale vs 2-scale) built from EXPLICIT, absolute length
 * scales -- deliberately NOT GPRegistrationCore.buildMultiscaleGP's divisor-of-bbox-diagonal scheme, since this
 * study's whole point is comparing two fixed kernel designs against each other, not auto-scaling with specimen size.
 *
 * Per-specimen pipeline reuses this project's already-validated stages directly (not reimplemented):
 *   Stage 1+2: RigidAlign.landmarkThenIcp -- landmark Procrustes (5 landmarks: GC, TS, IA, PLA, AC) then trimmed
 *              ICP refinement (Config.icpIterations, same defaults as the rest of this project).
 *   Stage 3:   GPRegistrationCore.icpGprRegister -- iterative closest-point + GP regression (same defaults as the
 *              rest of this project: Config.icpGprIterations/NumPoints/TrimFraction/Sigma2Schedule).
 *
 * Kernel cases (amplitudes match this project's own validated coarse/mid/fine defaults -- 15/10/5 -- only the
 * length scales are this study's explicit absolute values; Case B is Case A with the mid term simply removed, so
 * the comparison isolates exactly the effect of dropping that one scale):
 *   Case A (3-scale): Gaussian(sigma=75mm, amp=15) + Gaussian(sigma=30mm, amp=10) + Gaussian(sigma=15mm, amp=5)
 *   Case B (2-scale): Gaussian(sigma=75mm, amp=15) + Gaussian(sigma=15mm, amp=5)
 */
object ScapulaKernelCaseStudy {

  val referenceId = "paired_scapula_005_F_67_R"
  val targetIds: IndexedSeq[String] = IndexedSeq(
    "paired_scapula_001_M_64_L",
    "paired_scapula_002_M_56_L",
    "paired_scapula_006_F_60_R",
    "paired_scapula_007_M_26_L",
    "paired_scapula_008_F_73_L"
  )

  private val outDir = new File(sys.env.getOrElse(
    "SCAPULA_KERNEL_CASE_OUT_DIR",
    "/home/g25upadh/Documents/100 plus scapula data/scapula_kernel_case_study_out"
  ))

  final case class KernelCase(folderName: String, label: String, sigmasAndAmplitudes: IndexedSeq[(Double, Double)])
  val caseA: KernelCase = KernelCase("case_a", "Case A (3-scale: 75+30+15)", IndexedSeq(75.0 -> 15.0, 30.0 -> 10.0, 15.0 -> 5.0))
  val caseB: KernelCase = KernelCase("case_b", "Case B (2-scale: 75+15)", IndexedSeq(75.0 -> 15.0, 15.0 -> 5.0))

  /** Absolute-length-scale multiscale GP kernel -- sum of Gaussian kernels at the given (sigma_mm, amplitude) pairs. */
  def buildFixedScaleGP(reference: TriangleMesh[_3D], sigmasAndAmplitudes: IndexedSeq[(Double, Double)])
                        (implicit rng: Random): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    val kernelSum = sigmasAndAmplitudes
      .map { case (sigma, amp) => GaussianKernel3D(sigma, amp): PDKernel[_3D] }
      .reduce((a, b) => a + b)
    val kernel = DiagonalKernel3D(kernelSum, outputDim = 3)
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp = GaussianProcess(zeroMean, kernel)
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference, gp, relativeTolerance = Config.gpRelativeTolerance, interpolator = NearestNeighborInterpolator3D()
    )
    if (lowRankGP.rank > Config.gpMaxRank) lowRankGP.truncate(Config.gpMaxRank) else lowRankGP
  }

  private def runCase(
      kernelCase: KernelCase,
      reference: TriangleMesh[_3D],
      alignedTargets: IndexedSeq[(String, TriangleMesh[_3D])]
  )(implicit rng: Random): IndexedSeq[(String, TriangleMesh[_3D], Metrics.SurfaceStats)] = {
    val lowRankGP = buildFixedScaleGP(reference, kernelCase.sigmasAndAmplitudes)
    println(f"\n=== ${kernelCase.label} -- GP prior rank = ${lowRankGP.rank} ===")
    alignedTargets.map { case (id, target) =>
      val t0 = System.nanoTime()
      val registered = GPRegistrationCore.icpGprRegister(
        lowRankGP, reference, target,
        Config.icpGprIterations, Config.icpGprSigma2Schedule.toIndexedSeq, Config.icpGprNumPoints, Config.icpGprTrimFraction
      )
      val stats = Metrics.symmetric(registered, target)
      val seconds = (System.nanoTime() - t0) / 1e9
      println(f"  $id%-24s ${stats.render}   ($seconds%.1fs)")
      (id, registered, stats)
    }
  }

  private def writeCsv(file: File, rows: IndexedSeq[(String, Metrics.SurfaceStats)]): Unit = {
    val pw = new PrintWriter(file)
    try {
      pw.println("model_id,mean_mm,rms_mm,hd95_mm,hd_mm,chamfer_mm2")
      rows.foreach { case (id, s) => pw.println(f"$id,${s.mean}%.4f,${s.rms}%.4f,${s.hd95}%.4f,${s.hd}%.4f,${s.chamfer}%.4f") }
    } finally pw.close()
  }

  private def printTableAndFindBestWorst(caseLabel: String, rows: IndexedSeq[(String, Metrics.SurfaceStats)]): (String, String) = {
    println(f"\n[$caseLabel] metrics table (RMSE = symmetric RMS, CD = Chamfer distance, HD = Hausdorff):")
    println(f"  ${"Specimen"}%-24s ${"RMSE(mm)"}%10s ${"CD(mm^2)"}%10s ${"HD(mm)"}%9s ${"HD95(mm)"}%10s ${"Mean(mm)"}%10s")
    rows.foreach { case (id, s) =>
      println(f"  $id%-24s ${s.rms}%10.3f ${s.chamfer}%10.3f ${s.hd}%9.3f ${s.hd95}%10.3f ${s.mean}%10.3f")
    }
    val best = rows.minBy { case (_, s) => s.chamfer }
    val worst = rows.maxBy { case (_, s) => s.chamfer }
    println(f"  BEST  (lowest CD) : ${best._1}%-24s CD=${best._2.chamfer}%.3f mm^2")
    println(f"  WORST (highest CD): ${worst._1}%-24s CD=${worst._2.chamfer}%.3f mm^2")
    (best._1, worst._1)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csvFile = ScapulaData.csvFile(dir)
    println(s"Data directory : ${dir.getAbsolutePath}")
    println(s"Landmark CSV   : ${csvFile.getAbsolutePath}")
    println(s"Output dir     : ${outDir.getAbsolutePath}")

    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csvFile)
    val allSpecimens = ScapulaData.specimens(dir).map(s => s.modelId -> s).toMap
    val wanted = (targetIds :+ referenceId).toSet
    val missing = wanted -- allSpecimens.keySet
    require(missing.isEmpty, s"Specimen id(s) not found in ${dir.getPath}: ${missing.mkString(", ")}")

    final case class Canonical(mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]])
    def loadCanonical(id: String): Canonical = {
      val s = allSpecimens(id)
      val rawMesh = ScapulaData.loadMesh(s.file)
      val rawLms = landmarks.getOrElse(id, throw new RuntimeException(s"No landmarks for $id"))
      if (s.isRight) Canonical(ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
      else Canonical(rawMesh, rawLms)
    }

    println(s"\nReference (fixed): $referenceId")
    println(s"Targets (${targetIds.length}): ${targetIds.mkString(", ")}")

    val refCanonical = loadCanonical(referenceId)
    val referenceFull = refCanonical.mesh
    val reference = referenceFull.operations.decimate(Config.modelResolution)
    println(s"Reference has ${reference.pointSet.numberOfPoints} vertices (requested ${Config.modelResolution}).")

    println("\n=== Stage 1+2: landmark Procrustes + trimmed ICP (per target) ===")
    val alignedTargets: IndexedSeq[(String, TriangleMesh[_3D])] = targetIds.map { id =>
      val c = loadCanonical(id)
      val (aligned, _) = RigidAlign.landmarkThenIcp(c.mesh, c.lms, referenceFull, refCanonical.lms, Config.icpIterations)
      println(s"  $id: rigidly aligned onto $referenceId")
      id -> aligned
    }

    outDir.mkdirs()
    val alignedTargetsDir = new File(outDir, "aligned_targets")
    alignedTargetsDir.mkdirs()
    MeshIO.writeMesh(referenceFull, new File(outDir, "reference_full.vtk")).get
    MeshIO.writeMesh(reference, new File(outDir, "reference.vtk")).get
    alignedTargets.foreach { case (id, mesh) => MeshIO.writeMesh(mesh, new File(alignedTargetsDir, s"$id.vtk")).get }
    println(s"Wrote reference + ${alignedTargets.length} rigid-aligned (non-registered) targets to ${outDir.getPath}")

    println("\n=== Stage 3: GP non-rigid ICP -- Case A then Case B ===")
    val resultsA = runCase(caseA, reference, alignedTargets)
    val resultsB = runCase(caseB, reference, alignedTargets)

    def saveCase(kernelCase: KernelCase, results: IndexedSeq[(String, TriangleMesh[_3D], Metrics.SurfaceStats)]): (String, String) = {
      val caseDir = new File(outDir, kernelCase.folderName)
      caseDir.mkdirs()
      results.foreach { case (id, mesh, _) => MeshIO.writeMesh(mesh, new File(caseDir, s"$id.vtk")).get }
      val rows = results.map { case (id, _, s) => id -> s }
      writeCsv(new File(outDir, s"${kernelCase.folderName}_metrics.csv"), rows)
      printTableAndFindBestWorst(kernelCase.label, rows)
    }

    val (bestA, worstA) = saveCase(caseA, resultsA)
    val (bestB, worstB) = saveCase(caseB, resultsB)

    println(s"\nWrote case_a/, case_b/, case_a_metrics.csv, case_b_metrics.csv to ${outDir.getPath}")

    if (Config.showUi) {
      val ui = ScalismoUI()
      val refGroup = ui.createGroup("reference")
      ui.show(refGroup, reference, s"$referenceId (reference)")

      val rigidGroup = ui.createGroup("rigid-aligned (non-registered)")
      alignedTargets.foreach { case (id, mesh) => ui.show(rigidGroup, mesh, s"$id (rigid, non-registered)") }

      def showCaseGroups(
          caseLabel: String,
          results: IndexedSeq[(String, TriangleMesh[_3D], Metrics.SurfaceStats)],
          bestId: String,
          worstId: String
      ): Unit = {
        val targetById = alignedTargets.toMap
        val allGroup = ui.createGroup(s"$caseLabel - all specimens")
        results.foreach { case (id, mesh, _) =>
          ui.show(allGroup, mesh, s"$id (registered)")
          ui.show(allGroup, targetById(id), s"$id (real target)")
        }
        val (bestMesh, _, _) = results.find(_._1 == bestId).get
        val bestGroup = ui.createGroup(s"$caseLabel - BEST ($bestId)")
        ui.show(bestGroup, bestMesh, s"$bestId (registered)")
        ui.show(bestGroup, targetById(bestId), s"$bestId (real target)")

        val (worstMesh, _, _) = results.find(_._1 == worstId).get
        val worstGroup = ui.createGroup(s"$caseLabel - WORST ($worstId)")
        ui.show(worstGroup, worstMesh, s"$worstId (registered)")
        ui.show(worstGroup, targetById(worstId), s"$worstId (real target)")
      }

      showCaseGroups(caseA.label, resultsA, bestA, worstA)
      showCaseGroups(caseB.label, resultsB, bestB, worstB)

      println("\nScalismo-UI window opened: 'reference', 'rigid-aligned (non-registered)', and per-case " +
        "'all specimens' / 'BEST' / 'WORST' groups for Case A and Case B. Toggle each group's meshes on/off " +
        "to compare registered vs. real target -- hide all but one group at a time for a clean read. " +
        "Close the window when done.")
    } else {
      println("SCAPULA_UI=false -- skipping the interactive viewer.")
    }
  }
}
