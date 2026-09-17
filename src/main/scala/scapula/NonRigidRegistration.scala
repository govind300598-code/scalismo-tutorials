package scapula

import breeze.linalg.DenseVector
import scalismo.common.Field
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, LBFGSOptimizer}
import scalismo.registration.{GaussianProcessTransformationSpace, L2Regularizer, MeanSquaresMetric, Registration}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess}
import scalismo.io.MeshIO
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

// ============================================================
//  Stage 2 – Multiscale GP Non-Rigid Registration (scalismo 0.92)
// ============================================================
//
//  Pipeline per specimen
//  ─────────────────────
//  1. Mirror right-side scapulae into left space
//  2. Rigid alignment (landmark Procrustes + trimmed ICP) to reference
//  3. Show rigid-aligned surface in Scalismo UI ("unregistered_rigid" group)
//  4. 4-pass GP-ICP cascade (coarse → fine regularisation)
//  5. Show registered surface in Scalismo UI ("registered_nonrigid" group)
//  6. Compute pre/post surface-distance + point-to-point metrics
//  7. Save registered STL to date-stamped output folder
//  8. Write CSV metric table + kernel config + summary text
//
//  Multiscale kernel design for scapulae (bone ~150-180 mm long):
//  ──────────────────────────────────────────────────────────────
//  Coarse  σ=90 mm  scale=30  → global shape / spine / body
//  Mid     σ=35 mm  scale=15  → glenoid bowl / acromion
//  Fine    σ=15 mm  scale= 8  → rim / coracoid tip
//  VFine   σ= 6 mm  scale= 4  → articular surface texture
//
//  Four levels give the LRGP enough modes to resolve the glenoid
//  without globally distorting the rest of the bone.
// ============================================================

object NonRigidRegistration {

  // ----------------------------------------------------------
  //  Kernel configuration
  // ----------------------------------------------------------
  final case class KernelSpec(sigma: Double, scale: Double, tag: String)

  val kernelSpecs: Seq[KernelSpec] = Seq(
    KernelSpec(sigma = 90.0, scale = 30.0, tag = "Coarse"),
    KernelSpec(sigma = 35.0, scale = 15.0, tag = "Mid"),
    KernelSpec(sigma = 15.0, scale =  8.0, tag = "Fine"),
    KernelSpec(sigma =  6.0, scale =  4.0, tag = "VFine")
  )

  // ----------------------------------------------------------
  //  ICP cascade: 4 passes, decreasing regularisation
  // ----------------------------------------------------------
  final case class RegPass(regWeight: Double, iters: Int, nPoints: Int, label: String)

  val regPasses: Seq[RegPass] = Seq(
    RegPass(regWeight = 1e-1, iters =  60, nPoints =  500, label = "pass1-coarse"),
    RegPass(regWeight = 1e-2, iters =  60, nPoints = 1500, label = "pass2-mid"),
    RegPass(regWeight = 1e-4, iters =  80, nPoints = 4000, label = "pass3-fine"),
    RegPass(regWeight = 1e-6, iters = 100, nPoints = 8000, label = "pass4-vfine")
  )

  // ----------------------------------------------------------
  //  Build the low-rank GP prior on the reference mesh
  // ----------------------------------------------------------
  def buildLRGP(
    reference:   TriangleMesh[_3D],
    specs:       Seq[KernelSpec],
    relativeTol: Double
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {

    // Weighted sum of Gaussian kernels (scalar); each term: exp(-||x-y||^2 / 2*sigma^2) * scale
    val scalarKernel = specs
      .map(s => GaussianKernel3D(s.sigma, scalingFactor = s.scale))
      .reduce(_ + _)

    // Promote to a 3×3 diagonal (isotropic) matrix-valued kernel
    val kernel = DiagonalKernel3D(scalarKernel, outputDim = 3)

    // Zero-mean GP (standard prior for non-rigid deformations)
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp = GaussianProcess(zeroMean, kernel)

    // Pivoted Cholesky low-rank approximation evaluated at reference vertices
    LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = relativeTol,
      interpolator = NearestNeighborInterpolator3D()
    )
  }

  // ----------------------------------------------------------
  //  Reconstruct a deformed mesh from GP coefficients
  // ----------------------------------------------------------
  def gpInstance(
    reference:  TriangleMesh[_3D],
    lrgp:       LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    coeffs:     DenseVector[Double]
  ): TriangleMesh[_3D] = {
    val deformField = lrgp.instance(coeffs)
    reference.transform(pt => pt + deformField(pt))
  }

  // ----------------------------------------------------------
  //  One optimisation pass
  // ----------------------------------------------------------
  def runPass(
    lrgp:       LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference:  TriangleMesh[_3D],
    target:     TriangleMesh[_3D],
    initCoeffs: DenseVector[Double],
    p:          RegPass
  )(implicit rng: Random): DenseVector[Double] = {
    val tSpace     = GaussianProcessTransformationSpace(lrgp)
    val fixedImg   = reference.operations.toDistanceImage
    val movingImg  = target.operations.toDistanceImage
    val sampler    = FixedPointsUniformMeshSampler3D(reference, p.nPoints)
    val metric     = MeanSquaresMetric(fixedImg, movingImg, tSpace, sampler)
    val optimizer  = LBFGSOptimizer(maxNumberOfIterations = p.iters)
    val reg        = Registration(metric, L2Regularizer(tSpace), p.regWeight, optimizer)
    reg.iterator(initCoeffs).toSeq.last.parameters
  }

  // ----------------------------------------------------------
  //  Full 4-pass cascade → registered mesh
  // ----------------------------------------------------------
  def registerOne(
    lrgp:      LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference: TriangleMesh[_3D],
    target:    TriangleMesh[_3D],
    label:     String
  )(implicit rng: Random): (TriangleMesh[_3D], DenseVector[Double]) = {
    println(s"  [NR] Registering $label  (GP rank = ${lrgp.rank})")
    var coeffs = DenseVector.zeros[Double](lrgp.rank)
    regPasses.foreach { p =>
      print(s"      ${p.label} ... ")
      coeffs = runPass(lrgp, reference, target, coeffs, p)
      println("done")
    }
    (gpInstance(reference, lrgp, coeffs), coeffs)
  }

  // ----------------------------------------------------------
  //  Metric row
  // ----------------------------------------------------------
  final case class RegMetrics(
    modelId:   String,
    preMean:   Double, preRms:  Double, preHd95:  Double, preHd:  Double,
    postMean:  Double, postRms: Double, postHd95: Double, postHd: Double,
    p2pMean:   Double, p2pRms:  Double, p2pHd95:  Double,
    gpRank:    Int
  ) {
    def csvRow: String =
      Seq(
        modelId,
        f"$preMean%.3f",  f"$preRms%.3f",  f"$preHd95%.3f",  f"$preHd%.3f",
        f"$postMean%.3f", f"$postRms%.3f", f"$postHd95%.3f", f"$postHd%.3f",
        f"$p2pMean%.3f",  f"$p2pRms%.3f",  f"$p2pHd95%.3f",
        gpRank.toString
      ).mkString(",")
  }

  val csvHeader: String =
    "modelId," +
    "pre_mean_mm,pre_rms_mm,pre_HD95_mm,pre_HD_mm," +
    "post_mean_mm,post_rms_mm,post_HD95_mm,post_HD_mm," +
    "p2p_mean_mm,p2p_rms_mm,p2p_HD95_mm," +
    "gp_rank"

  // ----------------------------------------------------------
  //  Console tables
  // ----------------------------------------------------------
  def printKernelTable(specs: Seq[KernelSpec], rank: Int, tol: Double): Unit = {
    val notes = Seq("global shape / spine / body", "glenoid bowl / acromion", "rim / coracoid tip", "articular surface texture")
    println()
    println("=" * 72)
    println("MULTISCALE GP KERNEL CONFIGURATION")
    println("=" * 72)
    println(f"${"Level"}%-10s  ${"σ (mm)"}%8s  ${"scale"}%7s  ${"note"}")
    println("-" * 72)
    specs.zip(notes).foreach { case (s, note) =>
      println(f"${s.tag}%-10s  ${s.sigma}%8.1f  ${s.scale}%7.1f  $note")
    }
    println("-" * 72)
    println(f"  Combined GP rank (Cholesky, tol=$tol): $rank")
    println("=" * 72)
  }

  def printCascadeTable(): Unit = {
    println()
    println("=" * 68)
    println("REGISTRATION CASCADE")
    println("=" * 68)
    println(f"${"Pass"}%-16s  ${"regWeight"}%12s  ${"iters"}%6s  ${"nPoints"}%8s")
    println("-" * 68)
    regPasses.foreach { p =>
      println(f"${p.label}%-16s  ${p.regWeight}%12.1e  ${p.iters}%6d  ${p.nPoints}%8d")
    }
    println("=" * 68)
  }

  def printMetricsTable(rows: Seq[RegMetrics]): Unit = {
    if (rows.isEmpty) return
    val w = 32
    println()
    println("=" * 130)
    println("NON-RIGID REGISTRATION METRICS  (all distances in mm)")
    println("=" * 130)
    println(
      f"${"Model"}%-${w}s  " +
      f"${"pre-mean"}%8s  ${"pre-RMS"}%8s  ${"pre-HD95"}%9s  ${"pre-HD"}%8s  " +
      f"${"post-mean"}%9s  ${"post-RMS"}%9s  ${"post-HD95"}%10s  ${"post-HD"}%9s  " +
      f"${"p2p-mean"}%9s  ${"p2p-RMS"}%8s  ${"p2p-HD95"}%9s  ${"rank"}%5s"
    )
    println("-" * 130)
    rows.foreach { r =>
      println(
        f"${r.modelId}%-${w}s  " +
        f"${r.preMean}%8.2f  ${r.preRms}%8.2f  ${r.preHd95}%9.2f  ${r.preHd}%8.2f  " +
        f"${r.postMean}%9.2f  ${r.postRms}%9.2f  ${r.postHd95}%10.2f  ${r.postHd}%9.2f  " +
        f"${r.p2pMean}%9.2f  ${r.p2pRms}%8.2f  ${r.p2pHd95}%9.2f  ${r.gpRank}%5d"
      )
    }
    println("-" * 130)
    def avg(f: RegMetrics => Double) = rows.map(f).sum / rows.length
    println(
      f"${"MEAN"}%-${w}s  " +
      f"${avg(_.preMean)}%8.2f  ${avg(_.preRms)}%8.2f  ${avg(_.preHd95)}%9.2f  ${avg(_.preHd)}%8.2f  " +
      f"${avg(_.postMean)}%9.2f  ${avg(_.postRms)}%9.2f  ${avg(_.postHd95)}%10.2f  ${avg(_.postHd)}%9.2f  " +
      f"${avg(_.p2pMean)}%9.2f  ${avg(_.p2pRms)}%8.2f  ${avg(_.p2pHd95)}%9.2f  ${rows.head.gpRank}%5d"
    )
    println("=" * 130)
  }

  // ----------------------------------------------------------
  //  Main
  // ----------------------------------------------------------
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"))
    val runTag    = s"nonrigid_$timestamp"

    val dir    = Config.dataDir
    val outDir = new File(Config.outDir, runTag)
    val regDir = new File(outDir, "registered_meshes")
    val logDir = new File(outDir, "logs")
    Seq(outDir, regDir, logDir).foreach(_.mkdirs())

    println(s"===============================================================")
    println(s" Non-Rigid Registration  |  $timestamp")
    println(s"===============================================================")
    println(s" Data   : ${dir.getAbsolutePath}")
    println(s" Output : ${outDir.getAbsolutePath}")

    // ── Load landmarks and find all specimens ──────────────────────────────
    val csv = ScapulaData.csvFile(dir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("  WARN: landmark columns resolved by fallback offsets – verify CSV")

    val allSpecimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    require(allSpecimens.nonEmpty, s"No specimens with landmarks in ${dir.getAbsolutePath}")
    println(s" Specimens with landmarks: ${allSpecimens.length}")

    // ── Choose reference: first left-side specimen ─────────────────────────
    val refSpec = allSpecimens.find(!_.isRight).getOrElse(allSpecimens.head)
    val refRaw  = ScapulaData.loadMesh(refSpec.file)
    val reference = refRaw.operations.decimate(Config.modelResolution)
    val refLms    = landmarks(refSpec.modelId)
    println(s" Reference: ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices after decimation)")

    // ── Build multiscale LRGP prior ────────────────────────────────────────
    println("\nBuilding multiscale GP prior …")
    val lrgp = buildLRGP(reference, kernelSpecs, Config.gpRelativeTolerance)
    println(s"  GP rank = ${lrgp.rank}")

    printKernelTable(kernelSpecs, lrgp.rank, Config.gpRelativeTolerance)
    printCascadeTable()

    // Save kernel config
    val kw = new PrintWriter(new File(logDir, "kernel_config.csv"))
    kw.println("level,sigma_mm,scale,relativeTol,gp_rank")
    kernelSpecs.foreach(s => kw.println(s"${s.tag},${s.sigma},${s.scale},${Config.gpRelativeTolerance},${lrgp.rank}"))
    kw.close()

    // ── Scalismo UI ───────────────────────────────────────────────────────
    val ui = if (Config.showUi) Some(ScalismoUI()) else None

    val grpRef   = ui.map(_.createGroup("reference"))
    val grpUnreg = ui.map(_.createGroup("unregistered_rigid"))
    val grpReg   = ui.map(_.createGroup("registered_nonrigid"))
    val grpGp    = ui.map(_.createGroup("gp_prior_samples"))

    // Show reference
    ui.zip(grpRef).foreach { case (u, g) => u.show(g, reference, "reference") }

    // Show 3 GP prior samples so the user can see the deformation modes
    ui.zip(grpGp).foreach { case (u, g) =>
      val sample1 = gpInstance(reference, lrgp, DenseVector.fill(lrgp.rank)(0.5))
      val sample2 = gpInstance(reference, lrgp, DenseVector.fill(lrgp.rank)(-0.5))
      u.show(g, sample1, "gp_sample_+0.5")
      u.show(g, sample2, "gp_sample_-0.5")
    }

    // ── Register all other specimens ──────────────────────────────────────
    val targets    = allSpecimens.filterNot(_.modelId == refSpec.modelId)
    val allMetrics = scala.collection.mutable.ListBuffer[RegMetrics]()

    val mw = new PrintWriter(new File(logDir, "registration_metrics.csv"))
    mw.println(csvHeader)

    targets.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n[${idx + 1}/${targets.length}] ${spec.modelId}")

      // Mirror right → left space
      val rawMesh = ScapulaData.loadMesh(spec.file)
      val rawLms  = landmarks(spec.modelId)
      val (oriented, orientedLms) =
        if (spec.isRight)
          (ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
        else
          (rawMesh, rawLms)

      // Rigid alignment
      val (rigid, _) = RigidAlign.landmarkThenIcp(
        oriented, orientedLms, reference, refLms, icpIterations = Config.icpIterations
      )

      // Show rigid-only surface
      ui.zip(grpUnreg).foreach { case (u, g) => u.show(g, rigid, s"${spec.modelId}_rigid") }

      // Pre-registration metrics (rigid-aligned target vs reference)
      val preStats = Metrics.symmetric(rigid, reference)

      // Non-rigid registration
      val (regMesh, _) = registerOne(lrgp, reference, rigid, spec.modelId)

      // Show registered surface
      ui.zip(grpReg).foreach { case (u, g) => u.show(g, regMesh, s"${spec.modelId}_reg") }

      // Post-registration surface metrics (registered vs rigid target)
      val postStats = Metrics.symmetric(regMesh, rigid)

      // Point-to-point (regMesh is in reference topology → compare to reference vertex-by-vertex)
      val p2pDists = Metrics.correspondingDistances(regMesh, reference)
      val p2pMean  = p2pDists.sum / p2pDists.length
      val p2pRms   = math.sqrt(p2pDists.map(x => x * x).sum / p2pDists.length)
      val p2pHd95  = Metrics.percentile(p2pDists, 0.95)

      println(f"    PRE  (rigid vs ref)  : ${preStats.render}")
      println(f"    POST (reg vs target) : ${postStats.render}")
      println(f"    P2P  (reg vs ref)    : mean=${p2pMean}%.2f mm  RMS=${p2pRms}%.2f  HD95=${p2pHd95}%.2f")

      val row = RegMetrics(
        modelId  = spec.modelId,
        preMean  = preStats.mean,  preRms  = preStats.rms,
        preHd95  = preStats.hd95,  preHd   = preStats.hd,
        postMean = postStats.mean, postRms = postStats.rms,
        postHd95 = postStats.hd95, postHd  = postStats.hd,
        p2pMean  = p2pMean, p2pRms = p2pRms, p2pHd95 = p2pHd95,
        gpRank   = lrgp.rank
      )
      allMetrics += row
      mw.println(row.csvRow)
      mw.flush()

      // Save registered STL
      val outFile = new File(regDir, s"${spec.modelId}_registered.stl")
      MeshIO.writeMesh(regMesh, outFile)
        .recover { case ex => println(s"  WARN: failed to write ${outFile.getName}: ${ex.getMessage}") }
    }

    mw.close()

    // ── Print final metrics table ─────────────────────────────────────────
    val rows = allMetrics.toSeq
    printMetricsTable(rows)

    // ── Save summary report ───────────────────────────────────────────────
    val rw = new PrintWriter(new File(logDir, "summary.txt"))
    rw.println(s"Run: $runTag")
    rw.println(s"Data: ${dir.getAbsolutePath}")
    rw.println(s"Reference: ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices)")
    rw.println(s"Specimens registered: ${rows.length}")
    rw.println()
    rw.println("KERNEL CONFIGURATION")
    kernelSpecs.foreach(s => rw.println(f"  ${s.tag}%-10s  sigma=${s.sigma}%.1f mm  scale=${s.scale}%.1f"))
    rw.println(s"  GP rank: ${lrgp.rank}  (relativeTol=${Config.gpRelativeTolerance})")
    rw.println()
    rw.println("REGISTRATION CASCADE")
    regPasses.foreach(p => rw.println(s"  ${p.label}: regWeight=${p.regWeight}  iters=${p.iters}  nPoints=${p.nPoints}"))
    if (rows.nonEmpty) {
      rw.println()
      def avg(f: RegMetrics => Double) = rows.map(f).sum / rows.length
      rw.println(f"MEAN PRE-RIGID    mean=${avg(_.preMean)}%.2f  RMS=${avg(_.preRms)}%.2f  HD95=${avg(_.preHd95)}%.2f  HD=${avg(_.preHd)}%.2f mm")
      rw.println(f"MEAN POST-NONRIG  mean=${avg(_.postMean)}%.2f  RMS=${avg(_.postRms)}%.2f  HD95=${avg(_.postHd95)}%.2f  HD=${avg(_.postHd)}%.2f mm")
      rw.println(f"MEAN P2P (vs ref) mean=${avg(_.p2pMean)}%.2f  RMS=${avg(_.p2pRms)}%.2f  HD95=${avg(_.p2pHd95)}%.2f mm")
    }
    rw.close()

    println(s"\nAll outputs written to: ${outDir.getAbsolutePath}")
    println(s"  registered_meshes/  – ${rows.length} registered STL files")
    println(s"  logs/registration_metrics.csv")
    println(s"  logs/kernel_config.csv")
    println(s"  logs/summary.txt")

    if (Config.showUi) {
      println("\nScalismo UI groups")
      println("  reference             – decimated reference scapula")
      println("  gp_prior_samples      – two GP samples (±0.5 std); hide this group when done")
      println("  unregistered_rigid    – rigid-aligned targets BEFORE non-rigid (pink/red)")
      println("  registered_nonrigid   – GP-registered surfaces (green)")
      println("\nKeep the window open to inspect. Close to exit.")
    }
  }
}
