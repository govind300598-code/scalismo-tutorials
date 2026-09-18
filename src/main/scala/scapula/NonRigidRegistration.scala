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
//  Per-specimen pipeline
//  ─────────────────────
//  1. Mirror right-side scapulae into left space
//  2. Rigid alignment (landmark Procrustes + trimmed ICP)
//  3. Show rigid-aligned surface in Scalismo UI
//  4. 4-pass GP-ICP cascade (coarse → fine regularisation)
//  5. Compute surface-distance metrics (mean, RMS, Chamfer, HD)
//     AND per-landmark distance metrics (pre and post)
//  6. Show registered surface in Scalismo UI
//  7. Save STL + CSV metrics + kernel table + summary
//
//  Multiscale kernel design  (Dennis Madsen's calibrated scheme, L≈150 mm)
//  ─────────────────────────────────────────────────────────────────────────
//  Coarse  σ=L/2  =75 mm  scale=15  whole-bone shape / spine / body
//  Mid     σ=L/5  =30 mm  scale=10  glenoid bowl / acromion / coracoid region
//  Fine    σ=L/10 =15 mm  scale= 5  glenoid rim / coracoid tip detail
//  σ ratios (L/2, L/5, L/10) and scaleFactor step of 5 follow the anatomical
//  D/2–D/3 rule; scale accordingly if your scapulae differ from 150 mm.
// ============================================================

object NonRigidRegistration {

  // ----------------------------------------------------------
  //  Dataset selection
  //  The six specimens used in the validation run (Table 1).
  //  005_F_67_R is the fixed reference; the other five are targets.
  //  Set to None to process every specimen in the data directory.
  // ----------------------------------------------------------
  val selectedSpecimens: Option[Set[String]] = Some(Set(
    "001_M_64_L",
    "002_M_56_L",
    "005_F_67_R",
    "006_F_60_R",
    "007_M_26_L",
    "008_F_73_L"
  ))
  val fixedReferenceId: Option[String] = Some("005_F_67_R")

  // ----------------------------------------------------------
  //  Kernel configuration
  // ----------------------------------------------------------
  final case class KernelSpec(sigma: Double, scale: Double, tag: String)

  // Dennis Madsen's 3-term scheme: σ = L/2, L/5, L/10  (L≈150 mm for scapula)
  val kernelSpecs: Seq[KernelSpec] = Seq(
    KernelSpec(sigma = 75.0, scale = 15.0, tag = "Coarse"),
    KernelSpec(sigma = 30.0, scale = 10.0, tag = "Mid"),
    KernelSpec(sigma = 15.0, scale =  5.0, tag = "Fine")
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
    val scalarKernel = specs
      .map(s => GaussianKernel3D(s.sigma, scalingFactor = s.scale))
      .reduce(_ + _)
    val kernel   = DiagonalKernel3D(scalarKernel, outputDim = 3)
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp       = GaussianProcess(zeroMean, kernel)
    LowRankGaussianProcess.approximateGPCholesky(
      reference, gp,
      relativeTolerance = relativeTol,
      interpolator = NearestNeighborInterpolator3D()
    )
  }

  // ----------------------------------------------------------
  //  Reconstruct a deformed mesh from GP coefficients
  // ----------------------------------------------------------
  def gpInstance(
    reference: TriangleMesh[_3D],
    lrgp:      LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    coeffs:    DenseVector[Double]
  ): TriangleMesh[_3D] = {
    val def_ = lrgp.instance(coeffs)
    reference.transform(pt => pt + def_(pt))
  }

  // ----------------------------------------------------------
  //  One ICP optimisation pass
  // ----------------------------------------------------------
  def runPass(
    lrgp:       LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference:  TriangleMesh[_3D],
    target:     TriangleMesh[_3D],
    initCoeffs: DenseVector[Double],
    p:          RegPass
  )(implicit rng: Random): DenseVector[Double] = {
    val tSpace    = GaussianProcessTransformationSpace(lrgp)
    val fixedImg  = reference.operations.toDistanceImage
    val movingImg = target.operations.toDistanceImage
    val sampler   = FixedPointsUniformMeshSampler3D(reference, p.nPoints)
    val metric    = MeanSquaresMetric(fixedImg, movingImg, tSpace, sampler)
    val optimizer = LBFGSOptimizer(maxNumberOfIterations = p.iters)
    val reg       = Registration(metric, L2Regularizer(tSpace), p.regWeight, optimizer)
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
    println(s"  [NR] $label  (GP rank = ${lrgp.rank})")
    var coeffs = DenseVector.zeros[Double](lrgp.rank)
    regPasses.foreach { p =>
      print(s"      ${p.label} ... ")
      coeffs = runPass(lrgp, reference, target, coeffs, p)
      println("done")
    }
    (gpInstance(reference, lrgp, coeffs), coeffs)
  }

  // ----------------------------------------------------------
  //  Per-landmark distances
  // ----------------------------------------------------------

  // Distance between two sets of landmarks matched by id (e.g. rigid-aligned lms vs ref lms)
  def lmDistances(
    aLms: IndexedSeq[Landmark[_3D]],
    bLms: IndexedSeq[Landmark[_3D]]
  ): Map[String, Double] = {
    val byId = bLms.map(l => l.id -> l.point).toMap
    aLms.flatMap(l => byId.get(l.id).map(p => l.id -> (l.point - p).norm)).toMap
  }

  // After non-rigid registration, each reference landmark vertex has moved.
  // Report how far that moved vertex is from the corresponding rigid-target landmark.
  def lmErrorOnMesh(
    reference:  TriangleMesh[_3D],
    regMesh:    TriangleMesh[_3D],
    refLms:     IndexedSeq[Landmark[_3D]],
    targetLms:  IndexedSeq[Landmark[_3D]]
  ): Map[String, Double] = {
    val byId = targetLms.map(l => l.id -> l.point).toMap
    refLms.flatMap { rl =>
      byId.get(rl.id).map { tp =>
        val vid      = reference.pointSet.findClosestPoint(rl.point).id
        val regPoint = regMesh.pointSet.point(vid)
        rl.id -> (regPoint - tp).norm
      }
    }.toMap
  }

  // ----------------------------------------------------------
  //  Metric row
  // ----------------------------------------------------------
  val lmNames: Seq[String] = ScapulaData.landmarkNames.toSeq

  final case class RegMetrics(
    modelId:     String,
    preMean:     Double, preRms:  Double, preCd:  Double, preHd:  Double,
    postMean:    Double, postRms: Double, postCd: Double, postHd: Double,
    p2pMean:     Double, p2pRms:  Double, p2pCd:  Double,
    preLmDists:  Map[String, Double],
    postLmDists: Map[String, Double],
    gpRank:      Int
  ) {
    private def lmCols(m: Map[String, Double]) =
      lmNames.map(n => f"${m.getOrElse(n, Double.NaN)}%.3f").mkString(",")

    def csvRow: String =
      Seq(
        modelId,
        f"$preMean%.3f",  f"$preRms%.3f",  f"$preCd%.3f",  f"$preHd%.3f",
        f"$postMean%.3f", f"$postRms%.3f", f"$postCd%.3f", f"$postHd%.3f",
        f"$p2pMean%.3f",  f"$p2pRms%.3f",  f"$p2pCd%.3f",
        lmCols(preLmDists), lmCols(postLmDists),
        gpRank.toString
      ).mkString(",")
  }

  def csvHeader: String = {
    val preLmCols  = lmNames.map(n => s"pre_lm_${n}_mm").mkString(",")
    val postLmCols = lmNames.map(n => s"post_lm_${n}_mm").mkString(",")
    s"modelId," +
    s"pre_mean_mm,pre_rms_mm,pre_CD_mm,pre_HD_mm," +
    s"post_mean_mm,post_rms_mm,post_CD_mm,post_HD_mm," +
    s"p2p_mean_mm,p2p_rms_mm,p2p_CD_mm," +
    s"$preLmCols,$postLmCols," +
    s"gp_rank"
  }

  // ----------------------------------------------------------
  //  Console tables
  // ----------------------------------------------------------
  def printKernelTable(specs: Seq[KernelSpec], rank: Int, tol: Double): Unit = {
    // Notes indexed to match the 3-term Madsen scheme; extra terms get a generic label
    val baseNotes = Seq(
      "whole-bone shape / spine / body  (σ=L/2)",
      "glenoid bowl / acromion / coracoid  (σ=L/5)",
      "glenoid rim / coracoid tip detail  (σ=L/10)"
    )
    val notes = specs.zipWithIndex.map { case (_, i) => baseNotes.applyOrElse(i, (_: Int) => "additional term") }
    println()
    println("=" * 72)
    println("MULTISCALE GP KERNEL CONFIGURATION")
    println("=" * 72)
    println(f"${"Level"}%-10s  ${"σ (mm)"}%8s  ${"scale"}%7s  note")
    println("-" * 72)
    specs.zip(notes).foreach { case (s, note) =>
      println(f"${s.tag}%-10s  ${s.sigma}%8.1f  ${s.scale}%7.1f  $note")
    }
    println("-" * 72)
    println(f"  GP rank (Cholesky, tol=$tol): $rank")
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
    val w        = 32
    val lmHeader = lmNames.map(n => f"$n%6s").mkString("  ")

    // Identify best (lowest post-CD) and worst (highest post-CD) specimens
    val bestId  = rows.minBy(_.postCd).modelId
    val worstId = rows.maxBy(_.postCd).modelId

    println()
    println("=" * 150)
    println("NON-RIGID REGISTRATION METRICS  (all distances in mm)  [BEST=lowest post-CD  WORST=highest post-CD]")
    println("=" * 150)
    println(
      f"${"Model"}%-${w}s  " +
      f"${"pre-mean"}%8s  ${"pre-CD"}%8s  ${"pre-HD"}%8s  " +
      f"${"post-mean"}%9s  ${"post-CD"}%8s  ${"post-HD"}%9s  " +
      f"${"p2p-mean"}%9s  ${"p2p-CD"}%8s  " +
      s"pre-lm[$lmHeader]  post-lm[$lmHeader]  rank"
    )
    println("-" * 150)
    def lmStr(m: Map[String, Double]) =
      lmNames.map(n => f"${m.getOrElse(n, Double.NaN)}%6.2f").mkString("  ")
    rows.foreach { r =>
      val tag = if (r.modelId == bestId) " <BEST" else if (r.modelId == worstId) " <WORST" else ""
      println(
        f"${r.modelId}%-${w}s  " +
        f"${r.preMean}%8.2f  ${r.preCd}%8.2f  ${r.preHd}%8.2f  " +
        f"${r.postMean}%9.2f  ${r.postCd}%8.2f  ${r.postHd}%9.2f  " +
        f"${r.p2pMean}%9.2f  ${r.p2pCd}%8.2f  " +
        s"[${lmStr(r.preLmDists)}]  [${lmStr(r.postLmDists)}]  ${r.gpRank}$tag"
      )
    }
    println("-" * 150)
    def avg(f: RegMetrics => Double) = rows.map(f).sum / rows.length
    def avgLm(get: RegMetrics => Map[String, Double]) = lmNames.map { n =>
      val vs = rows.flatMap(r => get(r).get(n))
      if (vs.isEmpty) Double.NaN else vs.sum / vs.length
    }
    println(
      f"${"MEAN"}%-${w}s  " +
      f"${avg(_.preMean)}%8.2f  ${avg(_.preCd)}%8.2f  ${avg(_.preHd)}%8.2f  " +
      f"${avg(_.postMean)}%9.2f  ${avg(_.postCd)}%8.2f  ${avg(_.postHd)}%9.2f  " +
      f"${avg(_.p2pMean)}%9.2f  ${avg(_.p2pCd)}%8.2f  " +
      s"[${avgLm(_.preLmDists).map(v => f"$v%6.2f").mkString("  ")}]  " +
      s"[${avgLm(_.postLmDists).map(v => f"$v%6.2f").mkString("  ")}]  ${rows.head.gpRank}"
    )
    println("=" * 150)
    println(s"  BEST registration : $bestId  (post-CD = ${rows.minBy(_.postCd).postCd}%.2f mm)")
    println(s"  WORST registration: $worstId  (post-CD = ${rows.maxBy(_.postCd).postCd}%.2f mm)")
    println("=" * 150)
  }

  def printLmDistSummary(rows: Seq[RegMetrics]): Unit = {
    if (rows.isEmpty) return
    println()
    println("=" * 62)
    println("LANDMARK DISTANCE SUMMARY  (mm, mean across specimens)")
    println("=" * 62)
    println(f"${"Landmark"}%-8s  ${"pre-rigid"}%10s  ${"post-nonrig"}%12s  ${"improvement"}%12s")
    println("-" * 62)
    lmNames.foreach { n =>
      val pre  = rows.flatMap(_.preLmDists.get(n))
      val post = rows.flatMap(_.postLmDists.get(n))
      if (pre.nonEmpty && post.nonEmpty) {
        val preMean  = pre.sum  / pre.length
        val postMean = post.sum / post.length
        println(f"$n%-8s  $preMean%10.2f  $postMean%12.2f  ${preMean - postMean}%12.2f")
      }
    }
    println("=" * 62)
  }

  // ----------------------------------------------------------
  //  Main
  // ----------------------------------------------------------
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // Millisecond timestamp + 4-hex random tag → unique across parallel runs
    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss-SSS"))
    val runTag    = s"nr_${timestamp}_${"%04x".format(rng.scalaRandom.nextInt(0xffff))}"

    val dir    = Config.dataDir
    val outDir = new File(Config.outDir, runTag)
    val regDir = new File(outDir, "registered_meshes")
    val logDir = new File(outDir, "logs")
    Seq(outDir, regDir, logDir).foreach(_.mkdirs())

    println("================================================================")
    println(s" Multiscale GP Non-Rigid Registration")
    println(s" $timestamp")
    println("================================================================")
    println(s" Data   : ${dir.getAbsolutePath}")
    println(s" Output : ${outDir.getAbsolutePath}")

    // ── Landmarks and specimens ────────────────────────────────────────────
    val csv = ScapulaData.csvFile(dir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("  WARN: landmark columns resolved by fallback offsets – verify CSV")

    val allWithLandmarks = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    val allSpecimens = selectedSpecimens match {
      case None       => allWithLandmarks
      case Some(ids)  =>
        val filtered = allWithLandmarks.filter(s => ids.contains(s.modelId))
        val missing  = ids -- filtered.map(_.modelId).toSet
        if (missing.nonEmpty)
          println(s"  WARN: selected specimens not found in data dir: ${missing.mkString(", ")}")
        filtered
    }
    require(allSpecimens.nonEmpty, s"No specimens with landmarks in ${dir.getAbsolutePath}")
    println(s" Specimens selected: ${allSpecimens.length}  (of ${allWithLandmarks.length} total with landmarks)")
    allSpecimens.foreach(s => println(s"   ${if (s.isRight) "R" else "L"} ${s.modelId}"))

    // ── Reference ─────────────────────────────────────────────────────────
    val refSpec = fixedReferenceId match {
      case Some(id) =>
        allSpecimens.find(_.modelId == id).getOrElse {
          throw new RuntimeException(s"Fixed reference '$id' not found among selected specimens")
        }
      case None =>
        allSpecimens.find(!_.isRight).getOrElse(allSpecimens.head)
    }
    val refRaw    = ScapulaData.loadMesh(refSpec.file)
    val reference = refRaw.operations.decimate(Config.modelResolution)
    val refLms    = landmarks(refSpec.modelId)
    println(s" Reference: ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices)")

    // ── Build LRGP prior ──────────────────────────────────────────────────
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
    val ui       = if (Config.showUi) Some(ScalismoUI()) else None
    val grpRef   = ui.map(_.createGroup("reference"))
    val grpUnreg = ui.map(_.createGroup("unregistered_rigid"))
    val grpReg   = ui.map(_.createGroup("registered_nonrigid"))

    ui.zip(grpRef).foreach { case (u, g) =>
      u.show(g, reference, "reference")
      refLms.foreach(lm => u.show(g, lm, lm.id))
    }

    // ── Registration loop ─────────────────────────────────────────────────
    val targets    = allSpecimens.filterNot(_.modelId == refSpec.modelId)
    val allMetrics = scala.collection.mutable.ListBuffer[RegMetrics]()
    // Keep registered meshes so we can highlight best/worst in the UI after the loop
    val regMeshMap = scala.collection.mutable.Map[String, TriangleMesh[_3D]]()

    val mw = new PrintWriter(new File(logDir, "registration_metrics.csv"))
    mw.println(csvHeader)

    targets.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n[${idx + 1}/${targets.length}] ${spec.modelId}")
      println(s"  ── pre-processing ──────────────────────────────────────────")

      // ── 0. Mirror right → left space ────────────────────────────────────
      val rawMesh = ScapulaData.loadMesh(spec.file)
      val rawLms  = landmarks(spec.modelId)
      val (oriented, orientedLms) =
        if (spec.isRight) {
          println(s"  [0] Mirror right → left")
          (ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
        } else {
          println(s"  [0] Left side – no mirror needed")
          (rawMesh, rawLms)
        }

      // ── STEP 1. Landmark-based rigid registration (Procrustes) ──────────
      // Finds the best rigid transform that maps the 5 annotated landmarks
      // (GC, TS, IA, PLA, AC) on the moving mesh onto the same landmarks on
      // the reference.  This gives a coarse but anatomically grounded pose.
      println(s"  [1] Landmark rigid registration (Procrustes, ${refLms.length} landmarks)")
      val lmTransform  = ScapulaData.rigidFromLandmarks(orientedLms, refLms)
      val afterLm      = oriented.transform(lmTransform)
      val afterLmLandmarks = orientedLms.map(lm => lm.copy(point = lmTransform(lm.point)))
      val lmStats      = Metrics.symmetric(afterLm, reference)
      println(f"     after LM Procrustes : ${lmStats.render}")

      // ── STEP 2. Automatic trimmed ICP (rigid refinement) ─────────────────
      // Runs from the landmark-initialised pose.  Uses a spatially uniform
      // point sample so thin structures (acromion, coracoid) are not
      // systematically missed. Trims the worst 15 % of correspondences so
      // partially non-overlapping regions do not corrupt the rotation.
      println(s"  [2] Automatic trimmed ICP  (${Config.icpIterations} iters)")
      val rigid     = RigidAlign.rigidIcp(afterLm, reference, Config.icpIterations)
      // Carry the landmarks along with the ICP motion
      val icpMotion = scalismo.registration.LandmarkRegistration.rigid3DLandmarkRegistration(
        afterLm.pointSet.points.zip(rigid.pointSet.points).toIndexedSeq,
        center = scalismo.geometry.Point3D(0, 0, 0)
      )
      val rigidLms  = afterLmLandmarks.map(lm => lm.copy(point = icpMotion(lm.point)))
      val preStats  = Metrics.symmetric(rigid, reference)
      println(f"     after ICP           : ${preStats.render}")
      val preLmDists = lmDistances(rigidLms, refLms)
      println(s"     landmark errors     : " +
        ScapulaData.landmarkNames.map(n => f"$n=${preLmDists.getOrElse(n, Double.NaN)}%.2f").mkString("  "))

      // Show rigid surface + carried landmarks in UI
      ui.zip(grpUnreg).foreach { case (u, g) =>
        u.show(g, rigid, s"${spec.modelId}_rigid")
        rigidLms.foreach(lm => u.show(g, lm, s"${spec.modelId}_${lm.id}"))
      }

      // ── STEP 3. Multiscale GP non-rigid registration ──────────────────────
      // Uses the GP prior (3-term Madsen kernel) as the transformation space.
      // Runs 4 passes with decreasing regularisation weight so the deformation
      // moves from coarse global shape to fine local detail.
      println(s"  ── non-rigid (GP-ICP, 4 passes) ────────────────────────────")
      val (regMesh, _) = registerOne(lrgp, reference, rigid, spec.modelId)
      regMeshMap(spec.modelId) = regMesh

      // Show registered surface in UI
      ui.zip(grpReg).foreach { case (u, g) =>
        u.show(g, regMesh, s"${spec.modelId}_reg")
      }

      // Post-registration surface metrics (registered mesh vs the rigid target)
      val postStats   = Metrics.symmetric(regMesh, rigid)
      val postLmDists = lmErrorOnMesh(reference, regMesh, refLms, rigidLms)

      // Point-to-point metrics (regMesh shares reference topology — deformation magnitude)
      val p2pDists = Metrics.correspondingDistances(regMesh, reference)
      val p2pMean  = p2pDists.sum / p2pDists.length
      val p2pRms   = math.sqrt(p2pDists.map(x => x * x).sum / p2pDists.length)
      val p2pCd    = p2pMean * 2.0

      println(f"  ── results ─────────────────────────────────────────────────")
      println(f"     PRE  rigid   vs ref  : ${preStats.render}")
      println(f"     POST nonrig  vs rigid: ${postStats.render}")
      println(f"     P2P  nonrig  vs ref  : mean=${p2pMean}%.2f mm  RMSE=${p2pRms}%.2f mm  CD=${p2pCd}%.2f mm")
      println(s"     PRE  lm dists (mm)   : " +
        ScapulaData.landmarkNames.map(n => f"$n=${preLmDists.getOrElse(n, Double.NaN)}%.2f").mkString("  "))
      println(s"     POST lm dists (mm)   : " +
        ScapulaData.landmarkNames.map(n => f"$n=${postLmDists.getOrElse(n, Double.NaN)}%.2f").mkString("  "))

      val row = RegMetrics(
        modelId     = spec.modelId,
        preMean     = preStats.mean,   preRms  = preStats.rms,
        preCd       = preStats.chamfer, preHd  = preStats.hd,
        postMean    = postStats.mean,  postRms = postStats.rms,
        postCd      = postStats.chamfer, postHd = postStats.hd,
        p2pMean     = p2pMean, p2pRms = p2pRms, p2pCd = p2pCd,
        preLmDists  = preLmDists,
        postLmDists = postLmDists,
        gpRank      = lrgp.rank
      )
      allMetrics += row
      mw.println(row.csvRow)
      mw.flush()

      val outFile = new File(regDir, s"${spec.modelId}_registered.stl")
      MeshIO.writeMesh(regMesh, outFile)
        .recover { case ex => println(s"  WARN: could not write ${outFile.getName}: ${ex.getMessage}") }
    }

    mw.close()

    // ── Final tables ──────────────────────────────────────────────────────
    val rows = allMetrics.toSeq
    printMetricsTable(rows)
    printLmDistSummary(rows)

    // ── Best / worst highlight ────────────────────────────────────────────
    if (rows.nonEmpty) {
      val bestRow  = rows.minBy(_.postCd)
      val worstRow = rows.maxBy(_.postCd)

      println()
      println("*" * 90)
      println("  BEST REGISTRATION  (lowest post-CD — most accurate shape fit)")
      println("*" * 90)
      println(f"  Specimen : ${bestRow.modelId}")
      println(f"  POST  mean=${bestRow.postMean}%.2f mm  RMSE=${bestRow.postRms}%.2f mm  CD=${bestRow.postCd}%.2f mm  HD=${bestRow.postHd}%.2f mm")
      println(f"  P2P   mean=${bestRow.p2pMean}%.2f mm  RMSE=${bestRow.p2pRms}%.2f mm  CD=${bestRow.p2pCd}%.2f mm")
      println(s"  POST lm: " +
        lmNames.map(n => f"$n=${bestRow.postLmDists.getOrElse(n, Double.NaN)}%.2f").mkString("  "))
      println()
      println("*" * 90)
      println("  WORST REGISTRATION  (highest post-CD — hardest shape to fit)")
      println("*" * 90)
      println(f"  Specimen : ${worstRow.modelId}")
      println(f"  POST  mean=${worstRow.postMean}%.2f mm  RMSE=${worstRow.postRms}%.2f mm  CD=${worstRow.postCd}%.2f mm  HD=${worstRow.postHd}%.2f mm")
      println(f"  P2P   mean=${worstRow.p2pMean}%.2f mm  RMSE=${worstRow.p2pRms}%.2f mm  CD=${worstRow.p2pCd}%.2f mm")
      println(s"  POST lm: " +
        lmNames.map(n => f"$n=${worstRow.postLmDists.getOrElse(n, Double.NaN)}%.2f").mkString("  "))

      // Show best and worst in dedicated UI groups for visual comparison
      for {
        u           <- ui
        bestMesh    <- regMeshMap.get(bestRow.modelId)
        worstMesh   <- regMeshMap.get(worstRow.modelId)
      } {
        val grpBest  = u.createGroup(s"BEST_${bestRow.modelId}")
        val grpWorst = u.createGroup(s"WORST_${worstRow.modelId}")
        u.show(grpBest,  reference, "reference")
        u.show(grpBest,  bestMesh,  s"${bestRow.modelId}_reg")
        u.show(grpWorst, reference, "reference")
        u.show(grpWorst, worstMesh, s"${worstRow.modelId}_reg")
      }
    }

    // ── Summary file ──────────────────────────────────────────────────────
    val rw = new PrintWriter(new File(logDir, "summary.txt"))
    rw.println(s"Run       : $runTag")
    rw.println(s"Data      : ${dir.getAbsolutePath}")
    rw.println(s"Reference : ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices)")
    rw.println(s"Registered: ${rows.length} specimens")
    rw.println(s"Landmarks : ${lmNames.mkString(", ")}")
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
      rw.println(f"MEAN PRE-RIGID    mean=${avg(_.preMean)}%.2f  RMS=${avg(_.preRms)}%.2f  CD=${avg(_.preCd)}%.2f  HD=${avg(_.preHd)}%.2f mm")
      rw.println(f"MEAN POST-NONRIG  mean=${avg(_.postMean)}%.2f  RMS=${avg(_.postRms)}%.2f  CD=${avg(_.postCd)}%.2f  HD=${avg(_.postHd)}%.2f mm")
      rw.println(f"MEAN P2P (vs ref) mean=${avg(_.p2pMean)}%.2f  RMS=${avg(_.p2pRms)}%.2f  CD=${avg(_.p2pCd)}%.2f mm")
      rw.println()
      val bestRow  = rows.minBy(_.postCd)
      val worstRow = rows.maxBy(_.postCd)
      rw.println(f"BEST  registration: ${bestRow.modelId}  post-CD=${bestRow.postCd}%.2f  post-HD=${bestRow.postHd}%.2f  p2p-mean=${bestRow.p2pMean}%.2f mm")
      rw.println(f"WORST registration: ${worstRow.modelId}  post-CD=${worstRow.postCd}%.2f  post-HD=${worstRow.postHd}%.2f  p2p-mean=${worstRow.p2pMean}%.2f mm")
      rw.println()
      rw.println("MEAN LANDMARK DISTANCES")
      lmNames.foreach { n =>
        val pre  = rows.flatMap(_.preLmDists.get(n))
        val post = rows.flatMap(_.postLmDists.get(n))
        if (pre.nonEmpty && post.nonEmpty)
          rw.println(f"  $n%-6s  pre=${pre.sum / pre.length}%.2f mm  post=${post.sum / post.length}%.2f mm")
      }
    }
    rw.close()

    println(s"\nOutputs: ${outDir.getAbsolutePath}")
    println(s"  registered_meshes/             – ${rows.length} STL files")
    println(s"  logs/registration_metrics.csv  – surface + landmark distances")
    println(s"  logs/kernel_config.csv")
    println(s"  logs/summary.txt")

    if (Config.showUi) {
      println("\nScalismo UI groups")
      println("  reference             – reference scapula + landmark spheres")
      println("  unregistered_rigid    – rigid-aligned targets + their landmarks")
      println("  registered_nonrigid   – GP-registered surfaces")
      println("\nClose the window to exit.")
    }
  }
}
