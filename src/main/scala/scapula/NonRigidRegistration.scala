package scapula

import breeze.linalg.DenseVector
import scalismo.common.{Field, PointId}
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
//  Stage 2 – Landmark-conditioned Multiscale GP Non-Rigid
//             Registration  (scalismo 0.92)
// ============================================================
//
//  Per-specimen pipeline
//  ─────────────────────
//  1.  Mirror right-side scapulae into left space
//  2.  Rigid alignment (landmark Procrustes + trimmed ICP)
//  3.  Condition the GP posterior on rigid-aligned landmark
//      correspondences (σ²=1 mm² observation noise)
//      → the posterior mean already deforms toward all 5 landmarks
//  4.  4-pass ICP cascade on the posterior GP (coarse → fine)
//  5.  Compute surface-distance AND per-landmark distance metrics
//      both PRE and POST non-rigid registration
//  6.  Show unregistered and registered surfaces in Scalismo UI
//  7.  Save STL + CSV metrics + kernel table + summary
//
//  Landmark conditioning (why it matters for the glenoid)
//  ───────────────────────────────────────────────────────
//  The GP prior knows nothing about where the glenoid is.
//  Conditioning on GC (glenoid centre) forces the posterior mean
//  to already deform toward the correct glenoid location before
//  a single ICP step runs.  The posterior covariance is smaller
//  near each landmark so the optimiser spends its budget on
//  anatomy the landmarks do not constrain (blade surface, spine).
//
//  Multiscale kernel design  (bone ≈ 150-180 mm)
//  ──────────────────────────────────────────────
//  Coarse  σ=90 mm  scale=30  global shape / spine / body
//  Mid     σ=35 mm  scale=15  glenoid bowl / acromion
//  Fine    σ=15 mm  scale= 8  rim / coracoid tip
//  VFine   σ= 6 mm  scale= 4  articular surface texture
// ============================================================

object NonRigidRegistration {

  // ── Landmark observation noise (mm²) ──────────────────────
  // σ² = 1.0  →  ≈1 mm soft constraint; tight enough to pull
  // the GP toward the landmark, loose enough to absorb digitisation
  // error.  Reduce toward 0.1 for near-hard constraints.
  val lmSigma2: Double = 1.0

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
  def buildPriorLRGP(
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
  //  Condition the GP prior on landmark correspondences →
  //  posterior LRGP pinned at each landmark site.
  //
  //  trainingData: (reference-vertex-id, deformation-vector)
  //  where deformation = rigidTargetLandmark.point - refLandmark.point
  // ----------------------------------------------------------
  def conditionOnLandmarks(
    priorGP:    LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference:  TriangleMesh[_3D],
    refLms:     IndexedSeq[Landmark[_3D]],
    targetLms:  IndexedSeq[Landmark[_3D]]   // rigid-aligned target landmarks
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    val byId = targetLms.map(l => l.id -> l.point).toMap
    val trainingData: IndexedSeq[(PointId, EuclideanVector[_3D])] =
      refLms.flatMap { rl =>
        byId.get(rl.id).map { tp =>
          val ptId       = reference.pointSet.findClosestPoint(rl.point).id
          val deformation = tp - rl.point
          (ptId, deformation)
        }
      }
    priorGP.posterior(trainingData, lmSigma2)
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

  // The posterior mean (zero coefficients) already deforms toward landmarks
  def gpMeanInstance(
    reference: TriangleMesh[_3D],
    lrgp:      LowRankGaussianProcess[_3D, EuclideanVector[_3D]]
  ): TriangleMesh[_3D] =
    gpInstance(reference, lrgp, DenseVector.zeros[Double](lrgp.rank))

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
  //  Full 4-pass cascade on a (conditioned) posterior LRGP
  // ----------------------------------------------------------
  def registerOne(
    posteriorGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference:   TriangleMesh[_3D],
    target:      TriangleMesh[_3D],
    label:       String
  )(implicit rng: Random): (TriangleMesh[_3D], DenseVector[Double]) = {
    println(s"  [NR] $label  (posterior rank = ${posteriorGP.rank})")
    // Start from zero: the posterior MEAN already encodes landmark pull
    var coeffs = DenseVector.zeros[Double](posteriorGP.rank)
    regPasses.foreach { p =>
      print(s"      ${p.label} ... ")
      coeffs = runPass(posteriorGP, reference, target, coeffs, p)
      println("done")
    }
    (gpInstance(reference, posteriorGP, coeffs), coeffs)
  }

  // ----------------------------------------------------------
  //  Per-landmark distance between two aligned landmark sets
  // ----------------------------------------------------------
  def lmDistances(
    regMeshLms: IndexedSeq[Landmark[_3D]],  // landmarks mapped onto registered mesh
    targetLms:  IndexedSeq[Landmark[_3D]]   // rigid-aligned target landmarks
  ): Map[String, Double] = {
    val byId = targetLms.map(l => l.id -> l.point).toMap
    regMeshLms.flatMap { rl =>
      byId.get(rl.id).map(tp => rl.id -> (rl.point - tp).norm)
    }.toMap
  }

  // Map each reference landmark to the nearest point on a registered mesh,
  // then report how far those points are from the corresponding target landmarks.
  def lmErrorOnMesh(
    reference:   TriangleMesh[_3D],
    regMesh:     TriangleMesh[_3D],         // same topology as reference
    refLms:      IndexedSeq[Landmark[_3D]],
    targetLms:   IndexedSeq[Landmark[_3D]]
  ): Map[String, Double] = {
    // refLms are IN the reference; the registered mesh deforms those same vertices.
    // For each landmark, find the corresponding vertex id in the reference
    // and look up where it moved in regMesh.
    val byId = targetLms.map(l => l.id -> l.point).toMap
    refLms.flatMap { rl =>
      byId.get(rl.id).map { tp =>
        val vid       = reference.pointSet.findClosestPoint(rl.point).id
        val regPoint  = regMesh.pointSet.point(vid)
        rl.id -> (regPoint - tp).norm
      }
    }.toMap
  }

  // ----------------------------------------------------------
  //  Metric row (surface + per-landmark distances)
  // ----------------------------------------------------------
  val lmNames: Seq[String] = ScapulaData.landmarkNames.toSeq

  final case class RegMetrics(
    modelId:     String,
    preMean:     Double, preRms:  Double, preCd:   Double, preHd:  Double,
    postMean:    Double, postRms: Double, postCd:  Double, postHd: Double,
    p2pMean:     Double, p2pRms:  Double, p2pCd:  Double,
    // per-landmark distances: pre-rigid and post-nonrigid (mm)
    preLmDists:  Map[String, Double],
    postLmDists: Map[String, Double],
    gpRank:      Int
  ) {
    private def lmCols(m: Map[String, Double]) =
      lmNames.map(n => f"${m.getOrElse(n, Double.NaN)}%.3f").mkString(",")

    def csvRow: String =
      Seq(
        modelId,
        f"$preMean%.3f",  f"$preRms%.3f",  f"$preCd%.3f",   f"$preHd%.3f",
        f"$postMean%.3f", f"$postRms%.3f", f"$postCd%.3f",  f"$postHd%.3f",
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
    val notes = Seq(
      "global shape / spine / body",
      "glenoid bowl / acromion",
      "rim / coracoid tip",
      "articular surface texture"
    )
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
    println(f"  Landmark noise σ²=$lmSigma2 mm²  |  GP rank (Cholesky tol=$tol): $rank")
    println("=" * 72)
  }

  def printCascadeTable(): Unit = {
    println()
    println("=" * 68)
    println("REGISTRATION CASCADE  (on landmark-conditioned posterior GP)")
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
    println("=" * 150)
    println("NON-RIGID REGISTRATION METRICS  (all distances in mm)")
    println("=" * 150)
    // Header line 1: surface distances
    val lmHeader = lmNames.map(n => f"$n%6s").mkString("  ")
    println(
      f"${"Model"}%-${w}s  " +
      f"${"pre-mean"}%8s  ${"pre-CD"}%8s  ${"pre-HD"}%8s  " +
      f"${"post-mean"}%9s  ${"post-CD"}%8s  ${"post-HD"}%9s  " +
      f"${"p2p-mean"}%9s  ${"p2p-CD"}%8s  " +
      f"pre-lm[$lmHeader]  post-lm[$lmHeader]  ${"rank"}%5s"
    )
    println("-" * 150)
    def lmStr(m: Map[String, Double]) =
      lmNames.map(n => f"${m.getOrElse(n, Double.NaN)}%6.2f").mkString("  ")
    rows.foreach { r =>
      println(
        f"${r.modelId}%-${w}s  " +
        f"${r.preMean}%8.2f  ${r.preCd}%8.2f  ${r.preHd}%8.2f  " +
        f"${r.postMean}%9.2f  ${r.postCd}%8.2f  ${r.postHd}%9.2f  " +
        f"${r.p2pMean}%9.2f  ${r.p2pCd}%8.2f  " +
        s"[${lmStr(r.preLmDists)}]  [${lmStr(r.postLmDists)}]  ${r.gpRank}%5d"
      )
    }
    println("-" * 150)
    def avg(f: RegMetrics => Double) = rows.map(f).sum / rows.length
    def avgLm(m: RegMetrics => Map[String, Double]) =
      lmNames.map { n =>
        val vals = rows.flatMap(r => m(r).get(n))
        if (vals.isEmpty) Double.NaN else vals.sum / vals.length
      }
    val preAvgLm  = avgLm(_.preLmDists)
    val postAvgLm = avgLm(_.postLmDists)
    println(
      f"${"MEAN"}%-${w}s  " +
      f"${avg(_.preMean)}%8.2f  ${avg(_.preCd)}%8.2f  ${avg(_.preHd)}%8.2f  " +
      f"${avg(_.postMean)}%9.2f  ${avg(_.postCd)}%8.2f  ${avg(_.postHd)}%9.2f  " +
      f"${avg(_.p2pMean)}%9.2f  ${avg(_.p2pCd)}%8.2f  " +
      s"[${preAvgLm.map(v => f"$v%6.2f").mkString("  ")}]  " +
      s"[${postAvgLm.map(v => f"$v%6.2f").mkString("  ")}]  ${rows.head.gpRank}%5d"
    )
    println("=" * 150)
  }

  def printLmDistSummary(rows: Seq[RegMetrics]): Unit = {
    if (rows.isEmpty) return
    println()
    println("=" * 60)
    println("LANDMARK DISTANCE SUMMARY  (mm, mean across specimens)")
    println("=" * 60)
    println(f"${"Landmark"}%-8s  ${"pre-rigid"}%10s  ${"post-nonrig"}%12s  ${"improvement"}%12s")
    println("-" * 60)
    lmNames.foreach { n =>
      val preVals  = rows.flatMap(_.preLmDists.get(n))
      val postVals = rows.flatMap(_.postLmDists.get(n))
      if (preVals.nonEmpty && postVals.nonEmpty) {
        val pre  = preVals.sum  / preVals.length
        val post = postVals.sum / postVals.length
        val diff = pre - post
        println(f"$n%-8s  $pre%10.2f  $post%12.2f  $diff%12.2f")
      }
    }
    println("=" * 60)
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

    println("================================================================")
    println(s" Landmark-Conditioned Multiscale GP Non-Rigid Registration")
    println(s" $timestamp")
    println("================================================================")
    println(s" Data   : ${dir.getAbsolutePath}")
    println(s" Output : ${outDir.getAbsolutePath}")

    // ── Landmarks and specimens ────────────────────────────────────────────
    val csv = ScapulaData.csvFile(dir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("  WARN: landmark columns resolved by fallback offsets – verify CSV")

    val allSpecimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    require(allSpecimens.nonEmpty, s"No specimens with landmarks in ${dir.getAbsolutePath}")
    println(s" Specimens with landmarks: ${allSpecimens.length}")

    // ── Reference ─────────────────────────────────────────────────────────
    val refSpec   = allSpecimens.find(!_.isRight).getOrElse(allSpecimens.head)
    val refRaw    = ScapulaData.loadMesh(refSpec.file)
    val reference = refRaw.operations.decimate(Config.modelResolution)
    val refLms    = landmarks(refSpec.modelId)
    println(s" Reference: ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices)")

    // ── Build prior LRGP ──────────────────────────────────────────────────
    println("\nBuilding multiscale GP prior …")
    val priorGP = buildPriorLRGP(reference, kernelSpecs, Config.gpRelativeTolerance)
    println(s"  Prior rank = ${priorGP.rank}")

    printKernelTable(kernelSpecs, priorGP.rank, Config.gpRelativeTolerance)
    printCascadeTable()

    // Save kernel config
    val kw = new PrintWriter(new File(logDir, "kernel_config.csv"))
    kw.println(s"level,sigma_mm,scale,relativeTol,prior_rank,lm_sigma2")
    kernelSpecs.foreach { s =>
      kw.println(s"${s.tag},${s.sigma},${s.scale},${Config.gpRelativeTolerance},${priorGP.rank},$lmSigma2")
    }
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

    val mw = new PrintWriter(new File(logDir, "registration_metrics.csv"))
    mw.println(csvHeader)

    targets.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n[${idx + 1}/${targets.length}] ${spec.modelId}")

      // Mirror right → left
      val rawMesh = ScapulaData.loadMesh(spec.file)
      val rawLms  = landmarks(spec.modelId)
      val (oriented, orientedLms) =
        if (spec.isRight)
          (ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
        else
          (rawMesh, rawLms)

      // Rigid alignment – also returns rigid-aligned landmarks
      val (rigid, rigidLms) = RigidAlign.landmarkThenIcp(
        oriented, orientedLms, reference, refLms, icpIterations = Config.icpIterations
      )

      // Show rigid surface + its landmarks
      ui.zip(grpUnreg).foreach { case (u, g) =>
        u.show(g, rigid, s"${spec.modelId}_rigid")
        rigidLms.foreach(lm => u.show(g, lm, s"${spec.modelId}_${lm.id}"))
      }

      // Pre-registration surface metrics (rigid-aligned vs reference)
      val preStats = Metrics.symmetric(rigid, reference)

      // Pre-registration landmark distances (rigid-aligned lms vs ref lms)
      val preLmDists = lmDistances(rigidLms, refLms)

      // Condition the GP on rigid-aligned landmark correspondences
      println(s"  Conditioning GP on ${rigidLms.length} landmarks (σ²=$lmSigma2 mm²) …")
      val posteriorGP = conditionOnLandmarks(priorGP, reference, refLms, rigidLms)
      println(s"  Posterior rank = ${posteriorGP.rank}")

      // Non-rigid registration (starts from posterior mean → already pulls toward landmarks)
      val (regMesh, _) = registerOne(posteriorGP, reference, rigid, spec.modelId)

      // Show registered surface
      ui.zip(grpReg).foreach { case (u, g) =>
        u.show(g, regMesh, s"${spec.modelId}_reg")
      }

      // Post-registration surface metrics (registered vs rigid target)
      val postStats = Metrics.symmetric(regMesh, rigid)

      // Post-registration landmark distances
      // Each reference landmark vertex has moved; measure distance to rigid-target landmark
      val postLmDists = lmErrorOnMesh(reference, regMesh, refLms, rigidLms)

      // Point-to-point (regMesh shares reference topology)
      val p2pDists = Metrics.correspondingDistances(regMesh, reference)
      val p2pMean  = p2pDists.sum / p2pDists.length
      val p2pRms   = math.sqrt(p2pDists.map(x => x * x).sum / p2pDists.length)
      // p2p Chamfer: both directions (regMesh → reference and reference → regMesh vertex-to-vertex)
      val p2pCd    = p2pMean + (p2pDists.sum / p2pDists.length) // symmetric by construction; CD = 2*mean
      // For surface CD, use the symmetric mesh Chamfer (already in postStats.chamfer / preStats.chamfer)

      println(f"    PRE  (rigid vs ref)    : ${preStats.render}")
      println(f"    POST (reg vs target)   : ${postStats.render}")
      println(f"    P2P  (reg vs ref)      : mean=${p2pMean}%.2f  RMS=${p2pRms}%.2f  CD=${p2pCd}%.2f mm")
      println(s"    PRE  lm dists (mm)  : " +
        ScapulaData.landmarkNames.map(n => f"$n=${preLmDists.getOrElse(n, Double.NaN)}%.2f").mkString("  "))
      println(s"    POST lm dists (mm)  : " +
        ScapulaData.landmarkNames.map(n => f"$n=${postLmDists.getOrElse(n, Double.NaN)}%.2f").mkString("  "))

      val row = RegMetrics(
        modelId     = spec.modelId,
        preMean     = preStats.mean,  preRms  = preStats.rms,
        preCd       = preStats.chamfer, preHd = preStats.hd,
        postMean    = postStats.mean, postRms = postStats.rms,
        postCd      = postStats.chamfer, postHd = postStats.hd,
        p2pMean     = p2pMean, p2pRms = p2pRms, p2pCd = p2pCd,
        preLmDists  = preLmDists,
        postLmDists = postLmDists,
        gpRank      = posteriorGP.rank
      )
      allMetrics += row
      mw.println(row.csvRow)
      mw.flush()

      // Save registered STL
      val outFile = new File(regDir, s"${spec.modelId}_registered.stl")
      MeshIO.writeMesh(regMesh, outFile)
        .recover { case ex => println(s"  WARN: could not write ${outFile.getName}: ${ex.getMessage}") }
    }

    mw.close()

    // ── Final tables ──────────────────────────────────────────────────────
    val rows = allMetrics.toSeq
    printMetricsTable(rows)
    printLmDistSummary(rows)

    // ── Summary file ──────────────────────────────────────────────────────
    val rw = new PrintWriter(new File(logDir, "summary.txt"))
    rw.println(s"Run       : $runTag")
    rw.println(s"Data      : ${dir.getAbsolutePath}")
    rw.println(s"Reference : ${refSpec.modelId}  (${reference.pointSet.numberOfPoints} vertices)")
    rw.println(s"Registered: ${rows.length} specimens")
    rw.println(s"Landmarks : ${lmNames.mkString(", ")}")
    rw.println(s"LM σ²     : $lmSigma2 mm²")
    rw.println()
    rw.println("KERNEL CONFIGURATION")
    kernelSpecs.foreach(s => rw.println(f"  ${s.tag}%-10s  sigma=${s.sigma}%.1f mm  scale=${s.scale}%.1f"))
    rw.println(s"  Prior rank: ${priorGP.rank}  (relativeTol=${Config.gpRelativeTolerance})")
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
    println(s"  registered_meshes/         – ${rows.length} STL files")
    println(s"  logs/registration_metrics.csv  – per-specimen + landmark distances")
    println(s"  logs/kernel_config.csv")
    println(s"  logs/summary.txt")

    if (Config.showUi) {
      println("\nScalismo UI groups")
      println("  reference             – reference scapula + landmark spheres")
      println("  unregistered_rigid    – rigid-aligned targets + their landmarks")
      println("  registered_nonrigid   – landmark-conditioned GP registered surfaces")
      println("\nClose the window to exit.")
    }
  }
}
