package scapula

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.*
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.kernels.*
import breeze.linalg.{DenseMatrix, DenseVector}
import java.io.{File, PrintWriter}
import scalismo.utils.Random
import scalismo.ui.api.*

/**
 * Full scapula SSM pipeline — two registration passes with progressive-posterior ICP.
 *
 * Fixes relative to the original script
 * ──────────────────────────────────────
 * FIX 1 – Kernel parameters tuned for scapula scale.
 *   Original sigma=80 mm covered ~60 % of the bone and allowed anatomically
 *   implausible vertex drift (glenoid vertices could land on the spine).
 *   New combined std dev ≈ sqrt(30+12+4+2) ≈ 6.9 mm/axis — enough to span real
 *   inter-subject variability without producing implausible deformations.
 *
 * FIX 2 – Rigid ICP pre-alignment before every non-rigid registration.
 *   Without pose removal first, the GP prior has to "spend" modes on translation
 *   and rotation, leaving fewer effective DOFs for shape, and the max-distance
 *   filter rejects most valid correspondences.
 *
 * FIX 3 – Progressive posterior update inside the ICP loop (key fix).
 *   Original: model.posterior(corr) was computed from the ORIGINAL PRIOR every
 *   iteration — the model never tightened.  Each iteration discarded all the
 *   information from the previous posterior and restarted, producing oscillation.
 *   Fix: pass cur.posterior(corr) as the new current model.  The model tightens
 *   around the target at every step, correspondences become progressively more
 *   accurate, and the whole loop converges in far fewer iterations.
 *
 * FIX 4 – Geometrically annealed max-distance and noise.
 *   Original fixed maxDist=20 mm is too tight when there is any residual pose
 *   error, leaving too few correspondences.  New schedule: 40 mm → 5 mm for
 *   max-distance, 15 mm → 0.5 mm for noise sigma.
 *
 * FIX 5 – Outlier rejection before PCA.
 *   Even one badly registered mesh corrupts the PCA and produces the spiky
 *   artifacts visible in screenshot 1.  Any mesh whose corresponding-point RMSE
 *   exceeds 2.5× the median is discarded.
 *
 * FIX 6 – Two-pass registration for reference-bias removal.
 *   Building the SSM against an arbitrary reference biases the modes toward
 *   that single specimen.  Pass 2 uses the mean shape from pass 1 as reference.
 *
 * FIX 7 – Iteration count raised to Config.icpIterations (default 40, not 20).
 */
object ScapulaSSMPipeline {

  private val BaseDir    = new File(s"${sys.props("user.home")}/Documents/100 plus scapula data/aligned_scapulae_output")
  private val OutDir     = new File(s"${sys.props("user.home")}/Documents/100 plus scapula data/ssm_output")
  private val CorrespDir = new File(OutDir, "correspondences")
  private val ModesDir   = new File(OutDir, "modes_of_variation")

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    Seq(OutDir, CorrespDir, ModesDir).foreach(_.mkdirs())
    val ui = if (Config.showUi) Some(ScalismoUI()) else None

    // ── Load meshes ──────────────────────────────────────────────────────────
    val targetFiles = collectStls(BaseDir)
    require(targetFiles.nonEmpty, s"No STL files found under $BaseDir")
    println(s"Found ${targetFiles.size} meshes.")

    // ── Initial reference ────────────────────────────────────────────────────
    val refFile = targetFiles.find(_.getName.contains("001_M_64_L")).getOrElse(targetFiles.head)
    val refMesh = MeshIO.readMesh(refFile).get.operations.decimate(Config.modelResolution)
    println(s"Reference: ${refFile.getName} (${refMesh.pointSet.numberOfPoints} vertices after decimation)")

    // ── Pass 1: register all to initial reference ────────────────────────────
    println("\n── Pass 1/2: register to initial reference ──")
    val pass1Good = registrationPass(targetFiles, refMesh, "p1_", ui, "Pass 1 GPMM Prior")

    // Mean shape from pass 1 becomes the pass-2 reference (bias removal)
    val meanRef = PointDistributionModel
      .createUsingPCA(DataCollection.fromTriangleMesh3DSequence(refMesh, pass1Good))
      .mean

    // ── Pass 2: register all to mean shape ───────────────────────────────────
    println("\n── Pass 2/2: register to mean shape reference ──")
    val pass2Good = registrationPass(targetFiles, meanRef, "p2_", ui, "Pass 2 GPMM Prior")

    // ── Final SSM ────────────────────────────────────────────────────────────
    println(s"\nBuilding final SSM from ${pass2Good.size} shapes...")
    val ssm = PointDistributionModel.createUsingPCA(
      DataCollection.fromTriangleMesh3DSequence(meanRef, pass2Good))
    println(s"SSM: rank=${ssm.rank}, n=${pass2Good.size}")

    val ssmFile = new File(OutDir, "scapula_ssm_final.h5")
    if (ssmFile.exists()) ssmFile.delete()
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm, ssmFile)
    println(s"Saved: $ssmFile")

    // Export mean + first 5 modes ±2/3 SD
    MeshIO.writeMesh(ssm.mean, new File(ModesDir, "Mode_Mean.stl"))
    for (m <- 0 until math.min(5, ssm.rank); sd <- Seq(-3.0, -2.0, 2.0, 3.0)) {
      val c = DenseVector.zeros[Double](ssm.rank); c(m) = sd
      val tag = (if (sd > 0) "+" else "") + sd.toInt + "SD"
      MeshIO.writeMesh(ssm.instance(c), new File(ModesDir, s"Mode_${m + 1}_$tag.stl"))
    }

    ui.foreach { u =>
      val ssmGrp = u.createGroup("Scapula SSM")
      u.show(ssmGrp, ssm, "Scapula SSM")
      val cohortGrp = u.createGroup("Registered Cohort Meshes")
      pass2Good.take(15).zipWithIndex.foreach { case (m, i) =>
        u.show(cohortGrp, m, s"Subject_${i + 1}")
      }
    }

    // ── Validation ───────────────────────────────────────────────────────────
    println("\nComputing validation curves...")
    val variances = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val totalVar  = variances.sum
    val cumVar    = variances.scanLeft(0.0)(_ + _).tail.map(_ / totalVar * 100.0)
    val maxModes  = math.min(20, ssm.rank)

    println(s"  LOOCV generalization ($maxModes modes)...")
    val genErrs = loocvGeneralization(ssm, meanRef, pass2Good, maxModes)
    println(s"  Specificity ($maxModes modes, 35 samples each)...")
    val specErrs = specificity(ssm, pass2Good, maxModes, 35)

    val csvFile = new File(OutDir, "ssm_validation_curves.csv")
    val pw = new PrintWriter(csvFile)
    pw.println("Mode,Variance_Pct,Cumulative_Variance_Pct,Generalization_LOOCV_mm,Specificity_mm")
    for (m <- 1 to maxModes)
      pw.println(f"$m,${variances(m - 1) / totalVar * 100.0}%.3f,${cumVar(m - 1)}%.3f,${genErrs(m - 1)}%.4f,${specErrs(m - 1)}%.4f")
    pw.close()

    println(s"Saved: $csvFile")
    println("\nDone. Scalismo UI is open — use sliders to explore shape modes.")
  }

  // ── Private helpers ──────────────────────────────────────────────────────────

  private def collectStls(root: File): List[File] =
    Option(root.listFiles()).getOrElse(Array.empty).toList.flatMap { e =>
      if (e.isDirectory)
        Option(e.listFiles()).getOrElse(Array.empty)
          .filter(_.getName.toLowerCase.endsWith(".stl")).toList
      else if (e.getName.toLowerCase.endsWith(".stl")) List(e)
      else Nil
    }.sortBy(_.getName)

  /** One full registration pass: build GPMM on `reference`, register all files, filter outliers. */
  private def registrationPass(
    files:     List[File],
    reference: TriangleMesh[_3D],
    prefix:    String,
    ui:        Option[ScalismoUI],
    uiLabel:   String
  )(implicit rng: Random): IndexedSeq[TriangleMesh[_3D]] = {

    val gpmm  = buildGpmm(reference)
    val ptIds = RigidAlign.uniformIds(reference, 3000)
    println(s"  GPMM prior: ${gpmm.rank} modes")

    ui.foreach { u => u.show(u.createGroup(uiLabel), gpmm, uiLabel) }

    val registered: IndexedSeq[TriangleMesh[_3D]] =
      files.zipWithIndex.map { case (f, i) =>
        val name  = f.getName.stripSuffix(".stl")
        val cache = new File(CorrespDir, s"$prefix$name.stl")
        val mesh =
          if (cache.exists()) MeshIO.readMesh(cache).get
          else {
            val target       = MeshIO.readMesh(f).get
            val rigidAligned = RigidAlign.rigidIcp(target, reference) // FIX 2: pose removal first
            val fitted       = nonRigidICP(gpmm, rigidAligned, ptIds, Config.icpIterations)
            MeshIO.writeMesh(fitted, cache)
            fitted
          }
        print(f"\r  [${i + 1}%3d/${files.size}] $name%-55s")
        mesh
      }.toIndexedSeq
    println()

    val kept = filterOutliers(reference, registered, 2.5)  // FIX 5
    println(s"  Outlier filter: ${kept.size}/${registered.size} kept.")
    kept
  }

  /**
   * FIX 1 – Tightly-tuned multi-scale Gaussian kernel.
   *
   * Scale   sigma (mm)   scaleFactor   std/axis (mm)
   * coarse      50           30            ~5.5
   * mid         25           12            ~3.5
   * fine        10            4            ~2.0
   * very fine    5            2            ~1.4
   *
   * Combined std/axis ≈ sqrt(30+12+4+2) = 6.9 mm.  This is tight enough to prevent
   * cross-anatomical correspondence drift while still spanning real inter-subject
   * shape variability (~5–15 mm depending on region).
   */
  private def buildGpmm(ref: TriangleMesh[_3D])(implicit rng: Random): PointDistributionModel[_3D, TriangleMesh] = {
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val kernel =
      DiagonalKernel3D(GaussianKernel3D(50.0, 30.0), 3) +
      DiagonalKernel3D(GaussianKernel3D(25.0, 12.0), 3) +
      DiagonalKernel3D(GaussianKernel3D(10.0,  4.0), 3) +
      DiagonalKernel3D(GaussianKernel3D( 5.0,  2.0), 3)
    val lrGP = LowRankGaussianProcess.approximateGPCholesky(
      ref,
      GaussianProcess(zeroMean, kernel),
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )
    PointDistributionModel[_3D, TriangleMesh](ref, lrGP)
  }

  /**
   * FIX 3 + FIX 4 – Progressive-posterior non-rigid ICP with annealed schedules.
   *
   * Original problem: every iteration called `model.posterior(corr)` on the ORIGINAL
   * PRIOR, discarding all accumulated evidence.  The model never tightened; convergence
   * was slow and prone to oscillation between local minima.
   *
   * Fix: the current model `cur` is updated to `cur.posterior(corr)` each iteration.
   * The posterior IS the new prior for the next step, so the model progressively
   * constrains itself to the target surface.  Correspondences computed from an
   * already-close mean are far more anatomically consistent than those from the
   * original prior mean.
   *
   * Both noise sigma and max-distance are annealed geometrically so the first
   * iterations act as a global coarse fit and the last iterations refine locally.
   */
  private def nonRigidICP(
    model:  PointDistributionModel[_3D, TriangleMesh],
    target: TriangleMesh[_3D],
    ptIds:  IndexedSeq[PointId],
    nIter:  Int
  ): TriangleMesh[_3D] = {

    def noiseSigma(i: Int): Double = 15.0 * math.pow(0.5 / 15.0, i.toDouble / (nIter - 1))  // 15 → 0.5 mm
    def maxDist(i: Int): Double    = 40.0 * math.pow(5.0 / 40.0,  i.toDouble / (nIter - 1)) // 40 → 5 mm

    var cur = model
    for (i <- 0 until nIter) {
      val s     = noiseSigma(i)
      val noise = MultivariateNormalDistribution(
        DenseVector.zeros[Double](3),
        DenseMatrix.eye[Double](3) * (s * s)
      )
      val dMax   = maxDist(i)
      val moving = cur.mean
      val corr: IndexedSeq[(PointId, Point[_3D], MultivariateNormalDistribution)] =
        ptIds.flatMap { id =>
          val pt = moving.pointSet.point(id)
          val cp = target.operations.closestPointOnSurface(pt).point
          if ((pt - cp).norm < dMax) Some((id, cp, noise)) else None
        }.toIndexedSeq
      if (corr.size >= 30)
        cur = cur.posterior(corr) // FIX 3: tighten the model, not just the mesh
    }
    cur.mean
  }

  /**
   * FIX 5 – Drop meshes whose corresponding-point RMSE exceeds `factor` × median RMSE.
   *
   * A single badly-registered mesh creates extreme PCA modes that appear as sharp
   * radial spikes in SSM samples.  Removing outliers before PCA prevents this.
   */
  private def filterOutliers(
    ref:    TriangleMesh[_3D],
    meshes: IndexedSeq[TriangleMesh[_3D]],
    factor: Double
  ): IndexedSeq[TriangleMesh[_3D]] = {
    val rmses = meshes.map { m =>
      val d = Metrics.correspondingDistances(ref, m)
      math.sqrt(d.map(x => x * x).sum / d.size)
    }
    val median = rmses.sorted.apply(rmses.size / 2)
    val thresh = factor * median
    val kept   = meshes.zip(rmses).collect { case (m, r) if r <= thresh => m }
    if (kept.size < meshes.size)
      println(s"\n    [outlier] removed ${meshes.size - kept.size} mesh(es) with RMSE > ${"%.2f".format(thresh)} mm (median=${"%.2f".format(median)} mm, ×$factor)")
    kept
  }

  // ── LOOCV generalization ─────────────────────────────────────────────────────
  private def loocvGeneralization(
    ssm:      PointDistributionModel[_3D, TriangleMesh],
    ref:      TriangleMesh[_3D],
    meshes:   IndexedSeq[TriangleMesh[_3D]],
    maxModes: Int
  )(implicit rng: Random): IndexedSeq[Double] =
    (1 to maxModes).map { m =>
      val errs = meshes.indices.map { j =>
        val train = meshes.patch(j, Nil, 1)
        val loocv = PointDistributionModel.createUsingPCA(
          DataCollection.fromTriangleMesh3DSequence(ref, train))
        val c  = loocv.coefficients(meshes(j))
        val tc = DenseVector.tabulate(loocv.rank)(i => if (i < m) c(i) else 0.0)
        Metrics.symmetric(loocv.instance(tc), meshes(j)).mean
      }
      errs.sum / errs.size
    }.toIndexedSeq

  // ── Specificity ──────────────────────────────────────────────────────────────
  private def specificity(
    ssm:      PointDistributionModel[_3D, TriangleMesh],
    meshes:   IndexedSeq[TriangleMesh[_3D]],
    maxModes: Int,
    nSamples: Int
  )(implicit rng: Random): IndexedSeq[Double] =
    (1 to maxModes).map { m =>
      val errs = (0 until nSamples).map { _ =>
        val c = DenseVector.zeros[Double](ssm.rank)
        for (i <- 0 until m) c(i) = rng.scalaRandom.nextGaussian()
        meshes.map(t => Metrics.symmetric(ssm.instance(c), t).mean).min
      }
      errs.sum / nSamples
    }.toIndexedSeq
}
