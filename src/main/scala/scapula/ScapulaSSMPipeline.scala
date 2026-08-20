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
 * Full scapula SSM pipeline: two registration passes with progressive-posterior ICP.
 *
 * Root causes fixed (relative to the original script that produced a spiky, corrupted model)
 * ──────────────────────────────────────────────────────────────────────────────────────────
 *
 * FIX 1 – Kernel parameters tuned for scapula scale (~130 mm).
 *   Original sigma=80 mm covered 60 % of the bone; scaleFactor=100 gave ~10 mm std/axis.
 *   At that flexibility a glenoid vertex can drift to the spine of the scapula during ICP,
 *   producing anatomically wrong correspondences.  New 4-scale kernel:
 *     sigma 50/25/10/5 mm, scaleFactor 30/12/4/2 → combined std ≈ 6.9 mm/axis.
 *   Tight enough to prevent cross-anatomical drift; wide enough to span real variability.
 *
 * FIX 2 – Rigid ICP pre-alignment before every non-rigid registration.
 *   Without pose removal first the GP prior must waste DOFs on translation/rotation,
 *   leaving too few effective shape DOFs, and the maxDist filter rejects the bulk of
 *   correspondences.
 *
 * FIX 3 – Progressive posterior update (the principal fix for the spiky artifact).
 *   Original loop: model.posterior(corr) was called on the ORIGINAL PRIOR every iteration.
 *   The model never tightened; each iteration discarded the previous posterior and
 *   restarted cold, producing oscillation around local minima.
 *   Fix: cur = cur.posterior(corr) so the model tightens around the target surface at
 *   every step.  Correspondences computed from an already-close mean are far more
 *   anatomically consistent across subjects, which is what the PCA needs.
 *
 * FIX 4 – Geometrically annealed max-distance and noise sigma.
 *   Original fixed maxDist=20 mm rejected most correspondences whenever any residual pose
 *   error remained.  New schedule: maxDist 40→5 mm, noise sigma 15→0.5 mm.
 *
 * FIX 5 – Outlier rejection before PCA.
 *   A single badly-registered mesh creates extreme PCA modes that appear as radial spikes
 *   in SSM samples (exactly screenshot 1).  Any mesh with corresponding-point RMSE >
 *   2.5 × the median is discarded before PCA.
 *
 * FIX 6 – Two-pass registration (reference-bias removal).
 *   Pass 2 uses the population mean from pass 1 as the reference, so SSM modes capture
 *   inter-subject variation rather than deviation from one arbitrary specimen.
 *
 * FIX 7 – Iteration count: Config.icpIterations (default 40) instead of the hardcoded 20.
 */
object ScapulaSSMPipeline {

  // ── Directories ─────────────────────────────────────────────────────────────
  // Overridable at runtime via the same env-var scheme as Config.
  private val baseDir: File = new File(
    sys.env.getOrElse(
      "SCAPULA_ALIGN_DIR",
      s"${sys.props("user.home")}/Documents/100 plus scapula data/aligned_scapulae_output"
    )
  )
  // Config.outDir already respects the SCAPULA_OUT_DIR env var.
  private val outDir     = Config.outDir
  private val correspDir = new File(outDir, "correspondences")
  private val modesDir   = new File(outDir, "modes_of_variation")

  // ── Entry point ──────────────────────────────────────────────────────────────
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    Seq(outDir, correspDir, modesDir).foreach(_.mkdirs())
    val ui = if (Config.showUi) Some(ScalismoUI()) else None

    // ── Collect STL files ────────────────────────────────────────────────────
    val targetFiles = collectStls(baseDir)
    require(targetFiles.nonEmpty, s"No STL meshes found under ${baseDir.getAbsolutePath}")
    println(s"Found ${targetFiles.size} meshes.")

    // ── Initial reference (decimated to model resolution) ────────────────────
    val refFile = targetFiles.find(_.getName.contains("001_M_64_L")).getOrElse(targetFiles.head)
    val refMesh = MeshIO.readMesh(refFile).get.operations.decimate(Config.modelResolution)
    println(s"Reference: ${refFile.getName} (${refMesh.pointSet.numberOfPoints} vertices)")

    // ── Pass 1: register everything to the initial reference ─────────────────
    println("\n── Pass 1 / 2: registering to initial reference ──")
    val pass1Meshes = registrationPass(
      files = targetFiles, reference = refMesh, cachePrefix = "p1_", ui = ui, uiLabel = "Pass 1 GPMM"
    )
    require(pass1Meshes.nonEmpty, "Pass 1 produced zero valid registrations; check your data.")

    // Build an intermediate SSM from pass 1 to extract the population mean shape.
    val meanRef: TriangleMesh[_3D] = PointDistributionModel
      .createUsingPCA(DataCollection.fromTriangleMesh3DSequence(refMesh, pass1Meshes))
      .mean

    // ── Pass 2: re-register to mean shape (removes reference bias) ───────────
    println("\n── Pass 2 / 2: registering to mean-shape reference ──")
    val pass2Meshes = registrationPass(
      files = targetFiles, reference = meanRef, cachePrefix = "p2_", ui = ui, uiLabel = "Pass 2 GPMM"
    )
    require(pass2Meshes.nonEmpty, "Pass 2 produced zero valid registrations; check your data.")

    // ── Final SSM via PCA ────────────────────────────────────────────────────
    println(s"\nBuilding final SSM from ${pass2Meshes.size} registered shapes...")
    val ssm = PointDistributionModel.createUsingPCA(
      DataCollection.fromTriangleMesh3DSequence(meanRef, pass2Meshes)
    )
    println(s"SSM: rank = ${ssm.rank}, training set = ${pass2Meshes.size}")

    val ssmFile = new File(outDir, "scapula_ssm_final.h5")
    if (ssmFile.exists()) ssmFile.delete()
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(ssm, ssmFile).get
    println(s"Saved SSM: $ssmFile")

    // Export mean and first 5 modes at ±2 SD, ±3 SD.
    exportModes(ssm, modesDir)

    // Show in Scalismo UI.
    ui.foreach { u =>
      val ssmGrp = u.createGroup("Scapula SSM")
      u.show(ssmGrp, ssm, "Scapula SSM")
      val cohortGrp = u.createGroup("Registered Cohort Meshes")
      pass2Meshes.take(15).zipWithIndex.foreach { case (m, i) =>
        u.show(cohortGrp, m, s"Subject_${i + 1}")
      }
    }

    // ── Validation curves ────────────────────────────────────────────────────
    println("\nComputing validation curves (LOOCV generalization + specificity)...")
    val variances  = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val totalVar   = variances.sum
    val cumVarPct  = variances.scanLeft(0.0)(_ + _).tail.map(_ / totalVar * 100.0)
    val maxModes   = math.min(20, ssm.rank)

    println(s"  Generalization ($maxModes modes, LOOCV)...")
    val genErrs = loocvGeneralization(meanRef, pass2Meshes, maxModes)

    println(s"  Specificity ($maxModes modes, 35 random samples each)...")
    val specErrs = specificity(ssm, pass2Meshes, maxModes, nSamples = 35)

    val csvFile = new File(outDir, "ssm_validation_curves.csv")
    val pw = new PrintWriter(csvFile)
    pw.println("Mode,Variance_Pct,Cumulative_Variance_Pct,Generalization_LOOCV_mm,Specificity_mm")
    for (m <- 1 to maxModes)
      pw.println(f"$m,${variances(m - 1) / totalVar * 100.0}%.3f,${cumVarPct(m - 1)}%.3f,${genErrs(m - 1)}%.4f,${specErrs(m - 1)}%.4f")
    pw.close()
    println(s"Saved CSV:  $csvFile")
    println("\nPipeline complete. Scalismo UI is open — use sliders to explore shape modes.")
  }

  // ── collectStls ──────────────────────────────────────────────────────────────

  private def collectStls(root: File): List[File] = {
    val entries = Option(root.listFiles()).getOrElse(Array.empty[File])
    entries.toList.flatMap { e =>
      if (e.isDirectory)
        Option(e.listFiles()).getOrElse(Array.empty[File])
          .filter(_.getName.toLowerCase.endsWith(".stl"))
          .toList
      else if (e.getName.toLowerCase.endsWith(".stl")) List(e)
      else Nil
    }.sortBy(_.getName)
  }

  // ── registrationPass ─────────────────────────────────────────────────────────

  /**
   * Build a GPMM prior on `reference`, rigidly then non-rigidly register every target,
   * filter outliers, and return the kept correspondence meshes (topology = reference).
   */
  private def registrationPass(
    files:       List[File],
    reference:   TriangleMesh[_3D],
    cachePrefix: String,
    ui:          Option[ScalismoUI],
    uiLabel:     String
  )(implicit rng: Random): IndexedSeq[TriangleMesh[_3D]] = {

    val gpmm  = buildGpmm(reference)
    val ptIds = RigidAlign.uniformIds(reference, 3000)
    println(s"  GPMM prior: ${gpmm.rank} modes  (relativeTolerance = ${Config.gpRelativeTolerance})")
    ui.foreach { u => u.show(u.createGroup(uiLabel), gpmm, uiLabel) }

    val registered: IndexedSeq[TriangleMesh[_3D]] = files.zipWithIndex.map { case (f, idx) =>
      val name      = f.getName.stripSuffix(".stl")
      val cacheFile = new File(correspDir, s"$cachePrefix$name.stl")

      val mesh: TriangleMesh[_3D] =
        if (cacheFile.exists()) {
          MeshIO.readMesh(cacheFile).get
        } else {
          val rawTarget    = MeshIO.readMesh(f).get
          val rigidTarget  = RigidAlign.rigidIcp(rawTarget, reference)  // FIX 2: pose removal
          val fittedMesh   = nonRigidICP(gpmm, rigidTarget, ptIds, Config.icpIterations) // FIX 3+4
          MeshIO.writeMesh(fittedMesh, cacheFile).get
          fittedMesh
        }

      print(f"\r  [${idx + 1}%3d / ${files.size}%3d]  $name%-55s")
      mesh
    }.toIndexedSeq
    println()

    val kept = filterOutliers(reference, registered, outlierFactor = 2.5)  // FIX 5
    println(s"  Outlier filter: ${kept.size} / ${registered.size} meshes kept.")
    kept
  }

  // ── buildGpmm ────────────────────────────────────────────────────────────────

  /**
   * Build a tightly-tuned multi-scale GPMM prior (FIX 1).
   *
   * Each Gaussian kernel k(x,y) = scaleFactor · exp(−‖x−y‖² / 2σ²) contributes
   * scaleFactor mm² of variance per axis.  Combined std/axis ≈ √(30+12+4+2) ≈ 6.9 mm,
   * vs. the original √(100+25+5) ≈ 11.4 mm — tight enough to prevent cross-anatomical
   * drift, while still spanning the ~5–15 mm regional variability of the scapula.
   */
  private def buildGpmm(ref: TriangleMesh[_3D]): PointDistributionModel[_3D, TriangleMesh] = {
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val kernel =
      DiagonalKernel3D(GaussianKernel3D(sigma = 50.0, scaleFactor = 30.0), outputDim = 3) +
      DiagonalKernel3D(GaussianKernel3D(sigma = 25.0, scaleFactor = 12.0), outputDim = 3) +
      DiagonalKernel3D(GaussianKernel3D(sigma = 10.0, scaleFactor =  4.0), outputDim = 3) +
      DiagonalKernel3D(GaussianKernel3D(sigma =  5.0, scaleFactor =  2.0), outputDim = 3)
    val lrGP = LowRankGaussianProcess.approximateGPCholesky(
      ref,
      GaussianProcess(zeroMean, kernel),
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )
    PointDistributionModel[_3D, TriangleMesh](ref, lrGP)
  }

  // ── nonRigidICP ──────────────────────────────────────────────────────────────

  /**
   * Non-rigid ICP with progressive posterior update and geometrically annealed schedules.
   *
   * The key change from the original (FIX 3):
   *   Original:  val posterior = model.posterior(corr)   ← always the ORIGINAL prior
   *              iterate(posterior.mean, iter + 1)        ← model discarded each step
   *
   *   Fixed:     cur = cur.posterior(corr)               ← model TIGHTENS each step
   *
   * Because `cur` accumulates all previous observations, correspondences from
   * its mean are progressively closer to the target, making each next round's
   * correspondences more anatomically consistent.
   *
   * Anneal schedules (FIX 4) — geometrically decayed over nIter steps:
   *   noise sigma  : 15 mm → 0.5 mm   (large noise = trusts prior more initially)
   *   max distance : 40 mm → 5 mm     (wide search initially; tight refinement at end)
   */
  private def nonRigidICP(
    model:  PointDistributionModel[_3D, TriangleMesh],
    target: TriangleMesh[_3D],
    ptIds:  IndexedSeq[PointId],
    nIter:  Int
  ): TriangleMesh[_3D] = {

    // Geometric anneal: value at step i out of nIter total steps.
    // Guard against nIter == 1 to avoid division by zero.
    def anneal(start: Double, end: Double, i: Int): Double =
      if (nIter <= 1) end
      else start * math.pow(end / start, i.toDouble / (nIter - 1))

    val targetOps = target.operations  // cache the surface query structure once

    var cur = model
    for (i <- 0 until nIter) {
      val sigma = anneal(start = 15.0, end = 0.5, i)
      val dMax  = anneal(start = 40.0, end = 5.0,  i)
      val noise = MultivariateNormalDistribution(
        DenseVector.zeros[Double](3),
        DenseMatrix.eye[Double](3) * (sigma * sigma)
      )
      val moving = cur.mean

      val corr: IndexedSeq[(PointId, Point[_3D], MultivariateNormalDistribution)] =
        ptIds.flatMap { id =>
          val pt = moving.pointSet.point(id)
          val cp = targetOps.closestPointOnSurface(pt).point
          if ((pt - cp).norm < dMax) Some((id, cp, noise)) else None
        }.toIndexedSeq

      // Only update if enough correspondences exist; otherwise keep current model.
      if (corr.size >= 30)
        cur = cur.posterior(corr)  // FIX 3: posterior becomes next iteration's prior
    }
    cur.mean
  }

  // ── filterOutliers ───────────────────────────────────────────────────────────

  /**
   * Discard any registered mesh whose corresponding-point RMSE against `reference`
   * exceeds `outlierFactor` × the median RMSE (FIX 5).
   *
   * Precondition: every mesh in `meshes` has the same number of vertices as `reference`
   * (guaranteed because each is the mean of a PDM whose reference topology = `reference`).
   */
  private def filterOutliers(
    reference:     TriangleMesh[_3D],
    meshes:        IndexedSeq[TriangleMesh[_3D]],
    outlierFactor: Double
  ): IndexedSeq[TriangleMesh[_3D]] = {
    if (meshes.isEmpty) return meshes

    val rmses: IndexedSeq[Double] = meshes.map { m =>
      val dists = Metrics.correspondingDistances(reference, m)
      math.sqrt(dists.map(d => d * d).sum / dists.size)
    }
    val sorted = rmses.sorted
    val median = sorted(sorted.size / 2)
    val thresh = outlierFactor * median

    val kept = meshes.zip(rmses).collect { case (m, r) if r <= thresh => m }
    if (kept.size < meshes.size)
      println(s"\n    [outlier] dropped ${meshes.size - kept.size} mesh(es)  " +
        f"(threshold = $thresh%.2f mm = $outlierFactor × median $median%.2f mm)")
    kept
  }

  // ── exportModes ──────────────────────────────────────────────────────────────

  private def exportModes(ssm: PointDistributionModel[_3D, TriangleMesh], dir: File): Unit = {
    MeshIO.writeMesh(ssm.mean, new File(dir, "Mode_Mean.stl")).get
    val nExport = math.min(5, ssm.rank)
    for (m <- 0 until nExport; sd <- Seq(-3.0, -2.0, 2.0, 3.0)) {
      val coeffs = DenseVector.zeros[Double](ssm.rank)
      coeffs(m) = sd
      val tag  = (if (sd > 0) "+" else "") + sd.toInt.toString + "SD"
      MeshIO.writeMesh(ssm.instance(coeffs), new File(dir, s"Mode_${m + 1}_$tag.stl")).get
    }
    println(s"Exported mean + first $nExport modes (±2 SD, ±3 SD) to ${dir.getName}/")
  }

  // ── loocvGeneralization ──────────────────────────────────────────────────────

  /**
   * Leave-one-out cross-validation generalization error.
   *
   * For each number of retained modes m, and for each training shape j:
   *   1. Build a LOO model from the remaining shapes.
   *   2. Project the left-out shape onto the first m modes of the LOO model.
   *   3. Measure the average surface distance between the reconstruction and the target.
   *
   * The final value is the mean over all left-out shapes.
   *
   * Note: loocv.rank = pass2Meshes.size − 2 (PCA on n−1 shapes gives rank n−2).
   * We clamp m to loocv.rank to avoid out-of-bounds coefficient access.
   */
  private def loocvGeneralization(
    reference: TriangleMesh[_3D],
    meshes:    IndexedSeq[TriangleMesh[_3D]],
    maxModes:  Int
  )(implicit rng: Random): IndexedSeq[Double] = {
    require(meshes.size >= 3, "Need at least 3 shapes for LOOCV generalization.")
    (1 to maxModes).map { m =>
      val distPerShape = meshes.indices.map { j =>
        val trainMeshes = meshes.patch(j, Nil, 1)
        val loocv       = PointDistributionModel.createUsingPCA(
          DataCollection.fromTriangleMesh3DSequence(reference, trainMeshes)
        )
        val allCoeffs = loocv.coefficients(meshes(j))
        // Clamp to loocv.rank to avoid IndexOutOfBoundsException when m > loocv.rank.
        val mClamped  = math.min(m, loocv.rank)
        val truncated = DenseVector.tabulate(loocv.rank) { i =>
          if (i < mClamped) allCoeffs(i) else 0.0
        }
        Metrics.symmetric(loocv.instance(truncated), meshes(j)).mean
      }
      distPerShape.sum / distPerShape.size
    }.toIndexedSeq
  }

  // ── specificity ──────────────────────────────────────────────────────────────

  /**
   * Specificity: how well random SSM samples resemble real scapulae.
   *
   * For each number of retained modes m, draw nSamples random instances (first m
   * coefficients ~ N(0,1), rest 0) and record the closest training shape distance.
   * Low specificity error = model only generates realistic-looking shapes.
   */
  private def specificity(
    ssm:      PointDistributionModel[_3D, TriangleMesh],
    meshes:   IndexedSeq[TriangleMesh[_3D]],
    maxModes: Int,
    nSamples: Int
  )(implicit rng: Random): IndexedSeq[Double] = {
    (1 to maxModes).map { m =>
      val minDists = (0 until nSamples).map { _ =>
        val coeffs = DenseVector.zeros[Double](ssm.rank)
        for (i <- 0 until m) coeffs(i) = rng.scalaRandom.nextGaussian()
        val sample = ssm.instance(coeffs)
        meshes.map(t => Metrics.symmetric(sample, t).mean).min
      }
      minDists.sum / nSamples
    }.toIndexedSeq
  }
}
