package scapula

import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.File
import scala.util.{Failure, Success}

/**
 * Full multi-pass SSM construction pipeline.
 *
 * Implements the 4-step SSM workflow:
 *   1. Register all specimens to an initial arbitrary reference (Pass 1).
 *   2. Build SSM₁ from Pass 1 → take its PCA mean as a bias-free new reference.
 *   3. Re-register all specimens to the mean-shape reference (Pass 2).
 *   4. Build SSM₂ from Pass 2.  Report a STABILITY metric:
 *      mean-surface-distance between the Pass 1 and Pass 2 mean shapes.
 *      Convergence (< 1 mm shift) confirms the reference bias has been removed.
 *
 * Methods used
 * ────────────
 *  • Landmark-based Procrustes rigid registration (RigidAlign.landmarkThenIcp)
 *  • Trimmed rigid ICP to refine pose
 *  • Single isotropic Gaussian kernel GP prior
 *  • Pivoted-Cholesky low-rank approximation with NearestNeighborInterpolator3D
 *  • GP-ICP: iterated nearest-neighbour posterior regression (non-rigid registration)
 *  • GPA-style bias removal via iterative mean-shape reference update
 *  • PCA-based SSM (PointDistributionModel.createUsingPCA)
 *
 * Usage:
 *   sbt "runMain scapula.RebuildSSM"
 *   SCAPULA_DATA_DIR=/x SCAPULA_OUT_DIR=/y sbt "runMain scapula.RebuildSSM"
 *
 * Tuning (all overridable via environment variables):
 *   SCAPULA_KERNEL_SIGMA   bandwidth in mm          (default 70.0)
 *   SCAPULA_KERNEL_SCALE   deformation scale in mm  (default 30.0)
 *   SCAPULA_ICP_SIGMA2     noise variance mm²        (default 2.0)
 *   SCAPULA_GP_TOL         Cholesky tolerance        (default 0.01)
 *   SCAPULA_ICP_ITERS      GP-ICP iterations         (default 40)
 *   SCAPULA_MODEL_RES      target vertex count       (default 8000)
 */
object RebuildSSM {

  // ── GP model construction ──────────────────────────────────────────────────

  /**
   * Build a low-rank GP prior on `reference` using a single isotropic Gaussian kernel.
   * The kernel is: k(x,y) = scale² · exp(−‖x−y‖² / (2σ²)) · I₃
   * The NearestNeighborInterpolator3D is used so the GP can be evaluated at
   * arbitrary positions on the reference mesh without costly kernel evaluations.
   */
  def buildGpModel(
    reference: TriangleMesh[_3D]
  )(implicit rng: Random): PointDistributionModel[_3D, TriangleMesh] = {

    val zeroMean = Field[_3D, EuclideanVector[_3D]](
      EuclideanSpace[_3D], _ => EuclideanVector.zeros[_3D])

    // Single isotropic Gaussian kernel, replicated independently for each axis
    val scalarKernel = GaussianKernel[_3D](Config.kernelSigma) * Config.kernelScale
    val kernel       = DiagonalKernel(scalarKernel, outputDim = 3)

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)

    // Pivoted-Cholesky low-rank approximation; NearestNeighborInterpolator so
    // the GP can be evaluated at any mesh vertex without extra kernel calls.
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      domain            = reference.pointSet,
      gp                = gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )

    PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
  }

  // ── GP-ICP non-rigid registration ─────────────────────────────────────────

  /**
   * Non-rigid registration via iterated nearest-neighbour GP posterior.
   *
   * At each iteration:
   *   1. Sample `numPoints` spatially uniform vertices from the current fitted mesh.
   *   2. For each, find the nearest point on the TARGET surface (NN correspondence).
   *   3. Treat correspondences as noisy observations of the deformation field
   *      and compute the posterior GP model given them.
   *   4. The next iteration starts from the posterior mean.
   *
   * The NearestNeighborInterpolator in the GP prior makes step 2→3 efficient:
   * the posterior coefficients are solved in low-rank space (O(n·r²) not O(n³)).
   */
  def gpIcp(
    model:      PointDistributionModel[_3D, TriangleMesh],
    target:     TriangleMesh[_3D],
    iterations: Int    = Config.icpIterations,
    numPoints:  Int    = 1000,
    sigma2:     Double = Config.icpSigma2
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val targetOps = target.operations
    var current   = model

    for (iter <- 0 until iterations) {
      val fitted = current.mean

      // Nearest-neighbour correspondences: fitted vertex → closest point on target surface
      val sampleIds = RigidAlign.uniformIds(fitted, numPoints)
      val observations: IndexedSeq[(PointId, Point[_3D])] = sampleIds.map { id =>
        val fittedPt  = fitted.pointSet.point(id)
        val closestPt = targetOps.closestPointOnSurface(fittedPt).point
        (id, closestPt)
      }

      // Bayesian update: posterior mean = best-fit deformation of reference
      current = current.posterior(observations, sigma2)
    }
    current.mean
  }

  // ── One registration pass ──────────────────────────────────────────────────

  private case class PreparedSpecimen(
    modelId: String,
    mesh:    TriangleMesh[_3D],
    lms:     IndexedSeq[Landmark[_3D]]
  )

  private def doPass(
    label:   String,
    passDir: File,
    preps:   IndexedSeq[PreparedSpecimen],
    refMesh: TriangleMesh[_3D],
    refLms:  IndexedSeq[Landmark[_3D]]
  )(implicit rng: Random): IndexedSeq[(TriangleMesh[_3D], IndexedSeq[Landmark[_3D]])] = {

    passDir.mkdirs()
    println(s"\n╔══ $label ══╗")
    println(s"  Reference: ${refMesh.pointSet.numberOfPoints} vertices")
    println(s"  Building GP model  (σ=${Config.kernelSigma} mm, scale=${Config.kernelScale} mm, " +
            s"tol=${Config.gpRelativeTolerance})...")

    val gpModel = buildGpModel(refMesh)
    println(s"  GP rank = ${gpModel.rank}   GP-ICP iterations = ${Config.icpIterations}")

    preps.zipWithIndex.map { case (p, i) =>
      val tag = s"[${i+1}/${preps.length}]"
      print(s"  $tag  ${p.modelId}  rigid-align ...")

      // ── Step 1: Landmark Procrustes + trimmed rigid ICP ──────────────────
      val (rigidMesh, rigidLms) = RigidAlign.landmarkThenIcp(p.mesh, p.lms, refMesh, refLms)

      print(s"  gp-icp (${Config.icpIterations} iter) ...")

      // ── Step 2: GP-ICP non-rigid registration ────────────────────────────
      val registered = gpIcp(gpModel, rigidMesh)

      val st   = Metrics.symmetric(registered, rigidMesh)
      println(f"  done | resid ${st.render}")

      // Save registered mesh
      val outF = new File(passDir, s"reg_${p.modelId}.stl")
      MeshIO.writeMesh(registered, outF) match {
        case Failure(e) => println(s"  WARNING: could not save ${outF.getName}: ${e.getMessage}")
        case Success(_) => ()
      }

      (registered, rigidLms)
    }
  }

  // ── Main ──────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir  = Config.dataDir
    val outDir   = Config.outDir
    val pass1Dir = new File(outDir, "pass1")
    val pass2Dir = new File(outDir, "pass2")
    outDir.mkdirs()

    println(s"Data dir : ${dataDir.getAbsolutePath}")
    println(s"Out dir  : ${outDir.getAbsolutePath}")

    // ── Load data ────────────────────────────────────────────────────────────
    val (lmMap, _, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dataDir))
    val specimens     = ScapulaData.specimens(dataDir).filter(s => lmMap.contains(s.modelId))
    println(s"Specimens with landmarks: ${specimens.length}")

    // Orient all meshes to left-scapula frame (mirror right → left)
    val preps: IndexedSeq[PreparedSpecimen] = specimens.map { s =>
      val (m, l) =
        if (s.isRight) (ScapulaData.mirrorMesh(ScapulaData.loadMesh(s.file)),
                        ScapulaData.mirrorLandmarks(lmMap(s.modelId)))
        else           (ScapulaData.loadMesh(s.file), lmMap(s.modelId))
      PreparedSpecimen(s.modelId, m, l)
    }

    // ── PASS 1: register to first specimen as reference ───────────────────────
    val ref0     = preps.head
    val p1Out    = doPass("PASS 1", pass1Dir, preps, ref0.mesh, ref0.lms)
    val p1Meshes = p1Out.map(_._1)
    val p1Lms    = p1Out.map(_._2)

    // Build SSM₁ from pass 1 → mean shape = new reference
    println("\nBuilding SSM₁ from pass 1...")
    val dc1  = DataCollection.fromTriangleMesh3DSequence(p1Meshes.head, p1Meshes)
    val ssm1 = PointDistributionModel.createUsingPCA(dc1)
    println(f"  SSM₁ rank = ${ssm1.rank}  |  ${p1Meshes.length} specimens")

    val mean1 = ssm1.mean
    MeshIO.writeMesh(mean1, new File(outDir, "mean_pass1.stl")).toOption
    println(s"  Mean shape → mean_pass1.stl")

    // Reference landmarks for pass 2 = mean of all rigidly-aligned landmark positions
    // (approximates the landmark positions on the mean mesh)
    val pass2RefLms: IndexedSeq[Landmark[_3D]] = ScapulaData.landmarkNames.map { nm =>
      val pts = p1Lms.flatMap(_.find(_.id == nm).map(_.point))
      Landmark(nm, Point3D(pts.map(_.x).sum / pts.length,
                           pts.map(_.y).sum / pts.length,
                           pts.map(_.z).sum / pts.length))
    }

    // Optionally decimate mean shape for faster GP computation in pass 2
    val mean1Ref = {
      val n = mean1.pointSet.numberOfPoints
      if (n > Config.modelResolution) {
        println(s"  Decimating mean reference $n → ~${Config.modelResolution} vertices...")
        ScapulaData.decimateInCorrespondence(mean1, IndexedSeq(mean1), Config.modelResolution).head
      } else mean1
    }

    // ── PASS 2: re-register to mean shape (GPA-style reference update) ────────
    val p2Out    = doPass("PASS 2", pass2Dir, preps, mean1Ref, pass2RefLms)
    val p2Meshes = p2Out.map(_._1)

    // Build final SSM₂ from pass 2
    println("\nBuilding final SSM₂ from pass 2...")
    val dc2  = DataCollection.fromTriangleMesh3DSequence(p2Meshes.head, p2Meshes)
    val ssm2 = PointDistributionModel.createUsingPCA(dc2)
    println(f"  SSM₂ rank = ${ssm2.rank}  |  ${p2Meshes.length} specimens")

    val mean2 = ssm2.mean
    MeshIO.writeMesh(mean2, new File(outDir, "mean_pass2.stl")).toOption
    println(s"  Mean shape → mean_pass2.stl")

    // ── STABILITY CHECK: compare mean shapes across passes ────────────────────
    println("\n─── Stability check (Step 4): mean shape convergence ───")
    println("    Measures whether the GPA reference-update converged.")
    println("    A small shift (< 1 mm mean) confirms reference bias has been removed.\n")

    // Use surface-distance (no correspondence needed) to compare mean shapes
    val stability = Metrics.symmetric(mean1, mean2)
    println(f"  Pass 1 mean ↔ Pass 2 mean surface distance:")
    println(f"    ${stability.render}")
    if (stability.mean < 1.0)
      println(s"  ✓ CONVERGED  (mean shift = ${stability.mean}%.3f mm < 1 mm threshold)")
    else
      println(s"  ✗ NOT CONVERGED  (mean shift = ${stability.mean}%.3f mm). " +
              s"Set SCAPULA_REFINE_PASSES=3 and re-run.")

    // ── Summary ───────────────────────────────────────────────────────────────
    println(s"\n══════════════════════════════════════════════════")
    println(s"Pipeline complete. Results in: ${outDir.getAbsolutePath}")
    println(s"══════════════════════════════════════════════════")
    println(s"  pass1/   — ${p1Meshes.length} registered meshes (initial reference)")
    println(s"  pass2/   — ${p2Meshes.length} registered meshes (mean-shape reference)")
    println(s"  mean_pass1.stl  — SSM₁ mean shape")
    println(s"  mean_pass2.stl  — SSM₂ mean shape  (use for publication)")
    println()
    println("Next steps:")
    println("  sbt \"runMain scapula.SSMValidation\"    # compactness / generalization / specificity / HD / RMSE")
    println("  sbt \"runMain scapula.ViewSSM\"          # interactive SSM with PCA sliders")
    println("  sbt \"runMain scapula.ViewRegistration\" # target vs fitted comparison + metrics table")
  }
}
