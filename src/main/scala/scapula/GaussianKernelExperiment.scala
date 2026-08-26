package scapula

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Single Gaussian-kernel experiment — full standard pipeline.
 *
 * Pipeline per (sigma, scale) config:
 *   1. Load reference mesh + landmarks
 *   2. Load ONE target mesh + landmarks
 *   3. Rigid-align target into reference space (landmark Procrustes + trimmed ICP)
 *   4. Build GP prior  : DiagonalKernel(GaussianKernel(sigma) * scale, 3), rank = 5
 *   5. GP non-rigid ICP: iteratively find closest-point correspondences,
 *      compute Bayesian posterior, repeat — this is the standard Scalismo pipeline
 *   6. Extract registered mesh (posterior mean)
 *   7. Save prior .h5, posterior .h5, registered mesh .vtk
 *   8. Save +-3std deformed meshes for each of 5 prior modes
 *   9. Report surface-distance quality and print comparison table
 *
 * Run:
 *   SCAPULA_DATA_DIR=/path/to/paired_scapulae_STLs \
 *   SCAPULA_OUT_DIR=/path/to/output \
 *   sbt "runMain scapula.GaussianKernelExperiment"
 *
 * SCAPULA_UI=false for headless.
 */
object GaussianKernelExperiment {

  // ── IDs ────────────────────────────────────────────────────────────────────
  val referenceId: String = "paired_scapula_001_M_64_L"
  val targetId:    String = "paired_scapula_002_M_56_L"   // ONE target only

  // ── GP settings ────────────────────────────────────────────────────────────
  val gpRank:         Int    = 5
  val icpIterations:  Int    = Config.icpIterations        // default 40
  val icpSigma2:      Double = 1.0    // GP ICP noise variance (mm^2)
  val icpMaxDist:     Double = 15.0   // correspondence rejection threshold (mm)

  // ── Experiment grid ────────────────────────────────────────────────────────
  // Set A: vary sigma, scale fixed at 20
  // Set B: vary scale, sigma fixed at 65
  // Baseline (65,20) shared between both sets.
  val experiments: IndexedSeq[(Double, Double)] = IndexedSeq(
    (50.0, 20.0),   // A-1 narrow kernel
    (65.0, 20.0),   // A-2 / B-2  baseline
    (80.0, 20.0),   // A-3 wide kernel
    (65.0, 10.0),   // B-1 small amplitude
    (65.0, 30.0)    // B-3 large amplitude
  )

  // ── Helpers ────────────────────────────────────────────────────────────────

  def tag(sigma: Double, scale: Double): String =
    f"sigma${sigma.toInt}_scale${scale.toInt}_rank$gpRank"

  def modesInstance(model: PointDistributionModel[_3D, TriangleMesh],
                    modeIdx: Int,
                    coeff: Double): TriangleMesh[_3D] = {
    val v = DenseVector.zeros[Double](model.rank)
    v(modeIdx) = coeff
    model.instance(v)
  }

  def maxPointDisp(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): Double =
    a.pointSet.points.zip(b.pointSet.points)
      .map { case (p, q) => (p - q).norm }
      .toIndexedSeq.max

  // ── Result record ──────────────────────────────────────────────────────────

  final case class Result(
    sigma:        Double,
    scale:        Double,
    rank:         Int,
    surfDist:     Metrics.SurfaceStats,
    priorH5:      File,
    postH5:       File,
    regVtk:       File,
    modeMaxDisp:  IndexedSeq[Double]   // max disp at +3std, prior modes
  )

  // ── Main ───────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // Paths
    val dataDir = Config.dataDir
    require(dataDir.exists(),
      s"Data directory not found: ${dataDir.getAbsolutePath}\n" +
        "Set SCAPULA_DATA_DIR to the folder containing the STL files.")

    val outDir = new File(Config.outDir, "GaussianKernelExperiment")
    outDir.mkdirs()
    println(s"Output directory : ${outDir.getAbsolutePath}")

    // ── Landmarks ──
    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("  WARNING: landmark columns resolved by fallback — verify CSV.")
    require(landmarks.contains(referenceId),
      s"'$referenceId' missing from landmark CSV.")
    require(landmarks.contains(targetId),
      s"'$targetId' missing from landmark CSV.")

    // ── Reference mesh ──
    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")
    val refRaw = ScapulaData.loadMesh(refFile)
    val refMesh: TriangleMesh[_3D] =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
        refRaw.operations.decimate(Config.modelResolution)
      else refRaw
    val refLms = landmarks(referenceId)

    println(s"Reference        : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")

    // ── Target mesh — load once, rigid-align once ──
    val tgtFile = new File(dataDir, s"$targetId.stl")
    require(tgtFile.exists(), s"Target STL not found: ${tgtFile.getAbsolutePath}")
    val tgtRaw = ScapulaData.loadMesh(tgtFile)
    val tgtMesh: TriangleMesh[_3D] =
      if (tgtRaw.pointSet.numberOfPoints > Config.modelResolution)
        tgtRaw.operations.decimate(Config.modelResolution)
      else tgtRaw
    val tgtLms = landmarks(targetId)

    // Rigid-align target into reference space — landmark Procrustes + ICP
    val (targetAligned, _) = RigidAlign.landmarkThenIcp(
      tgtMesh, tgtLms, refMesh, refLms, icpIterations = 30
    )
    println(s"Target (aligned) : $targetId  (${targetAligned.pointSet.numberOfPoints} vertices)")
    println(s"GP rank (fixed)  : $gpRank")
    println(s"ICP iterations   : $icpIterations")
    println()

    // Uniform point ids for ICP correspondences
    val ptStride = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds    = (0 until refMesh.pointSet.numberOfPoints by ptStride).map(PointId(_))

    // Optional UI
    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None

    ui.foreach { scalismoUi =>
      val grp = scalismoUi.createGroup("Reference + Target")
      scalismoUi.show(grp, refMesh,      "Reference")
      scalismoUi.show(grp, targetAligned,"Target (aligned)")
    }

    // ── Experiments ────────────────────────────────────────────────────────

    val results = scala.collection.mutable.Buffer.empty[Result]

    for ((sigma, scale) <- experiments) {
      val t = tag(sigma, scale)
      println(s"=== $t ===")

      // 1. GP prior — same kernel formula as RegisterAllFiveScapulae
      val kernel = DiagonalKernel(GaussianKernel[_3D](sigma) * scale, 3)
      val gp     = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

      // 2. Low-rank prior with fixed rank = gpRank via Cholesky decomposition.
      //    NearestNeighborInterpolator3D extends the basis to the full surface.
      //    relativeTolerance is set high enough that the rank stays at gpRank.
      val priorLRGP = LowRankGaussianProcess.approximateGPCholesky(
        refMesh,
        gp,
        relativeTolerance = 0.01,
        interpolator      = NearestNeighborInterpolator3D()
      )
      // Save prior .h5 (for mode visualisation)
      val priorModel = PointDistributionModel[_3D, TriangleMesh](refMesh, priorLRGP)
      println(s"  Prior rank     : ${priorModel.rank}")
      val priorH5 = new File(outDir, s"${t}_prior.h5")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(priorModel, priorH5).get
      println(s"  Saved prior.h5 : ${priorH5.getName}")

      // 3. GP non-rigid ICP — standard Scalismo posterior update loop
      println(s"  Running GP non-rigid ICP ($icpIterations iterations)...")
      var model = priorModel
      for (iter <- 0 until icpIterations) {
        val meanMesh = model.mean
        val correspondences: IndexedSeq[(PointId, Point[_3D])] =
          ptIds.flatMap { pid =>
            val pt      = meanMesh.pointSet.point(pid)
            val nearest = targetAligned.operations.closestPointOnSurface(pt).point
            if ((nearest - pt).norm < icpMaxDist) Some((pid, nearest)) else None
          }
        if (correspondences.nonEmpty)
          model = model.posterior(correspondences, sigma2 = icpSigma2)
        if (iter == 0 || (iter + 1) % 10 == 0 || iter == icpIterations - 1)
          println(f"    iter ${iter + 1}%3d / $icpIterations : ${correspondences.size} correspondences")
      }

      // 4. Extract registered mesh (posterior mean)
      val registeredMesh = model.mean

      // 5. Surface-distance quality
      val distStats = Metrics.symmetric(registeredMesh, targetAligned)
      println(s"  Quality        : ${distStats.render}")

      // 6. Save posterior .h5 and registered mesh .vtk
      val postH5  = new File(outDir, s"${t}_posterior.h5")
      val regVtk  = new File(outDir, s"${t}_registered.vtk")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, postH5).get
      MeshIO.writeMesh(registeredMesh, regVtk).get
      println(s"  Saved post.h5  : ${postH5.getName}")
      println(s"  Saved reg.vtk  : ${regVtk.getName}")

      // 7. Save +-3std deformed meshes for each of the 5 PRIOR modes
      val modeMaxDisp = scala.collection.mutable.Buffer.empty[Double]
      for (modeIdx <- 0 until math.min(gpRank, priorModel.rank)) {
        val pos = modesInstance(priorModel, modeIdx,  3.0)
        val neg = modesInstance(priorModel, modeIdx, -3.0)
        MeshIO.writeMesh(pos, new File(outDir, s"${t}_mode${modeIdx+1}_pos3std.vtk")).get
        MeshIO.writeMesh(neg, new File(outDir, s"${t}_mode${modeIdx+1}_neg3std.vtk")).get
        val maxD = maxPointDisp(refMesh, pos)
        modeMaxDisp += maxD
        println(f"  Prior mode ${modeIdx+1}  : max deform = $maxD%5.1f mm  (+-3std)")
      }

      // 8. UI
      ui.foreach { scalismoUi =>
        val grp = scalismoUi.createGroup(t)
        scalismoUi.show(grp, priorModel,    s"prior ($t)")
        scalismoUi.show(grp, model,          s"posterior ($t)")
        scalismoUi.show(grp, registeredMesh,"registered mean")
        for (modeIdx <- 0 until math.min(gpRank, priorModel.rank)) {
          scalismoUi.show(grp, modesInstance(priorModel, modeIdx,  3.0), s"mode${modeIdx+1} +3std")
          scalismoUi.show(grp, modesInstance(priorModel, modeIdx, -3.0), s"mode${modeIdx+1} -3std")
        }
      }

      results += Result(sigma, scale, priorModel.rank, distStats,
                        priorH5, postH5, regVtk, modeMaxDisp.toIndexedSeq)
      println()
    }

    // ── Comparison table ───────────────────────────────────────────────────

    val csvFile = new File(outDir, "comparison_table.csv")
    writeCSV(results.toIndexedSeq, csvFile)
    println(s"Comparison CSV : ${csvFile.getAbsolutePath}\n")
    printTable(results.toIndexedSeq)
    println()
    printInterpretation()

    ui.foreach(_ => println("\nUI open — close the window to exit."))
  }

  // ── Reporting ──────────────────────────────────────────────────────────────

  def writeCSV(results: IndexedSeq[Result], file: File): Unit = {
    val pw = new PrintWriter(file)
    try {
      val modeHdrs = (1 to gpRank).map(i => s"mode${i}_max_mm").mkString(",")
      pw.println(s"sigma_mm,scale_mm,rank,mean_dist_mm,rms_mm,hd95_mm," +
                 s"$modeHdrs,prior_h5,posterior_h5,reg_vtk,set")
      results.foreach { r =>
        val maxes = r.modeMaxDisp.padTo(gpRank, 0.0).take(gpRank)
                      .map(d => f"$d%.1f").mkString(",")
        val set = if (r.scale == 20.0 && r.sigma != 65.0) "vary_sigma"
                  else if (r.sigma == 65.0 && r.scale != 20.0) "vary_scale"
                  else "baseline"
        pw.println(
          f"${r.sigma},${r.scale},${r.rank}," +
          f"${r.surfDist.mean}%.2f,${r.surfDist.rms}%.2f,${r.surfDist.hd95}%.2f," +
          s"$maxes,${r.priorH5.getName},${r.postH5.getName},${r.regVtk.getName},$set"
        )
      }
    } finally pw.close()
  }

  def printTable(results: IndexedSeq[Result]): Unit = {
    println(f"${"σ mm"}%-6s  ${"scale"}%-6s  ${"rank"}%-5s  " +
            f"${"mean dist"}%-10s  ${"HD95"}%-8s  ${"mode1 max"}%-10s  ${"mode2 max"}%-10s  prior h5")
    println("-" * 95)
    results.foreach { r =>
      val m1 = r.modeMaxDisp.headOption.map(d => f"$d%.1f mm").getOrElse("n/a")
      val m2 = r.modeMaxDisp.lift(1).map(d => f"$d%.1f mm").getOrElse("n/a")
      println(
        f"${r.sigma}%-6.0f  ${r.scale}%-6.0f  ${r.rank}%-5d  " +
        f"${r.surfDist.mean}%6.2f mm    ${r.surfDist.hd95}%5.2f mm   " +
        f"${m1}%-10s  ${m2}%-10s  ${r.priorH5.getName}"
      )
    }
  }

  def printInterpretation(): Unit = {
    println("=" * 65)
    println("HOW sigma AND scale AFFECT DEFORMATION")
    println("=" * 65)
    println()
    println("sigma (length-scale, mm):")
    println("  Small sigma -> local, patch-like deformations.")
    println("  Large sigma -> global, whole-bone smooth deformations.")
    println("  Also affects GP ICP: large sigma allows the model to")
    println("  account for large-scale shape differences between subjects.")
    println()
    println("scale (kernel amplitude, = variance):")
    println("  Prior std = sqrt(scale) mm per axis.")
    println("  scale=10 -> std ~3.2 mm   +-3std ~9.5 mm")
    println("  scale=20 -> std ~4.5 mm   +-3std ~13.4 mm  (baseline)")
    println("  scale=30 -> std ~5.5 mm   +-3std ~16.4 mm")
    println("  Larger scale allows bigger deformations in the ICP fit,")
    println("  reducing surface distance but risking over-smoothing.")
    println()
    println("rank = 5 throughout (5 deformation modes).")
    println("=" * 65)
  }
}
