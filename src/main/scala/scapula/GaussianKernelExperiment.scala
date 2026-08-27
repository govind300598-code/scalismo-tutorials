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
 * Gaussian-kernel parameter sweep for scapula FFDM (free-form deformation model).
 *
 * 9-experiment 3×3 grid (one parameter varied at a time from the baseline T5):
 *
 *   Exp  σ (mm)  Scale (mm)  prior h5
 *   T1     40       10       scapula_ffdm_sig40_s10_r5_prior.h5
 *   T2     40       20       scapula_ffdm_sig40_s20_r5_prior.h5
 *   T3     40       30       scapula_ffdm_sig40_s30_r5_prior.h5
 *   T4     65       10       scapula_ffdm_sig65_s10_r5_prior.h5
 *   T5*    65       20       scapula_ffdm_sig65_s20_r5_prior.h5  ← baseline
 *   T6     65       30       scapula_ffdm_sig65_s30_r5_prior.h5
 *   T7     90       10       scapula_ffdm_sig90_s10_r5_prior.h5
 *   T8     90       20       scapula_ffdm_sig90_s20_r5_prior.h5
 *   T9     90       30       scapula_ffdm_sig90_s30_r5_prior.h5
 *
 * Reference:  paired_scapula_001_M_64_L (smooth adult male, no artifacts)
 * Targets (5, chosen for anatomical diversity):
 *   M56 paired_scapula_002_M_56_L
 *   F67 paired_scapula_004_F_67_L
 *   M26 paired_scapula_007_M_26_L
 *   F43 paired_scapula_010_F_43_L
 *   M68 paired_scapula_012_M_68_L
 *
 * Run with:
 *   SCAPULA_DATA_DIR=<stl_dir> SCAPULA_OUT_DIR=<out_dir> sbt "runMain scapula.GaussianKernelExperiment"
 */
object GaussianKernelExperiment {

  val referenceId: String = "paired_scapula_001_M_64_L"

  val targetIds: IndexedSeq[(String, String)] = IndexedSeq(
    ("paired_scapula_002_M_56_L", "M56"),
    ("paired_scapula_004_F_67_L", "F67"),
    ("paired_scapula_007_M_26_L", "M26"),
    ("paired_scapula_010_F_43_L", "F43"),
    ("paired_scapula_012_M_68_L", "M68")
  )

  // 3×3 grid: (expId, sigma_mm, scale_mm)
  val experiments: IndexedSeq[(String, Double, Double)] = IndexedSeq(
    ("T1", 40.0, 10.0),
    ("T2", 40.0, 20.0),
    ("T3", 40.0, 30.0),
    ("T4", 65.0, 10.0),
    ("T5", 65.0, 20.0), // baseline
    ("T6", 65.0, 30.0),
    ("T7", 90.0, 10.0),
    ("T8", 90.0, 20.0),
    ("T9", 90.0, 30.0)
  )

  val targetRank: Int    = 5   // nominal mode count (actual rank from Cholesky may differ)
  val icpMaxDist: Double = 15.0
  val icpSigma2:  Double = 1.0

  final case class RunResult(
    expId: String, sigma: Double, scale: Double, gpRank: Int,
    targetId: String, label: String,
    stats: Metrics.SurfaceStats,
    regVtk: File, postH5: File
  )

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    require(dataDir.exists(), s"Data dir not found: ${dataDir.getAbsolutePath}")

    val outDir = new File(Config.outDir, "GaussianKernelExperiment")
    outDir.mkdirs()
    println(s"Output directory : ${outDir.getAbsolutePath}")

    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader) println("WARNING: landmark columns resolved by fallback.")

    // Reference mesh
    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")
    require(landmarks.contains(referenceId), s"'$referenceId' missing from CSV.")
    val refRaw  = ScapulaData.loadMesh(refFile)
    val refMesh = if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
                    refRaw.operations.decimate(Config.modelResolution) else refRaw
    val refLms  = landmarks(referenceId)
    println(s"Reference : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")

    // Rigid-align all 5 targets into reference space (done once, shared across all experiments)
    println("Rigid-aligning targets into reference space...")
    val alignedTargets: IndexedSeq[(String, String, TriangleMesh[_3D])] = targetIds.flatMap {
      case (tId, label) =>
        val tFile = new File(dataDir, s"$tId.stl")
        if (!tFile.exists()) {
          println(s"  SKIP $tId — STL not found"); None
        } else if (!landmarks.contains(tId)) {
          println(s"  SKIP $tId — not in CSV"); None
        } else {
          val raw  = ScapulaData.loadMesh(tFile)
          val mesh = if (raw.pointSet.numberOfPoints > Config.modelResolution)
                       raw.operations.decimate(Config.modelResolution) else raw
          val (aligned, _) = RigidAlign.landmarkThenIcp(
            mesh, landmarks(tId), refMesh, refLms, icpIterations = 30)
          println(s"  $label  ($tId) — ${aligned.pointSet.numberOfPoints} vertices")
          Some((tId, label, aligned))
        }
    }
    println(s"${alignedTargets.size} targets ready.\n")

    val ptStride = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds    = (0 until refMesh.pointSet.numberOfPoints by ptStride).map(PointId(_))

    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None
    ui.foreach { u =>
      val g = u.createGroup("Reference + Targets (rigid)")
      u.show(g, refMesh, "Reference")
      alignedTargets.foreach { case (id, lbl, m) => u.show(g, m, s"$lbl $id") }
    }

    val allResults = scala.collection.mutable.Buffer.empty[RunResult]

    for ((expId, sigma, scale) <- experiments) {
      println(s"=== $expId  σ=${sigma.toInt} mm  scale=${scale.toInt} mm ===")

      val kernel = DiagonalKernel(GaussianKernel[_3D](sigma) * scale, 3)
      val gp     = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

      val priorLRGP = LowRankGaussianProcess.approximateGPCholesky(
        refMesh, gp,
        relativeTolerance = 0.01,
        interpolator      = NearestNeighborInterpolator3D()
      )
      val priorModel = PointDistributionModel[_3D, TriangleMesh](refMesh, priorLRGP)
      val gpRank = priorModel.rank
      println(s"  GP rank: $gpRank")

      // Save prior model (named by parameters for easy comparison)
      val priorH5 = new File(outDir,
        s"scapula_ffdm_sig${sigma.toInt}_s${scale.toInt}_r${targetRank}_prior.h5")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(priorModel, priorH5).get
      println(s"  Prior h5 : ${priorH5.getName}")

      // ±3std VTKs for the first (up to 5) modes — visual inspection of deformation reach
      for (mi <- 0 until math.min(targetRank, gpRank)) {
        val v = DenseVector.zeros[Double](gpRank)
        v(mi) = 3.0;  MeshIO.writeMesh(priorModel.instance(v),
          new File(outDir, s"${expId}_mode${mi + 1}_pos3std.vtk")).get
        v(mi) = -3.0; MeshIO.writeMesh(priorModel.instance(v),
          new File(outDir, s"${expId}_mode${mi + 1}_neg3std.vtk")).get
      }

      // GP non-rigid ICP — one run per target
      for ((tId, label, targetAligned) <- alignedTargets) {
        var model = priorModel

        for (iter <- 0 until Config.icpIterations) {
          val meanMesh = model.mean
          val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
            val pt      = meanMesh.pointSet.point(pid)
            val nearest = targetAligned.operations.closestPointOnSurface(pt).point
            if ((nearest - pt).norm < icpMaxDist) Some((pid, nearest)) else None
          }
          if (correspondences.nonEmpty)
            model = model.posterior(correspondences, sigma2 = icpSigma2)
        }

        val regMesh = model.mean
        val stats   = Metrics.symmetric(regMesh, targetAligned)
        println(f"  $label  ${stats.render}")

        val regVtk = new File(outDir, s"${expId}_${label}_registered.vtk")
        val postH5 = new File(outDir, s"${expId}_${label}_posterior.h5")
        MeshIO.writeMesh(regMesh, regVtk).get
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, postH5).get

        ui.foreach { u =>
          val g = u.createGroup(s"$expId - $label")
          u.show(g, targetAligned, "target")
          u.show(g, regMesh, "registered")
        }

        allResults += RunResult(expId, sigma, scale, gpRank, tId, label, stats, regVtk, postH5)
      }
      println()
    }

    // ---- Per-experiment summary (averaged over 5 targets) --------------------
    println()
    println(f"${"Exp"}%-4s  ${"σ mm"}%-6s  ${"scale"}%-6s  ${"rank"}%-5s  " +
            f"${"mean mm"}%-9s  ${"rms mm"}%-9s  ${"HD95 mm"}%-9s  note")
    println("-" * 78)
    for ((expId, sigma, scale) <- experiments) {
      val rows = allResults.filter(_.expId == expId)
      if (rows.nonEmpty) {
        val n       = rows.size
        val avgMean = rows.map(_.stats.mean).sum / n
        val avgRms  = rows.map(_.stats.rms).sum  / n
        val avgHd95 = rows.map(_.stats.hd95).sum / n
        val r       = rows.head.gpRank
        val note    = if (expId == "T5") "baseline" else ""
        println(f"$expId%-4s  $sigma%-6.0f  $scale%-6.0f  $r%-5d  " +
                f"$avgMean%7.2f mm   $avgRms%7.2f mm   $avgHd95%7.2f mm   $note")
      }
    }

    // ---- Per-target detail for best and worst experiment --------------------
    val byExp = allResults.groupBy(_.expId).view.mapValues { rows =>
      val n = rows.size.toDouble
      rows.map(_.stats.mean).sum / n
    }.toMap

    val bestExp  = byExp.minBy(_._2)._1
    val worstExp = byExp.maxBy(_._2)._1
    println(s"\nBest experiment  : $bestExp  (avg mean ${byExp(bestExp)%.2f} mm)")
    println(s"Worst experiment : $worstExp  (avg mean ${byExp(worstExp)%.2f} mm)")

    for (tag <- Seq("BEST" -> bestExp, "WORST" -> worstExp)) {
      println(s"\n  ${tag._1} ($${tag._2}) per target:")
      allResults.filter(_.expId == tag._2).foreach { r =>
        println(f"    ${r.label}%-5s ${r.stats.render}")
      }
    }

    // ---- CSV -----------------------------------------------------------------
    val csvOut = new File(outDir, "comparison_table.csv")
    val pw = new PrintWriter(csvOut)
    try {
      pw.println("exp_id,sigma_mm,scale_mm,gp_rank,target_id,target_label," +
                 "mean_mm,rms_mm,hd95_mm,hd_mm")
      allResults.foreach { r =>
        pw.println(f"${r.expId},${r.sigma},${r.scale},${r.gpRank}," +
                   f"${r.targetId},${r.label}," +
                   f"${r.stats.mean}%.3f,${r.stats.rms}%.3f," +
                   f"${r.stats.hd95}%.3f,${r.stats.hd}%.3f")
      }
    } finally pw.close()
    println(s"\nCSV : ${csvOut.getAbsolutePath}")
    println("""
Interpretation guide
  σ (sigma)  : spatial reach of the kernel.
               Small (40 mm) -> deformation stays local (individual ridges, processes).
               Large (90 mm) -> global bending of the whole blade.
  scale      : amplitude (std ≈ sqrt(scale) mm of pointwise displacement).
               Small scale -> tight prior, model resists large deformations.
               Large scale -> looser prior, model follows the target surface more aggressively.
  Best fit   : lowest mean symmetric surface distance after GP-ICP.
  Worst fit  : highest mean distance — usually paired with σ too small (can't reach global shape)
               or scale too small (posterior doesn't move far enough from reference).
""")
    ui.foreach(_ => println("UI open — close the window to exit."))
  }
}
