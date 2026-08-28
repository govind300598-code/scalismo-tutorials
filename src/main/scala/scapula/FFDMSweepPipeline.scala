package scapula

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.mesh.*
import scalismo.registration.*
import scalismo.io.MeshIO
import scalismo.numerics.*
import scalismo.kernels.*
import scalismo.statisticalmodel.*
import breeze.linalg.DenseVector
import scalismo.ui.api.*
import scalismo.utils.Random
import scalismo.utils.Random.FixedSeed.randBasis

import java.io.{File, PrintWriter}
import java.awt.Color

/**
 * Free-Form Deformation Model (FFDM) sweep pipeline.
 *
 * A FFDM is a GP deformation model built from a single isotropic Gaussian kernel.
 * Its two hyperparameters are:
 *   sigma        – length scale (mm): controls spatial reach of deformations.
 *   scaleFactor  – amplitude: controls maximum deformation magnitude.
 *
 * Phase 1 — Grid sweep: 7 sigma × 3 scaleFactor = 21 kernel combinations.
 *            Each is fitted (20 iterations) to 5 diverse target scapulae and
 *            evaluated by symmetric surface-distance metrics.
 *
 * Phase 2 — Full registration: the winning (sigma, scaleFactor) pair is used to
 *            register all 5 targets (Config.icpIterations), with UI visualisation
 *            and CSV export of mean / RMS / HD95 for best and worst cases.
 *
 * Outputs (written to Config.outDir):
 *   ffdm_sweep_report.csv   – full grid results (one row per kernel combination)
 *   ffdm_final_results.csv  – per-target metrics with optimal parameters
 *   ffdm_fitted_<id>.stl    – registered meshes
 */
object FFDMSweepPipeline extends App {

  scalismo.initialize()
  implicit val rng: Random = Random(Config.seed)

  val dir    = Config.dataDir
  val outDir = Config.outDir
  outDir.mkdirs()

  val (allLandmarks, _, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dir))

  // -------------------------------------------------------------------------
  // Reference scapula: a smooth specimen without surface artefacts.
  // Switch via SCAPULA_DATA_DIR / SCAPULA_REF_ID env vars or edit below.
  // -------------------------------------------------------------------------
  val referenceId = sys.env.getOrElse("SCAPULA_REF_ID", "paired_scapula_001_M_64_L")
  require(allLandmarks.contains(referenceId),
    s"Reference '$referenceId' not in CSV. Check SCAPULA_REF_ID or edit FFDMSweepPipeline.scala.")

  val refRaw  = ScapulaData.loadMesh(new File(dir, s"$referenceId.stl"))
  val refMesh = if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
                  refRaw.operations.decimate(Config.modelResolution)
                else refRaw
  val refLms  = allLandmarks(referenceId)

  println(s"Reference  : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices after decimation)")

  // -------------------------------------------------------------------------
  // 5 diverse target scapulae — varied sex, age, and anticipated bone shape.
  // These are deliberately spread across the dataset so the sweep reflects a
  // realistic cross-section of anatomical variability.
  // -------------------------------------------------------------------------
  val targetIds = IndexedSeq(
    "paired_scapula_002_M_56_L",   // M 56 y — similar age to reference
    "paired_scapula_004_F_67_L",   // F 67 y — different sex, older
    "paired_scapula_007_M_26_L",   // M 26 y — young, likely smaller/slimmer
    "paired_scapula_010_F_43_L",   // F 43 y — mid-age female
    "paired_scapula_012_M_68_L"    // M 68 y — oldest male
  )

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  /** Landmark Procrustes to the reference frame (no ICP — shape is preserved). */
  def loadAndAlign(id: String): TriangleMesh[_3D] = {
    val raw  = ScapulaData.loadMesh(new File(dir, s"$id.stl"))
    val mesh = if (raw.pointSet.numberOfPoints > Config.modelResolution)
                 raw.operations.decimate(Config.modelResolution)
               else raw
    val t    = ScapulaData.rigidFromLandmarks(allLandmarks(id), refLms)
    mesh.transform(t)
  }

  /**
   * Build a single-kernel Gaussian FFDM prior on the reference surface.
   * The kernel covariance is  k(x,y) = scaleFactor² · exp(−‖x−y‖² / 2σ²) · I₃.
   */
  def buildFFDM(sigma: Double,
                scaleFactor: Double): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    val kernel   = DiagonalKernel3D(GaussianKernel[_3D](sigma = sigma, scaleFactor = scaleFactor), 3)
    val zeroMean = Field(EuclideanSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp       = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)
    LowRankGaussianProcess.approximateGPCholesky(
      refMesh, gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )
  }

  /**
   * Non-rigid registration: find the FFDM parameters that deform refMesh
   * to best match the target, then return the deformed reference mesh.
   */
  def fitFFDM(target: TriangleMesh[_3D],
              gp: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
              iterations: Int,
              regWeight: Double): TriangleMesh[_3D] = {
    val tSpace   = GaussianProcessTransformationSpace[_3D](gp)
    val fixedImg = refMesh.operations.toDistanceImage
    val movImg   = target.operations.toDistanceImage
    val sampler  = FixedPointsUniformMeshSampler3D(refMesh, 2000)
    val metric   = MeanSquaresMetric(fixedImg, movImg, tSpace, sampler)
    val optim    = LBFGSOptimizer(maxNumberOfIterations = iterations)
    val reg      = Registration(metric, L2Regularizer[_3D](tSpace),
                                regularizationWeight = regWeight, optim)
    val params   = reg.iterator(DenseVector.zeros[Double](gp.rank)).toSeq.last.parameters
    refMesh.transform(tSpace.transformationForParameters(params))
  }

  // Pre-load and align all targets once (outside the sweep loop to avoid redundant I/O).
  println("\nLoading and aligning targets ...")
  val alignedTargets: IndexedSeq[(String, TriangleMesh[_3D])] = targetIds.map { id =>
    require(allLandmarks.contains(id),
      s"Target '$id' not in CSV. Check SCAPULA_DATA_DIR or edit FFDMSweepPipeline.scala.")
    println(s"  $id")
    id -> loadAndAlign(id)
  }

  // =========================================================================
  // Phase 1 – Grid sweep
  //
  //   sigma (length scale, mm) : 20  40  60  80  100  120  140
  //   scaleFactor (amplitude)  :  5  10  15
  //
  // Each combination is evaluated with 20 optimisation iterations to keep
  // sweep time manageable; full-resolution fitting is done in Phase 2.
  // =========================================================================
  val sigmaValues  = Seq(20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0)
  val scaleFactors = Seq(5.0, 10.0, 15.0)

  println(s"\n=== Phase 1 – FFDM Grid Sweep ===")
  println(s"  ${sigmaValues.length} sigma × ${scaleFactors.length} scaleFactor = " +
          s"${sigmaValues.length * scaleFactors.length} combinations × ${targetIds.length} targets")
  println(s"  Iterations per fit (sweep): 20\n")

  case class SweepRow(sigma: Double, scale: Double, rank: Int,
                      meanMean: Double, meanRms: Double, meanHd95: Double,
                      perTarget: IndexedSeq[Metrics.SurfaceStats])

  val sweepWriter = new PrintWriter(new File(outDir, "ffdm_sweep_report.csv"))
  val targetCols  = targetIds.flatMap(id =>
    Seq(s"Mean_${id.takeRight(8)}", s"RMS_${id.takeRight(8)}", s"HD95_${id.takeRight(8)}"))
  sweepWriter.println(("Sigma_mm,ScaleFactor,Rank" +: targetCols :+ "AllTargets_MeanMean").mkString(","))

  var allRows = Seq.empty[SweepRow]

  for (sigma <- sigmaValues; scale <- scaleFactors) {
    val gp    = buildFFDM(sigma, scale)
    val stats = alignedTargets.map { case (_, tgt) =>
      val fitted = fitFFDM(tgt, gp, iterations = 20, regWeight = 1e-4)
      Metrics.symmetric(fitted, tgt)
    }
    val meanMean = stats.map(_.mean).sum / stats.length
    val meanRms  = stats.map(_.rms ).sum / stats.length
    val meanHd95 = stats.map(_.hd95).sum / stats.length
    val row      = SweepRow(sigma, scale, gp.rank, meanMean, meanRms, meanHd95, stats)
    allRows :+= row

    val detail = stats.flatMap(s => Seq(f"${s.mean}%.4f", f"${s.rms}%.4f", f"${s.hd95}%.4f")).mkString(",")
    sweepWriter.println(f"${sigma}%.0f,${scale}%.0f,${gp.rank},$detail,${meanMean}%.4f")
    sweepWriter.flush()
    println(f"  σ=${sigma}%4.0f mm   amp=${scale}%2.0f   rank=${gp.rank}%3d   " +
            f"mean(all 5 targets)=${meanMean}%.4f mm   HD95=${meanHd95}%.4f mm")
  }
  sweepWriter.close()

  val bestRow  = allRows.minBy(_.meanMean)
  val worstRow = allRows.maxBy(_.meanMean)
  println(f"\n  Best  : σ=${bestRow.sigma}%.0f mm  amp=${bestRow.scale}%.0f  " +
          f"→ mean=${bestRow.meanMean}%.4f mm  HD95=${bestRow.meanHd95}%.4f mm")
  println(f"  Worst : σ=${worstRow.sigma}%.0f mm  amp=${worstRow.scale}%.0f  " +
          f"→ mean=${worstRow.meanMean}%.4f mm  HD95=${worstRow.meanHd95}%.4f mm")

  // Print the sweep grid as a table (mean surface distance, mm) for the report
  println("\n  === Sweep Table: mean surface distance (mm) — rows = sigma, cols = scaleFactor ===")
  print(f"  ${"σ \\ amp"}%-8s")
  scaleFactors.foreach(s => print(f"  $s%6.0f"))
  println()
  for (sigma <- sigmaValues) {
    print(f"  ${sigma}%8.0f")
    for (scale <- scaleFactors) {
      val r = allRows.find(r => r.sigma == sigma && r.scale == scale).get
      print(f"  ${r.meanMean}%6.4f")
    }
    println()
  }

  // =========================================================================
  // Phase 2 – Full registration with optimal (sigma, scaleFactor)
  // =========================================================================
  println(s"\n=== Phase 2 – Full FFDM Registration ===")
  println(f"  Optimal: σ=${bestRow.sigma}%.0f mm   scaleFactor=${bestRow.scale}%.0f   rank=${bestRow.rank}")
  println(s"  Iterations: ${Config.icpIterations}\n")

  val optGP = buildFFDM(bestRow.sigma, bestRow.scale)

  val finalWriter = new PrintWriter(new File(outDir, "ffdm_final_results.csv"))
  finalWriter.println("ID,Mean_mm,RMS_mm,HD95_mm,HD_mm")

  val ui       = ScalismoUI()
  val regGroup = ui.createGroup("FFDM Registration – sigma=%.0f amp=%.0f".format(bestRow.sigma, bestRow.scale))

  val refView  = ui.show(regGroup, refMesh, s"Reference ($referenceId)")
  refView.color = Color.LIGHT_GRAY

  val finalResults: IndexedSeq[(String, Metrics.SurfaceStats)] =
    alignedTargets.zipWithIndex.map { case ((id, aligned), idx) =>
      println(s"  [${idx + 1}/${alignedTargets.length}] Fitting $id ...")
      val fitted = fitFFDM(aligned, optGP, iterations = Config.icpIterations, regWeight = 1e-4)
      MeshIO.writeMesh(fitted, new File(outDir, s"ffdm_fitted_$id.stl")).get
      val st = Metrics.symmetric(fitted, aligned)
      finalWriter.println(f"$id,${st.mean}%.4f,${st.rms}%.4f,${st.hd95}%.4f,${st.hd}%.4f")
      finalWriter.flush()
      println(f"  ${st.render}")

      val tv = ui.show(regGroup, aligned, s"Target   $id")
      tv.color = Color.RED
      val fv = ui.show(regGroup, fitted,  s"Fitted   $id")
      fv.color = Color.GRAY

      id -> st
    }
  finalWriter.close()

  val bestCase  = finalResults.minBy(_._2.mean)
  val worstCase = finalResults.maxBy(_._2.mean)

  println(s"\n  Best  case: ${bestCase._1}")
  println(s"    ${bestCase._2.render}")
  println(s"  Worst case: ${worstCase._1}")
  println(s"    ${worstCase._2.render}")
  println(s"\nOutputs written to: ${outDir.getAbsolutePath}")
  println("Press ENTER to close the UI and exit.")
  scala.io.StdIn.readLine()
}
