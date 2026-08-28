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

import java.io.{File, PrintWriter}
import java.awt.Color

object FFDMSweepPipeline extends App {

  scalismo.initialize()
  implicit val rng: Random = Random(Config.seed)

  val dir    = Config.dataDir
  val outDir = Config.outDir
  outDir.mkdirs()

  // Avoid Scala 3 cyclic-value inference: pull the tuple apart with explicit types
  val landmarkData                                         = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dir))
  val allLandmarks: Map[String, IndexedSeq[Landmark[_3D]]] = landmarkData._1

  val referenceId = sys.env.getOrElse("SCAPULA_REF_ID", "paired_scapula_001_M_64_L")
  require(allLandmarks.contains(referenceId), s"Reference '$referenceId' not in CSV.")

  val refRaw: TriangleMesh[_3D] = ScapulaData.loadMesh(new File(dir, s"$referenceId.stl"))
  val refMesh: TriangleMesh[_3D] =
    if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
      refRaw.operations.decimate(Config.modelResolution)
    else refRaw
  val refLms = allLandmarks(referenceId)

  println(s"Reference  : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")

  val targetIds = IndexedSeq(
    "paired_scapula_002_M_56_L",
    "paired_scapula_004_F_67_L",
    "paired_scapula_007_M_26_L",
    "paired_scapula_010_F_43_L",
    "paired_scapula_012_M_68_L"
  )

  def loadAndAlign(id: String): TriangleMesh[_3D] = {
    val raw: TriangleMesh[_3D] = ScapulaData.loadMesh(new File(dir, s"$id.stl"))
    val mesh: TriangleMesh[_3D] =
      if (raw.pointSet.numberOfPoints > Config.modelResolution)
        raw.operations.decimate(Config.modelResolution)
      else raw
    val t = ScapulaData.rigidFromLandmarks(allLandmarks(id), refLms)
    mesh.transform(t)
  }

  def buildFFDM(sigma: Double, scaleFactor: Double): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
    val kernel   = DiagonalKernel3D(GaussianKernel[_3D](sigma, scaleFactor), 3)
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp       = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)
    LowRankGaussianProcess.approximateGPCholesky(
      refMesh, gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )
  }

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
    val reg      = Registration(metric, L2Regularizer[_3D](tSpace), regularizationWeight = regWeight, optim)
    val params   = reg.iterator(DenseVector.zeros[Double](gp.rank)).toSeq.last.parameters
    refMesh.transform(tSpace.transformationForParameters(params))
  }

  println("\nLoading and aligning targets ...")
  val alignedTargets: IndexedSeq[(String, TriangleMesh[_3D])] = targetIds.map { id =>
    require(allLandmarks.contains(id), s"Target '$id' not in CSV.")
    println(s"  $id")
    id -> loadAndAlign(id)
  }

  val sigmaValues  = Seq(20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0)
  val scaleFactors = Seq(5.0, 10.0, 15.0)

  println(s"\n=== Phase 1 - FFDM Grid Sweep ===")
  println(s"  ${sigmaValues.length} sigma x ${scaleFactors.length} scaleFactor = " +
          s"${sigmaValues.length * scaleFactors.length} combinations x ${targetIds.length} targets")
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
    val stats: IndexedSeq[Metrics.SurfaceStats] = alignedTargets.map { case (_, tgt) =>
      val fitted = fitFFDM(tgt, gp, iterations = 20, regWeight = 1e-4)
      Metrics.symmetric(fitted, tgt)
    }
    // Use foldLeft to avoid Scala 3 Numeric[Double] implicit ambiguity on .sum
    val meanMean: Double = stats.foldLeft(0.0)(_ + _.mean) / stats.length
    val meanRms:  Double = stats.foldLeft(0.0)(_ + _.rms)  / stats.length
    val meanHd95: Double = stats.foldLeft(0.0)(_ + _.hd95) / stats.length
    val row = SweepRow(sigma, scale, gp.rank, meanMean, meanRms, meanHd95, stats)
    allRows :+= row

    val detail = stats.flatMap(s => Seq(f"${s.mean}%.4f", f"${s.rms}%.4f", f"${s.hd95}%.4f")).mkString(",")
    sweepWriter.println(f"${sigma}%.0f,${scale}%.0f,${gp.rank},$detail,${meanMean}%.4f")
    sweepWriter.flush()
    println(f"  sigma=${sigma}%4.0f mm   amp=${scale}%2.0f   rank=${gp.rank}%3d   " +
            f"mean(all 5 targets)=${meanMean}%.4f mm   HD95=${meanHd95}%.4f mm")
  }
  sweepWriter.close()

  val bestRow  = allRows.minBy(_.meanMean)
  val worstRow = allRows.maxBy(_.meanMean)
  println(f"\n  Best  : sigma=${bestRow.sigma}%.0f mm  amp=${bestRow.scale}%.0f  " +
          f"-> mean=${bestRow.meanMean}%.4f mm  HD95=${bestRow.meanHd95}%.4f mm")
  println(f"  Worst : sigma=${worstRow.sigma}%.0f mm  amp=${worstRow.scale}%.0f  " +
          f"-> mean=${worstRow.meanMean}%.4f mm  HD95=${worstRow.meanHd95}%.4f mm")

  println("\n  === Sweep Table: mean surface distance (mm) - rows=sigma, cols=scaleFactor ===")
  print(f"  ${"sigma \\ amp"}%-12s")
  scaleFactors.foreach(s => print(f"  $s%8.0f"))
  println()
  for (sigma <- sigmaValues) {
    print(f"  ${sigma}%12.0f")
    for (scale <- scaleFactors) {
      val r = allRows.find(r => r.sigma == sigma && r.scale == scale).get
      print(f"  ${r.meanMean}%8.4f")
    }
    println()
  }

  println(s"\n=== Phase 2 - Full FFDM Registration ===")
  println(f"  Optimal: sigma=${bestRow.sigma}%.0f mm   scaleFactor=${bestRow.scale}%.0f   rank=${bestRow.rank}")
  println(s"  Iterations: ${Config.icpIterations}\n")

  val optGP = buildFFDM(bestRow.sigma, bestRow.scale)

  val finalWriter = new PrintWriter(new File(outDir, "ffdm_final_results.csv"))
  finalWriter.println("ID,Mean_mm,RMS_mm,HD95_mm,HD_mm")

  val ui       = ScalismoUI()
  val regGroup = ui.createGroup(
    "FFDM Registration sigma=%.0f amp=%.0f".format(bestRow.sigma, bestRow.scale))

  val refView = ui.show(regGroup, refMesh: TriangleMesh[_3D], s"Reference ($referenceId)")
  refView.color = Color.LIGHT_GRAY

  val finalResults: IndexedSeq[(String, Metrics.SurfaceStats)] =
    alignedTargets.zipWithIndex.map { case ((id, aligned), idx) =>
      println(s"  [${idx + 1}/${alignedTargets.length}] Fitting $id ...")
      val fitted = fitFFDM(aligned, optGP, iterations = Config.icpIterations, regWeight = 1e-4)
      MeshIO.writeMesh(fitted, new File(outDir, s"ffdm_fitted_$id.stl")).get
      val st: Metrics.SurfaceStats = Metrics.symmetric(fitted, aligned)
      finalWriter.println(f"$id,${st.mean}%.4f,${st.rms}%.4f,${st.hd95}%.4f,${st.hd}%.4f")
      finalWriter.flush()
      println(f"  ${st.render}")

      val tv = ui.show(regGroup, aligned: TriangleMesh[_3D], s"Target   $id")
      tv.color = Color.RED
      val fv = ui.show(regGroup, fitted: TriangleMesh[_3D], s"Fitted   $id")
      fv.color = Color.GRAY

      id -> st
    }
  finalWriter.close()

  val bestCase  = finalResults.minBy(_._2.mean: Double)
  val worstCase = finalResults.maxBy(_._2.mean: Double)

  println(s"\n  Best  case: ${bestCase._1}")
  println(s"    ${bestCase._2.render}")
  println(s"  Worst case: ${worstCase._1}")
  println(s"    ${worstCase._2.render}")
  println(s"\nOutputs written to: ${outDir.getAbsolutePath}")
  println("Press ENTER to close the UI and exit.")
  scala.io.StdIn.readLine()
}
