package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Full SSM evaluation with all standard metrics:
 *
 *   A. Compactness      — cumulative variance explained by first N modes
 *   B. Generalization   — leave-one-out: build model without specimen i,
 *                          project it back, measure surface distance
 *   C. Specificity      — sample random shapes, measure distance to nearest training shape
 *   D. Per-specimen     — Hausdorff / HD95 / surface RMSE / Chamfer L1 & L2 /
 *                          point-to-point mean, RMSE, max
 *   E. Stability        — surface distance between pass-1 mean and pass-2 mean
 */
object SSMEval {

  type PDM = PointDistributionModel[_3D, TriangleMesh]

  /**
   * Run the full evaluation and write results to `outDir`.
   *
   * @param finalSsm       The final (pass-2) SSM.
   * @param meanRef        The final SSM mean mesh (pass-2 reference).
   * @param registered     Per-specimen: (Specimen, registered mesh from pass 2).
   * @param rigidOriginals Per-specimen: (Specimen → rigid-aligned original mesh).
   * @param stability      Surface-distance stats between pass-1 and pass-2 means.
   */
  def evaluate(
    finalSsm:       PDM,
    meanRef:        TriangleMesh[_3D],
    registered:     IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D])],
    rigidOriginals: Map[ScapulaData.Specimen, TriangleMesh[_3D]],
    stability:      Metrics.SurfaceStats,
    outDir:         File
  )(implicit rng: Random): Unit = {

    val regMeshes = registered.map(_._2)

    // ── A. Compactness ────────────────────────────────────────────────────────
    val evs   = finalSsm.gp.klBasis.map(_.eigenvalue)
    val total = evs.sum
    def varAt(n: Int): Double = evs.take(n).sum / total * 100.0

    println("\n[A] COMPACTNESS")
    val compRows = Seq(1,2,3,5,10,20,30,50,100,150,200).map { n =>
      val v = varAt(n)
      println(f"    modes 1-$n%3d : $v%5.1f%%")
      (n, v)
    }

    // ── B. Leave-one-out generalization ──────────────────────────────────────
    println("\n[B] GENERALIZATION (leave-one-out, surface distance to projected shape)")
    val looResults: IndexedSeq[(String, Double)] = registered.zipWithIndex.map { case ((spec, testMesh), i) =>
      val trainMeshes = regMeshes.patch(i, Nil, 1)
      val trainRef    = meanRef  // keep the same reference for topology consistency
      val trainDC     = DataCollection.fromTriangleMeshSequence(trainRef, trainMeshes)
      val trainModel  = PointDistributionModel.createUsingPCA(trainDC)
      val projected   = trainModel.instance(trainModel.coefficients(testMesh))
      val d           = Metrics.symmetric(testMesh, projected).mean
      print(f"\r    [${i + 1}/${registered.length}] ${spec.modelId}%-28s  dist=${d}%.3f mm")
      System.out.flush()
      (spec.modelId, d)
    }
    println()
    val looMean = looResults.map(_._2).sum / looResults.length
    val looMax  = looResults.map(_._2).max
    println(f"    LOO mean=${looMean}%.3f mm   max=${looMax}%.3f mm")

    // ── C. Specificity ────────────────────────────────────────────────────────
    val nSpecSamples = 200
    println(s"\n[C] SPECIFICITY ($nSpecSamples random samples)")
    val specDistsBuf = scala.collection.mutable.ArrayBuffer[Double]()
    (0 until nSpecSamples).foreach { i =>
      val sample  = finalSsm.sample()
      val minDist = regMeshes.map(t => Metrics.symmetric(sample, t).mean).min
      specDistsBuf += minDist
      if ((i + 1) % 20 == 0) {
        val rm = specDistsBuf.sum / specDistsBuf.length
        print(f"\r    sample ${i + 1}%3d/$nSpecSamples  running-mean=${rm}%.3f mm")
        System.out.flush()
      }
    }
    val specDists = specDistsBuf.toIndexedSeq
    println()
    val specMean = specDists.sum / specDists.length
    val specMax  = specDists.max
    println(f"    Specificity mean=${specMean}%.3f mm   max=${specMax}%.3f mm")

    // ── D. Per-specimen metrics ───────────────────────────────────────────────
    println("\n[D] PER-SPECIMEN METRICS")
    println(s"    ${Metrics.FullDistStats.csvHeader}")
    val perSpecRows: IndexedSeq[Metrics.FullDistStats] = registered.map { case (spec, regMesh) =>
      val origMesh = rigidOriginals(spec)
      val stats    = Metrics.fullStats(regMesh, origMesh, meanRef)
      println(s"    ${stats.csvRow(spec.modelId)}")
      stats
    }

    // aggregate
    def mean(f: Metrics.FullDistStats => Double): Double = perSpecRows.map(f).sum / perSpecRows.length
    println(f"\n    MEAN  surf=${mean(_.surfMean)}%.3f  HD95=${mean(_.surfHd95)}%.3f  HD=${mean(_.surfHd)}%.3f" +
      f"  p2p=${mean(_.p2pMean)}%.3f  ChamferL1=${mean(_.chamferL1)}%.3f  ChamferL2=${mean(_.chamferL2)}%.3f mm")

    // ── E. Stability ──────────────────────────────────────────────────────────
    println(s"\n[E] STABILITY")
    println(f"    pass-1 mean vs pass-2 mean: mean=${stability.mean}%.3f  HD95=${stability.hd95}%.3f  HD=${stability.hd}%.3f mm")
    if (stability.mean < 0.5) println("    => Model is STABLE (< 0.5 mm mean drift; further passes not needed)")
    else println("    => Model may still benefit from a 3rd pass (mean drift >= 0.5 mm)")

    // ── Write CSV / TXT outputs ───────────────────────────────────────────────
    writePerSpecCsv(registered, rigidOriginals, meanRef, outDir)
    writeQualityReport(
      outDir, registered.length, finalSsm.rank, evs, total,
      compRows, looMean, looMax, specMean, specMax, stability, mean
    )
  }

  // ── I/O helpers ─────────────────────────────────────────────────────────────

  private def writePerSpecCsv(
    registered:     IndexedSeq[(ScapulaData.Specimen, TriangleMesh[_3D])],
    rigidOriginals: Map[ScapulaData.Specimen, TriangleMesh[_3D]],
    meanRef:        TriangleMesh[_3D],
    outDir:         File
  ): Unit = {
    val f = new File(outDir, "metrics_per_spec.csv")
    val pw = new PrintWriter(f)
    pw.println(Metrics.FullDistStats.csvHeader)
    registered.foreach { case (spec, regMesh) =>
      val orig  = rigidOriginals(spec)
      val stats = Metrics.fullStats(regMesh, orig, meanRef)
      pw.println(stats.csvRow(spec.modelId))
    }
    pw.close()
    println(s"\n  Saved per-specimen CSV: ${f.getAbsolutePath}")
  }

  private def writeQualityReport(
    outDir:     File,
    nShapes:    Int,
    rank:       Int,
    evs:        IndexedSeq[Double],
    total:      Double,
    compRows:   Seq[(Int, Double)],
    looMean:    Double,
    looMax:     Double,
    specMean:   Double,
    specMax:    Double,
    stability:  Metrics.SurfaceStats,
    aggMean:    (Metrics.FullDistStats => Double) => Double
  ): Unit = {
    val f  = new File(outDir, "ssm_quality.txt")
    val pw = new PrintWriter(f)
    pw.println("=" * 60)
    pw.println("SSM QUALITY REPORT")
    pw.println("=" * 60)
    pw.println(s"Shapes: $nShapes   Model rank: $rank")
    pw.println()
    pw.println("COMPACTNESS (% variance explained):")
    compRows.foreach { case (n, v) => pw.println(f"  modes 1-$n%3d : $v%5.1f%%") }
    pw.println()
    pw.println("GENERALIZATION (leave-one-out, mm):")
    pw.println(f"  mean = $looMean%.4f   max = $looMax%.4f")
    pw.println()
    pw.println("SPECIFICITY (200 random samples, mm to nearest training):")
    pw.println(f"  mean = $specMean%.4f   max = $specMax%.4f")
    pw.println()
    pw.println("AGGREGATE PER-SPECIMEN DISTANCES (registered vs rigid-aligned original):")
    pw.println(f"  Surface mean  : ${aggMean(_.surfMean)}%.4f mm")
    pw.println(f"  Surface RMSE  : ${aggMean(_.surfRms)}%.4f mm")
    pw.println(f"  Surface HD95  : ${aggMean(_.surfHd95)}%.4f mm")
    pw.println(f"  Surface HD    : ${aggMean(_.surfHd)}%.4f mm")
    pw.println(f"  P2P mean      : ${aggMean(_.p2pMean)}%.4f mm")
    pw.println(f"  P2P RMSE      : ${aggMean(_.p2pRmse)}%.4f mm")
    pw.println(f"  P2P max       : ${aggMean(_.p2pMax)}%.4f mm")
    pw.println(f"  Chamfer L1    : ${aggMean(_.chamferL1)}%.4f mm")
    pw.println(f"  Chamfer L2    : ${aggMean(_.chamferL2)}%.4f mm²")
    pw.println()
    pw.println("STABILITY (pass-1 mean vs pass-2 mean):")
    pw.println(f"  mean = ${stability.mean}%.4f   HD95 = ${stability.hd95}%.4f   HD = ${stability.hd}%.4f mm")
    pw.println("=" * 60)
    pw.close()
    println(s"  Saved quality report : ${f.getAbsolutePath}")
  }
}
