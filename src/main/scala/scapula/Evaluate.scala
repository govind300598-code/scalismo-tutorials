package scapula

import breeze.linalg.DenseVector
import scalismo.mesh.TriangleMesh
import scalismo.geometry._3D
import scalismo.statisticalmodel.{PointDistributionModel, dataset}
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, PrintWriter}

object Evaluate {

  final case class ModelMetrics(
    compactness90pct: Int,
    compactness95pct: Int,
    generalizationMean: Double,
    generalizationStd: Double,
    specificityMean: Double,
    specificityStd: Double
  )

  def saveCsv(rows: Seq[Seq[String]], file: File): Unit = {
    val pw = new PrintWriter(file)
    rows.foreach(row => pw.println(row.mkString(",")))
    pw.close()
  }

  // ── Compactness ─────────────────────────────────────────────────────────────
  def compactness(
    model: PointDistributionModel[_3D, TriangleMesh],
    outFile: File
  ): Unit = {
    val eigenvalues = model.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val total       = eigenvalues.sum
    var cumulative  = 0.0
    val pw = new PrintWriter(outFile)
    pw.println("Modes,Eigenvalue,CumulativeVariancePct")
    eigenvalues.zipWithIndex.foreach { case (ev, i) =>
      cumulative += ev
      val pct = if (total > 0) cumulative / total * 100.0 else 0.0
      pw.println(s"${i + 1},$ev,$pct")
    }
    pw.close()
  }

  def modesFor(model: PointDistributionModel[_3D, TriangleMesh], threshold: Double): Int = {
    val eigenvalues = model.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val total = eigenvalues.sum
    var cum   = 0.0
    var count = 0
    while (count < eigenvalues.length && cum / total < threshold) {
      cum += eigenvalues(count)
      count += 1
    }
    count
  }

  // ── Generalization (LOO reconstruction error) ────────────────────────────────
  def generalization(
    reference: TriangleMesh[_3D],
    shapes:    IndexedSeq[TriangleMesh[_3D]],
    modeList:  Seq[Int],
    outFile:   File
  )(implicit rng: Random): IndexedSeq[(Int, Double, Double)] = {
    val pw = new PrintWriter(outFile)
    pw.println("Modes,MeanLOO_mm,StdLOO_mm")

    val results = modeList.toIndexedSeq.map { nModes =>
      val errors: IndexedSeq[Double] = shapes.indices.map { leaveOutIdx =>
        val train  = shapes.patch(leaveOutIdx, Nil, 1)
        val dc     = DataCollection.fromTriangleMesh3DSequence(reference, train)
        val loModel = PointDistributionModel.createUsingPCA(dc)
        val nM      = math.min(nModes, loModel.rank)
        val testShape = shapes(leaveOutIdx)
        val coeffs  = loModel.coefficients(testShape)
        val trunc   = DenseVector.tabulate[Double](loModel.rank)(j => if (j < nM) coeffs(j) else 0.0)
        val fitted  = loModel.instance(trunc)
        Metrics.correspondingDistances(fitted, testShape).sum / testShape.pointSet.numberOfPoints
      }
      val mu  = errors.sum / errors.size
      val std = math.sqrt(errors.map(e => (e - mu) * (e - mu)).sum / errors.size)
      pw.println(s"$nModes,$mu,$std")
      (nModes, mu, std)
    }
    pw.close()
    results
  }

  // ── Specificity (random samples vs training shapes) ───────────────────────────
  def specificity(
    model:    PointDistributionModel[_3D, TriangleMesh],
    shapes:   IndexedSeq[TriangleMesh[_3D]],
    modeList: Seq[Int],
    nSamples: Int = 30,
    outFile:  File
  ): IndexedSeq[(Int, Double, Double)] = {
    val pw = new PrintWriter(outFile)
    pw.println("Modes,MeanSpec_mm,StdSpec_mm")

    val results = modeList.toIndexedSeq.map { nModes =>
      val specRng = new java.util.Random(42L + nModes)
      val dists: IndexedSeq[Double] = (0 until nSamples).toIndexedSeq.map { _ =>
        val coeffs = DenseVector.tabulate[Double](model.rank)(j =>
          if (j < nModes) specRng.nextGaussian() else 0.0)
        val sample = model.instance(coeffs)
        shapes.map(tm => Metrics.symmetric(sample, tm).mean).min
      }
      val mu  = dists.sum / dists.size
      val std = math.sqrt(dists.map(d => (d - mu) * (d - mu)).sum / dists.size)
      pw.println(s"$nModes,$mu,$std")
      (nModes, mu, std)
    }
    pw.close()
    results
  }

  // ── Run all evaluations and print a summary table ─────────────────────────────
  def evaluateModel(
    label:     String,
    model:     PointDistributionModel[_3D, TriangleMesh],
    reference: TriangleMesh[_3D],
    shapes:    IndexedSeq[TriangleMesh[_3D]],
    outDir:    File,
    maxLooN:   Int = 15
  )(implicit rng: Random): ModelMetrics = {
    println(s"\n  --- Compactness ($label) ---")
    compactness(model, new File(outDir, s"${label}_compactness.csv"))
    val m90 = modesFor(model, 0.90)
    val m95 = modesFor(model, 0.95)
    println(f"    90%% variance: $m90 modes   95%% variance: $m95 modes")

    val step      = math.max(1, shapes.size / maxLooN)
    val looShapes = shapes.zipWithIndex.filter(_._2 % step == 0).map(_._1)
    val modeList  = Seq(1, 2, 3, 5, 10, 20, 50, model.rank).distinct.filter(_ <= model.rank)

    println(s"  --- Generalization ($label, LOO on ${looShapes.size} shapes) ---")
    val gen = generalization(reference, looShapes, modeList, new File(outDir, s"${label}_generalization.csv"))
    val (_, genMu, genStd) = gen.last
    println(f"    Best (all modes): mean=$genMu%.4f  std=$genStd%.4f mm")

    println(s"  --- Specificity ($label) ---")
    val spec = specificity(model, shapes, modeList, outFile = new File(outDir, s"${label}_specificity.csv"))
    val (_, specMu, specStd) = spec.last
    println(f"    Best (all modes): mean=$specMu%.4f  std=$specStd%.4f mm")

    ModelMetrics(m90, m95, genMu, genStd, specMu, specStd)
  }
}
