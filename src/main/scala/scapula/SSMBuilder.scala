package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.io.StatisticalModelIO
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel

import java.io.{File, PrintWriter}

/** Builds and validates a PCA-based Statistical Shape Model from a set of registered meshes. */
object SSMBuilder {

  /**
   * Build SSM via PCA from registered meshes that are already in dense correspondence
   * (same topology, same vertex ordering) with each other.
   *
   * All meshes MUST share the same reference mesh topology.
   */
  def buildFromCorrespondences(
    registeredMeshes: IndexedSeq[TriangleMesh[_3D]]
  ): PointDistributionModel[_3D, TriangleMesh] = {
    require(registeredMeshes.nonEmpty, "no registered meshes supplied")
    PointDistributionModel.createUsingPCA(registeredMeshes)
  }

  /** Compute and print basic variance/compactness metrics. Returns the table as a string. */
  def varianceReport(model: PointDistributionModel[_3D, TriangleMesh], label: String): String = {
    val v         = model.variance
    val total     = v.sum
    val pct       = v.map(_ / total * 100.0)
    val cumPct    = pct.toArray.scanLeft(0.0)(_ + _).tail

    val mode1     = pct(0)
    val top5      = cumPct(math.min(4, cumPct.length - 1))
    val top10     = cumPct(math.min(9, cumPct.length - 1))

    val sb = new StringBuilder
    sb.append(s"\n[$label] PCA variance report  (rank = ${model.rank})\n")
    sb.append(f"  Mode 1 : $mode1%5.1f%%\n")
    sb.append(f"  Top  5 : $top5%5.1f%%\n")
    sb.append(f"  Top 10 : $top10%5.1f%%\n")
    val modes = math.min(10, model.rank)
    sb.append("  Cumulative: " + (0 until modes).map(i => f"${cumPct(i)}%.1f").mkString("  ") + "\n")

    val report = sb.toString
    print(report)
    report
  }

  /**
   * Generalization error: leave-one-out reconstruction.
   * For each mesh, fit the model built WITHOUT that mesh by computing the closest point
   * in model-space, then measure the residual surface distance.
   *
   * This is the standard generalization metric: lower is better.
   */
  def generalizationError(
    meshes: IndexedSeq[TriangleMesh[_3D]],
    label: String
  ): Double = {
    if (meshes.length < 3) {
      println(s"  [$label] Skipping generalization (need >= 3 meshes)")
      return Double.NaN
    }
    val errors = meshes.zipWithIndex.map { case (testMesh, i) =>
      val trainMeshes = meshes.patch(i, Nil, 1)
      val loo         = PointDistributionModel.createUsingPCA(trainMeshes)
      // Project test mesh onto model: condition on all vertex observations, σ²=0.5
      val observations = testMesh.pointSet.pointIds.toIndexedSeq.map { ptId =>
        (ptId, testMesh.pointSet.point(ptId))
      }
      val posterior  = loo.posterior(observations, 0.5)
      val projected  = posterior.mean
      val d          = Metrics.surfaceDistances(testMesh, projected)
      d.sum / d.length
    }
    val meanError = errors.sum / errors.length
    println(f"  [$label] Generalization (LOO mean surface dist) = $meanError%5.3f mm")
    meanError
  }

  /**
   * Specificity: sample random instances from the model and measure how close they
   * are to the nearest training shape. Lower is better (samples look like real shapes).
   */
  def specificityError(
    model: PointDistributionModel[_3D, TriangleMesh],
    trainingMeshes: IndexedSeq[TriangleMesh[_3D]],
    nSamples: Int = 50,
    label: String
  )(implicit rng: scalismo.utils.Random): Double = {
    if (trainingMeshes.isEmpty) return Double.NaN
    val errors = (0 until nSamples).map { _ =>
      val sample  = model.sample()
      val minDist = trainingMeshes.map { ref =>
        val d = Metrics.surfaceDistances(sample, ref)
        d.sum / d.length
      }.min
      minDist
    }
    val meanError = errors.sum / errors.length
    println(f"  [$label] Specificity ($nSamples samples, mean dist to nearest training) = $meanError%5.3f mm")
    meanError
  }

  final case class SSMMetrics(
    label: String,
    rank: Int,
    mode1Pct: Double,
    top5Pct: Double,
    top10Pct: Double,
    generalization: Double,
    specificity: Double
  ) {
    def tableRow: String =
      f"| $label%-5s | $mode1Pct%6.1f%% | $top5Pct%6.1f%% | $top10Pct%7.1f%% | $generalization%14.3f | $specificity%11.3f | N/A |"
  }

  def computeMetrics(
    model: PointDistributionModel[_3D, TriangleMesh],
    trainingMeshes: IndexedSeq[TriangleMesh[_3D]],
    label: String
  )(implicit rng: scalismo.utils.Random): SSMMetrics = {
    val v      = model.variance
    val total  = v.sum
    val pct    = v.map(_ / total * 100.0)
    val cumPct = pct.toArray.scanLeft(0.0)(_ + _).tail

    val mode1  = pct(0)
    val top5   = cumPct(math.min(4, cumPct.length - 1))
    val top10  = cumPct(math.min(9, cumPct.length - 1))
    val gen    = generalizationError(trainingMeshes, label)
    val spc    = specificityError(model, trainingMeshes, label = label)

    SSMMetrics(label, model.rank, mode1, top5, top10, gen, spc)
  }

  def saveModel(model: PointDistributionModel[_3D, TriangleMesh], file: File): Unit = {
    file.getParentFile.mkdirs()
    StatisticalModelIO.writeStatisticalMeshModel(model, file)
      .getOrElse(throw new RuntimeException(s"Failed to save model to ${file.getPath}"))
    println(s"  Saved model → ${file.getPath}")
  }

  def saveMean(model: PointDistributionModel[_3D, TriangleMesh], file: File): Unit = {
    file.getParentFile.mkdirs()
    scalismo.io.MeshIO.writeMesh(model.mean, file)
      .getOrElse(throw new RuntimeException(s"Failed to save mean to ${file.getPath}"))
    println(s"  Saved mean  → ${file.getPath}")
  }

  def writeMetricsTable(metrics: IndexedSeq[SSMMetrics], file: File): Unit = {
    file.getParentFile.mkdirs()
    val header =
      "| Model | Mode 1 | Top 5 | Top 10 | Generalization | Specificity | Reconstruction |"
    val sep =
      "|-------|-------:|------:|-------:|---------------:|------------:|---------------:|"
    val pw = new PrintWriter(file)
    pw.println(header)
    pw.println(sep)
    metrics.foreach(m => pw.println(m.tableRow))
    pw.close()
    println(s"  SSM comparison table → ${file.getPath}")
  }
}
