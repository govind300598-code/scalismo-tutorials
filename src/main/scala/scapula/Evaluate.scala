package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/** Compactness, generalization, and specificity for a PointDistributionModel. */
object Evaluate {

  // ---------------------------------------------------------------------------
  // Compactness
  // ---------------------------------------------------------------------------

  /**
   * Returns (modeIndex, cumulativeExplainedVarianceFraction) for each mode.
   * A more compact model reaches a high fraction with fewer modes.
   *
   * `klBasis` in Scalismo 0.92 returns a Seq, so we call `.toIndexedSeq` explicitly.
   * `createUsingPCA` in Scalismo 0.92 returns the model directly (not a Try).
   */
  def compactness(model: PointDistributionModel[_3D, TriangleMesh]): IndexedSeq[(Int, Double)] = {
    val eigenvalues = model.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val total       = eigenvalues.sum
    var cumulative  = 0.0
    eigenvalues.zipWithIndex.map { case (v, i) =>
      cumulative += v
      (i + 1, if (total > 0.0) cumulative / total else 0.0)
    }
  }

  // ---------------------------------------------------------------------------
  // Generalization  (leave-one-out cross-validation)
  // ---------------------------------------------------------------------------

  /**
   * For every mesh in `meshes`: rebuild the model from the remaining N-1 meshes, project the
   * left-out mesh onto it, and measure the point-to-point reconstruction error.
   *
   * All `meshes` are already in correspondence with `reference` (same topology, same vertex order),
   * so no additional GPA is applied — the registered shapes are already in a consistent frame.
   * Point-to-point distance is valid for the same reason.
   *
   * Returns (meanError_mm, stdError_mm).
   */
  def generalization(
    reference : TriangleMesh[_3D],
    meshes    : IndexedSeq[TriangleMesh[_3D]]
  ): (Double, Double) = {
    require(meshes.length >= 3, "Need at least 3 meshes for LOO generalization")

    val errors: IndexedSeq[Double] = meshes.indices.map { leaveOutIdx =>
      val training = meshes.indices.filterNot(_ == leaveOutIdx).map(meshes)

      // Build without an extra GPA pass: shapes are already rigidly registered.
      // In Scalismo 0.92 createUsingPCA returns the model directly (no Try wrapper).
      val dc      = DataCollection.fromTriangleMesh3DSequence(reference, training)
      val loModel = PointDistributionModel.createUsingPCA(dc)

      val target       = meshes(leaveOutIdx)
      val coefficients = loModel.coefficients(target)
      val projected    = loModel.instance(coefficients)

      // Point-to-point RMSE — valid because both meshes share reference topology
      val dists = Metrics.correspondingDistances(projected, target)
      dists.sum / dists.length
    }

    meanAndStd(errors)
  }

  // ---------------------------------------------------------------------------
  // Specificity
  // ---------------------------------------------------------------------------

  /**
   * Draw `nSamples` random instances from the model (coefficients clamped to ±3σ), find the
   * nearest training shape for each by mean point-to-point distance, and return (mean, std).
   * Lower = more realistic samples.
   */
  def specificity(
    model         : PointDistributionModel[_3D, TriangleMesh],
    trainingMeshes: IndexedSeq[TriangleMesh[_3D]],
    nSamples      : Int = 100
  )(implicit rng: Random): (Double, Double) = {
    val errors: IndexedSeq[Double] = (0 until nSamples).toIndexedSeq.map { _ =>
      val raw     = DenseVector.fill(model.rank)(rng.scalaRandom.nextGaussian())
      val clamped = raw.map(c => math.max(-3.0, math.min(3.0, c)))
      val sample  = model.instance(clamped)

      val minDist = trainingMeshes.map { t =>
        val d = Metrics.correspondingDistances(sample, t)
        d.sum / d.length
      }.min
      minDist
    }

    meanAndStd(errors)
  }

  // ---------------------------------------------------------------------------
  // CSV helper
  // ---------------------------------------------------------------------------

  def saveCsv(rows: Seq[Seq[String]], file: File): Unit = {
    val pw = new PrintWriter(file)
    try rows.foreach(r => pw.println(r.mkString(",")))
    finally pw.close()
  }

  // ---------------------------------------------------------------------------
  private def meanAndStd(xs: IndexedSeq[Double]): (Double, Double) = {
    val mean = xs.sum / xs.length
    val std  = math.sqrt(xs.map(e => (e - mean) * (e - mean)).sum / xs.length)
    (mean, std)
  }
}
