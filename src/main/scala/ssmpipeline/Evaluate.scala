package ssmpipeline

import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.utils.Random

/**
 * SSM quality metrics: compactness, generalization, and specificity.
 *
 * All three operate on meshes that are already in point-to-point correspondence (registered to the same reference).
 * Distances are mean per-vertex distances in mm.
 */
object Evaluate {

  /**
   * Number of principal components needed to explain at least `threshold` of total GP variance.
   * Returns the component count (1-indexed), never zero.
   */
  def compactnessAt(model: PointDistributionModel[_3D, TriangleMesh], threshold: Double): Int = {
    val variances = model.gp.klBasis.map(_.eigenvalue)
    val total     = variances.sum
    var acc       = 0.0
    var idx       = 0
    while (idx < variances.length && acc / total < threshold) {
      acc += variances(idx)
      idx += 1
    }
    idx
  }

  /**
   * Generalization (leave-one-out reconstruction error).
   *
   * For each shape, build a PCA model on the remaining shapes, project the left-out shape onto it, and measure the
   * mean per-vertex distance between the original and the reconstruction. Returns (mean, std) across all LOO folds.
   *
   * This quantifies how well the model represents unseen shapes. Lower is better.
   */
  def generalization(shapes: IndexedSeq[TriangleMesh[_3D]]): (Double, Double) = {
    val errors = shapes.zipWithIndex.map { case (target, i) =>
      val trainShapes = shapes.patch(i, Nil, 1)
      val loModel     = PointDistributionModel.createUsingPCA(trainShapes)
      val recon       = loModel.project(target)
      Metrics.correspondingDistances(recon, target).sum / target.pointSet.numberOfPoints
    }
    (vecMean(errors), vecStd(errors))
  }

  /**
   * Specificity: average distance from random model samples to their nearest training shape.
   *
   * Samples a shape from the model, finds the training shape closest to it (by mean per-vertex distance), and
   * records that distance. Returns (mean, std) over `numSamples` draws. Lower is better.
   */
  def specificity(model: PointDistributionModel[_3D, TriangleMesh],
                  shapes: IndexedSeq[TriangleMesh[_3D]],
                  numSamples: Int = 100
  )(implicit rng: Random): (Double, Double) = {
    val errors = IndexedSeq.fill(numSamples) {
      val sample = model.sample()
      shapes
        .map(s => Metrics.correspondingDistances(sample, s).sum / sample.pointSet.numberOfPoints)
        .min
    }
    (vecMean(errors), vecStd(errors))
  }

  private def vecMean(xs: IndexedSeq[Double]): Double = xs.sum / xs.length
  private def vecStd(xs: IndexedSeq[Double]): Double = {
    val m = vecMean(xs)
    math.sqrt(xs.map(x => (x - m) * (x - m)).sum / xs.length)
  }
}
