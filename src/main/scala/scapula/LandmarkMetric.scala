package scapula

import breeze.linalg.DenseVector
import scalismo.geometry.{NDSpace, Point, _3D}
import scalismo.registration.RegistrationMetric
import scalismo.registration.RegistrationMetric.ValueAndDerivative
import scalismo.transformations.TransformationSpace

/**
 * Mean squared distance between named landmark correspondences under the current transform:
 * mean_i ||transform(referencePoint_i) - targetPoint_i||^2.
 *
 * Scalismo's built-in registration metrics (MeanSquaresMetric, ...) are all image-to-image, so surface-based
 * GPMM fitting normally has no landmark term at all -- only the anonymous closest-surface-point data term. This
 * adds the missing piece: the same named anatomical landmarks used for rigid pre-alignment also pull the
 * non-rigid fit, instead of being discarded after that first step.
 */
case class LandmarkMetric(correspondences: IndexedSeq[(Point[_3D], Point[_3D])], transformationSpace: TransformationSpace[_3D])(
  implicit val ndSpace: NDSpace[_3D]
) extends RegistrationMetric[_3D] {

  require(correspondences.nonEmpty, "LandmarkMetric needs at least one correspondence")

  def value(parameters: DenseVector[Double]): Double = valueAndDerivative(parameters).value

  def derivative(parameters: DenseVector[Double]): DenseVector[Double] = valueAndDerivative(parameters).derivative

  def valueAndDerivative(parameters: DenseVector[Double]): ValueAndDerivative = {
    val t = transformationSpace.transformationForParameters(parameters)
    val jacobian = t.derivativeWRTParameters
    val n = correspondences.length.toDouble

    var v = 0.0
    var grad = DenseVector.zeros[Double](parameters.length)
    correspondences.foreach { case (referencePoint, targetPoint) =>
      val residual = t(referencePoint) - targetPoint
      v += residual.norm2
      // d(||transform(x) - target||^2)/dp = 2 * J(x)^T * (transform(x) - target)
      grad = grad + (jacobian(referencePoint).t * residual.toBreezeVector) * 2.0
    }
    ValueAndDerivative(v / n, grad / n)
  }
}

/** Weighted sum of several registration metrics over the same transformation space. */
case class SumMetric(terms: Seq[(RegistrationMetric[_3D], Double)])(implicit val ndSpace: NDSpace[_3D]) extends RegistrationMetric[_3D] {
  require(terms.nonEmpty, "SumMetric needs at least one term")

  def transformationSpace: TransformationSpace[_3D] = terms.head._1.transformationSpace

  def value(parameters: DenseVector[Double]): Double = terms.map { case (m, w) => w * m.value(parameters) }.sum

  def derivative(parameters: DenseVector[Double]): DenseVector[Double] = valueAndDerivative(parameters).derivative

  def valueAndDerivative(parameters: DenseVector[Double]): ValueAndDerivative = {
    val parts = terms.map { case (m, w) =>
      val vd = m.valueAndDerivative(parameters)
      (vd.value * w, vd.derivative * w)
    }
    ValueAndDerivative(parts.map(_._1).sum, parts.map(_._2).reduce(_ + _))
  }
}
