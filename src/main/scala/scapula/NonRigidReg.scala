package scapula

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.*
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.*
import scalismo.mesh.*
import scalismo.statisticalmodel.*
import scalismo.utils.Random

/**
 * GP-ICP non-rigid registration.
 *
 * Registers a target mesh onto a reference by iteratively:
 *   1. Finding the closest point on the target surface for each sampled reference vertex.
 *   2. Computing the Bayesian posterior of the deformation GP given those correspondences.
 *   3. Applying the posterior mean deformation to update the working mesh.
 *
 * The output mesh has the same topology and vertex count as the reference (dense correspondence).
 */
object NonRigidReg {

  /**
   * Multi-scale GP prior on deformation fields.
   * Three Gaussian kernels at coarse / medium / fine scales are summed so the model
   * can capture both global shape differences and local surface details.
   */
  def buildPrior(reference: TriangleMesh[_3D])(
      implicit rng: Random
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {

    val kernel: MatrixValuedPDKernel[_3D] =
      DiagonalKernel[_3D](GaussianKernel[_3D](100.0) * 80.0, 3) +
      DiagonalKernel[_3D](GaussianKernel[_3D](50.0)  * 40.0, 3) +
      DiagonalKernel[_3D](GaussianKernel[_3D](20.0)  * 10.0, 3)

    val zeroMean = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector3D.zero)
    val gp       = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)

    LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator       = NearestNeighborInterpolator3D()
    )
  }

  /**
   * Register one target mesh to the reference using GP-ICP.
   *
   * @param reference     Fixed reference mesh (defines output topology).
   * @param target        Moving target mesh (full-resolution; rigid-aligned beforehand).
   * @param prior         Low-rank GP prior built by [[buildPrior]].
   * @param iterations    Number of ICP iterations.
   * @param numSamples    Number of reference vertices sampled per iteration.
   * @param noiseVariance Isotropic noise variance (mm²) on correspondences.
   * @return Reference mesh deformed to match the target (same topology as reference).
   */
  def registerOne(
      reference:     TriangleMesh[_3D],
      target:        TriangleMesh[_3D],
      prior:         LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
      iterations:    Int    = Config.icpIterations,
      numSamples:    Int    = 2000,
      noiseVariance: Double = 2.0
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val noiseModel = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * noiseVariance
    )
    val targetOps = target.operations
    val sampleIds = RigidAlign.uniformIds(reference, numSamples)

    // Outlier rejection: threshold starts large and tightens each iteration.
    def threshold(iter: Int): Double = math.max(5.0, 30.0 - iter.toDouble)

    var currentGP = prior

    for (iter <- 0 until iterations) {
      val meanField = currentGP.mean
      val currentPts = reference.pointSet.points.toIndexedSeq.map(p => p + meanField(p))

      val observations: IndexedSeq[(Point[_3D], EuclideanVector[_3D], MultivariateNormalDistribution)] =
        sampleIds.flatMap { id =>
          val currentPt = currentPts(id.id)
          val closestPt = targetOps.closestPointOnSurface(currentPt).point
          val dist      = (currentPt - closestPt).norm
          if (dist < threshold(iter)) {
            val refPt          = reference.pointSet.point(id)
            val observedDefVec = closestPt - refPt
            Some((refPt, observedDefVec, noiseModel))
          } else None
        }

      if (observations.nonEmpty)
        currentGP = currentGP.posterior(observations)
    }

    val finalField = currentGP.mean
    TriangleMesh3D(
      reference.pointSet.points.toIndexedSeq.map(p => p + finalField(p)),
      reference.triangulation
    )
  }

  /** Element-wise mean of a set of meshes in dense point correspondence with `reference`. */
  def meanMesh(reference: TriangleMesh[_3D], meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    require(meshes.nonEmpty, "Cannot compute mean of empty mesh list")
    require(
      meshes.forall(_.pointSet.numberOfPoints == reference.pointSet.numberOfPoints),
      "All meshes must have the same vertex count as the reference"
    )
    val n    = meshes.size.toDouble
    val nPts = reference.pointSet.numberOfPoints
    val sumX = Array.fill(nPts)(0.0)
    val sumY = Array.fill(nPts)(0.0)
    val sumZ = Array.fill(nPts)(0.0)
    meshes.foreach { m =>
      m.pointSet.points.zipWithIndex.foreach { case (pt, i) =>
        sumX(i) += pt.x; sumY(i) += pt.y; sumZ(i) += pt.z
      }
    }
    val meanPts = (0 until nPts).map(i => Point3D(sumX(i) / n, sumY(i) / n, sumZ(i) / n))
    TriangleMesh3D(meanPts, reference.triangulation)
  }
}
