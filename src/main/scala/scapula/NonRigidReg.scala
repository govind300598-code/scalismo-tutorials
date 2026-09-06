package scapula

import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.common.{Field, RealSpace}
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

object NonRigidReg {

  /**
   * Build a single-Gaussian GP kernel on the given reference mesh, then return a PointDistributionModel whose
   * reference is the DECIMATED mesh. The Nystrom quadrature uses the original (higher-res) reference for accuracy,
   * and a NearestNeighborInterpolator evaluates the continuous GP at the decimated mesh's vertices.
   *
   * Kernel: k(x,y) = gpScale · exp(−‖x−y‖² / 2·gpSigma²) · I₃
   */
  private def buildModel(
    originalRef: TriangleMesh[_3D],
    decimatedRef: TriangleMesh[_3D]
  )(implicit rng: Random): PointDistributionModel[_3D, TriangleMesh] = {
    val scalarKernel = GaussianKernel[_3D](Config.gpSigma) * Config.gpScale
    val kernel       = DiagonalKernel(scalarKernel, 3)
    val zeroMean     = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp           = GaussianProcess(zeroMean, kernel)

    // Nystrom approximation using the denser original mesh as quadrature domain.
    // The resulting LowRankGaussianProcess is continuous and can be evaluated anywhere.
    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(
      gp,
      originalRef.pointSet,
      numBasisFunctions = Config.gpBasis
    )

    // Evaluate (via NN interpolation) at the decimated mesh's vertices to get the PDM.
    PointDistributionModel[_3D, TriangleMesh](decimatedRef, lowRankGP)
  }

  /**
   * GP-ICP: iteratively find closest-point correspondences between the current model instance and the target,
   * then update the model via GP posterior. Returns a mesh with the same topology as `decimatedRef`.
   *
   * @param originalRef    full-resolution reference (used for Nystrom quadrature only)
   * @param decimatedRef   decimated reference (the SSM lives at this resolution)
   * @param target         rigidly-aligned target specimen
   */
  def register(
    originalRef: TriangleMesh[_3D],
    decimatedRef: TriangleMesh[_3D],
    target: TriangleMesh[_3D]
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val model     = buildModel(originalRef, decimatedRef)
    val targetOps = target.operations
    var current   = decimatedRef

    for (_ <- 0 until Config.gpIcpIter) {
      val correspondences = current.pointSet.pointsWithId.map { case (pt, id) =>
        (id, targetOps.closestPointOnSurface(pt).point)
      }.toIndexedSeq
      current = model.posterior(correspondences, Config.gpNoise).mean
    }
    current
  }

  /** Pointwise mean of a set of meshes that are already in correspondence (same topology). */
  def meanMesh(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    require(meshes.nonEmpty)
    val n     = meshes.length
    val pts   = (0 until meshes.head.pointSet.numberOfPoints).map { i =>
      val sum = meshes.foldLeft(EuclideanVector.zeros[_3D])((acc, m) => acc + m.pointSet.point(scalismo.common.PointId(i)).toVector)
      (sum * (1.0 / n)).toPoint
    }
    scalismo.mesh.TriangleMesh3D(pts, meshes.head.triangulation)
  }
}
