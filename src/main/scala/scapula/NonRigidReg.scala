package scapula

import scalismo.common.{DiscreteField, Field, RealSpace}
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random
import scalismo.common.PointId

object NonRigidReg {

  /**
   * GP-ICP non-rigid registration.
   *
   * Kernel: k(x,y) = gpScale · exp(−‖x−y‖² / 2·gpSigma²) · I₃
   *
   * Nystrom approximation: uniform random samples on the reference mesh surface are
   * used as inducing points (this is the nearest-neighbour interpolation step the
   * approximation performs internally when evaluating the GP at mesh vertices).
   */
  def register(
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D]
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val scalarKernel = GaussianKernel[_3D](Config.gpSigma) * Config.gpScale
    val kernel       = DiagonalKernel(scalarKernel, 3)
    val zeroMean     = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp           = GaussianProcess(zeroMean, kernel)

    // Nystrom: sample Config.gpBasis * 10 surface points as inducing points,
    // then keep Config.gpBasis basis functions.
    val sampler   = UniformMeshSampler3D(reference, Config.gpBasis * 10)
    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(gp, sampler, Config.gpBasis)
    val model     = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)

    val targetOps = target.operations
    var current   = reference

    for (_ <- 0 until Config.gpIcpIter) {
      val correspondences = current.pointSet.pointsWithId.map { case (pt, id) =>
        (id, targetOps.closestPointOnSurface(pt).point)
      }.toIndexedSeq
      current = model.posterior(correspondences, Config.gpNoise).mean
    }
    current
  }

  /** Pointwise mean of meshes already in correspondence (same topology). */
  def meanMesh(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    require(meshes.nonEmpty)
    val n   = meshes.length
    val pts = (0 until meshes.head.pointSet.numberOfPoints).map { i =>
      val id  = PointId(i)
      val sum = meshes.foldLeft(EuclideanVector.zeros[_3D])((acc, m) => acc + m.pointSet.point(id).toVector)
      (sum * (1.0 / n)).toPoint
    }
    TriangleMesh3D(pts, meshes.head.triangulation)
  }
}
