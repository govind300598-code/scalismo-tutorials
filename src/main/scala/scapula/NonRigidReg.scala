package scapula

import scalismo.common.{Field, RealSpace}
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random
import scalismo.common.PointId

object NonRigidReg {

  /**
   * GP-ICP non-rigid registration.
   *
   * Kernel (single Gaussian, diagonal):
   *   k(x,y) = gpScale · exp(−‖x−y‖² / 2·gpSigma²) · I₃
   *
   * Each iteration:
   *  1. For every vertex of the current mesh, find closest surface point on target.
   *  2. Treat those as noisy observations and update the GP posterior.
   *  3. The posterior mean becomes the next mesh estimate.
   *
   * @param reference decimated reference mesh (SSM lives at this resolution)
   * @param target    rigidly-aligned target specimen
   */
  def register(
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D]
  )(implicit rng: Random): TriangleMesh[_3D] = {

    // ── Build kernel and GP ───────────────────────────────────────────────────
    val scalarKernel = GaussianKernel[_3D](Config.gpSigma) * Config.gpScale
    val kernel       = DiagonalKernel(scalarKernel, 3)
    val zeroMean     = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp           = GaussianProcess(zeroMean, kernel)

    // Low-rank Nystrom approximation anchored at the reference mesh's vertices.
    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(
      gp,
      reference.pointSet,
      numBasisFunctions = Config.gpBasis
    )

    val model     = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
    val targetOps = target.operations
    var current   = reference

    // ── GP-ICP iterations ─────────────────────────────────────────────────────
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
      val sum = meshes.foldLeft(EuclideanVector.zeros[_3D]) { (acc, m) =>
        acc + m.pointSet.point(id).toVector
      }
      (sum * (1.0 / n)).toPoint
    }
    TriangleMesh3D(pts, meshes.head.triangulation)
  }
}
