package scapula

import scalismo.common.{PointId, UnstructuredPoints3D}
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess}
import scalismo.utils.Random

object NonRigidReg {

  def registerAll(
    reference    : TriangleMesh[_3D],
    targets      : IndexedSeq[(String, TriangleMesh[_3D])],
    icpIterations: Int          = 40,
    outDir       : java.io.File = new java.io.File(".")
  )(implicit rng: Random): IndexedSeq[(String, TriangleMesh[_3D])] = {

    println("  Building GP prior (sigma=20, scale=5, Cholesky)...")

    // Single Gaussian kernel: sigma = 20 mm, scaleFactor = 5
    val kernel    = DiagonalKernel[_3D](GaussianKernel[_3D](sigma = 20.0, scaleFactor = 5.0), 3)
    val gp        = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = 0.01,
      interpolator      = NearestNeighborInterpolator3D[TriangleMesh, EuclideanVector[_3D]]()
    )
    val discreteGP = lowRankGP.discretize(reference)
    println(s"  GP rank: ${discreteGP.rank}")

    def runIcp(target: TriangleMesh[_3D]): TriangleMesh[_3D] = {
      val tgtOps  = target.operations
      var current = reference

      for (iter <- 0 until icpIterations) {
        // Anneal noise: loose at iter 0 (sigma2=4), tight at last iter (sigma2=0.25)
        val alpha  = iter.toDouble / math.max(1, icpIterations - 1)
        val sigma2 = 4.0 * (1.0 - alpha) + 0.25 * alpha

        val sampleIds = UniformMeshSampler3D(current, 1200)
          .sample()
          .map { case (pt, _) => reference.pointSet.findClosestPoint(pt).id }
          .distinct

        val obs: IndexedSeq[(PointId, EuclideanVector[_3D])] = sampleIds.map { ptId =>
          val curPt = current.pointSet.point(ptId)
          val tgtPt = tgtOps.closestPointOnSurface(curPt).point
          ptId -> (tgtPt - reference.pointSet.point(ptId))
        }

        val post   = discreteGP.posterior(obs, sigma2)
        val newPts = reference.pointSet.pointIds.toIndexedSeq.map { ptId =>
          reference.pointSet.point(ptId) + post.mean(ptId)
        }
        current = TriangleMesh3D(UnstructuredPoints3D(newPts), reference.triangulation)
      }
      current
    }

    targets.zipWithIndex.map { case ((id, target), i) =>
      println(s"  GP-ICP [${i + 1}/${targets.size}] $id")
      val reg = runIcp(target)
      scalismo.io.MeshIO.writeMesh(reg, new java.io.File(outDir, s"reg_$id.stl")).get
      (id, reg)
    }
  }

  /** Convenience wrapper for a single target. */
  def register(
    reference    : TriangleMesh[_3D],
    target       : TriangleMesh[_3D],
    icpIterations: Int = 40
  )(implicit rng: Random): TriangleMesh[_3D] =
    registerAll(reference, IndexedSeq(("_tmp", target)), icpIterations).head._2
}
