package scapula

import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.utils.Random

/**
 * GP-ICP non-rigid registration.
 *
 * Standard Gaussian-Process Iterative Closest Point algorithm:
 *   1. From the current deformed reference, find the closest point on the target surface
 *      for each sampled reference vertex.
 *   2. Treat those closest points as noisy observations of the GP deformation field.
 *   3. Compute the GP posterior mean — this is the MAP-optimal deformation.
 *   4. The deformed reference under this MAP deformation is the new current estimate.
 *   5. Repeat.
 *
 * The result has the SAME TOPOLOGY as the reference mesh, giving point-to-point
 * correspondence between all registered shapes.
 */
object NonRigidReg {

  /**
   * Run GP-ICP.
   *
   * @param priorModel        FFDM prior model (reference mesh + low-rank GP prior).
   * @param target            Rigid-aligned target mesh to register to.
   * @param iterations        Number of ICP iterations.
   * @param numCorrespondences Number of reference points sampled per iteration.
   * @param sigma2            Isotropic observation noise variance (mm²).
   *                          Larger = less confident in correspondences = smoother deformation.
   */
  def gpIcp(
    priorModel:        PointDistributionModel[_3D, TriangleMesh],
    target:            TriangleMesh[_3D],
    iterations:        Int,
    numCorrespondences: Int    = 1000,
    sigma2:            Double  = 1.0
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val targetOps = target.operations
    // Start from the prior mean (= reference shape)
    var currentMesh = priorModel.mean

    for (iter <- 0 until iterations) {
      // Sample reference vertices uniformly (spatially uniform, not by index stride)
      val sampleIds: IndexedSeq[PointId] =
        RigidAlign.uniformIds(priorModel.reference, numCorrespondences)

      // Build observations: (PointId in reference, closest point on target surface).
      // PointDistributionModel.posterior takes (PointId, Point[D]) pairs — observed target positions,
      // not deformation vectors; the internal conversion to deformation happens inside the method.
      val observations: IndexedSeq[(PointId, Point[_3D])] = sampleIds.map { pid =>
        val currentPt = currentMesh.pointSet.point(pid)
        val closestPt = targetOps.closestPointOnSurface(currentPt).point
        (pid, closestPt)
      }

      // Posterior model given the correspondences; posterior mean = MAP-optimal deformation.
      val posteriorModel = priorModel.posterior(observations, sigma2)
      currentMesh = posteriorModel.mean

      if ((iter + 1) % 10 == 0) {
        val d = Metrics.symmetric(currentMesh, target).mean
        print(f"\r      iter ${iter + 1}%3d/$iterations  mean-surf-dist=${d}%.2f mm")
        System.out.flush()
      }
    }
    println()
    currentMesh
  }

  /**
   * Compute the point-wise mean mesh from a collection of registered (in-correspondence) meshes.
   * All meshes must share the same triangulation as `reference`.
   */
  def meanMesh(
    reference:         TriangleMesh[_3D],
    registeredMeshes:  IndexedSeq[TriangleMesh[_3D]]
  ): TriangleMesh[_3D] = {
    require(registeredMeshes.nonEmpty, "Cannot compute mean of empty collection")
    val n = registeredMeshes.length.toDouble
    val meanPts = reference.pointSet.pointIds.toIndexedSeq.map { pid =>
      val cx = registeredMeshes.map(_.pointSet.point(pid).x).sum / n
      val cy = registeredMeshes.map(_.pointSet.point(pid).y).sum / n
      val cz = registeredMeshes.map(_.pointSet.point(pid).z).sum / n
      Point3D(cx, cy, cz)
    }
    TriangleMesh3D(meanPts, reference.triangulation)
  }
}
