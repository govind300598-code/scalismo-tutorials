package scapula

import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.numerics.UniformMeshSampler3D
import scalismo.registration.LandmarkRegistration
import scalismo.utils.Random

/** Rigid (pose-only) alignment. This stage removes position and orientation; it must NOT remove shape. */
object RigidAlign {

  /**
   * Spatially uniform sample of point ids.
   *
   * Sampling by index stride (allIds.filter(_ % step == 0)) is NOT equivalent: STL vertex order reflects
   * scan/mesh-generation order, so a stride can systematically under-sample thin structures such as the acromion and
   * the coracoid process.
   */
  def uniformIds(mesh: TriangleMesh[_3D], n: Int)(implicit rng: Random): IndexedSeq[PointId] = {
    val requested = math.min(n, mesh.pointSet.numberOfPoints)
    UniformMeshSampler3D(mesh, requested)
      .sample()
      .map { case (pt, _) => mesh.pointSet.findClosestPoint(pt).id }
      .distinct
  }

  /**
   * Trimmed rigid ICP.
   *
   * The worst `trimFraction` of correspondences is dropped before each Procrustes solve, so that a handful of
   * non-corresponding points (thin structures present on one bone and not the other) cannot drag the whole pose.
   *
   * IMPORTANT: like every ICP variant, this minimises a MEAN criterion. It gives no guarantee at all about the maximum
   * (Hausdorff) distance, which may legitimately increase while the fit improves.
   */
  def rigidIcp(moving: TriangleMesh[_3D],
               target: TriangleMesh[_3D],
               iterations: Int = 30,
               numPoints: Int = 2000,
               trimFraction: Double = 0.15
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val ops = target.operations
    val ids = uniformIds(moving, numPoints)
    var current = moving

    for (_ <- 0 until iterations) {
      val pairs = ids.map { id =>
        val p = current.pointSet.point(id)
        (p, ops.closestPointOnSurface(p).point)
      }
      val keep = math.max(10, (pairs.length * (1.0 - trimFraction)).toInt)
      val trimmed = pairs.sortBy { case (m, t) => (m - t).norm }.take(keep)
      val trans = LandmarkRegistration.rigid3DLandmarkRegistration(trimmed, center = Point3D(0, 0, 0))
      current = current.transform(trans)
    }
    current
  }

  /** Landmark Procrustes followed by trimmed rigid ICP. Returns the aligned mesh and the aligned landmarks. */
  def landmarkThenIcp(mesh: TriangleMesh[_3D],
                      lms: IndexedSeq[Landmark[_3D]],
                      targetMesh: TriangleMesh[_3D],
                      targetLms: IndexedSeq[Landmark[_3D]],
                      icpIterations: Int = 30
  )(implicit rng: Random): (TriangleMesh[_3D], IndexedSeq[Landmark[_3D]]) = {
    val lmTrans = ScapulaData.rigidFromLandmarks(lms, targetLms)
    val preAligned = mesh.transform(lmTrans)
    val preLms = lms.map(lm => lm.copy(point = lmTrans(lm.point)))

    if (icpIterations <= 0) (preAligned, preLms)
    else {
      // Recover the rigid motion ICP applied, so the landmarks can be carried along with the mesh.
      val refined = rigidIcp(preAligned, targetMesh, icpIterations)
      val motion = LandmarkRegistration.rigid3DLandmarkRegistration(
        preAligned.pointSet.points.zip(refined.pointSet.points).toIndexedSeq,
        center = Point3D(0, 0, 0)
      )
      (refined, preLms.map(lm => lm.copy(point = motion(lm.point))))
    }
  }
}
