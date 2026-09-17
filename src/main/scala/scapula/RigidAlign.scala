package scapula

import breeze.linalg.{det, eigSym, DenseMatrix, DenseVector}
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

  /**
   * Principal axes (columns, unit length, sorted by descending eigenvalue) and centroid of a mesh's point cloud,
   * via eigendecomposition of the covariance matrix.
   */
  private def principalAxes(mesh: TriangleMesh[_3D]): (Point[_3D], DenseMatrix[Double]) = {
    val pts = mesh.pointSet.points.toIndexedSeq
    val n = pts.length
    val cx = pts.map(_.x).sum / n
    val cy = pts.map(_.y).sum / n
    val cz = pts.map(_.z).sum / n
    val centroid = Point3D(cx, cy, cz)
    var cov = DenseMatrix.zeros[Double](3, 3)
    pts.foreach { p =>
      val d = DenseVector(p.x - cx, p.y - cy, p.z - cz)
      cov = cov + (d * d.t)
    }
    cov = cov / n.toDouble
    val es = eigSym(cov) // ascending eigenvalues
    val axes = DenseMatrix.zeros[Double](3, 3)
    Seq(2, 1, 0).zipWithIndex.foreach { case (srcIdx, dstIdx) => axes(::, dstIdx) := es.eigenvectors(::, srcIdx) }
    (centroid, axes)
  }

  /**
   * Rigid alignment of two meshes that do NOT share landmarks -- needed for an external template (e.g. a generic
   * scapula STL downloaded separately) that was never digitised with the population's landmark protocol.
   *
   * Plain ICP alone is not safe here: without a decent initial pose it converges to the nearest local optimum, which
   * for a thin, strongly asymmetric bone like a scapula is very often the wrong one entirely. Coarse pose is instead
   * obtained by matching principal axes (PCA) between the two point clouds. PCA determines each axis only up to
   * sign, so there are 2^3 = 8 candidate rotations; requiring the rotation to be a proper one (determinant +1, i.e.
   * no reflection -- a rigid motion cannot turn a left scapula into a right one) leaves exactly 4. Each of the 4 is
   * refined with trimmed ICP and scored by symmetric surface distance; the best-scoring one is returned together
   * with that residual, so the caller can print/inspect it rather than silently trust an alignment that may have
   * failed.
   *
   * If the template mesh is the wrong side (mirrored) relative to the population, no candidate here will score well
   * -- mirror it first (see ScapulaData.mirrorMesh) and retry.
   */
  def robustTemplateAlign(moving: TriangleMesh[_3D],
                          target: TriangleMesh[_3D],
                          icpIterations: Int = 60,
                          icpNumPoints: Int = 3000,
                          icpTrimFraction: Double = 0.2
  )(implicit rng: Random): (TriangleMesh[_3D], Metrics.SurfaceStats) = {
    val (movingCentroid, movingAxes) = principalAxes(moving)
    val (targetCentroid, targetAxes) = principalAxes(target)

    // det(R) = det(targetAxes) * det(D) * det(movingAxes) must equal +1 for R to be a proper rotation.
    val requiredDetD = math.signum(det(targetAxes)) * math.signum(det(movingAxes))

    def rotateAroundCentroids(p: Point[_3D], r: DenseMatrix[Double]): Point[_3D] = {
      val d = DenseVector(p.x - movingCentroid.x, p.y - movingCentroid.y, p.z - movingCentroid.z)
      val rd = r * d
      Point3D(targetCentroid.x + rd(0), targetCentroid.y + rd(1), targetCentroid.z + rd(2))
    }

    val signCombos = for (sx <- Seq(1.0, -1.0); sy <- Seq(1.0, -1.0); sz <- Seq(1.0, -1.0)) yield (sx, sy, sz)
    val properSignCombos = signCombos.filter { case (sx, sy, sz) => math.abs(sx * sy * sz - requiredDetD) < 1e-6 }
    require(properSignCombos.length == 4, s"expected 4 proper rotation candidates, got ${properSignCombos.length}")

    val candidates = properSignCombos.map { case (sx, sy, sz) =>
      val d = DenseMatrix.eye[Double](3)
      d(0, 0) = sx; d(1, 1) = sy; d(2, 2) = sz
      val r = targetAxes * d * movingAxes.t
      val coarse = moving.transform(p => rotateAroundCentroids(p, r))
      val refined = rigidIcp(coarse, target, icpIterations, icpNumPoints, icpTrimFraction)
      (refined, Metrics.symmetric(refined, target))
    }
    candidates.minBy(_._2.mean)
  }

  /**
   * Generalized Procrustes Analysis (Gower 1975) over landmark configurations only -- the classical definition of
   * GPA, applied here because the raw specimens are not yet in point correspondence (different STL vertex
   * counts/order), so scalismo's own mesh-level DataCollection.gpa (which requires a shared domain) does not apply
   * until AFTER Stage 2's non-rigid registration has already run.
   *
   * Without this, "the reference frame" is whichever one real specimen happens to be picked as the pivot -- every
   * other specimen is aligned to THAT INDIVIDUAL's own raw pose, which is a bias (arbitrary though the individual's
   * shape itself is well-motivated to pick). GPA instead iterates: align every specimen's landmarks to the current
   * mean configuration, recompute the mean from the newly aligned landmarks, repeat until the mean stops moving.
   * The result is a population mean pose that is not tied to any single specimen's own coordinate frame.
   *
   * Returns the converged mean landmark configuration -- not a real mesh, since it is now a purely statistical
   * construct with no associated surface, hence still requires a real specimen's mesh (whichever is closest to it)
   * to serve as the actual ICP-refinement target surface downstream.
   */
  def landmarkGPA(
      landmarksById: Map[String, IndexedSeq[Landmark[_3D]]],
      maxIterations: Int = 10,
      haltDistanceMm: Double = 0.01
  ): IndexedSeq[Landmark[_3D]] = {
    val ids = landmarksById.keys.toIndexedSeq
    require(ids.nonEmpty, "landmarkGPA needs at least one specimen")
    val landmarkNames = landmarksById(ids.head).map(_.id)

    def meanOf(configs: IndexedSeq[IndexedSeq[Landmark[_3D]]]): IndexedSeq[Landmark[_3D]] =
      landmarkNames.map { name =>
        val pts = configs.map(cfg => cfg.find(_.id == name).get.point)
        Landmark(name, Point3D(pts.map(_.x).sum / pts.length, pts.map(_.y).sum / pts.length, pts.map(_.z).sum / pts.length))
      }

    // Start from an arbitrary specimen's own configuration as the initial alignment target -- GPA's result does not
    // depend on this choice, only the path to convergence does.
    var target: IndexedSeq[Landmark[_3D]] = landmarksById(ids.head)
    var previousMeanPoints: Option[IndexedSeq[Point[_3D]]] = None
    var iteration = 0
    var converged = false

    while (iteration < maxIterations && !converged) {
      val aligned = ids.map { id =>
        val lms = landmarksById(id)
        val t = ScapulaData.rigidFromLandmarks(lms, target)
        lms.map(lm => lm.copy(point = t(lm.point)))
      }
      val newMean = meanOf(aligned)
      val newMeanPoints = newMean.map(_.point)
      previousMeanPoints.foreach { prev =>
        val maxShift = prev.zip(newMeanPoints).map { case (a, b) => (a - b).norm }.max
        if (maxShift < haltDistanceMm) converged = true
      }
      previousMeanPoints = Some(newMeanPoints)
      target = newMean
      iteration += 1
    }
    target
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
