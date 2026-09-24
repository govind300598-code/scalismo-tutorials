package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.registration.LandmarkRegistration
import scalismo.utils.Random

import java.io.File

/**
 * Cortical-anchor co-registration strategy.
 *
 * When cortical and trabecular bone meshes are segmented from the same scan they
 * live in the same coordinate frame: every mm of translation or degree of rotation
 * applied to one is implicitly applied to the other.  This makes the trabecular
 * alignment trivial once the cortical alignment has been computed:
 *
 *   1. Align the cortical mesh to a reference (landmark Procrustes + trimmed ICP).
 *      Cortical bone has clear anatomical landmarks and a well-defined surface, so
 *      standard rigid registration is reliable.
 *
 *   2. Recover the single composed rigid transform T that maps the original cortical
 *      mesh to its aligned position.
 *
 *   3. Apply T to the trabecular mesh of the same specimen.  No landmarks are needed
 *      for the trabecular structure.
 *
 * Reference: Rasmussen & Williams, Gaussian Processes for Machine Learning (MIT
 * Press, 2006) — recommended by the collaborator who originated this strategy.
 */
object CorticalAnchorAlign {

  /**
   * A matched cortical + trabecular mesh pair from one specimen.
   *
   * Naming conventions supported by [[BonePairLoader]]:
   *   • sibling directories  : <root>/cortical/<id>.stl  + <root>/trabecular/<id>.stl
   *   • shared directory     : <dir>/<id>_cortical.stl   + <dir>/<id>_trabecular.stl
   */
  final case class BonePair(
    modelId: String,
    corticalFile: File,
    trabecularFile: File,
    isRight: Boolean,
    subject: String
  )

  /**
   * The result of aligning one BonePair.
   *
   * @param cortical   cortical mesh aligned to the reference pose
   * @param trabecular trabecular mesh aligned by the same transform (no extra landmarks)
   * @param landmarks  cortical landmarks carried through the full transform
   */
  final case class AlignedPair(
    cortical: TriangleMesh[_3D],
    trabecular: TriangleMesh[_3D],
    landmarks: IndexedSeq[Landmark[_3D]]
  )

  /**
   * Align one BonePair using the cortical bone as an anchor.
   *
   * The full composed transform (landmark Procrustes → trimmed ICP) is recovered
   * from a dense set of before/after point correspondences on the cortical mesh
   * and then applied verbatim to the trabecular mesh.  A rigid transform has only
   * 6 DOF, so even a subsample of 200 point pairs determines it to floating-point
   * precision.
   *
   * @param cortical       cortical mesh of the moving specimen
   * @param trabecular     trabecular mesh of the SAME specimen (same scan coordinates)
   * @param corticalLms    landmarks on the cortical mesh
   * @param refCortical    cortical reference mesh (the fixed target)
   * @param refCorticalLms landmarks on the reference cortical mesh
   * @param icpIterations  ICP iterations applied after landmark pre-alignment (0 = skip ICP)
   */
  def alignPair(
    cortical: TriangleMesh[_3D],
    trabecular: TriangleMesh[_3D],
    corticalLms: IndexedSeq[Landmark[_3D]],
    refCortical: TriangleMesh[_3D],
    refCorticalLms: IndexedSeq[Landmark[_3D]],
    icpIterations: Int = 30
  )(implicit rng: Random): AlignedPair = {

    // Step 1: align cortical with landmark Procrustes + ICP
    val (alignedCortical, alignedLms) =
      RigidAlign.landmarkThenIcp(cortical, corticalLms, refCortical, refCorticalLms, icpIterations)

    // Step 2: recover the composed rigid transform from before/after point pairs.
    //   200 pairs is far more than the 3 needed to pin a 6-DOF transform; the
    //   Procrustes solve will return essentially the exact composed motion.
    val sampleSize = math.min(200, cortical.pointSet.numberOfPoints)
    val stride = math.max(1, cortical.pointSet.numberOfPoints / sampleSize)
    val pairs = cortical.pointSet.points
      .zip(alignedCortical.pointSet.points)
      .zipWithIndex
      .collect { case (pair, i) if i % stride == 0 => pair }
      .toIndexedSeq

    val composedTransform = LandmarkRegistration.rigid3DLandmarkRegistration(
      pairs,
      center = Point3D(0, 0, 0)
    )

    // Step 3: apply the identical transform to the trabecular mesh — the only
    //   thing that makes this valid is that both meshes came from the same scan.
    val alignedTrabecular = trabecular.transform(composedTransform)

    AlignedPair(alignedCortical, alignedTrabecular, alignedLms)
  }
}

/**
 * Discovers BonePair files from two common directory layouts.
 *
 *   Layout A — sibling directories:
 *     <root>/
 *       cortical/   specimen_L.stl  specimen_R.stl  …
 *       trabecular/ specimen_L.stl  specimen_R.stl  …
 *
 *   Layout B — shared directory with suffixes:
 *     <dir>/  specimen_L_cortical.stl  specimen_L_trabecular.stl  …
 */
object BonePairLoader {

  def fromSiblingDirs(root: File): IndexedSeq[CorticalAnchorAlign.BonePair] = {
    val corticalDir    = new File(root, "cortical")
    val trabecularDir  = new File(root, "trabecular")
    require(corticalDir.isDirectory,   s"Expected cortical dir at ${corticalDir.getPath}")
    require(trabecularDir.isDirectory, s"Expected trabecular dir at ${trabecularDir.getPath}")

    val corticalFiles = stlMap(corticalDir)
    val trabecularFiles = stlMap(trabecularDir)

    corticalFiles.keySet.intersect(trabecularFiles.keySet).toIndexedSeq.sorted.map { id =>
      CorticalAnchorAlign.BonePair(
        modelId       = id,
        corticalFile  = corticalFiles(id),
        trabecularFile = trabecularFiles(id),
        isRight       = id.endsWith("_R"),
        subject       = ScapulaData.subjectKey(id)
      )
    }
  }

  def fromSuffixedDir(dir: File): IndexedSeq[CorticalAnchorAlign.BonePair] = {
    val corticalFiles   = stlMapBySuffix(dir, "_cortical")
    val trabecularFiles = stlMapBySuffix(dir, "_trabecular")

    corticalFiles.keySet.intersect(trabecularFiles.keySet).toIndexedSeq.sorted.map { id =>
      CorticalAnchorAlign.BonePair(
        modelId       = id,
        corticalFile  = corticalFiles(id),
        trabecularFile = trabecularFiles(id),
        isRight       = id.endsWith("_R"),
        subject       = ScapulaData.subjectKey(id)
      )
    }
  }

  private def stlMap(dir: File): Map[String, File] =
    Option(dir.listFiles()).getOrElse(Array.empty)
      .filter(_.getName.toLowerCase.endsWith(".stl"))
      .map(f => f.getName.stripSuffix(".stl") -> f)
      .toMap

  private def stlMapBySuffix(dir: File, suffix: String): Map[String, File] =
    Option(dir.listFiles()).getOrElse(Array.empty)
      .filter(f => f.getName.toLowerCase.endsWith(s"$suffix.stl"))
      .map(f => f.getName.stripSuffix(s"$suffix.stl") -> f)
      .toMap
}
