package scapula

import scalismo.geometry.{Landmark, _3D}
import scalismo.mesh.TriangleMesh
import scalismo.utils.Random

import java.io.File

/**
 * Picks the specimen to use as the non-rigid registration TEMPLATE.
 *
 * A template must be an actual, watertight scapula: you cannot average vertex positions across specimens before
 * correspondence exists, which is exactly the chicken-and-egg problem non-rigid registration solves. The best
 * substitute for "the population mean" that needs no correspondence to compute is the MEDOID -- the specimen
 * whose rigidly-aligned shape is, on average, closest (smallest mean surface distance) to every other specimen
 * in the pool. It is also, by construction, the least extreme/outlier shape available, which keeps the GP
 * prior's zero-deformation state close to a typical scapula instead of an outlier.
 *
 * This medoid is only the STARTING POINT, not the final answer to "most average shape". Stage2ReferenceRefinement
 * uses it to bootstrap the unbiased-template procedure that Config.refinePasses documents: fit every subject to
 * it, average the now-corresponded results into a new mean shape, and repeat -- which removes the residual bias
 * of having anchored on one real specimen.
 */
object ReferenceSelection {

  final case class Pool(specimen: ScapulaData.Specimen,
                        mesh: TriangleMesh[_3D],
                        landmarks: IndexedSeq[Landmark[_3D]],
                        wasMirrored: Boolean
  )

  final case class Ranked(specimen: ScapulaData.Specimen, meanDistanceToOthers: Double, worstPairHd: Double)

  /**
   * One mesh per subject (or per side if `Config.buildIndependentModel` is false), decimated to `resolution`, all
   * expressed in the LEFT-scapula frame (right-side specimens mirrored). Restricting to one side per subject by
   * default avoids a subject's near-identical left+right pair acting as two votes for the same shape, both in
   * the medoid search and in the mean-shape averaging.
   */
  def loadPool(dir: File, resolution: Int): IndexedSeq[Pool] = {
    val csv = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))

    def toPool(s: ScapulaData.Specimen): Pool = {
      val raw = ScapulaData.loadMesh(s.file).operations.decimate(resolution)
      val lms = landmarks(s.modelId)
      if (s.isRight) Pool(s, ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms), wasMirrored = true)
      else Pool(s, raw, lms, wasMirrored = false)
    }

    if (Config.buildIndependentModel) {
      specimens.groupBy(_.subject).toIndexedSeq.sortBy(_._1).flatMap { case (_, group) =>
        group.find(!_.isRight).orElse(group.headOption).map(toPool)
      }
    } else {
      specimens.sortBy(_.modelId).map(toPool)
    }
  }

  private def alignedDistance(moving: Pool, fixed: Pool)(implicit rng: Random): Metrics.SurfaceStats = {
    val (aligned, _) =
      RigidAlign.landmarkThenIcp(moving.mesh, moving.landmarks, fixed.mesh, fixed.landmarks, Config.icpIterations)
    Metrics.symmetric(aligned, fixed.mesh)
  }

  /**
   * Full O(n^2) pairwise rigid-alignment distance matrix over an already-loaded pool. Returns the medoid and the
   * full ranking (best/most-average first).
   */
  def medoid(pool: IndexedSeq[Pool])(implicit rng: Random): (Pool, IndexedSeq[Ranked]) = {
    require(pool.size >= 3, s"Need at least 3 subjects to pick a medoid reference, found ${pool.size}")

    val ranked = pool.map { a =>
      val others = pool.filter(_.specimen.modelId != a.specimen.modelId)
      val distances = others.map(b => alignedDistance(a, b))
      val meanOfMeans = distances.map(_.mean).sum / distances.length
      val worstHd = distances.map(_.hd).max
      println(f"    ${a.specimen.modelId}%-24s mean-to-others=$meanOfMeans%6.2f mm   worst-pair-HD=$worstHd%7.2f mm")
      Ranked(a.specimen, meanOfMeans, worstHd)
    }.sortBy(_.meanDistanceToOthers)

    val best = ranked.head
    println(f"\n  Medoid (bootstrap reference): ${best.specimen.modelId}  " +
      f"(mean distance to the rest of the pool = ${best.meanDistanceToOthers}%.2f mm, the smallest in the pool)")

    (pool.find(_.specimen.modelId == best.specimen.modelId).get, ranked)
  }

  /** Convenience: loads the pool and picks the medoid in one call. */
  def chooseReference(dir: File, resolution: Int)(implicit rng: Random): (Pool, IndexedSeq[Ranked]) =
    medoid(loadPool(dir, resolution))
}
