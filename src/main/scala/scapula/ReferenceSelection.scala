package scapula

import scalismo.geometry.{Landmark, _3D}
import scalismo.mesh.TriangleMesh
import scalismo.utils.Random

import java.io.File
import scala.io.Source
import scala.util.Using

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

  /** One entry of the full pairwise distance matrix: `a` rigidly aligned onto `b`. */
  final case class Pairwise(a: String, b: String, stats: Metrics.SurfaceStats)

  /**
   * One mesh per subject (or per side if `Config.buildIndependentModel` is false), decimated to `resolution`, all
   * expressed in the LEFT-scapula frame (right-side specimens mirrored). Restricting to one side per subject by
   * default avoids a subject's near-identical left+right pair acting as two votes for the same shape, both in
   * the medoid search and in the mean-shape averaging.
   *
   * `dirs`: one or more dataset directories, pooled into a single population (see `Config.dataDirs`). Each
   * directory's own landmark CSV is resolved independently before pooling, so folders with different naming
   * conventions can be combined; subject grouping (for `buildIndependentModel`) then runs over the COMBINED
   * list, which is safe here because every subject id already carries its own dataset's naming prefix (e.g.
   * "hill_sachs_001" vs. "paired_scapula_002"), so there is no cross-directory collision risk.
   */
  def loadPool(dirs: IndexedSeq[File], resolution: Int): IndexedSeq[Pool] = {
    val specimens: IndexedSeq[(ScapulaData.Specimen, IndexedSeq[Landmark[_3D]])] = dirs.flatMap { dir =>
      val csv = ScapulaData.csvFile(dir)
      val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
      ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId)).map(s => s -> landmarks(s.modelId))
    }

    def toPool(entry: (ScapulaData.Specimen, IndexedSeq[Landmark[_3D]])): Pool = {
      val (s, lms) = entry
      val raw = ScapulaData.loadMesh(s.file).operations.decimate(resolution)
      if (s.isRight) Pool(s, ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms), wasMirrored = true)
      else Pool(s, raw, lms, wasMirrored = false)
    }

    val full =
      if (Config.buildIndependentModel) {
        specimens.groupBy(_._1.subject).toIndexedSeq.sortBy(_._1).flatMap { case (_, group) =>
          group.find(!_._1.isRight).orElse(group.headOption).map(toPool)
        }
      } else {
        specimens.sortBy(_._1.modelId).map(toPool)
      }

    if (Config.subjectLimit > 0) full.take(Config.subjectLimit) else full
  }

  private def alignedDistance(moving: Pool, fixed: Pool)(implicit rng: Random): Metrics.SurfaceStats = {
    val (aligned, _) = RigidAlign.landmarkThenIcp(moving.mesh, moving.landmarks, fixed.mesh, fixed.landmarks,
      Config.icpIterations, useScaling = true)
    Metrics.symmetric(aligned, fixed.mesh)
  }

  /** The full O(n^2) pairwise rigid-alignment distance matrix over an already-loaded pool (asymmetric: a->b uses
    * a aligned onto b, which need not equal b aligned onto a). This is the underlying "distance error matrix". */
  def pairwiseDistanceMatrix(pool: IndexedSeq[Pool])(implicit rng: Random): IndexedSeq[Pairwise] =
    for {
      a <- pool
      b <- pool
      if a.specimen.modelId != b.specimen.modelId
    } yield Pairwise(a.specimen.modelId, b.specimen.modelId, alignedDistance(a, b))

  /** Ranks every pool member by mean distance to everyone else and returns the medoid (rank #1). */
  def medoid(pool: IndexedSeq[Pool], matrix: IndexedSeq[Pairwise]): (Pool, IndexedSeq[Ranked]) = {
    require(pool.size >= 2, s"Need at least 2 subjects to pick a medoid reference, found ${pool.size}")
    val byA = matrix.groupBy(_.a)

    val ranked = pool.map { a =>
      val distances = byA.getOrElse(a.specimen.modelId, IndexedSeq.empty).map(_.stats)
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

  /** Convenience: loads the pool, computes the full pairwise matrix, and picks the medoid in one call. */
  def chooseReference(dirs: IndexedSeq[File], resolution: Int)(implicit rng: Random): (Pool, IndexedSeq[Ranked], IndexedSeq[Pairwise]) = {
    val pool = loadPool(dirs, resolution)
    val matrix = pairwiseDistanceMatrix(pool)
    val (best, ranked) = medoid(pool, matrix)
    (best, ranked, matrix)
  }

  /**
   * Reuses a medoid/bootstrap choice already written (as `medoid_ranking.csv`) by a PRIOR run against the
   * exact same pool -- e.g. every grid point of a kernel-selection sweep shares the identical bootstrap step,
   * since which subject is the medoid (and the whole ranking) only depends on rigid+scale alignment, never on
   * the GPMM kernel being swept. Skips the expensive O(n^2) pairwise-alignment search (`pairwiseDistanceMatrix`
   * + `medoid`) entirely when this succeeds -- for a ~154-subject combined pool that search is the single most
   * expensive fixed step, so avoiding N redundant copies of it across an N-point sweep matters a lot.
   *
   * `cacheDir` must be another run's `SCAPULA_OUT_DIR`. Returns None (caller MUST fall back to a fresh
   * `pairwiseDistanceMatrix`/`medoid` computation) if `medoid_ranking.csv` is missing, empty, or any subject it
   * names isn't found in the freshly-loaded `pool` -- i.e. this can only skip work it can PROVE is still valid
   * for this exact pool (same directories, resolution, seed, subject filters), never silently produce a wrong
   * answer for a pool that has actually changed.
   */
  def cachedMedoid(pool: IndexedSeq[Pool], cacheDir: File): Option[(Pool, IndexedSeq[Ranked])] = {
    val rankingFile = new File(cacheDir, "medoid_ranking.csv")
    if (!rankingFile.exists()) None
    else {
      val byId = pool.map(p => p.specimen.modelId -> p).toMap
      val lines = Using.resource(Source.fromFile(rankingFile))(_.getLines().toIndexedSeq)
      val rows = lines.drop(1).filter(_.trim.nonEmpty).map(_.split(",", -1))

      val ranked = rows.flatMap { c =>
        if (c.length < 4) None
        else
          for {
            p       <- byId.get(c(1))
            meanD   <- c(2).toDoubleOption
            worstHd <- c(3).toDoubleOption
          } yield Ranked(p.specimen, meanD, worstHd)
      }

      if (rows.isEmpty || ranked.length != pool.length || ranked.length != rows.length) None
      else byId.get(rows.head(1)).map(best => (best, ranked))
    }
  }
}
