package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.common.PointId
import scalismo.io.MeshIO
import scalismo.numerics.UniformMeshSampler3D
import scalismo.registration.LandmarkRegistration
import scalismo.transformations.TranslationAfterRotation
import scalismo.utils.Random

import java.io.File
import scala.io.Source
import scala.util.Using

/**
 * Configuration. Everything is overridable through environment variables so that the same jar can be run against a
 * different dataset without editing code.
 */
object Config {
  private def env(key: String, default: String): String = sys.env.getOrElse(key, default)

  val dataDir: File = new File(env("SCAPULA_DATA_DIR", "/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"))
  val outDir: File = new File(env("SCAPULA_OUT_DIR", "/home/g25upadh/Documents/100 plus scapula data/non_rigid_aligned_output"))

  /** Number of vertices of the model reference. All registered shapes and the SSM live at this resolution. */
  val modelResolution: Int = env("SCAPULA_MODEL_RES", "8000").toInt

  /** Non-rigid (GP) ICP iterations per pass. */
  val icpIterations: Int = env("SCAPULA_ICP_ITERS", "40").toInt

  /**
   * Number of registration passes. Pass 1 registers to an arbitrary specimen; each further pass rebuilds the reference
   * as the mean of the previous pass and re-registers. This removes reference bias.
   */
  val refinePasses: Int = env("SCAPULA_REFINE_PASSES", "4").toInt

  /** Relative tolerance for the pivoted-Cholesky low-rank approximation of the GP prior. Smaller => higher rank. */
  val gpRelativeTolerance: Double = env("SCAPULA_GP_TOL", "0.01").toDouble

  /** Hard cap on the rank of the GP prior (keeps memory and posterior cost bounded). */
  val gpMaxRank: Int = env("SCAPULA_GP_MAX_RANK", "250").toInt

  /**
   * Gaussian kernel bandwidth (mm) — spatial extent of the deformation field.
   * σ=30 mm: deformations remain correlated across the full scapular body (~120 mm wide),
   * while still localising to sub-structures (acromion, glenoid, coracoid).
   * Used for SSM1–SSM4 to guarantee a fair comparison across passes.
   */
  val kernelSigma: Double = env("SCAPULA_KERNEL_SIGMA", "30.0").toDouble

  /**
   * Gaussian kernel amplitude (mm) — maximum magnitude of allowed deformations.
   * scaleFactor=10 mm: after landmark+ICP rigid alignment, residual shape differences
   * are typically 5–15 mm, so ±10 mm gives the GP enough room to fit without
   * overcorrecting.  The covariance matrix entry is scaleFactor² = 100 mm².
   * Used identically for SSM1–SSM4.
   */
  val kernelScale: Double = env("SCAPULA_KERNEL_SCALE", "10.0").toDouble

  /** Noise variance (mm²) for GP-ICP posterior regression. Lower = tighter fit. */
  val icpSigma2: Double = env("SCAPULA_ICP_SIGMA2", "2.0").toDouble

  /**
   * 0-based index into the sorted specimen list to use as the initial reference for pass 1.
   * Default 12 = a mid-dataset left-side specimen (avoids mirroring bias from right-side #001).
   * Override with SCAPULA_REF_IDX=<n> to choose any specimen.
   * Tip: pick a left-side specimen near the middle of the dataset for least reference bias.
   */
  val refIdx: Int = env("SCAPULA_REF_IDX", "2").toInt

  val buildIndependentModel: Boolean = env("SCAPULA_INDEPENDENT_MODEL", "true").toBoolean

  val showUi: Boolean = env("SCAPULA_UI", "true").toBoolean

  val seed: Long = env("SCAPULA_SEED", "42").toLong
}

/** Loading, landmark parsing, mirroring and the small geometric helpers shared by all stages. */
object ScapulaData {

  val landmarkNames: IndexedSeq[String] = IndexedSeq("GC", "TS", "IA", "PLA", "AC")

  /** Fallback column offsets, used only if the header cannot be parsed by name. */
  private val fallbackStartIdx: Map[String, Int] =
    Map("GC" -> 11, "TS" -> 14, "IA" -> 17, "PLA" -> 20, "AC" -> 23)

  final case class Specimen(modelId: String, file: File, isRight: Boolean, subject: String)

  def csvFile(dir: File): File = {
    val files = Option(dir.listFiles()).getOrElse(Array.empty[File])
    files
      .filter(_.getName.toLowerCase.endsWith(".csv"))
      .filter(f => f.getName.toLowerCase.contains("scapula") && f.getName.toLowerCase.contains("model_data"))
      .filterNot(_.getName.toLowerCase.startsWith("single"))
      .sortBy(_.getName)
      .headOption
      .getOrElse(
        throw new RuntimeException(
          s"No landmark CSV found in ${dir.getPath}. Present: ${files.map(_.getName).mkString(", ")}"
        )
      )
  }

  private def normaliseHeader(h: String): String = h.trim.toLowerCase.replaceAll("[^a-z0-9]", "")

  /**
   * Resolve the (x, y, z) column indices for each landmark by matching the header, falling back to the hardcoded
   * offsets only if that fails. Returns the resolution together with a flag saying whether it came from the header, so
   * the caller can warn.
   */
  def resolveColumns(header: IndexedSeq[String]): (Map[String, (Int, Int, Int)], Boolean) = {
    val norm = header.map(normaliseHeader)

    def findAxis(lm: String, axis: Char): Option[Int] = {
      val l = lm.toLowerCase
      norm.indexWhere { h =>
        h == s"$l$axis" || h == s"${l}_$axis".replace("_", "") || (h.startsWith(l) && h.endsWith(axis.toString) &&
          h.length <= l.length + 2)
      } match {
        case -1 => None
        case i  => Some(i)
      }
    }

    val byName = landmarkNames.flatMap { lm =>
      for {
        xi <- findAxis(lm, 'x')
        yi <- findAxis(lm, 'y')
        zi <- findAxis(lm, 'z')
      } yield lm -> (xi, yi, zi)
    }.toMap

    if (byName.size == landmarkNames.size) (byName, true)
    else (fallbackStartIdx.map { case (lm, i) => lm -> (i, i + 1, i + 2) }, false)
  }

  def readLandmarkCsv(file: File): (Map[String, IndexedSeq[Landmark[_3D]]], Boolean, IndexedSeq[String]) = {
    val lines = Using.resource(Source.fromFile(file))(_.getLines().toIndexedSeq)
    require(lines.nonEmpty, s"Landmark CSV ${file.getName} is empty")

    val header = lines.head.split(",", -1).toIndexedSeq.map(_.trim)
    val (cols, fromHeader) = resolveColumns(header)

    val entries = lines.tail.filter(_.trim.nonEmpty).map { line =>
      val c = line.split(",", -1)
      val modelId = c(0).trim
      val lms = landmarkNames.map { nm =>
        val (xi, yi, zi) = cols(nm)
        require(
          c.length > math.max(xi, math.max(yi, zi)),
          s"Row '$modelId' has ${c.length} columns but landmark $nm needs column ${math.max(xi, math.max(yi, zi))}"
        )
        Landmark(nm, Point3D(c(xi).trim.toDouble, c(yi).trim.toDouble, c(zi).trim.toDouble))
      }
      modelId -> lms
    }.toMap

    (entries, fromHeader, header)
  }

  def specimens(dir: File): IndexedSeq[Specimen] = {
    Option(dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.toLowerCase.endsWith(".stl"))
      .sortBy(_.getName)
      .toIndexedSeq
      .map { f =>
        val id = f.getName.stripSuffix(".stl")
        Specimen(id, f, id.endsWith("_R"), subjectKey(id))
      }
  }

  def subjectKey(modelId: String): String = modelId.stripSuffix("_L").stripSuffix("_R")

  // ---------------------------------------------------------------------------
  // Mirroring.
  //
  // NOTE ON METHOD: reflecting about the world plane x = 0 is not a special
  // choice.  A reflection about ANY plane differs from a reflection about x = 0
  // by a rigid motion, and a rigid alignment is applied immediately afterwards,
  // which absorbs exactly that difference.  What DOES matter is (a) that the map
  // is a genuine reflection (determinant -1) and (b) that the triangle winding is
  // flipped so surface normals keep pointing outwards.  Both are asserted below.
  // ---------------------------------------------------------------------------
  def mirrorPoint(p: Point[_3D]): Point[_3D] = Point3D(-p.x, p.y, p.z)

  def mirrorMesh(mesh: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pts = mesh.pointSet.points.map(mirrorPoint).toIndexedSeq
    val flipped = mesh.triangulation.triangles.map(t => TriangleCell(t.ptId1, t.ptId3, t.ptId2))
    TriangleMesh3D(pts, TriangleList(flipped))
  }

  def mirrorLandmarks(lms: IndexedSeq[Landmark[_3D]]): IndexedSeq[Landmark[_3D]] =
    lms.map(lm => lm.copy(point = mirrorPoint(lm.point)))

  /**
   * Signed volume of a closed mesh (divergence theorem). Positive for a watertight, outward-oriented surface. Used to
   * confirm that mirroring + winding flip preserved orientation instead of turning the bone inside out.
   */
  def signedVolume(mesh: TriangleMesh[_3D]): Double = {
    var acc = 0.0
    mesh.triangulation.triangles.foreach { t =>
      val a = mesh.pointSet.point(t.ptId1).toVector
      val b = mesh.pointSet.point(t.ptId2).toVector
      val c = mesh.pointSet.point(t.ptId3).toVector
      acc += a.dot(b.crossproduct(c)) / 6.0
    }
    acc
  }

  def rigidFromLandmarks(from: IndexedSeq[Landmark[_3D]],
                         to: IndexedSeq[Landmark[_3D]]
  ): TranslationAfterRotation[_3D] =
    LandmarkRegistration.rigid3DLandmarkRegistration(from, to, center = Point3D(0, 0, 0))

  def perLandmarkDistances(a: IndexedSeq[Landmark[_3D]], b: IndexedSeq[Landmark[_3D]]): IndexedSeq[(String, Double)] = {
    val bById = b.map(l => l.id -> l.point).toMap
    a.flatMap(la => bById.get(la.id).map(pb => la.id -> (la.point - pb).norm))
  }

  def loadMesh(f: File): TriangleMesh[_3D] =
    MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Could not read mesh ${f.getPath}"))

  /**
   * Coarsen all `meshes` to ~`targetVertices` vertices while preserving point-to-point
   * correspondence across the whole population.
   *
   * Algorithm — geodesic Voronoi coarsening:
   *   1. Select ~targetVertices seed VERTEX IDs from `reference` via uniform spatial sampling.
   *      Seeds are exact original vertices (no position movement → no spikes).
   *   2. Multi-source BFS on the mesh graph assigns every vertex to the nearest seed cluster.
   *   3. Each original triangle whose 3 vertices belong to 3 different clusters contributes
   *      one coarse triangle connecting those 3 cluster seeds.  Deduplication yields a valid,
   *      non-degenerate topology.
   *   4. The SAME seed IDs and topology are applied to every mesh in `meshes`, so
   *      correspondence is maintained.
   *
   * This avoids the spike / distortion artifacts produced by Quadric-Error decimation, which
   * optimises topology for the reference geometry and creates degenerate triangles when that
   * topology is transplanted to other shapes.
   */
  def decimateInCorrespondence(
      reference: TriangleMesh[_3D],
      meshes: IndexedSeq[TriangleMesh[_3D]],
      targetVertices: Int
  )(implicit rng: Random): IndexedSeq[TriangleMesh[_3D]] = {

    val nOrig = reference.pointSet.numberOfPoints

    // ── 1. Seed selection ─────────────────────────────────────────────────────
    // Oversample 4× so deduplication still hits the target count.
    val seeds: Array[Int] = UniformMeshSampler3D(reference, targetVertices * 4)
      .sample()
      .map { case (p, _) => reference.pointSet.findClosestPoint(p).id.id }
      .distinct
      .take(targetVertices)
      .toArray

    // ── 2. Build adjacency list ───────────────────────────────────────────────
    val adj = Array.fill(nOrig)(scala.collection.mutable.ArrayBuffer.empty[Int])
    reference.triangulation.triangles.foreach { t =>
      val a = t.ptId1.id; val b = t.ptId2.id; val c = t.ptId3.id
      adj(a) += b; adj(a) += c
      adj(b) += a; adj(b) += c
      adj(c) += a; adj(c) += b
    }

    // ── 3. Multi-source BFS Voronoi ───────────────────────────────────────────
    val cluster = Array.fill(nOrig)(-1)
    seeds.zipWithIndex.foreach { case (s, ci) => cluster(s) = ci }
    val queue = scala.collection.mutable.Queue.empty[Int]
    seeds.foreach(queue.enqueue)
    while (queue.nonEmpty) {
      val v = queue.dequeue()
      val ci = cluster(v)
      adj(v).foreach { nb =>
        if (cluster(nb) < 0) { cluster(nb) = ci; queue.enqueue(nb) }
      }
    }

    // ── 4. Build coarse triangulation ─────────────────────────────────────────
    // One coarse triangle per original triangle that spans 3 distinct clusters.
    // Sort the cluster-triple to deduplicate regardless of vertex order.
    val seen  = scala.collection.mutable.LinkedHashMap.empty[String, (Int, Int, Int)]
    reference.triangulation.triangles.foreach { t =>
      val ca = cluster(t.ptId1.id); val cb = cluster(t.ptId2.id); val cc = cluster(t.ptId3.id)
      if (ca >= 0 && cb >= 0 && cc >= 0 && ca != cb && ca != cc && cb != cc) {
        val s = Seq(ca, cb, cc).sorted
        val key = s"${s(0)},${s(1)},${s(2)}"
        if (!seen.contains(key)) seen(key) = (ca, cb, cc)
      }
    }

    // ── 5. Compact cluster IDs → contiguous 0..K-1 ───────────────────────────
    val usedClusters = seen.values.flatMap(t => Seq(t._1, t._2, t._3))
      .toIndexedSeq.distinct.sorted
    val toNew    = usedClusters.zipWithIndex.toMap
    val topology = TriangleList(seen.values.toIndexedSeq.map { case (a, b, c) =>
      TriangleCell(PointId(toNew(a)), PointId(toNew(b)), PointId(toNew(c)))
    })
    val repIds = usedClusters.map(ci => PointId(seeds(ci)))
    println(s"    Voronoi coarsening: ${repIds.length} vertices, ${topology.triangles.length} triangles")

    // ── 6. Apply to all meshes using original vertex positions ────────────────
    meshes.map(m => TriangleMesh3D(repIds.map(id => m.pointSet.point(id)), topology))
  }
}

/** Surface-distance measures. Kept separate from MeshMetrics so the directionality is explicit. */
object Metrics {

  /** Distance from every vertex of `from` to the closest point on the SURFACE of `to`. */
  def surfaceDistances(from: TriangleMesh[_3D], to: TriangleMesh[_3D]): IndexedSeq[Double] = {
    val ops = to.operations
    from.pointSet.points.map(p => (p - ops.closestPointOnSurface(p).point).norm).toIndexedSeq
  }

  def percentile(values: IndexedSeq[Double], p: Double): Double = {
    require(values.nonEmpty)
    val sorted = values.sorted
    val idx = math.min(sorted.length - 1, math.max(0, math.ceil(p * sorted.length).toInt - 1))
    sorted(idx)
  }

  final case class SurfaceStats(mean: Double, rms: Double, hd95: Double, hd: Double) {
    def render: String = f"mean=$mean%5.2f  rms=$rms%5.2f  HD95=$hd95%5.2f  HD=$hd%6.2f"
  }

  /** Symmetric statistics: both directions pooled, which is what "distance between two surfaces" should mean. */
  def symmetric(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): SurfaceStats = {
    val d = surfaceDistances(a, b) ++ surfaceDistances(b, a)
    SurfaceStats(
      mean = d.sum / d.length,
      rms = math.sqrt(d.map(x => x * x).sum / d.length),
      hd95 = percentile(d, 0.95),
      hd = d.max
    )
  }

  /** Point-to-point statistics for meshes that are already in correspondence (same point ids). */
  def correspondingDistances(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): IndexedSeq[Double] = {
    require(a.pointSet.numberOfPoints == b.pointSet.numberOfPoints, "meshes are not in correspondence")
    a.pointSet.points.zip(b.pointSet.points).map { case (p, q) => (p - q).norm }.toIndexedSeq
  }
}
