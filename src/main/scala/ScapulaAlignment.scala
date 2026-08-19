import scalismo.geometry.*
import scalismo.common.*
import scalismo.mesh.*
import scalismo.registration.*
import scalismo.io.MeshIO
import scalismo.ui.api.*

import java.io.{File, PrintWriter}
import scala.io.Source
import scalismo.utils.Random.FixedSeed.randBasis

/**
 * Alignment of paired scapula STLs to a common reference, with robust ICP refinement.
 *
 * ---------------------------------------------------------------------------
 * WHY THE HAUSDORFF DISTANCE WAS SO HIGH
 * ---------------------------------------------------------------------------
 * The previous run reported mean = 1.81 mm, HD95 = 4.90 mm but HD = 19.90 mm
 * (up to 37.7 mm). A mean of 1.8 mm with a max of 30+ mm is not a broken
 * alignment - it is a max statistic being driven by four separate things:
 *
 *   1. NO SIZE NORMALISATION (dominant cause).
 *      A rigid transform has no scale parameter, so it cannot absorb the
 *      10-15 % linear size difference between specimens. Aligning every
 *      scapula to one 64-year-old male reference leaves that size residual on
 *      the extremities (inferior angle, acromion, medial border), which is
 *      exactly where Hausdorff samples. The evidence is in the old log: every
 *      female specimen got WORSE after ICP (17.65 -> 19.89, 30.07 -> 34.74)
 *      while every male specimen got BETTER (18.87 -> 13.55, 15.70 -> 13.54),
 *      and the one same-subject pair, 001_M_64_R against its own contralateral
 *      side, scored HD 7.33 mm / Dice 0.822 - far better than anything else.
 *      The pipeline was fine; it was measuring biology.
 *
 *   2. ICP MINIMISED THE MEAN, WHICH INFLATES THE MAX.
 *      Plain least-squares ICP slides a smaller mesh to sit centrally inside a
 *      larger one. That lowers the average error and raises the worst-case
 *      error. With no outlier rejection, non-corresponding regions also drag
 *      the fit.
 *
 *   3. HD AND HD95 WERE NOT THE SAME MEASUREMENT.
 *      MeshMetrics.hausdorffDistance is SYMMETRIC (max over both directions),
 *      while MeshMetrics.avgDistance and the hand-rolled HD95 were
 *      ONE-DIRECTIONAL (fitted -> target). So HD included reference-side
 *      surface that the fitted mesh never covers, and HD95 did not. They were
 *      not comparable numbers.
 *
 *   4. MESH DEFECTS ON INDIVIDUAL SPECIMENS.
 *      003_M_61_L is off-trend: landmark error 7.60 mm (as good as specimens
 *      scoring HD ~13 mm) yet HD 37.72 mm and the lowest Dice, 0.341. That
 *      signature - good landmarks, catastrophic max - means a handful of stray
 *      vertices: a disconnected island or segmentation spike, not a pose error.
 *
 * WHAT THIS VERSION CHANGES
 *   - `useScaling` switches landmark alignment and ICP to a similarity
 *     (Procrustes) transform, removing the size term.
 *   - ICP now rejects correspondences by distance quantile and by surface
 *     normal agreement, and stops on convergence.
 *   - All metrics are symmetric point-to-surface, so mean / HD95 / HD are the
 *     same measurement at different quantiles, plus the directed halves.
 *   - Meshes are screened for disconnected components before alignment.
 *   - Reference specimen defaults to the median-size specimen, not the first.
 *   - BUGFIX: the ICP transform is now applied to the landmarks too. Before,
 *     `finalLandmarks` were the pre-ICP positions while `finalMesh` was
 *     post-ICP, which is why landmarks float off the bone in the viewer.
 *
 * READ THE RIGHT NUMBER: Hausdorff is a maximum over ~10^5 vertices, so it is
 * a defect detector, not an alignment score. Judge alignment on mean and HD95.
 */
object ScapulaAlignment extends App {

  scalismo.initialize()

  // ===========================================================================
  // 1. Configuration
  // ===========================================================================

  val dataDir   = new File("/home/g25upadh/Documents/database_v1.11/paired_scapulae_STLs")
  val outputDir = new File("/home/g25upadh/Documents/database_v1.11/aligned_scapulae_STLs")
  outputDir.mkdirs()

  /**
   * Normalise SIZE as well as pose.
   *
   * true  -> similarity (Procrustes) alignment. Shape-only comparison, which is
   *          the standard preprocessing for a statistical SHAPE model. Expect
   *          Hausdorff to drop sharply, especially on the female specimens.
   * false -> rigid alignment, size preserved. Use this only if specimen size is
   *          variation you intend to keep in the model - and then accept that
   *          HD of 15-25 mm across subjects is the expected biological spread,
   *          not a bug. Judge quality by HD95 in that case.
   */
  val useScaling: Boolean = true

  /**
   * Reference specimen. "auto" picks the specimen whose landmark centroid size
   * is closest to the median, so the worst-case size mismatch is halved
   * relative to picking an arbitrary specimen. Or hard-code an ID, e.g.
   * "paired_scapula_001_M_64_L".
   */
  val referenceSelection: String = "auto"

  // --- robust ICP ------------------------------------------------------------
  val icpIterations   = 60
  val icpSamplePoints = 4000
  /** Keep this fraction of the best correspondences each iteration (trimmed ICP). */
  val icpTrimQuantile = 0.85
  /** Reject correspondences whose surface normals disagree by more than ~60 deg. */
  val icpMinNormalDot = 0.5
  val icpToleranceMm  = 1e-4

  // --- mesh quality control --------------------------------------------------
  /** Drop disconnected islands (segmentation debris) before aligning. */
  val keepLargestComponent = true

  val launchViewer = true

  // --- landmark CSV layout ---------------------------------------------------
  val landmarkNames       = IndexedSeq("GC", "TS", "IA", "PLA", "AC")
  val landmarkStartColumn = Map("GC" -> 11, "TS" -> 14, "IA" -> 17, "PLA" -> 20, "AC" -> 23)

  // ===========================================================================
  // 2. Landmark CSV
  // ===========================================================================

  val csvFile: File = dataDir.listFiles()
    .filter(f => f.getName.toLowerCase.endsWith(".csv"))
    .filter(f => f.getName.toLowerCase.contains("scapulae") && f.getName.toLowerCase.contains("model_data"))
    .filterNot(f => f.getName.toLowerCase.startsWith("single"))
    .headOption
    .getOrElse(throw new RuntimeException(
      s"No 'paired_scapulae...model_data...csv' file found inside ${dataDir.getPath}. " +
      s"Files present: ${dataDir.listFiles().map(_.getName).mkString(", ")}"
    ))

  println(s"Using landmark CSV: ${csvFile.getAbsolutePath}")

  def readLandmarkCsv(file: File): Map[String, IndexedSeq[Landmark[_3D]]] = {
    val source = Source.fromFile(file)
    val lines  = try source.getLines().toIndexedSeq finally source.close()

    val header      = lines.head.split(",", -1)
    val maxColumn   = landmarkStartColumn.values.max + 2
    require(header.length > maxColumn,
      s"CSV has only ${header.length} columns but landmark coordinates are expected up to column $maxColumn. " +
      s"Header: ${header.mkString(" | ")}")

    // Print the columns being read once, so the mapping can be eyeballed.
    println("  landmark column mapping:")
    landmarkNames.foreach { nm =>
      val i = landmarkStartColumn(nm)
      println(f"    $nm%-4s -> cols $i%2d,${i + 1}%2d,${i + 2}%2d  [${header(i)}, ${header(i + 1)}, ${header(i + 2)}]")
    }

    lines.tail.filter(_.trim.nonEmpty).map { line =>
      val cols    = line.split(",", -1)
      val modelId = cols(0).trim
      val lms = landmarkNames.map { nm =>
        val i = landmarkStartColumn(nm)
        Landmark(nm, Point3D(cols(i).trim.toDouble, cols(i + 1).trim.toDouble, cols(i + 2).trim.toDouble))
      }
      modelId -> lms
    }.toMap
  }

  val allLandmarks = readLandmarkCsv(csvFile)
  println(s"Landmark CSV loaded: ${allLandmarks.size} model entries, ${landmarkNames.size} landmarks each " +
          s"(${landmarkNames.mkString(", ")})")

  // ===========================================================================
  // 3. Geometry helpers
  // ===========================================================================

  /**
   * Reflect through the plane x = 0 to bring a right scapula into left-handed
   * form. Any reflection plane works: two reflections differ by a rigid motion,
   * which the subsequent Procrustes step absorbs. Triangle winding is flipped
   * so that outward normals stay outward - the ICP normal test relies on this.
   */
  def mirrorPoint(p: Point[_3D]): Point[_3D] = Point3D(-p.x, p.y, p.z)

  def mirrorMesh(mesh: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val mirroredPoints = mesh.pointSet.points.map(mirrorPoint).toIndexedSeq
    val flippedCells   = mesh.triangulation.triangles.map(t => TriangleCell(t.ptId1, t.ptId3, t.ptId2))
    TriangleMesh3D(mirroredPoints, TriangleList(flippedCells))
  }

  def mirrorLandmarks(lms: IndexedSeq[Landmark[_3D]]): IndexedSeq[Landmark[_3D]] =
    lms.map(lm => lm.copy(point = mirrorPoint(lm.point)))

  /** Procrustes centroid size of a landmark configuration - a scale-free size proxy. */
  def centroidSize(lms: IndexedSeq[Landmark[_3D]]): Double = {
    val pts    = lms.map(_.point)
    val centre = pts.map(_.toVector).reduce(_ + _) * (1.0 / pts.size)
    math.sqrt(pts.map(p => (p.toVector - centre).norm2).sum)
  }

  /**
   * Best-fitting transform mapping `from` onto `to`.
   *
   * NOTE: the rotation centre must stay at the origin for the similarity case.
   * Scalismo composes the similarity as translate(scale(rotate(p))) where the
   * rotation is about `center` but the scaling is about the origin; those two
   * centres only agree when `center` is the origin. The rigid case is correct
   * for any centre, but we use the origin throughout for consistency.
   */
  def fitTransform(pairs: IndexedSeq[(Point[_3D], Point[_3D])], withScaling: Boolean): Point[_3D] => Point[_3D] = {
    val origin = Point3D(0, 0, 0)
    if (withScaling) LandmarkRegistration.similarity3DLandmarkRegistration(pairs, origin)
    else LandmarkRegistration.rigid3DLandmarkRegistration(pairs, origin)
  }

  def percentile(values: IndexedSeq[Double], p: Double): Double = {
    if (values.isEmpty) 0.0
    else {
      val sorted = values.sorted
      val rank   = math.ceil(p * sorted.length).toInt
      sorted(math.min(sorted.length - 1, math.max(0, rank - 1)))
    }
  }

  // ===========================================================================
  // 4. Mesh quality control
  // ===========================================================================

  /** Vertex-connectivity components, largest first. */
  def connectedComponents(mesh: TriangleMesh[_3D]): IndexedSeq[Array[Int]] = {
    val n       = mesh.pointSet.numberOfPoints
    val visited = Array.fill(n)(false)
    val out     = scala.collection.mutable.ArrayBuffer.empty[Array[Int]]

    var seed = 0
    while (seed < n) {
      if (!visited(seed)) {
        val component = scala.collection.mutable.ArrayBuffer.empty[Int]
        val stack     = scala.collection.mutable.Stack.empty[Int]
        stack.push(seed)
        visited(seed) = true
        while (stack.nonEmpty) {
          val current = stack.pop()
          component += current
          mesh.triangulation.adjacentPointsForPoint(PointId(current)).foreach { nb =>
            if (nb.id < n && !visited(nb.id)) {
              visited(nb.id) = true
              stack.push(nb.id)
            }
          }
        }
        out += component.toArray
      }
      seed += 1
    }
    out.sortBy(-_.length).toIndexedSeq
  }

  final case class MeshQc(mesh: TriangleMesh[_3D], numComponents: Int, droppedVertices: Int)

  def qualityControl(mesh: TriangleMesh[_3D]): MeshQc = {
    // adjacentPointsForPoint is indexed over the triangulation's point range; if
    // that does not line up with the point set (unreferenced vertices), skip QC
    // rather than risk an out-of-bounds on a mesh we do not fully understand.
    if (!keepLargestComponent || mesh.triangulation.pointIds.length != mesh.pointSet.numberOfPoints)
      MeshQc(mesh, 1, 0)
    else {
      val components = connectedComponents(mesh)
      if (components.length <= 1) MeshQc(mesh, 1, 0)
      else {
        val largest = components.head.toSet
        val cleaned = mesh.operations.maskPoints(pid => largest.contains(pid.id)).transformedMesh
        MeshQc(cleaned, components.length, mesh.pointSet.numberOfPoints - largest.size)
      }
    }
  }

  // ===========================================================================
  // 5. Robust ICP
  // ===========================================================================

  def subsampleIds(mesh: TriangleMesh[_3D], n: Int): IndexedSeq[PointId] = {
    val all = mesh.pointSet.pointIds.toIndexedSeq
    if (all.length <= n) all
    else {
      // Deterministic uniform sample; a fixed stride can align with the vertex
      // ordering an STL inherits from marching cubes and sample unevenly.
      val rng = new scala.util.Random(42)
      rng.shuffle(all).take(n)
    }
  }

  final case class IcpResult(
    mesh: TriangleMesh[_3D],
    transform: Point[_3D] => Point[_3D],
    iterationsRun: Int,
    rmsMm: Double,
    keptFraction: Double
  )

  /**
   * Trimmed ICP with normal-compatibility rejection.
   *
   * Two guards keep the fit from being dragged by regions that have no true
   * counterpart (differing segmentation extent, genuine shape difference):
   *   - normal test: drop pairs whose surface normals disagree, which stops the
   *     anterior surface being matched to the posterior surface across the thin
   *     scapular blade;
   *   - distance trimming: drop the worst (1 - icpTrimQuantile) of pairs, so a
   *     minority of non-corresponding points cannot steer the transform.
   */
  def icpRefine(moving: TriangleMesh[_3D], target: TriangleMesh[_3D], withScaling: Boolean): IcpResult = {
    val targetOps     = target.operations
    val targetNormals = target.vertexNormals
    val movingNormals = moving.vertexNormals
    val sampleIds     = subsampleIds(moving, icpSamplePoints)

    var points  = sampleIds.map(id => moving.pointSet.point(id))
    var normals = sampleIds.map(id => movingNormals.atPoint(id))

    var accumulated: Point[_3D] => Point[_3D] = identity
    var iterationsRun = 0
    var lastRms       = Double.MaxValue
    var keptFraction  = 1.0
    var converged     = false

    while (iterationsRun < icpIterations && !converged) {
      val candidates = points.indices.map { i =>
        val p       = points(i)
        val q       = targetOps.closestPointOnSurface(p).point
        val nTarget = targetNormals.atPoint(target.pointSet.findClosestPoint(q).id)
        (p, q, (p - q).norm, normals(i).dot(nTarget))
      }

      val normalOk = candidates.filter(_._4 >= icpMinNormalDot)
      val pool     = if (normalOk.length >= 20) normalOk else candidates

      val cutoff = percentile(pool.map(_._3), icpTrimQuantile)
      val kept   = pool.filter(_._3 <= cutoff)
      val used   = if (kept.length >= 20) kept else pool

      keptFraction = used.length.toDouble / candidates.length

      val step = fitTransform(used.map(c => (c._1, c._2)), withScaling)

      // Advance the samples and rotate their normals with the same transform.
      // For a similarity transform, t(p + n) - t(p) is the rotated normal
      // scaled by s, so normalising recovers the rotated unit normal exactly.
      val advanced = points.map(step)
      normals = points.indices.map(i => (step(points(i) + normals(i)) - advanced(i)).normalize)
      points  = advanced

      val previous = accumulated
      accumulated = (p: Point[_3D]) => step(previous(p))

      val rms = math.sqrt(used.map(c => c._3 * c._3).sum / used.length)
      if (math.abs(lastRms - rms) < icpToleranceMm) converged = true
      lastRms = rms
      iterationsRun += 1
    }

    IcpResult(moving.transform(accumulated), accumulated, iterationsRun, lastRms, keptFraction)
  }

  // ===========================================================================
  // 6. Metrics - symmetric, point-to-surface, one consistent definition
  // ===========================================================================

  final case class Metrics(
    mean: Double,       // symmetric
    hd95: Double,       // symmetric
    hd99: Double,       // symmetric
    hd: Double,         // symmetric max
    meanFittedToRef: Double,
    meanRefToFitted: Double,
    hdFittedToRef: Double,
    hdRefToFitted: Double,
    fractionOver5mm: Double,
    fractionOver10mm: Double,
    verticesOver10mm: Int,
    worstPoint: Point[_3D],
    dice: Double
  )

  def surfaceDistances(from: TriangleMesh[_3D], to: TriangleMesh[_3D]): IndexedSeq[Double] = {
    val ops = to.operations
    from.pointSet.points.map(p => (p - ops.closestPointOnSurface(p).point).norm).toIndexedSeq
  }

  def computeMetrics(fitted: TriangleMesh[_3D], target: TriangleMesh[_3D]): Metrics = {
    val dFittedToRef = surfaceDistances(fitted, target)
    val dRefToFitted = surfaceDistances(target, fitted)
    val all          = dFittedToRef ++ dRefToFitted

    val worstIdxFitted = dFittedToRef.indices.maxBy(dFittedToRef)
    val worstIdxRef    = dRefToFitted.indices.maxBy(dRefToFitted)
    val worstPoint =
      if (dFittedToRef(worstIdxFitted) >= dRefToFitted(worstIdxRef)) fitted.pointSet.point(PointId(worstIdxFitted))
      else target.pointSet.point(PointId(worstIdxRef))

    Metrics(
      mean             = all.sum / all.length,
      hd95             = percentile(all, 0.95),
      hd99             = percentile(all, 0.99),
      hd               = math.max(dFittedToRef.max, dRefToFitted.max),
      meanFittedToRef  = dFittedToRef.sum / dFittedToRef.length,
      meanRefToFitted  = dRefToFitted.sum / dRefToFitted.length,
      hdFittedToRef    = dFittedToRef.max,
      hdRefToFitted    = dRefToFitted.max,
      fractionOver5mm  = all.count(_ > 5.0).toDouble / all.length,
      fractionOver10mm = all.count(_ > 10.0).toDouble / all.length,
      verticesOver10mm = all.count(_ > 10.0),
      worstPoint       = worstPoint,
      dice             = MeshMetrics.diceCoefficient(fitted, target)
    )
  }

  // ===========================================================================
  // 7. Reference selection
  // ===========================================================================

  def modelIdOf(f: File): String = f.getName.stripSuffix(".stl")
  def isRight(name: String): Boolean = name.endsWith("_R")

  val stlFiles = dataDir.listFiles().filter(_.getName.endsWith(".stl")).sortBy(_.getName).toIndexedSeq
  require(stlFiles.nonEmpty, s"No STL files found in ${dataDir.getPath}")

  /** Landmarks in left-handed form, so sizes are comparable across sides. */
  val canonicalLandmarks: Map[String, IndexedSeq[Landmark[_3D]]] =
    stlFiles.map(modelIdOf).flatMap { id =>
      allLandmarks.get(id).map { lms => id -> (if (isRight(id)) mirrorLandmarks(lms) else lms) }
    }.toMap

  val sizes = canonicalLandmarks.view.mapValues(centroidSize).toMap

  val referenceModelId: String =
    if (referenceSelection != "auto") referenceSelection
    else {
      val sorted = sizes.toIndexedSeq.sortBy(_._2)
      val median = sorted(sorted.length / 2)._2
      sorted.minBy { case (_, s) => math.abs(s - median) }._1
    }

  val referenceFile = new File(dataDir, s"$referenceModelId.stl")
  require(referenceFile.exists(), s"Reference STL not found: ${referenceFile.getPath}")

  val referenceRaw = MeshIO.readMesh(referenceFile).get
  val referenceQc  = qualityControl(if (isRight(referenceModelId)) mirrorMesh(referenceRaw) else referenceRaw)
  val referenceMesh = referenceQc.mesh
  val refLandmarks  = canonicalLandmarks(referenceModelId)
  val refSize       = sizes(referenceModelId)

  println()
  println("[SPECIMEN SIZE] Procrustes centroid size of the landmark configuration")
  println(f"  smallest: ${sizes.minBy(_._2)._1}%-28s ${sizes.values.min}%7.2f")
  println(f"  largest:  ${sizes.maxBy(_._2)._1}%-28s ${sizes.values.max}%7.2f")
  println(f"  spread:   ${(sizes.values.max / sizes.values.min - 1.0) * 100}%.1f%% linear difference between extremes")
  println()
  println(s"Reference: $referenceModelId (${if (referenceSelection == "auto") "median size, auto-selected" else "user-specified"})")
  println(f"  ${referenceMesh.pointSet.numberOfPoints} points | centroid size $refSize%.2f | landmarks: ${refLandmarks.map(_.id).mkString(", ")}")
  if (referenceQc.numComponents > 1)
    println(f"  [QC] reference had ${referenceQc.numComponents} disconnected components, dropped ${referenceQc.droppedVertices} vertices")
  println(s"  Alignment mode: ${if (useScaling) "SIMILARITY (size normalised)" else "RIGID (size preserved)"}")

  // ===========================================================================
  // 8. Process every specimen
  // ===========================================================================

  final case class AlignResult(
    name: String,
    wasMirrored: Boolean,
    relativeSize: Double,
    landmarkErrorMm: Double,
    icpScale: Double,
    icpIterations: Int,
    icpKeptFraction: Double,
    numComponents: Int,
    before: Metrics,
    after: Metrics,
    finalMesh: TriangleMesh[_3D],
    finalLandmarks: IndexedSeq[Landmark[_3D]]
  )

  def processOne(targetFile: File): AlignResult = {
    val name    = modelIdOf(targetFile)
    val mirrored = isRight(name)

    val rawLandmarks = allLandmarks.getOrElse(name,
      throw new RuntimeException(s"No landmark row in CSV for Model ID '$name'"))
    val rawMesh = MeshIO.readMesh(targetFile).get

    val (orientedMesh, orientedLandmarks) =
      if (mirrored) (mirrorMesh(rawMesh), mirrorLandmarks(rawLandmarks))
      else (rawMesh, rawLandmarks)

    val qc = qualityControl(orientedMesh)

    // --- landmark (Procrustes) alignment -------------------------------------
    val lmPairs = refLandmarks.map(_.id).flatMap { id =>
      for {
        src <- orientedLandmarks.find(_.id == id)
        dst <- refLandmarks.find(_.id == id)
      } yield (src.point, dst.point)
    }
    val lmTransform      = fitTransform(lmPairs, useScaling)
    val landmarkAligned  = qc.mesh.transform(lmTransform)
    val alignedLandmarks = orientedLandmarks.map(lm => lm.copy(point = lmTransform(lm.point)))

    val lmErrors = refLandmarks.flatMap { r =>
      alignedLandmarks.find(_.id == r.id).map(a => (r.point - a.point).norm)
    }
    val landmarkErrorMm = lmErrors.sum / lmErrors.length
    val before = computeMetrics(landmarkAligned, referenceMesh)

    // --- robust ICP refinement ------------------------------------------------
    val icp   = icpRefine(landmarkAligned, referenceMesh, useScaling)
    val after = computeMetrics(icp.mesh, referenceMesh)

    // BUGFIX: carry the ICP transform through to the landmarks. Previously the
    // reported/displayed landmarks were the pre-ICP ones, so they no longer sat
    // on the mesh that was written out.
    val finalLandmarks = alignedLandmarks.map(lm => lm.copy(point = icp.transform(lm.point)))

    // Effective scale of the ICP step, recovered from how it stretches a unit vector.
    val probeOrigin = Point3D(0, 0, 0)
    val icpScale    = (icp.transform(Point3D(1, 0, 0)) - icp.transform(probeOrigin)).norm

    MeshIO.writeMesh(icp.mesh, new File(outputDir, s"aligned_$name.stl")).get

    val relativeSize = sizes.getOrElse(name, refSize) / refSize

    println(f"  $name%-28s | mir=$mirrored%-5s | size=${relativeSize * 100}%5.1f%% | LM=$landmarkErrorMm%5.2f | " +
            f"before: mean=${before.mean}%5.2f HD95=${before.hd95}%5.2f HD=${before.hd}%6.2f | " +
            f"after: mean=${after.mean}%5.2f HD95=${after.hd95}%5.2f HD=${after.hd}%6.2f Dice=${after.dice}%.3f" +
            (if (qc.numComponents > 1) f"  [QC: ${qc.numComponents} components, ${qc.droppedVertices} verts dropped]" else ""))

    AlignResult(name, mirrored, relativeSize, landmarkErrorMm, icpScale, icp.iterationsRun, icp.keptFraction,
                qc.numComponents, before, after, icp.mesh, finalLandmarks)
  }

  val targetFiles = stlFiles.filter(_.getName != referenceFile.getName)

  println()
  println(s"[ALIGNMENT] ${if (useScaling) "similarity" else "rigid"} + trimmed ICP, processing ${targetFiles.length} meshes")
  println()

  val results = targetFiles.map(processOne)

  // ===========================================================================
  // 9. Summary
  // ===========================================================================

  println()
  println("[VALIDATION SUMMARY -- after ICP refinement, all metrics symmetric point-to-surface]")
  println(f"  ${"specimen"}%-28s ${"mean"}%6s ${"HD95"}%6s ${"HD99"}%6s ${"HD"}%7s ${">5mm"}%7s ${">10mm"}%7s ${"Dice"}%6s")
  results.foreach { r =>
    val m = r.after
    println(f"  ${r.name}%-28s ${m.mean}%6.2f ${m.hd95}%6.2f ${m.hd99}%6.2f ${m.hd}%7.2f " +
            f"${m.fractionOver5mm * 100}%6.2f%% ${m.fractionOver10mm * 100}%6.2f%% ${m.dice}%6.3f")
  }

  def avg(f: AlignResult => Double): Double = results.map(f).sum / results.length

  println()
  println(f"[AGGREGATE -- averaged across all ${results.length} meshes]")
  println(f"  Mean distance:     ${avg(_.after.mean)}%.2f mm")
  println(f"  HD95:              ${avg(_.after.hd95)}%.2f mm")
  println(f"  HD99:              ${avg(_.after.hd99)}%.2f mm")
  println(f"  Hausdorff (max):   ${avg(_.after.hd)}%.2f mm")
  println(f"  Dice coefficient:  ${avg(_.after.dice)}%.3f")
  println(f"  Landmark residual: ${avg(_.landmarkErrorMm)}%.2f mm")
  println(f"  ICP correspondences kept: ${avg(_.icpKeptFraction) * 100}%.1f%%")

  // --- did ICP help or hurt? -------------------------------------------------
  val icpWorsenedHd = results.count(r => r.after.hd > r.before.hd)
  println()
  println("[ICP EFFECT]")
  println(f"  mean: ${avg(_.before.mean)}%.2f -> ${avg(_.after.mean)}%.2f mm")
  println(f"  HD95: ${avg(_.before.hd95)}%.2f -> ${avg(_.after.hd95)}%.2f mm")
  println(f"  HD:   ${avg(_.before.hd)}%.2f -> ${avg(_.after.hd)}%.2f mm")
  println(f"  ICP increased HD on $icpWorsenedHd of ${results.length} specimens")
  if (icpWorsenedHd > results.length / 2)
    println("  -> ICP is still trading max error for mean error. If useScaling is false, that is expected:\n" +
            "     a rigid fit cannot remove the size term, so ICP centres the mismatch instead.")

  // --- separate defects from biological spread -------------------------------
  println()
  println("[WORST HAUSDORFF -- and what is driving it]")
  results.sortBy(-_.after.hd).take(5).foreach { r =>
    val m = r.after
    val verdict =
      if (m.verticesOver10mm < 50)
        f"SPIKE: only ${m.verticesOver10mm} vertices exceed 10 mm -> inspect this mesh for segmentation debris"
      else if (math.abs(r.relativeSize - 1.0) > 0.05)
        f"SIZE: specimen is ${(r.relativeSize - 1.0) * 100}%+.1f%% of reference and ${m.fractionOver10mm * 100}%.2f%% of vertices exceed 10 mm"
      else
        f"SHAPE: broad mismatch, ${m.fractionOver10mm * 100}%.2f%% of vertices exceed 10 mm"
    println(f"  ${r.name}%-28s HD=${m.hd}%6.2f (HD95=${m.hd95}%5.2f)  $verdict")
    println(f"      worst point at (${m.worstPoint.x}%.1f, ${m.worstPoint.y}%.1f, ${m.worstPoint.z}%.1f), " +
            f"directed max: fitted->ref ${m.hdFittedToRef}%.2f, ref->fitted ${m.hdRefToFitted}%.2f")
  }

  results.filter(_.numComponents > 1).foreach { r =>
    println(f"  [QC] ${r.name} contained ${r.numComponents} disconnected components - debris was removed before alignment")
  }

  // --- metrics CSV -----------------------------------------------------------
  val metricsCsv = new File(outputDir, "alignment_metrics.csv")
  val writer     = new PrintWriter(metricsCsv)
  try {
    writer.println("model_id,mirrored,relative_size,landmark_error_mm,icp_scale,icp_iterations,icp_kept_fraction," +
                   "components,mean_mm,hd95_mm,hd99_mm,hd_mm,hd_fitted_to_ref_mm,hd_ref_to_fitted_mm," +
                   "frac_over_5mm,frac_over_10mm,dice")
    results.foreach { r =>
      val m = r.after
      writer.println(s"${r.name},${r.wasMirrored},${r.relativeSize},${r.landmarkErrorMm},${r.icpScale}," +
                     s"${r.icpIterations},${r.icpKeptFraction},${r.numComponents},${m.mean},${m.hd95},${m.hd99}," +
                     s"${m.hd},${m.hdFittedToRef},${m.hdRefToFitted},${m.fractionOver5mm},${m.fractionOver10mm},${m.dice}")
    }
  } finally writer.close()

  println()
  println(s"  Aligned STLs written to: ${outputDir.getAbsolutePath}")
  println(s"  Per-specimen metrics:    ${metricsCsv.getAbsolutePath}")

  println()
  println("[HOW TO READ THESE NUMBERS]")
  println("  Hausdorff is a maximum over ~10^5 vertices: one bad vertex sets it. Use it to FIND defects,")
  println("  not to score alignment. Mean and HD95 are the alignment quality numbers.")
  if (!useScaling)
    println("  useScaling = false, so specimen size is still in the residual. A 10-15% size spread across")
    println("  subjects puts 15-25 mm on the extremities by itself. Set useScaling = true to remove it.")

  // ===========================================================================
  // 10. Viewer
  // ===========================================================================

  if (launchViewer) {
    println()
    println("Launching Scalismo-UI viewer...")
    val ui    = ScalismoUI()
    val group = ui.createGroup("Rigid_Alignment_And_ICP")

    ui.show(group, referenceMesh, "REFERENCE_" + referenceModelId)
    ui.show(group, refLandmarks, "REFERENCE_landmarks")

    results.foreach { r =>
      try ui.show(group, r.finalMesh, r.name)
      catch { case e: Exception => println(s"  [warn] failed to show mesh ${r.name}: ${e.getMessage}") }
      try ui.show(group, r.finalLandmarks, s"${r.name}_landmarks")
      catch { case e: Exception => println(s"  [warn] failed to show landmarks for ${r.name}: ${e.getMessage}") }
    }

    println()
    println("Viewer is open. Press ENTER in this terminal to close it and exit.")
    scala.io.StdIn.readLine()
  }
}
