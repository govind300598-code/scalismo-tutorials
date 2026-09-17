package scapula

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.mesh.{TriangleList, TriangleMesh, TriangleMesh3D}
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * STAGE 2 -- build point-to-point correspondence across the whole population by non-rigidly registering a common
 * reference to every specimen (see GPRegistrationCore for the fitting algorithm itself, ported from the mailing
 * list). Run Stage1Diagnostics first: this stage assumes the rigid pipeline (mirroring, landmark frame, orientation)
 * is already known to be sound.
 *
 * Output layout (under Config.outDir), all in VTK (never STL) because STL is an unindexed triangle soup -- reloading
 * one does NOT reproduce the original vertex order, which would silently destroy the correspondence Stage 3 depends
 * on:
 *   reference.vtk              the final (post-refinement) reference topology, in the rigid landmark frame
 *   registered/<modelId>.vtk   every specimen warped into that reference's correspondence
 *   registration_accuracy.csv  per-specimen surface-distance residual between the registered mesh and the real target
 */
object Stage2GPNonRigidRegistration {

  // Every Stage 2 run writes a complete, self-consistent population (one reference topology, one registered mesh
  // per specimen IN THIS RUN). A directory left over from a PRIOR run (e.g. a full 24-specimen run, followed by a
  // SCAPULA_ONLY_SPECIMENS-restricted run) can still contain other specimens' .vtk files registered against a
  // DIFFERENT reference.vtk (different pivot specimen / different decimation -> different vertex correspondence,
  // even at the same vertex COUNT). Stage3's loadRegistered only checks vertex count, so those stale files would be
  // silently loaded alongside the fresh ones and PCA'd as if they were in correspondence with the current
  // reference -- producing a shattered, self-intersecting "noise" mesh with no relation to real anatomy. Clearing
  // the directory at the start of every run is what actually prevents that, not any registration/kernel parameter.
  private def clearStaleVtkFiles(dir: File): Unit = {
    val stale = Option(dir.listFiles()).getOrElse(Array.empty[File]).filter(_.getName.toLowerCase.endsWith(".vtk"))
    if (stale.nonEmpty) {
      stale.foreach(_.delete())
      println(s"  cleared ${stale.length} stale .vtk file(s) from a previous run in ${dir.getPath}")
    }
  }

  private def meanMesh(meshes: IndexedSeq[TriangleMesh[_3D]], triangles: TriangleList): TriangleMesh[_3D] = {
    val n = meshes.head.pointSet.numberOfPoints
    val m = meshes.length.toDouble
    val meanPoints = (0 until n).map { i =>
      val pts = meshes.map(_.pointSet.point(PointId(i)))
      Point3D(pts.map(_.x).sum / m, pts.map(_.y).sum / m, pts.map(_.z).sum / m)
    }
    TriangleMesh3D(meanPoints, triangles)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    println(s"Data directory : ${dir.getAbsolutePath}")
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")
    println(s"Output dir     : ${Config.outDir.getAbsolutePath}")

    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val allSpecimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    require(allSpecimens.nonEmpty, s"No specimens with both a mesh and a landmark row found in ${dir.getPath}")

    val specimens = Config.onlySpecimenIds match {
      case Some(ids) =>
        val wanted = ids ++ Config.referenceSpecimenId.toSet // the reference must be in the run regardless
        val filtered = allSpecimens.filter(s => wanted.contains(s.modelId))
        val missing = wanted -- filtered.map(_.modelId).toSet
        require(missing.isEmpty, s"SCAPULA_ONLY_SPECIMENS/SCAPULA_REFERENCE_SPECIMEN name unknown id(s): " +
          s"${missing.mkString(", ")} -- available: ${allSpecimens.map(_.modelId).mkString(", ")}")
        println(s"SCAPULA_ONLY_SPECIMENS set -- restricting this run to ${filtered.length} of " +
          s"${allSpecimens.length} specimens: ${filtered.map(_.modelId).mkString(", ")}")
        filtered
      case None => allSpecimens
    }
    println(s"${specimens.length} specimens with mesh + landmarks (expect 24 = 12 subjects x L/R, unless " +
      "SCAPULA_ONLY_SPECIMENS restricted this run)")

    // ---------------------------------------------------------------------------------------- canonicalize side
    // Left and right scapulae are mirror images of each other, not related by any rotation. Averaging their
    // landmarks directly (as loaded) is meaningless -- a right specimen's raw GC sits on the opposite side of the
    // body's midline from a left specimen's GC -- and rigidly aligning a right specimen onto a left reference (or
    // vice versa) with a pure rotation cannot work; only a reflection can turn one into the other, which is exactly
    // the doubled/backwards look you get overlaying un-mirrored L and R specimens together. So every specimen is
    // mirrored into a single canonical side (left) here, before pivot selection or any alignment happens.
    final case class Canonical(mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]])
    val canonicalById: Map[String, Canonical] = specimens.map { s =>
      val rawMesh = ScapulaData.loadMesh(s.file)
      val rawLms = landmarks(s.modelId)
      val c = if (s.isRight) Canonical(ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
              else Canonical(rawMesh, rawLms)
      s.modelId -> c
    }.toMap
    println(s"Mirrored ${specimens.count(_.isRight)} right-side specimens onto the left-side convention before " +
      "any alignment, so every specimen being compared/averaged/aligned is in the same anatomical convention.")

    // -------------------------------------------------------------------------------------------------- reference
    // Generalized Procrustes Analysis over landmarks (RigidAlign.landmarkGPA): iteratively re-estimate the
    // population's mean landmark configuration, instead of aligning everyone to one arbitrary real specimen's own
    // raw pose. That mean is a statistical construct with no surface of its own, so the specimen whose landmarks
    // land closest to it is still picked to supply the actual reference MESH -- but that specimen's mesh is then
    // itself rigidly moved onto the GPA mean (via landmark Procrustes) before anyone, including itself, is ICP-
    // refined against it, which is what actually removes the "one real specimen defines the frame" bias.
    val canonicalLandmarksById = specimens.map(s => s.modelId -> canonicalById(s.modelId).lms).toMap
    val gpaMeanLandmarks = RigidAlign.landmarkGPA(canonicalLandmarksById)
    val gpaMeanByName = gpaMeanLandmarks.map(lm => lm.id -> lm.point).toMap
    val referenceSpecimen = Config.referenceSpecimenId match {
      case Some(id) =>
        val s = specimens.find(_.modelId == id).getOrElse(throw new RuntimeException(
          s"SCAPULA_REFERENCE_SPECIMEN='$id' not found among specimens: ${specimens.map(_.modelId).mkString(", ")}"
        ))
        println(s"Reference specimen: $id (forced via SCAPULA_REFERENCE_SPECIMEN, not GPA-selected)")
        s
      case None =>
        val s = specimens.minBy { s =>
          canonicalById(s.modelId).lms.map { lm =>
            val d = (lm.point - gpaMeanByName(lm.id)).norm
            d * d
          }.sum
        }
        println(s"Reference specimen (closest to GPA mean landmark configuration, left-side convention): " +
          s"${s.modelId}")
        s
    }

    // ------------------------------------------------------------------------------------------------ rigid align
    val refCanonical = canonicalById(referenceSpecimen.modelId)
    val refToGpaMean = ScapulaData.rigidFromLandmarks(refCanonical.lms, gpaMeanLandmarks)
    val referenceRawMesh = refCanonical.mesh.transform(refToGpaMean)
    val referenceLms = refCanonical.lms.map(lm => lm.copy(point = refToGpaMean(lm.point)))

    val alignedTargets: IndexedSeq[(String, TriangleMesh[_3D])] = specimens.map { s =>
      val c = canonicalById(s.modelId)
      if (s.modelId == referenceSpecimen.modelId) s.modelId -> referenceRawMesh
      else {
        val (aligned, _) = RigidAlign.landmarkThenIcp(c.mesh, c.lms, referenceRawMesh, referenceLms, Config.icpIterations)
        s.modelId -> aligned
      }
    }
    println(s"Rigidly aligned all ${alignedTargets.length} specimens into the GPA mean landmark frame " +
      s"(${Config.icpIterations} ICP iterations after landmark Procrustes).")

    // Persisted so downstream viewers (e.g. ViewRegistrationHeatmap) can compute per-vertex registered-vs-real-target
    // error without recomputing rigid alignment themselves -- a recompute would not be bit-identical anyway, since
    // ICP's point sampling depends on how many prior calls have already advanced the shared RNG stream.
    Config.outDir.mkdirs()
    val alignedTargetsDir = new File(Config.outDir, "aligned_targets")
    alignedTargetsDir.mkdirs()
    clearStaleVtkFiles(alignedTargetsDir)
    alignedTargets.foreach { case (modelId, mesh) =>
      MeshIO.writeMesh(mesh, new File(alignedTargetsDir, s"$modelId.vtk")).get
    }
    println(s"Wrote ${alignedTargets.length} rigidly-aligned (pre-registration) target meshes to ${alignedTargetsDir.getPath}")

    // --------------------------------------------------------------------------------------- reference topology
    // Prefer an external template mesh (default: the Wikimedia scapula from the mailing-list thread) over one of
    // the 24 paired specimens, so the SSM's topology is not biased toward any one subject's anatomy. The template
    // was never digitised with the population's landmark protocol, so it cannot be landmark-aligned; instead it is
    // robustly rigid-aligned (PCA coarse pose + trimmed ICP, see RigidAlign.robustTemplateAlign) onto the pivot
    // specimen's mesh, which is already known to be in the population's landmark frame.
    val pivotAligned = alignedTargets.find(_._1 == referenceSpecimen.modelId).get._2

    var reference: TriangleMesh[_3D] = Config.referenceMeshFile match {
      case Some(file) if file.exists() =>
        println(s"\nExternal reference template: ${file.getPath}")
        val rawTemplate = ScapulaData.loadMesh(file)

        // A wrong-chirality template (opposite side from the population) cannot be fixed by any rigid ROTATION --
        // only a reflection can turn a left scapula into a right one, and robustTemplateAlign deliberately never
        // tries a reflection (a rigid motion can't do that to a real bone). So chirality is a real ambiguity that
        // has to be settled by trying both orientations and keeping whichever one actually fits, not by guessing.
        val orientationsToTry: Seq[(String, TriangleMesh[_3D])] = Config.referenceOrientation match {
          case "asis"     => Seq("as-is" -> rawTemplate)
          case "mirrored" => Seq("mirrored" -> ScapulaData.mirrorMesh(rawTemplate))
          case _          => Seq("as-is" -> rawTemplate, "mirrored" -> ScapulaData.mirrorMesh(rawTemplate))
        }
        val candidates = orientationsToTry.map { case (label, templateMesh) =>
          val templateDecimated = templateMesh.operations.decimate(Config.modelResolution)
          println(s"  trying orientation '$label' (${templateDecimated.pointSet.numberOfPoints} vertices) onto " +
            s"pivot specimen ${referenceSpecimen.modelId} (PCA coarse pose + trimmed ICP, 4 candidate rotations)...")
          val (aligned, stats) = RigidAlign.robustTemplateAlign(templateDecimated, pivotAligned, Config.icpIterations)
          println(f"    residual: ${stats.render}")
          (label, aligned, stats)
        }
        val (bestLabel, bestAligned, bestStats) = candidates.minBy { case (_, _, stats) => stats.hd95 }
        println(f"  best orientation: '$bestLabel' -- ${bestStats.render}")

        // HD95, not mean, is what actually catches a chirality mismatch: a wrong-side template can still look
        // deceptively OK on mean residual (similar overall bounding envelope) while its anatomical features --
        // glenoid, spine, angles -- are systematically misplaced, which HD95 exposes and a population-wide mean
        // blurs out. Non-rigid registration CANNOT fix a bad correspondence like this; it will make it worse, and
        // do so uniformly across every specimen (as opposed to a few noisy outliers), while burning ~1-2 hours.
        if (bestStats.hd95 > Config.referenceAlignAbortHD95Mm && !Config.referenceAlignForce) {
          throw new RuntimeException(
            f"Best external-reference alignment still has HD95=${bestStats.hd95}%.1f mm, above the " +
              f"${Config.referenceAlignAbortHD95Mm}%.1f mm threshold, in BOTH orientations tried " +
              s"(${candidates.map { case (l, _, s) => f"$l: HD95=${s.hd95}%.1f" }.mkString(", ")}). Aborting " +
              "before the non-rigid registration loop instead of spending ~1-2 hours registering onto a reference " +
              "that's already known to fit badly. Either: (1) set SCAPULA_REFERENCE_MESH=\"\" to use an " +
              "in-population reference instead (recommended), or (2) if you're confident this template is right, " +
              "rerun with SCAPULA_REFERENCE_FORCE=true."
          )
        }
        if (bestStats.mean > Config.referenceAlignWarnMeanMm)
          println(f"  !! WARNING: mean residual ${bestStats.mean}%.1f mm exceeds ${Config.referenceAlignWarnMeanMm}%.1f mm -- " +
            "inspect reference.vtk in ParaView/MeshLab before trusting the rest of this run.")
        bestAligned
      case Some(file) =>
        println(s"\nSCAPULA_REFERENCE_MESH points at ${file.getPath}, which does not exist -- falling back to an " +
          s"in-population reference (${referenceSpecimen.modelId}).")
        referenceRawMesh.operations.decimate(Config.modelResolution)
      case None =>
        referenceRawMesh.operations.decimate(Config.modelResolution)
    }
    println(s"Reference has ${reference.pointSet.numberOfPoints} vertices (requested ${Config.modelResolution}).")

    // -------------------------------------------------------------------------------------------- refinement loop
    var registered: IndexedSeq[(String, TriangleMesh[_3D])] = IndexedSeq.empty
    for (pass <- 1 to Config.refinePasses) {
      println(s"\n=== Registration pass $pass / ${Config.refinePasses} " +
        s"(reference has ${reference.pointSet.numberOfPoints} vertices) ===")
      val lowRankGP = GPRegistrationCore.buildMultiscaleGP(reference)
      println(f"  GP prior rank = ${lowRankGP.rank} (relativeTolerance=${Config.gpRelativeTolerance}, " +
        f"maxRank=${Config.gpMaxRank})")
      val cascade = GPRegistrationCore.defaultCascade(reference)
      val initialCoefficients = DenseVector.zeros[Double](lowRankGP.rank)

      registered = alignedTargets.zipWithIndex.map { case ((modelId, target), i) =>
        val t0 = System.nanoTime()
        val coeffs =
          GPRegistrationCore.registerToTarget(lowRankGP, reference, target, initialCoefficients, cascade)
        val mesh = GPRegistrationCore.warpedMesh(lowRankGP, reference, coeffs)
        val seconds = (System.nanoTime() - t0) / 1e9
        println(f"  [${i + 1}%2d/${alignedTargets.length}] $modelId%-20s registered in $seconds%6.1fs")
        modelId -> mesh
      }

      if (pass < Config.refinePasses) {
        reference = meanMesh(registered.map(_._2), reference.triangulation)
        println(s"  Rebuilt reference as the mean of pass $pass's registered meshes (removes reference bias).")
      }
    }

    // ------------------------------------------------------------------------------------------------------- save
    Config.outDir.mkdirs()
    val registeredDir = new File(Config.outDir, "registered")
    registeredDir.mkdirs()
    clearStaleVtkFiles(registeredDir)
    MeshIO.writeMesh(reference, new File(Config.outDir, "reference.vtk")).get
    registered.foreach { case (modelId, mesh) =>
      MeshIO.writeMesh(mesh, new File(registeredDir, s"$modelId.vtk")).get
    }
    println(s"\nWrote reference.vtk and ${registered.length} registered meshes to ${registeredDir.getPath}")

    // --------------------------------------------------------------------------------------- registration accuracy
    // Surface distance between each registered (reference-topology) mesh and its ACTUAL target mesh. This is the
    // standard "how good is the correspondence" number reported in SSM papers -- not to be confused with the SSM's
    // own generalization/specificity (Stage 3), which measure the *model*, not the per-specimen registration fit.
    println("\n[registration accuracy] registered mesh vs. real target surface, symmetric, after rigid alignment:")
    val targetById = alignedTargets.toMap
    val accuracy = registered.map { case (modelId, mesh) =>
      val stats = Metrics.symmetric(mesh, targetById(modelId))
      println(f"  $modelId%-20s ${stats.render}")
      modelId -> stats
    }

    val accCsv = new File(Config.outDir, "registration_accuracy.csv")
    val pw = new PrintWriter(accCsv)
    try {
      pw.println("model_id,mean_mm,rms_mm,hd95_mm,hd_mm")
      accuracy.foreach { case (id, s) => pw.println(f"$id,${s.mean}%.4f,${s.rms}%.4f,${s.hd95}%.4f,${s.hd}%.4f") }
    } finally pw.close()

    val n = accuracy.length
    def avg(f: Metrics.SurfaceStats => Double): Double = accuracy.map { case (_, s) => f(s) }.sum / n
    def sd(f: Metrics.SurfaceStats => Double): Double = {
      val m = avg(f)
      math.sqrt(accuracy.map { case (_, s) => math.pow(f(s) - m, 2) }.sum / n)
    }
    println(f"\n  POPULATION (n=$n): mean=${avg(_.mean)}%.3f+-${sd(_.mean)}%.3f mm  " +
      f"rms=${avg(_.rms)}%.3f+-${sd(_.rms)}%.3f mm  hd95=${avg(_.hd95)}%.3f+-${sd(_.hd95)}%.3f mm  " +
      f"hd=${avg(_.hd)}%.3f+-${sd(_.hd)}%.3f mm")
    println(s"  wrote ${accCsv.getPath}")

    // ------------------------------------------------------------------------------------------ watchlist specimens
    // Specimens called out separately (e.g. flagged as outliers by an external variation analysis) -- same numbers
    // as in registration_accuracy.csv above, just pulled out into their own section/file since these are exactly
    // the specimens most likely to expose a registration weakness (an outlier's real anatomy is, by definition,
    // furthest from the reference the GP deformation has to stretch to reach).
    if (Config.watchSpecimenIds.nonEmpty) {
      val accuracyById = accuracy.toMap
      println(s"\n[watchlist] ${Config.watchSpecimenIds.length} specimen(s) called out via SCAPULA_WATCH_SPECIMENS:")
      val watchRows = Config.watchSpecimenIds.map { id =>
        accuracyById.get(id) match {
          case Some(s) =>
            println(f"  $id%-24s ${s.render}")
            id -> Some(s)
          case None =>
            println(f"  $id%-24s !! not found among registered specimens")
            id -> None
        }
      }
      val watchCsv = new File(Config.outDir, "registration_accuracy_watchlist.csv")
      val wpw = new PrintWriter(watchCsv)
      try {
        wpw.println("model_id,mean_mm,rms_mm,hd95_mm,hd_mm")
        watchRows.foreach {
          case (id, Some(s)) => wpw.println(f"$id,${s.mean}%.4f,${s.rms}%.4f,${s.hd95}%.4f,${s.hd}%.4f")
          case (id, None)    => wpw.println(s"$id,,,,")
        }
      } finally wpw.close()
      println(s"  wrote ${watchCsv.getPath}")
    }

    println("\nNext: run Stage3SSMModelValidation.")
  }
}
