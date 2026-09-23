package scapula

import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector3D, Landmark, Point, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh3D
import scalismo.utils.Random as ScalismoRandom

import java.io.File

/**
 * STAGE 2 -- UNBIASED REFERENCE + LANDMARK-INFORMED NON-RIGID GPMM FITTING ON THE REAL DATASET.
 *
 * "Most average shape" cannot be answered by picking a specimen and calling it done: any single real scapula is
 * still one person's anatomy. This runs the standard unbiased-template procedure instead:
 *
 *   1. Bootstrap: pick the MEDOID (see ReferenceSelection) as a good, non-outlier starting template -- better
 *      than an arbitrary specimen, but still just one real subject.
 *   2. Build a GPMM on the current template, rigidly (+ scale) pre-align every subject to it via landmarks then
 *      trimmed ICP, then non-rigidly fit the GPMM to each (Config.refinePasses times) -- both steps use the
 *      named landmarks, not just anonymous closest-point correspondences (see below). Because every fit shares
 *      the template's own point indices, averaging the fitted meshes point-by-point is now a valid
 *      correspondence-based mean shape -- unlike averaging raw, uncorresponded vertices, which is meaningless.
 *   3. That mean becomes the next pass's template. Repeating removes the residual bias of having anchored the
 *      first pass on one real specimen (this is a Generalized-Procrustes-style iteration, done in shape space
 *      instead of point space because the specimens only have sparse landmark correspondence to start with).
 *
 * Where the landmarks are used:
 *   - RIGID pre-alignment: `RigidAlign.landmarkThenIcp(..., useScaling = true)` -- landmark Procrustes WITH an
 *     isotropic scale term (a similarity transform, not just rigid), because different subjects' scapulae
 *     genuinely differ in size and classical Generalized Procrustes Analysis removes that before shape
 *     comparison. Trimmed ICP then refines the pose (rigid-only: unconstrained scale plus closest-point search
 *     is not numerically stable).
 *   - NON-RIGID GPMM fitting: scalismo's registration metrics are all image-to-image (MeanSquaresMetric etc.),
 *     so a plain GPMM fit has no landmark term -- only the anonymous closest-surface-point data term. This adds
 *     one: `LandmarkMetric` (see LandmarkMetric.scala) penalizes the distance between each named landmark's
 *     position on the deforming reference and its corresponding position on the target, added to the surface
 *     term with weight Config.landmarkWeight.
 *
 * This is deliberately headless (no ScalismoUI window): keeping a live UI scene graph across every pass and
 * every one of the (subjects x passes) fitted meshes is what accumulates memory over a long batch run -- exactly
 * what killed the first attempt at this (OOM, "Killed" with no Java stack trace = the OS killed the process, not
 * a graceful OutOfMemoryError). Everything needed to inspect the result afterwards -- the error-metric CSVs and
 * the meshes themselves -- is written to Config.outDir; use ViewResults (a separate, lightweight program) to open
 * them in the Scalismo viewer without re-running the registration.
 */
object Stage2ReferenceRefinement {

  /** Root-mean-square distance between corresponding points, in mm. */
  private def rmse(a: IndexedSeq[Point[_3D]], b: IndexedSeq[Point[_3D]]): Double = {
    require(a.length == b.length && a.nonEmpty)
    math.sqrt(a.zip(b).map { case (p, q) => (p - q).norm2 }.sum / a.length)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: ScalismoRandom = ScalismoRandom(Config.seed)

    val dir = Config.dataDir
    println(s"Data directory  : ${dir.getAbsolutePath}")
    println(s"Model resolution: ${Config.modelResolution} vertices, refine passes: ${Config.refinePasses}, " +
      s"rigid ICP iterations: ${Config.icpIterations}, landmark weight: ${Config.landmarkWeight}")
    Config.outDir.mkdirs()

    // Written first so every output directory is self-documenting -- essential once you start comparing runs
    // made with different env vars (e.g. SCAPULA_KERNEL_TERMS=2 vs 3) against each other later.
    CsvWriter.write(
      new File(Config.outDir, "run_config.txt"),
      Seq("key", "value"),
      Seq(
        Seq("dataDir", Config.dataDir.getAbsolutePath),
        Seq("modelResolution", Config.modelResolution),
        Seq("icpIterations", Config.icpIterations),
        Seq("refinePasses", Config.refinePasses),
        Seq("gpRelativeTolerance", Config.gpRelativeTolerance),
        Seq("gpMaxRank", Config.gpMaxRank),
        Seq("kernelTerms", Config.kernelTerms),
        Seq("buildIndependentModel", Config.buildIndependentModel),
        Seq("seed", Config.seed),
        Seq("landmarkWeight", Config.landmarkWeight),
        Seq("subjectLimit", Config.subjectLimit),
        Seq("topKMostAverage", Config.topKMostAverage)
      )
    )

    var pool = ReferenceSelection.loadPool(dir, Config.modelResolution)
    println(s"\n${pool.length} subjects in the pool " +
      s"(${if (Config.buildIndependentModel) "one side per subject" else "both sides"}, right mirrored to left" +
      s"${if (Config.subjectLimit > 0) s", capped to SCAPULA_SUBJECT_LIMIT=${Config.subjectLimit}" else ""})\n")
    require(pool.size >= 2, s"Need at least 2 subjects, found ${pool.size}. Check SCAPULA_DATA_DIR / SCAPULA_SUBJECT_LIMIT.")

    println("[Step 1] Bootstrap reference: medoid of the pool (pairwise similarity+ICP rigid-alignment distance matrix)")
    val (bootstrap, ranking, pairwiseMatrix) = ReferenceSelection.chooseReference(dir, Config.modelResolution)

    CsvWriter.write(
      new File(Config.outDir, "pairwise_distance_matrix.csv"),
      Seq("subject_a", "subject_b", "mean_mm", "rms_mm", "hd95_mm", "hd_mm"),
      pairwiseMatrix.map(p => Seq(p.a, p.b, p.stats.mean, p.stats.rms, p.stats.hd95, p.stats.hd))
    )
    CsvWriter.write(
      new File(Config.outDir, "medoid_ranking.csv"),
      Seq("rank", "subject", "mean_distance_to_others_mm", "worst_pair_hd_mm"),
      ranking.zipWithIndex.map { case (r, i) => Seq(i + 1, r.specimen.modelId, r.meanDistanceToOthers, r.worstPairHd) }
    )
    println("\nFull ranking (most to least 'average'):")
    ranking.zipWithIndex.foreach { case (r, i) =>
      println(f"    ${i + 1}%2d. ${r.specimen.modelId}%-24s mean-to-others=${r.meanDistanceToOthers}%6.2f mm")
    }

    // Restrict FITTING to the K most representative subjects, if asked -- the ranking above was still computed
    // over the whole pool (dropping subjects first would bias which one looks "average"). A smaller, more
    // homogeneous population fits tighter mechanically, not because the model got better -- report it as such.
    if (Config.topKMostAverage > 0) {
      val topIds = ranking.take(Config.topKMostAverage).map(_.specimen.modelId).toSet
      pool = pool.filter(p => topIds.contains(p.specimen.modelId))
      require(pool.size >= 2, s"SCAPULA_TOP_K=${Config.topKMostAverage} left only ${pool.size} subjects, need at least 2.")
      println(s"\nSCAPULA_TOP_K=${Config.topKMostAverage}: fitting only the most representative subjects: " +
        s"${pool.map(_.specimen.modelId).mkString(", ")}")
    }

    var currentReference: TriangleMesh3D = bootstrap.mesh
    // Landmarks aren't re-estimated on the running mean mesh (a mean mesh has no natural named-point
    // correspondence), so we keep using the bootstrap's landmarks to get a good STARTING pose for the rigid ICP
    // pre-alignment in every pass; ICP then refines the pose regardless of how good that start was.
    val poseLandmarks = bootstrap.landmarks

    // The reference vertex closest to each named landmark, found ONCE on the bootstrap mesh. Every later pass's
    // reference is a point-by-point mean built from THIS mesh's topology, so the same point ids stay the right
    // landmark locations throughout -- only their coordinates move as the reference is refined.
    val referenceLandmarkIds: Map[String, PointId] =
      bootstrap.landmarks.map(lm => lm.id -> bootstrap.mesh.pointSet.findClosestPoint(lm.point).id).toMap
    println(s"\nReference landmark ids (nearest vertex to each named landmark on the bootstrap mesh): " +
      referenceLandmarkIds.map { case (name, id) => s"$name=${id.id}" }.mkString(", "))

    /** (reference point id, this subject's corresponding target point) for every landmark name known on both sides. */
    def matchLandmarks(alignedTargetLandmarks: IndexedSeq[Landmark[_3D]]): IndexedSeq[(PointId, Point[_3D])] =
      alignedTargetLandmarks.flatMap(lm => referenceLandmarkIds.get(lm.id).map(id => (id, lm.point)))

    val passRows = scala.collection.mutable.ArrayBuffer.empty[Seq[Any]]

    for (pass <- 1 to Config.refinePasses) {
      println(s"\n${"=" * 100}")
      println(s"PASS $pass / ${Config.refinePasses}")
      println("=" * 100)

      val (lowRankGP, gpmm) = GpmmFitting.buildGpmm(currentReference)
      println(f"  GPMM rank: ${gpmm.rank}, reference vertices: ${currentReference.pointSet.numberOfPoints}")
      MeshIO.writeMesh(currentReference, new File(Config.outDir, s"pass${pass}_reference.stl")).get

      val fitted = pool.map { subject =>
        val (rigidlyAligned, alignedLandmarks) = RigidAlign.landmarkThenIcp(subject.mesh, subject.landmarks,
          currentReference, poseLandmarks, Config.icpIterations, useScaling = true)
        val matches = matchLandmarks(alignedLandmarks)
        val targetLmPoints = matches.map(_._2)

        val beforeStats = Metrics.symmetric(rigidlyAligned, currentReference)
        val lmBefore = rmse(matches.map { case (id, _) => currentReference.pointSet.point(id) }, targetLmPoints)

        val coefficients = GpmmFitting.fit(lowRankGP, currentReference, rigidlyAligned,
          landmarkCorrespondences = matches.map { case (id, target) => (currentReference.pointSet.point(id), target) })
        val fittedMesh = gpmm.instance(coefficients)
        val afterStats = Metrics.symmetric(fittedMesh, rigidlyAligned)
        val lmAfter = rmse(matches.map { case (id, _) => fittedMesh.pointSet.point(id) }, targetLmPoints)

        println(f"    ${subject.specimen.modelId}%-24s rigid-only ${beforeStats.render} lm-RMSE=$lmBefore%5.2f   |   " +
          f"fitted ${afterStats.render} lm-RMSE=$lmAfter%5.2f")
        passRows += Seq(pass, subject.specimen.modelId,
          beforeStats.mean, beforeStats.rms, beforeStats.hd95, beforeStats.hd, lmBefore,
          afterStats.mean, afterStats.rms, afterStats.hd95, afterStats.hd, lmAfter)
        fittedMesh
      }

      // Correspondence-based mean shape: every fitted mesh shares the reference's own point indices, so
      // averaging point i across all subjects is a valid mean -- this is NOT true of the raw, uncorresponded
      // input meshes.
      val fittedPoints = fitted.map(_.pointSet.points.toIndexedSeq)
      val n = fittedPoints.length.toDouble
      val meanPoints = (0 until currentReference.pointSet.numberOfPoints).map { i =>
        val sum = fittedPoints.foldLeft(EuclideanVector3D(0, 0, 0)) { (acc, pts) => acc + pts(i).toVector }
        (sum * (1.0 / n)).toPoint
      }
      currentReference = TriangleMesh3D(meanPoints, currentReference.triangulation)
    }

    CsvWriter.write(
      new File(Config.outDir, "fit_quality_by_pass.csv"),
      Seq("pass", "subject", "rigid_mean_mm", "rigid_rms_mm", "rigid_hd95_mm", "rigid_hd_mm", "rigid_landmark_rmse_mm",
        "fit_mean_mm", "fit_rms_mm", "fit_hd95_mm", "fit_hd_mm", "fit_landmark_rmse_mm"),
      passRows.toSeq
    )

    println(s"\n${"=" * 100}")
    println("FINAL MODEL")
    println("=" * 100)
    val (finalGP, finalGpmm) = GpmmFitting.buildGpmm(currentReference)
    println(f"  Final GPMM rank: ${finalGpmm.rank}, reference vertices: ${currentReference.pointSet.numberOfPoints}")

    // Only clear a PREVIOUS run's final_<subject>_*.stl output now, once every pass has actually finished --
    // not at the start of the run. Deleting it upfront (this used to) means a run that dies partway through
    // (e.g. OOM during the expensive per-subject fitting loop) wipes out a previous GOOD run's results for
    // nothing, leaving ViewResults with no final_*.stl files at all even though nothing new was ever written.
    // pass<N>_reference.stl files aren't touched here: this run already (re)wrote its own during the loop above.
    Option(Config.outDir.listFiles((_, name) => name.matches("final_.*\\.stl")))
      .foreach(_.foreach(_.delete()))

    val finalRows = pool.map { subject =>
      val (rigidlyAligned, alignedLandmarks) = RigidAlign.landmarkThenIcp(subject.mesh, subject.landmarks,
        currentReference, poseLandmarks, Config.icpIterations, useScaling = true)
      val matches = matchLandmarks(alignedLandmarks)
      val targetLmPoints = matches.map(_._2)

      val beforeStats = Metrics.symmetric(rigidlyAligned, currentReference)
      val lmBefore = rmse(matches.map { case (id, _) => currentReference.pointSet.point(id) }, targetLmPoints)

      val coefficients = GpmmFitting.fit(finalGP, currentReference, rigidlyAligned,
        landmarkCorrespondences = matches.map { case (id, target) => (currentReference.pointSet.point(id), target) })
      val fittedMesh = finalGpmm.instance(coefficients)
      val afterStats = Metrics.symmetric(fittedMesh, rigidlyAligned)
      val lmAfter = rmse(matches.map { case (id, _) => fittedMesh.pointSet.point(id) }, targetLmPoints)

      println(f"    ${subject.specimen.modelId}%-24s rigid-only ${beforeStats.render} lm-RMSE=$lmBefore%5.2f   |   " +
        f"fitted ${afterStats.render} lm-RMSE=$lmAfter%5.2f")

      MeshIO.writeMesh(rigidlyAligned, new File(Config.outDir, s"final_${subject.specimen.modelId}_target.stl")).get
      MeshIO.writeMesh(fittedMesh, new File(Config.outDir, s"final_${subject.specimen.modelId}_fit.stl")).get

      (subject.specimen.modelId, beforeStats, lmBefore, afterStats, lmAfter)
    }

    CsvWriter.write(
      new File(Config.outDir, "final_fit_quality.csv"),
      Seq("subject", "rigid_mean_mm", "rigid_rms_mm", "rigid_hd95_mm", "rigid_hd_mm", "rigid_landmark_rmse_mm",
        "fit_mean_mm", "fit_rms_mm", "fit_hd95_mm", "fit_hd_mm", "fit_landmark_rmse_mm", "fit_chamfer_distance_mm"),
      // fit_chamfer_distance_mm is the SAME quantity as fit_mean_mm (Metrics.symmetric's mean field) -- an explicit
      // alias column, since "Chamfer distance" (CV/point-cloud terminology) and "mean surface distance" / ASSD
      // (medical-imaging terminology) name the identical computation. Not a second, independently-computed metric.
      finalRows.map { case (id, b, lb, a, la) => Seq(id, b.mean, b.rms, b.hd95, b.hd, lb, a.mean, a.rms, a.hd95, a.hd, la, a.mean) }
    )

    val beforeMean = finalRows.map(_._2.mean).sum / finalRows.length
    val afterMean = finalRows.map(_._4.mean).sum / finalRows.length
    val beforeHd95 = finalRows.map(_._2.hd95).sum / finalRows.length
    val afterHd95 = finalRows.map(_._4.hd95).sum / finalRows.length
    val beforeLm = finalRows.map(_._3).sum / finalRows.length
    val afterLm = finalRows.map(_._5).sum / finalRows.length
    println(f"\n  Average across all ${finalRows.length} subjects:")
    println(f"    rigid-only : mean=$beforeMean%5.2f mm  HD95=$beforeHd95%6.2f mm  landmark-RMSE=$beforeLm%5.2f mm")
    println(f"    fitted GPMM: mean=$afterMean%5.2f mm  HD95=$afterHd95%6.2f mm  landmark-RMSE=$afterLm%5.2f mm")
    println(f"    Non-rigid fitting reduced mean surface distance by ${(1 - afterMean / beforeMean) * 100}%.1f%%" +
      f" and landmark RMSE by ${(1 - afterLm / beforeLm) * 100}%.1f%%")

    val meshOut = new File(Config.outDir, "reference_mean_shape.stl")
    // NOTE: despite the "HDF5" name, scalismo 0.92.1's writer (HDF5Writer.write -> HDF5Json.writeToFile) always
    // serializes to its own JSON-based format, never real binary HDF5 -- and the reader picks its parser purely
    // by file EXTENSION (".h5" -> a real binary-HDF5 parser that then fails on this JSON content, ".json" -> the
    // matching JSON parser). So the file this writes MUST be named ".json", not ".h5", or reading it back fails
    // with "No valid HDF5 signature found" even though the write itself succeeded.
    val modelOut = new File(Config.outDir, "scapula_gpmm.json")
    MeshIO.writeMesh(currentReference, meshOut).get
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(finalGpmm, modelOut).get

    println(s"\nAll results written to ${Config.outDir.getAbsolutePath}:")
    println("  reference_mean_shape.stl     - the final unbiased template")
    println("  scapula_gpmm.json            - the final GPMM (scalismo's own JSON-based model format)")
    println("  final_<subject>_target.stl   - each subject rigidly (+scale) aligned to the final reference")
    println("  final_<subject>_fit.stl      - each subject's landmark-informed non-rigid GPMM fit")
    println("  pass<N>_reference.stl        - the reference used at the start of pass N")
    println("  pairwise_distance_matrix.csv - full N x N rigid-alignment distance error matrix")
    println("  medoid_ranking.csv           - mean distance to the rest of the pool, most to least average")
    println("  fit_quality_by_pass.csv      - rigid-only vs. fitted surface + landmark error per subject, per pass")
    println("  final_fit_quality.csv        - rigid-only vs. fitted surface + landmark error, final model")
    println("\nTo view everything in the Scalismo viewer, run:")
    println("  sbt \"runMain scapula.ViewResults\"")
  }
}
