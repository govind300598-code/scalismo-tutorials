package scapula.singlekernel

import scapula.{Config, CsvWriter, GpmmFitting, Metrics, RigidAlign, ReferenceSelection}
import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector3D, Landmark, Point, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh3D
import scalismo.utils.Random as ScalismoRandom

import java.io.File

/**
 * STAGE 2 (single-Gaussian-kernel pipeline) -- UNBIASED REFERENCE + LANDMARK-INFORMED NON-RIGID GPMM FITTING,
 * using a GPMM built from exactly ONE Gaussian kernel term instead of the multi-kernel pipeline's 2/3-term sum.
 *
 * This is a straight port of `scapula.Stage2ReferenceRefinement` -- same medoid bootstrap (Step 1), same
 * unbiased-template iteration (Config.refinePasses passes of: build GPMM on the current reference, rigidly
 * +scale pre-align every subject via landmarks then trimmed ICP, non-rigidly fit the GPMM using BOTH the
 * surface term and the named-landmark term, average the now-corresponded fits into the next reference), and
 * the same output files -- with the ONE deliberate difference being which kernel builds the GPMM:
 * `SingleGaussianGpmm.buildGpmm` (single term, sigma/scale given directly in mm) instead of
 * `scapula.GpmmFitting.buildGpmm` (2 or 3 terms, derived as fractions of the reference bounding box). See
 * `SingleKernelConfig` for how sigma/s are set, and `SingleGaussianGpmm` for why they're absolute mm here.
 *
 * No methodological step is skipped or simplified relative to the multi-kernel pipeline: same dataset loading
 * and CSV/landmark resolution (`ScapulaData`), same mirroring convention, same medoid-based reference bootstrap
 * (`ReferenceSelection`), same rigid+scale pre-alignment then trimmed ICP (`RigidAlign`), same landmark-informed
 * coarse-to-fine LBFGS registration schedule (`GpmmFitting.fit`, reused unchanged -- it only needs an
 * already-built low-rank GP, not a specific kernel), same distance-error and landmark-RMSE reporting
 * (`Metrics`, `CsvWriter`). Run `scapula.Stage3PCAModel` afterwards against the same `SCAPULA_OUT_DIR` to build
 * and validate the actual (PCA) statistical shape model, exactly as with the multi-kernel pipeline.
 *
 * Deliberately headless for the same reason as the multi-kernel Stage 2 (see its docstring): accumulating a live
 * UI scene graph across every pass and every subject's fitted mesh is what runs a long batch out of memory.
 */
object Stage2SingleKernelReferenceRefinement {

  /** Root-mean-square distance between corresponding points, in mm. */
  private def rmse(a: IndexedSeq[Point[_3D]], b: IndexedSeq[Point[_3D]]): Double = {
    require(a.length == b.length && a.nonEmpty)
    math.sqrt(a.zip(b).map { case (p, q) => (p - q).norm2 }.sum / a.length)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: ScalismoRandom = ScalismoRandom(Config.seed)

    val dirs = Config.dataDirs
    println(s"Data director${if (dirs.size > 1) "ies" else "y"} : ${dirs.map(_.getAbsolutePath).mkString(", ")}")
    println(s"Kernel          : single Gaussian term, sigma=${SingleKernelConfig.sigmaMm} mm, s=${SingleKernelConfig.scaleMm} mm")
    println(s"Model resolution: ${Config.modelResolution} vertices, refine passes: ${Config.refinePasses}, " +
      s"rigid ICP iterations: ${Config.icpIterations}, landmark weight: ${Config.landmarkWeight}")
    Config.outDir.mkdirs()

    // Written first so every output directory is self-documenting -- essential once you start comparing runs
    // made with different (sigma, s) pairs against each other, or against the multi-kernel pipeline's runs.
    CsvWriter.write(
      new File(Config.outDir, "run_config.txt"),
      Seq("key", "value"),
      Seq(
        Seq("pipeline", "single_gaussian_kernel"),
        Seq("dataDirs", dirs.map(_.getAbsolutePath).mkString(":")),
        Seq("sigma_mm", SingleKernelConfig.sigmaMm),
        Seq("scale_mm", SingleKernelConfig.scaleMm),
        Seq("modelResolution", Config.modelResolution),
        Seq("icpIterations", Config.icpIterations),
        Seq("refinePasses", Config.refinePasses),
        Seq("gpRelativeTolerance", Config.gpRelativeTolerance),
        Seq("gpMaxRank", Config.gpMaxRank),
        Seq("buildIndependentModel", Config.buildIndependentModel),
        Seq("seed", Config.seed),
        Seq("landmarkWeight", Config.landmarkWeight),
        Seq("subjectLimit", Config.subjectLimit),
        Seq("topKMostAverage", Config.topKMostAverage)
      )
    )

    var pool = ReferenceSelection.loadPool(dirs, Config.modelResolution)
    println(s"\n${pool.length} subjects in the pool " +
      s"(${if (Config.buildIndependentModel) "one side per subject" else "both sides"}, right mirrored to left" +
      s"${if (Config.subjectLimit > 0) s", capped to SCAPULA_SUBJECT_LIMIT=${Config.subjectLimit}" else ""})\n")
    require(pool.size >= 2, s"Need at least 2 subjects, found ${pool.size}. Check SCAPULA_DATA_DIR(S) / SCAPULA_SUBJECT_LIMIT.")

    println("[Step 1] Bootstrap reference: medoid of the pool (pairwise similarity+ICP rigid-alignment distance matrix)")
    val (bootstrap, ranking, pairwiseMatrix) = ReferenceSelection.chooseReference(dirs, Config.modelResolution)

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
      println(s"PASS $pass / ${Config.refinePasses}  (single Gaussian kernel: sigma=${SingleKernelConfig.sigmaMm} mm, s=${SingleKernelConfig.scaleMm} mm)")
      println("=" * 100)

      val (lowRankGP, gpmm) = SingleGaussianGpmm.buildGpmm(currentReference)
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
    val (finalGP, finalGpmm) = SingleGaussianGpmm.buildGpmm(currentReference)
    println(f"  Final GPMM rank: ${finalGpmm.rank}, reference vertices: ${currentReference.pointSet.numberOfPoints}")

    // Only clear a PREVIOUS run's final_<subject>_*.stl output now, once every pass has actually finished --
    // not at the start of the run. Deleting it upfront means a run that dies partway through wipes out a
    // previous GOOD run's results for nothing.
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
        "fit_mean_mm", "fit_rms_mm", "fit_hd95_mm", "fit_hd_mm", "fit_landmark_rmse_mm"),
      finalRows.map { case (id, b, lb, a, la) => Seq(id, b.mean, b.rms, b.hd95, b.hd, lb, a.mean, a.rms, a.hd95, a.hd, la) }
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
    // See scapula.Stage2ReferenceRefinement for why this must be named ".json", not ".h5" (scalismo 0.92.1
    // always writes its own JSON format regardless of name, but the reader dispatches purely by extension).
    val modelOut = new File(Config.outDir, "scapula_gpmm.json")
    MeshIO.writeMesh(currentReference, meshOut).get
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(finalGpmm, modelOut).get

    println(s"\nAll results written to ${Config.outDir.getAbsolutePath}:")
    println("  reference_mean_shape.stl     - the final unbiased template")
    println("  scapula_gpmm.json            - the final single-Gaussian-kernel GPMM")
    println("  final_<subject>_target.stl   - each subject rigidly (+scale) aligned to the final reference")
    println("  final_<subject>_fit.stl      - each subject's landmark-informed non-rigid GPMM fit")
    println("  pass<N>_reference.stl        - the reference used at the start of pass N")
    println("  pairwise_distance_matrix.csv - full N x N rigid-alignment distance error matrix")
    println("  medoid_ranking.csv           - mean distance to the rest of the pool, most to least average")
    println("  fit_quality_by_pass.csv      - rigid-only vs. fitted surface + landmark error per subject, per pass")
    println("  final_fit_quality.csv        - rigid-only vs. fitted surface + landmark error, final model")
    println("  run_config.txt               - every parameter this run used, including sigma_mm / scale_mm")
    println("\nNext: run `sbt \"runMain scapula.singlekernel.Stage3PCAModel\"` (same SCAPULA_OUT_DIR) to build")
    println("and validate the actual PCA statistical shape model from these registered subjects.")
  }
}
