package scapula

import scalismo.geometry.{EuclideanVector3D, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh3D
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random as ScalismoRandom

/**
 * STAGE 2 -- UNBIASED REFERENCE + NON-RIGID GPMM FITTING ON THE REAL DATASET.
 *
 * "Most average shape" cannot be answered by picking a specimen and calling it done: any single real scapula is
 * still one person's anatomy. This runs the standard unbiased-template procedure instead:
 *
 *   1. Bootstrap: pick the MEDOID (see ReferenceSelection) as a good, non-outlier starting template -- better
 *      than an arbitrary specimen, but still just one real subject.
 *   2. Build a GPMM on the current template, rigidly pre-align every subject to it, then non-rigidly fit the
 *      GPMM to each (Config.refinePasses times). Because every fit shares the template's own point indices,
 *      averaging the fitted meshes point-by-point is now a valid correspondence-based mean shape -- unlike
 *      averaging raw, uncorresponded vertices, which is meaningless.
 *   3. That mean becomes the next pass's template. Repeating removes the residual bias of having anchored the
 *      first pass on one real specimen (this is a Generalized-Procrustes-style iteration, done in shape space
 *      instead of point space because the specimens only have sparse landmark correspondence to start with).
 *
 * The final template is the population's mean shape in dense correspondence: not any one person's scapula, but
 * the "most average" one in the precise, defensible sense of being the arithmetic mean under this pipeline's own
 * established correspondence.
 */
object Stage2ReferenceRefinement {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: ScalismoRandom = ScalismoRandom(Config.seed)

    val dir = Config.dataDir
    println(s"Data directory  : ${dir.getAbsolutePath}")
    println(s"Model resolution: ${Config.modelResolution} vertices, refine passes: ${Config.refinePasses}, " +
      s"rigid ICP iterations: ${Config.icpIterations}")

    val pool = ReferenceSelection.loadPool(dir, Config.modelResolution)
    println(s"\n${pool.length} subjects in the pool " +
      s"(${if (Config.buildIndependentModel) "one side per subject" else "both sides"}, right mirrored to left)\n")
    require(pool.size >= 3, s"Need at least 3 subjects, found ${pool.size}. Check SCAPULA_DATA_DIR.")

    println("[Step 1] Bootstrap reference: medoid of the pool (pairwise rigid-alignment distance matrix)")
    val (bootstrap, ranking) = ReferenceSelection.medoid(pool)
    println("\nFull ranking (most to least 'average'):")
    ranking.zipWithIndex.foreach { case (r, i) =>
      println(f"    ${i + 1}%2d. ${r.specimen.modelId}%-24s mean-to-others=${r.meanDistanceToOthers}%6.2f mm")
    }

    val ui = ScalismoUI()

    var currentReference: TriangleMesh3D = bootstrap.mesh
    // Landmarks aren't re-estimated on the running mean mesh (a mean mesh has no natural named-point
    // correspondence), so we keep using the bootstrap's landmarks to get a good STARTING pose for the rigid ICP
    // pre-alignment in every pass; ICP then refines the pose regardless of how good that start was.
    val poseLandmarks = bootstrap.landmarks

    for (pass <- 1 to Config.refinePasses) {
      println(s"\n${"=" * 100}")
      println(s"PASS $pass / ${Config.refinePasses}")
      println("=" * 100)

      val (lowRankGP, gpmm) = GpmmFitting.buildGpmm(currentReference)
      println(f"  GPMM rank: ${gpmm.rank}, reference vertices: ${currentReference.pointSet.numberOfPoints}")

      val passGroup = ui.createGroup(s"pass-$pass")
      val modelView = ui.show(passGroup, gpmm, "reference")

      val fitted = pool.map { subject =>
        val (rigidlyAligned, _) =
          RigidAlign.landmarkThenIcp(subject.mesh, subject.landmarks, currentReference, poseLandmarks, Config.icpIterations)
        val coefficients = GpmmFitting.fit(
          lowRankGP,
          currentReference,
          rigidlyAligned,
          onIteration = pars => modelView.shapeModelTransformationView.shapeTransformationView.coefficients = pars
        )
        val fittedMesh = gpmm.instance(coefficients)
        val stats = Metrics.symmetric(fittedMesh, rigidlyAligned)
        println(f"    ${subject.specimen.modelId}%-24s ${stats.render}")
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
      ui.show(passGroup, currentReference, "mean-of-fits (next pass's reference)")
    }

    println(s"\n${"=" * 100}")
    println("FINAL MODEL")
    println("=" * 100)
    val (finalGP, finalGpmm) = GpmmFitting.buildGpmm(currentReference)
    println(f"  Final GPMM rank: ${finalGpmm.rank}, reference vertices: ${currentReference.pointSet.numberOfPoints}")

    val finalGroup = ui.createGroup("final")
    ui.show(finalGroup, finalGpmm, "final reference (unbiased mean shape)")

    val finalStats = pool.map { subject =>
      val (rigidlyAligned, _) =
        RigidAlign.landmarkThenIcp(subject.mesh, subject.landmarks, currentReference, poseLandmarks, Config.icpIterations)
      val coefficients = GpmmFitting.fit(finalGP, currentReference, rigidlyAligned)
      val fittedMesh = finalGpmm.instance(coefficients)
      val stats = Metrics.symmetric(fittedMesh, rigidlyAligned)
      println(f"    ${subject.specimen.modelId}%-24s ${stats.render}")
      ui.show(finalGroup, rigidlyAligned, s"${subject.specimen.modelId}_target")
      ui.show(finalGroup, fittedMesh, s"${subject.specimen.modelId}_fit")
      stats
    }

    val meanMean = finalStats.map(_.mean).sum / finalStats.length
    val meanHd95 = finalStats.map(_.hd95).sum / finalStats.length
    val meanHd = finalStats.map(_.hd).sum / finalStats.length
    println(f"\n  Average across all ${finalStats.length} subjects: mean=$meanMean%5.2f mm  " +
      f"HD95=$meanHd95%6.2f mm  HD=$meanHd%6.2f mm")

    if (Config.outDir.exists() || Config.outDir.mkdirs()) {
      val meshOut = new java.io.File(Config.outDir, "reference_mean_shape.stl")
      val modelOut = new java.io.File(Config.outDir, "scapula_gpmm.h5")
      MeshIO.writeMesh(currentReference, meshOut).get
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(finalGpmm, modelOut).get
      println(s"\n  Wrote reference mesh  -> ${meshOut.getAbsolutePath}")
      println(s"  Wrote GPMM            -> ${modelOut.getAbsolutePath}")
    } else {
      println(s"\n  !! Could not create output directory ${Config.outDir.getAbsolutePath}, skipped writing results.")
    }
  }
}
