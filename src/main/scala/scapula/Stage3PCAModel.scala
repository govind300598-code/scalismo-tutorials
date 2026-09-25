package scapula

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector3D, Point3D, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random as ScalismoRandom

import java.io.File

/**
 * STAGE 3 -- GPA ALIGNMENT + STATISTICAL SHAPE MODEL + STANDARD VALIDATION.
 *
 * Everything through Stage 2 used an ANALYTIC kernel purely as a REGISTRATION tool to get every subject into
 * correspondence with the reference. This stage:
 *
 *   1. GENERALISED PROCRUSTES ANALYSIS (GPA): iteratively aligns all registered shapes (in correspondence from
 *      Stage 2) to remove residual rigid-body variation before PCA. Each iteration aligns every shape to the
 *      current mean via rigid (rotation + translation) Procrustes, then updates the mean, until the mean
 *      shape shifts by < 1e-5 mm between iterations (typically 2-3 iterations suffice after Stage 2's own
 *      alignment). This removes the last residue of reference-frame choice from the shape space before PCA.
 *
 *   2. PCA: builds the real, data-driven statistical shape model from the GPA-aligned correspondences (Cootes'
 *      Point Distribution Model), using scalismo's DataCollection/PointDistributionModel pipeline.
 *
 *   3. VALIDATION: computes the three standard metrics (Styner et al. 2003):
 *      - COMPACTNESS  : cumulative variance explained vs. number of modes kept.
 *      - SPECIFICITY  : distance from a random model sample to the nearest real subject.
 *      - GENERALIZATION: leave-one-out reconstruction error vs. number of modes used.
 *
 * All three only need the correspondences Stage 2 already computed -- leave-one-out here means rebuilding a
 * small PCA (milliseconds) and projecting, NOT redoing registration.
 *
 * Writes gpa_mean_shape.stl, gpa_<subject>_fit.stl (per subject), scapula_pca_model.json, and 3 validation CSVs.
 * Run scripts/plot_ssm_validation.py afterwards to turn the CSVs into standard SSM validation figures.
 */

/** Generalised Procrustes Analysis on a set of meshes already in point correspondence. */
object GPA {

  def computeMean(shapes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    val n = shapes.length.toDouble
    val triangulation = shapes.head.triangulation
    val meanPts = (0 until shapes.head.pointSet.numberOfPoints).map { i =>
      val sum = shapes.foldLeft(EuclideanVector3D(0, 0, 0)) { (acc, m) =>
        acc + m.pointSet.point(PointId(i)).toVector
      }
      (sum / n).toPoint
    }
    TriangleMesh3D(meanPts, triangulation)
  }

  /**
   * Iteratively align every shape to the current mean via rigid Procrustes (rotation + translation only, no
   * scale -- after Stage 2's similarity pre-alignment, sizes are already normalised; dropping scale here means
   * the PCA captures genuine shape variation, not a mix of shape and size).
   *
   * Convergence criterion: max shift of any mean-shape point between iterations < `tol` mm.
   */
  def align(shapes: IndexedSeq[TriangleMesh[_3D]],
            maxIter: Int = 50,
            tol: Double = 1e-5): IndexedSeq[TriangleMesh[_3D]] = {
    require(shapes.length >= 2, "GPA needs at least 2 shapes")
    require(shapes.forall(_.pointSet.numberOfPoints == shapes.head.pointSet.numberOfPoints),
      "All shapes must have the same point count (in correspondence) for GPA")

    var current = shapes
    for (iter <- 1 to maxIter) {
      val mean = computeMean(current)
      val meanPts = mean.pointSet.points.toIndexedSeq

      val aligned = current.map { shape =>
        val srcPts = shape.pointSet.points.toIndexedSeq
        val correspondences = srcPts.zip(meanPts)
        val trans = LandmarkRegistration.rigid3DLandmarkRegistration(correspondences, center = Point3D(0, 0, 0))
        shape.transform(trans)
      }

      val newMean = computeMean(aligned)
      val delta = mean.pointSet.points.toIndexedSeq
        .zip(newMean.pointSet.points.toIndexedSeq)
        .map { case (a, b) => (a - b).norm }
        .max
      current = aligned
      println(f"  GPA iter $iter%2d: max mean-point shift = $delta%.6f mm")
      if (delta < tol) {
        println(s"  GPA converged in $iter iterations.")
        return current
      }
    }
    println(s"  GPA: reached $maxIter iterations (max allowed) without full convergence.")
    current
  }
}

object Stage3PCAModel {

  private val specificitySamplesPerK = Config.specificitySamples

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: ScalismoRandom = ScalismoRandom(Config.seed)

    val dir = Config.outDir
    val referenceFile = new File(dir, "reference_mean_shape.stl")
    require(referenceFile.exists(), s"$referenceFile not found -- run Stage2ReferenceRefinement first.")
    val reference = MeshIO.readMesh(referenceFile).get

    val fitFiles = Option(dir.listFiles((_, name) => name.startsWith("final_") && name.endsWith("_fit.stl")))
      .getOrElse(Array.empty[File])
      .sortBy(_.getName)
    require(fitFiles.length >= 4,
      s"Need at least 4 registered subjects (3 for PCA + 1 spare for a meaningful leave-one-out fold), found " +
        s"${fitFiles.length} in ${dir.getAbsolutePath}. (A SCAPULA_SUBJECT_LIMIT smoke test won't have enough.)")

    val subjectIds = fitFiles.map(_.getName.stripPrefix("final_").stripSuffix("_fit.stl"))
    val fittedMeshes = fitFiles.map(f => MeshIO.readMesh(f).get)
    println(s"Loaded ${fittedMeshes.length} registered subjects for GPA + PCA: ${subjectIds.mkString(", ")}")

    // ---------------------------------------------------------------- GPA
    println(s"\n[GPA] Generalised Procrustes Analysis -- removing residual pose variation from ${fittedMeshes.length} shapes...")
    println( "  (Shape pairs are already in correspondence from Stage 2; GPA refines the rigid frame.)")
    val gpaAligned = GPA.align(fittedMeshes.toIndexedSeq)

    val gpaMean = GPA.computeMean(gpaAligned)
    MeshIO.writeMesh(gpaMean, new File(dir, "gpa_mean_shape.stl")).get
    println(s"  Wrote GPA mean shape -> gpa_mean_shape.stl")

    gpaAligned.zip(subjectIds).foreach { case (m, id) =>
      MeshIO.writeMesh(m, new File(dir, s"gpa_${id}_fit.stl")).get
    }
    println(s"  Wrote ${gpaAligned.length} GPA-aligned fits -> gpa_<subject>_fit.stl")

    // Compare GPA mean to non-GPA reference for a sense of the shift
    val gpaMeanStats = Metrics.symmetric(gpaMean, reference)
    println(f"  GPA mean vs. Stage-2 reference: mean=${gpaMeanStats.mean}%.3f mm  HD95=${gpaMeanStats.hd95}%.3f mm")

    // ---------------------------------------------------------------- PCA on GPA-aligned shapes
    println(s"\n[PCA] Building statistical shape model from GPA-aligned correspondences...")
    val dc = DataCollection.fromTriangleMesh3DSequence(reference, gpaAligned)
    val pcaModel: PointDistributionModel[_3D, TriangleMesh] = PointDistributionModel.createUsingPCA(dc)
    println(f"  PCA model rank: ${pcaModel.rank} (at most N-1=${gpaAligned.length - 1} for N=${gpaAligned.length} subjects " +
      "-- PCA over N samples has only N-1 degrees of freedom around their mean)")

    // Load original (pre-GPA) rigidly-aligned target scans for specificity: comparing random model samples to
    // real anatomy rather than to the GPA-post-processed version more fairly tests whether model samples look
    // like actual scapulae.
    val targetFiles = subjectIds.map(id => new File(dir, s"final_${id}_target.stl"))
    val realTargets = targetFiles.filter(_.exists()).map(f => MeshIO.readMesh(f).get)

    // ---------------------------------------------------------------- 1. compactness
    val variance = pcaModel.gp.variance.toArray
    val totalVariance = variance.sum
    println("\n[1/3 Compactness] cumulative variance explained by the leading modes:")
    var cumulative = 0.0
    val compactnessRows = variance.zipWithIndex.map { case (v, i) =>
      cumulative += v
      val frac = if (totalVariance > 0) cumulative / totalVariance else 0.0
      println(f"    mode ${i + 1}%2d: eigenvalue=$v%8.2f   cumulative variance=${frac * 100}%5.1f%%")
      Seq(i + 1, v, frac)
    }
    CsvWriter.write(new File(dir, "pca_compactness.csv"),
      Seq("num_components", "eigenvalue", "cumulative_variance_fraction"), compactnessRows.toSeq)

    // ---------------------------------------------------------------- 2. specificity vs. number of components
    println(s"\n[2/3 Specificity] mean distance from a random sample (using only the first k modes) to the " +
      s"nearest real subject, $specificitySamplesPerK samples per k:")
    val specificityRows = (1 to pcaModel.rank).map { k =>
      val distances = (1 to specificitySamplesPerK).map { _ =>
        val coeffs = DenseVector.zeros[Double](pcaModel.rank)
        (0 until k).foreach(i => coeffs(i) = rng.scalaRandom.nextGaussian())
        val sample = pcaModel.instance(coeffs)
        if (realTargets.nonEmpty) realTargets.map(t => Metrics.symmetric(sample, t).mean).min
        else gpaAligned.map(t => Metrics.symmetric(sample, t).mean).min
      }
      val mean = distances.sum / distances.length
      println(f"    k=${k}%3d modes: mean nearest-real-subject distance = $mean%.2f mm")
      Seq(k, mean)
    }
    CsvWriter.write(new File(dir, "pca_specificity.csv"),
      Seq("num_components", "mean_distance_to_nearest_real_subject_mm"), specificityRows.toSeq)

    // ---------------------------------------------------------------- 3. generalization (leave-one-out)
    println(s"\n[3/3 Generalization] leave-one-out reconstruction error vs. number of components used " +
      s"(${gpaAligned.length} folds, on GPA-aligned shapes):")
    val perSubjectCurves = gpaAligned.indices.map { i =>
      val others = gpaAligned.indices.filter(_ != i).map(gpaAligned).toIndexedSeq
      val dcLoo = DataCollection.fromTriangleMesh3DSequence(reference, others)
      val modelLoo = PointDistributionModel.createUsingPCA(dcLoo)
      val fullCoeffs = modelLoo.coefficients(gpaAligned(i))
      val errorsPerK = (1 to modelLoo.rank).map { k =>
        val truncated = fullCoeffs.copy
        (k until truncated.length).foreach(j => truncated(j) = 0.0)
        val reconstruction = modelLoo.instance(truncated)
        Metrics.symmetric(reconstruction, gpaAligned(i)).mean
      }
      println(f"    left out ${subjectIds(i)}%-24s full-model (k=${errorsPerK.length}) reconstruction error = " +
        f"${errorsPerK.last}%.2f mm")
      errorsPerK
    }
    val maxKAvailable = perSubjectCurves.map(_.length).min
    val generalizationRows = (1 to maxKAvailable).map { k =>
      val errs = perSubjectCurves.map(_(k - 1))
      val mean = errs.sum / errs.length
      println(f"    k=${k}%3d modes: mean leave-one-out reconstruction error = $mean%.2f mm (across all " +
        s"${gpaAligned.length} folds)")
      Seq(k, mean)
    }
    CsvWriter.write(new File(dir, "pca_generalization.csv"),
      Seq("num_components", "mean_reconstruction_error_mm"), generalizationRows.toSeq)

    val modelOut = new File(dir, "scapula_pca_model.json")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(pcaModel, modelOut).get
    println(s"\nWrote the real, data-driven PCA model -> ${modelOut.getAbsolutePath}")
    println("Outputs in " + dir.getAbsolutePath + ":")
    println("  gpa_mean_shape.stl             - GPA mean (after pose normalisation)")
    println("  gpa_<subject>_fit.stl          - each subject's GPA-aligned registered shape")
    println("  scapula_pca_model.json         - data-driven SSM built on GPA-aligned shapes")
    println("  pca_compactness.csv            - cumulative variance per mode")
    println("  pca_specificity.csv            - mean distance random sample -> nearest real subject")
    println("  pca_generalization.csv         - leave-one-out reconstruction error per mode")
    println("Run `python3 scripts/plot_ssm_validation.py` for the standard validation figures.")
    println("Run `sbt \"runMain scapula.ViewResults\"` to view the PCA model alongside the GPMM.")
    println("Run `sbt \"runMain scapula.ViewAllRegistered\"` to overlay all GPA-aligned shapes.")
  }
}
