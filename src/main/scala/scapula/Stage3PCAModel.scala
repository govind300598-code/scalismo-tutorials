package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random as ScalismoRandom

import java.io.File

/**
 * STAGE 3 -- THE ACTUAL STATISTICAL SHAPE MODEL, AND ITS STANDARD VALIDATION.
 *
 * Everything through Stage 2 used an ANALYTIC kernel (hand-picked smoothness assumptions, see GpmmFitting) purely
 * as a REGISTRATION tool, to get every subject into correspondence with the reference. That model was never a
 * statistical model of this population: sampling from it (ScalismoUI's "Random" button on scapula_gpmm.json)
 * draws from a generic smoothness prior that has never seen these real scapulae.
 *
 * This stage builds the real thing: PCA over the already-registered final_<subject>_fit.stl shapes, which is what
 * "statistical shape model" means in the field (Cootes' Point Distribution Models). It then computes the three
 * metrics every SSM paper reports (Styner et al. 2003, "Evaluation of 3D correspondence methods..."):
 *
 *   - COMPACTNESS: cumulative variance explained vs. number of modes kept. Answers "how many modes matter".
 *   - SPECIFICITY: distance from a random model sample to the nearest real subject, vs. number of modes used.
 *     Answers "do samples look like real anatomy" -- should stay low and roughly flat/slowly rising.
 *   - GENERALIZATION: leave-one-out reconstruction error vs. number of modes used. Build the model WITHOUT
 *     subject i, project subject i's own (already-registered) shape onto it, measure the reconstruction error.
 *     Answers "can the model represent a shape it wasn't built from" -- should fall as more modes are kept.
 *
 * All three only need the correspondences Stage 2 already computed (fixed reference topology, same point count
 * everywhere) -- leave-one-out here means rebuilding a small PCA (milliseconds) and projecting, NOT redoing
 * registration, so it costs seconds, not hours.
 *
 * Writes 4 CSVs; run scripts/plot_ssm_validation.py afterwards to turn them into the standard figures.
 */
object Stage3PCAModel {

  private val specificitySamplesPerK = 30

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
    println(s"Building PCA model from ${fittedMeshes.length} registered subjects: ${subjectIds.mkString(", ")}")

    val dc = DataCollection.fromTriangleMesh3DSequence(reference, fittedMeshes.toIndexedSeq)
    val pcaModel: PointDistributionModel[_3D, TriangleMesh] = PointDistributionModel.createUsingPCA(dc)
    println(f"  PCA model rank: ${pcaModel.rank} (at most N-1=${fittedMeshes.length - 1} for N=${fittedMeshes.length} subjects " +
      "-- PCA over N samples has only N-1 degrees of freedom around their mean)")

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
        realTargets.map(t => Metrics.symmetric(sample, t).mean).min
      }
      val mean = distances.sum / distances.length
      println(f"    k=${k}%3d modes: mean nearest-real-subject distance = $mean%.2f mm")
      Seq(k, mean)
    }
    CsvWriter.write(new File(dir, "pca_specificity.csv"),
      Seq("num_components", "mean_distance_to_nearest_real_subject_mm"), specificityRows.toSeq)

    // ---------------------------------------------------------------- 3. generalization (leave-one-out)
    println(s"\n[3/3 Generalization] leave-one-out reconstruction error vs. number of components used " +
      s"(${fittedMeshes.length} folds):")
    val perSubjectCurves = fittedMeshes.indices.map { i =>
      val others = fittedMeshes.indices.filter(_ != i).map(fittedMeshes).toIndexedSeq
      val dcLoo = DataCollection.fromTriangleMesh3DSequence(reference, others)
      val modelLoo = PointDistributionModel.createUsingPCA(dcLoo)
      val fullCoeffs = modelLoo.coefficients(fittedMeshes(i))
      val errorsPerK = (1 to modelLoo.rank).map { k =>
        val truncated = fullCoeffs.copy
        (k until truncated.length).foreach(j => truncated(j) = 0.0)
        val reconstruction = modelLoo.instance(truncated)
        Metrics.symmetric(reconstruction, fittedMeshes(i)).mean
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
        s"${fittedMeshes.length} folds)")
      Seq(k, mean)
    }
    CsvWriter.write(new File(dir, "pca_generalization.csv"),
      Seq("num_components", "mean_reconstruction_error_mm"), generalizationRows.toSeq)

    val modelOut = new File(dir, "scapula_pca_model.json")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(pcaModel, modelOut).get
    println(s"\nWrote the real, data-driven PCA model -> ${modelOut.getAbsolutePath}")
    println("Wrote pca_compactness.csv, pca_specificity.csv, pca_generalization.csv to the same directory.")
    println("Run `python3 scripts/plot_ssm_validation.py` to turn those into the standard SSM validation figures.")
    println("Run `sbt \"runMain scapula.ViewResults\"` to load the PCA model alongside scapula_gpmm.json and")
    println("compare their 'Random' samples directly.")
  }
}
