package scapula

import scalismo.geometry._3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random as ScalismoRandom

import java.io.File

/**
 * STAGE 3 -- THE ACTUAL STATISTICAL SHAPE MODEL.
 *
 * Everything through Stage 2 used an ANALYTIC kernel (hand-picked smoothness assumptions, see GpmmFitting) purely
 * as a REGISTRATION tool, to get every subject into correspondence with the reference. That model was never a
 * statistical model of this population: sampling from it (ScalismoUI's "Random" button on scapula_gpmm.json)
 * draws from a generic smoothness prior that has never seen these 11 real scapulae, which is why those samples
 * look like wiggly, non-anatomical noise rather than plausible bones.
 *
 * This stage builds the real thing: PCA over the already-registered final_<subject>_fit.stl shapes, which is what
 * "statistical shape model" means in the field (Cootes' Point Distribution Models). Its covariance is LEARNED
 * from the population instead of designed by hand -- and it also gives the two standard SSM evaluation metrics
 * that a hand-designed prior cannot: compactness (how many modes actually matter) and specificity (do random
 * samples look like real anatomy).
 *
 * Reads what Stage2ReferenceRefinement already wrote; does not re-run any registration.
 */
object Stage3PCAModel {

  private val specificitySamples = 100

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
    require(fitFiles.length >= 3,
      s"Need at least 3 registered subjects to run PCA, found ${fitFiles.length} in ${dir.getAbsolutePath}. " +
        "(A SCAPULA_SUBJECT_LIMIT=2 smoke test doesn't have enough subjects for this step.)")

    val subjectIds = fitFiles.map(_.getName.stripPrefix("final_").stripSuffix("_fit.stl"))
    val fittedMeshes = fitFiles.map(f => MeshIO.readMesh(f).get)
    println(s"Building PCA model from ${fittedMeshes.length} registered subjects: ${subjectIds.mkString(", ")}")

    val dc = DataCollection.fromTriangleMesh3DSequence(reference, fittedMeshes.toIndexedSeq)
    val pcaModel: PointDistributionModel[_3D, TriangleMesh] = PointDistributionModel.createUsingPCA(dc)
    println(f"  PCA model rank: ${pcaModel.rank} (at most N-1=${fittedMeshes.length - 1} for N=${fittedMeshes.length} subjects " +
      "-- PCA over N samples has only N-1 degrees of freedom around their mean)")

    // ---------------------------------------------------------------- compactness
    val variance = pcaModel.gp.variance.toArray
    val totalVariance = variance.sum
    println("\n[Compactness] cumulative variance explained by the leading modes:")
    var cumulative = 0.0
    val compactnessRows = variance.zipWithIndex.map { case (v, i) =>
      cumulative += v
      val frac = if (totalVariance > 0) cumulative / totalVariance else 0.0
      println(f"    mode ${i + 1}%2d: eigenvalue=$v%8.2f   cumulative variance=${frac * 100}%5.1f%%")
      Seq(i + 1, v, frac)
    }
    CsvWriter.write(new File(dir, "pca_compactness.csv"),
      Seq("mode", "eigenvalue", "cumulative_variance_fraction"), compactnessRows.toSeq)

    // ---------------------------------------------------------------- specificity
    val targetFiles = subjectIds.map(id => new File(dir, s"final_${id}_target.stl"))
    val realTargets = targetFiles.filter(_.exists()).map(f => MeshIO.readMesh(f).get)
    println(s"\n[Specificity] drawing $specificitySamples random samples from the PCA model, measuring each one's " +
      "distance to the nearest REAL subject:")
    val specificityDistances = (1 to specificitySamples).map { _ =>
      val sample = pcaModel.sample()
      realTargets.map(t => Metrics.symmetric(sample, t).mean).min
    }
    val specMean = specificityDistances.sum / specificityDistances.length
    val specStd = math.sqrt(specificityDistances.map(d => (d - specMean) * (d - specMean)).sum / specificityDistances.length)
    println(f"    mean nearest-real-subject distance: $specMean%.2f mm  (std $specStd%.2f mm) over $specificitySamples samples")
    println("    Lower = random samples look more like plausible real anatomy. Compare this number against samples")
    println("    from scapula_gpmm.json (the analytic kernel) to see the difference a data-driven model makes.")
    CsvWriter.write(new File(dir, "pca_specificity.csv"),
      Seq("sample", "distance_to_nearest_real_subject_mm"),
      specificityDistances.zipWithIndex.map { case (d, i) => Seq(i + 1, d) })

    val modelOut = new File(dir, "scapula_pca_model.json")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(pcaModel, modelOut).get
    println(s"\nWrote the real, data-driven PCA model -> ${modelOut.getAbsolutePath}")
    println("Run `sbt \"runMain scapula.ViewResults\"` to load it alongside scapula_gpmm.json -- its 'Random'")
    println("samples should look like plausible scapula variants, not the analytic kernel's wiggly spikes.")
  }
}
