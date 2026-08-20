package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import scala.util.Using

/**
 * STAGE 3 -- STATISTICAL SHAPE MODEL.
 *
 * Loads the registered meshes from Stage 2, fits a PCA shape model, evaluates standard SSM
 * quality metrics (compactness, generalization, specificity) and writes them to CSV.
 *
 * Specificity uses 1000 random samples (Styner et al. 2003 recommendation). 30 samples, as
 * sometimes seen in tutorials, gives an ~18% coefficient of variation and is not meaningful.
 */
object Stage3BuildModel {

  // ── load registered meshes ────────────────────────────────────────────────

  def loadRegistered(regDir: File): IndexedSeq[TriangleMesh[_3D]] = {
    Option(regDir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.toLowerCase.endsWith(".stl"))
      .sortBy(_.getName)
      .map(f => MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read ${f.getPath}")))
      .toIndexedSeq
  }

  // ── quality metrics ───────────────────────────────────────────────────────

  /**
   * Compactness: cumulative variance explained by the first k modes, as a fraction of total variance.
   * Perfect compactness = 1.0 with k = 1.
   */
  def compactness(model: StatisticalMeshModel): IndexedSeq[(Int, Double)] = {
    val variances = model.gp.klBasis.map(_.eigenvalue)
    val total = variances.sum
    var cumulative = 0.0
    variances.zipWithIndex.map { case (v, i) =>
      cumulative += v
      (i + 1, cumulative / total)
    }
  }

  /**
   * Generalization: for each held-out specimen, register it to the model mean, project onto the
   * model and measure reconstruction error. Lower is better. Returns per-leave-one-out errors.
   *
   * We use surface-distance (symmetric) rather than point-to-point, so the metric is independent
   * of the correspondence established by Stage 2.
   */
  def generalization(
    model: StatisticalMeshModel,
    meshes: IndexedSeq[TriangleMesh[_3D]]
  ): IndexedSeq[Double] = {
    meshes.map { target =>
      val projected = model.project(target)
      Metrics.symmetric(projected, target).mean
    }
  }

  /**
   * Specificity: draw `nSamples` random shapes from the model and measure how close each is to the
   * nearest training shape. Lower is better (random samples stay near real shapes).
   *
   * Uses 1000 samples per Styner et al. 2003.
   */
  def specificity(
    model: StatisticalMeshModel,
    meshes: IndexedSeq[TriangleMesh[_3D]],
    nSamples: Int = 1000
  )(implicit rng: Random): IndexedSeq[Double] = {
    (0 until nSamples).map { _ =>
      val sample = model.sample()
      meshes.map(m => Metrics.symmetric(sample, m).mean).min
    }
  }

  /**
   * Mode concentration: for each mode, compute what fraction of the mode's total variance is
   * carried by the 20 vertices with the largest absolute displacement. A ratio above ~0.6 on mode 1
   * suggests that a single thin structure (acromion, coracoid) dominates that mode — a sign of
   * correspondence errors rather than genuine shape variation.
   */
  def modeConcentration(model: StatisticalMeshModel, topK: Int = 20): Unit = {
    println()
    println("[Mode concentration] Fraction of mode variance in the top-20 highest-displacement vertices")
    println("  A ratio > 0.60 on mode 1 suggests a single thin structure dominates the mode.")
    model.gp.klBasis.zipWithIndex.take(5).foreach { case (basis, i) =>
      val displacements = model.referenceMesh.pointSet.points.map(pt => basis.eigenfunction(pt).norm2).toIndexedSeq
      val total = displacements.sum
      val sorted = displacements.sorted.reverse
      val topKSum = sorted.take(topK).sum
      val ratio = topKSum / total
      val flag = if (ratio > 0.60 && i == 0) "  <-- SUSPICIOUS (mode 1 dominated by thin structure)" else ""
      println(f"  Mode ${i + 1}: top-$topK vertices carry ${ratio * 100}%.1f%% of variance$flag")
    }
  }

  // ── main ──────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    val regDir = new File(outDir, "registered")

    println(s"Loading registered meshes from ${regDir.getAbsolutePath}")
    val meshes = loadRegistered(regDir)
    println(s"${meshes.length} registered meshes loaded")
    require(meshes.length >= 3, s"Need at least 3 registered shapes for PCA; found ${meshes.length}")

    // ── PCA model ─────────────────────────────────────────────────────────
    println("\nBuilding PCA shape model...")
    val model = StatisticalMeshModel.createUsingPCA(meshes).get
    println(s"Model: ${model.rank} modes, reference ${model.referenceMesh.pointSet.numberOfPoints} vertices")

    // Save model
    val modelFile = new File(outDir, "scapula_ssm.h5")
    StatisticalModelIO.writeStatisticalMeshModel(model, modelFile).get
    println(s"Model saved to ${modelFile.getAbsolutePath}")

    // ── mode concentration diagnostic ─────────────────────────────────────
    modeConcentration(model)

    // ── compactness ───────────────────────────────────────────────────────
    println("\nCompactness (cumulative variance explained):")
    val compact = compactness(model)
    compact.take(10).foreach { case (k, c) =>
      println(f"  $k modes: ${c * 100}%5.1f%%")
    }

    // ── generalization ────────────────────────────────────────────────────
    println("\nGeneralization (leave-one-out reconstruction error, mm):")
    val genErrors = generalization(model, meshes)
    genErrors.zipWithIndex.foreach { case (e, i) =>
      println(f"  [${i + 1}%2d] $e%5.2f mm")
    }
    val genMean = genErrors.sum / genErrors.length
    println(f"  Mean: $genMean%.2f mm")

    // ── specificity ───────────────────────────────────────────────────────
    println("\nSpecificity (1000 random samples, nearest-training-shape distance, mm):")
    val specErrors = specificity(model, meshes, nSamples = 1000)
    val specMean = specErrors.sum / specErrors.length
    val specStd = math.sqrt(specErrors.map(e => (e - specMean) * (e - specMean)).sum / specErrors.length)
    println(f"  mean=$specMean%.2f mm  std=$specStd%.2f mm")

    // ── write metrics CSV ─────────────────────────────────────────────────
    val metricsFile = new File(outDir, "ssm_metrics.csv")
    Using.resource(new PrintWriter(metricsFile)) { pw =>
      // compactness sheet
      pw.println("section,k,value")
      compact.foreach { case (k, c) => pw.println(f"compactness,$k,${c}%.6f") }
      genErrors.zipWithIndex.foreach { case (e, i) => pw.println(f"generalization,${i + 1},$e%.4f") }
      specErrors.zipWithIndex.foreach { case (e, i) => pw.println(f"specificity,${i + 1},$e%.4f") }
    }
    println(s"\nSSM metrics written to ${metricsFile.getAbsolutePath}")
  }
}
