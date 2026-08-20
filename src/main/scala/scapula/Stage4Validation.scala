package scapula

import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.numerics.PivotedCholesky
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Stage 4 — SSM Validation.
 *
 * Three standard metrics (Styner et al., IPMI 2003):
 *
 *   Compactness    — cumulative fraction of training variance vs number of modes.
 *
 *   Generalization — leave-one-out: build a model on N-1 shapes, project the
 *                    held-out shape, report symmetric mean surface distance.
 *
 *   Specificity    — random samples from the model are close to training shapes.
 *                    For each sample find the nearest training shape; average distance.
 */
object Stage4Validation {

  type SSMModel = PointDistributionModel[_3D, TriangleMesh]

  // ---------------------------------------------------------------------------
  // Compactness
  // ---------------------------------------------------------------------------
  def compactness(model: SSMModel): IndexedSeq[(Int, Double)] = {
    val ev     = model.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val total  = ev.sum
    val cumVar = ev.scanLeft(0.0)(_ + _).tail.map(_ / total).toIndexedSeq
    cumVar.zipWithIndex.map { case (v, i) => (i + 1, v) }
  }

  // ---------------------------------------------------------------------------
  // Generalization (leave-one-out)
  // ---------------------------------------------------------------------------
  def generalization(
    reference:  TriangleMesh[_3D],
    registered: IndexedSeq[(String, TriangleMesh[_3D])],
    maxModes:   Int
  ): IndexedSeq[(Int, Double)] = {

    val n = registered.length
    if (n < 3) {
      println("  [Generalization] Fewer than 3 shapes — skipping LOO.")
      return IndexedSeq.empty
    }

    println(s"  [Generalization] LOO over $n shapes (maxModes=$maxModes)…")
    val modeRange = (1 to math.min(maxModes, n - 2)).toIndexedSeq

    val errorsByMode: IndexedSeq[IndexedSeq[Double]] = registered.zipWithIndex.flatMap { case ((id, heldOut), i) =>
      val trainingMeshes = registered.zipWithIndex.collect { case ((_, m), j) if j != i => m }
      val dc             = DataCollection.fromTriangleMesh3DSequence(reference, trainingMeshes)
      val looModel       = PointDistributionModel.createUsingPCA(dc, PivotedCholesky.RelativeTolerance(0.01))
      Some(modeRange.map { k =>
        val projected = looModel.truncate(k).project(heldOut)
        Metrics.symmetric(projected, heldOut).mean
      })
    }

    if (errorsByMode.isEmpty) return IndexedSeq.empty
    modeRange.map { k =>
      (k, errorsByMode.map(_(k - 1)).sum / errorsByMode.length)
    }
  }

  // ---------------------------------------------------------------------------
  // Specificity
  // ---------------------------------------------------------------------------
  def specificity(
    model:      SSMModel,
    training:   IndexedSeq[TriangleMesh[_3D]],
    numSamples: Int = 100,
    maxModes:   Int = 10
  )(implicit rng: Random): IndexedSeq[(Int, Double)] = {

    println(s"  [Specificity] $numSamples samples per mode count (maxModes=$maxModes)…")
    val modeRange = (1 to math.min(maxModes, model.rank)).toIndexedSeq

    modeRange.map { k =>
      val truncated = model.truncate(k)
      val dists = (0 until numSamples).map { _ =>
        val sample      = truncated.sample()
        val nearestDist = training.map(t => Metrics.symmetric(sample, t).mean).min
        nearestDist
      }
      (k, dists.sum / dists.length)
    }
  }

  // ---------------------------------------------------------------------------
  // Print + CSV export
  // ---------------------------------------------------------------------------
  def printAndSave(
    compactnessData:    IndexedSeq[(Int, Double)],
    generalizationData: IndexedSeq[(Int, Double)],
    specificityData:    IndexedSeq[(Int, Double)],
    outDir:             File,
    tag:                String = ""
  ): Unit = {

    println("\n  [Compactness] Cumulative variance vs #modes:")
    compactnessData.take(20).foreach { case (k, v) =>
      val bar = "#" * (v * 40).toInt
      println(f"    $k%3d  ${v * 100}%5.1f%%  $bar")
    }

    if (generalizationData.nonEmpty) {
      println("\n  [Generalization] Mean surface distance of held-out shape vs #modes:")
      generalizationData.foreach { case (k, d) => println(f"    $k%3d modes  $d%6.3f mm") }
    }

    if (specificityData.nonEmpty) {
      println("\n  [Specificity] Mean distance of random sample to nearest training shape vs #modes:")
      specificityData.foreach { case (k, d) => println(f"    $k%3d modes  $d%6.3f mm") }
    }

    val suffix  = if (tag.nonEmpty) s"_$tag" else ""
    val csvFile = new File(outDir, s"ssm_validation$suffix.csv")
    val pw      = new PrintWriter(csvFile)
    try {
      pw.println("modes,compactness_cumvar,generalization_mm,specificity_mm")
      val maxK = Seq(compactnessData.length, generalizationData.length, specificityData.length).max
      for (i <- 0 until maxK) {
        val k    = i + 1
        val comp = compactnessData.lift(i).map(_._2).getOrElse(Double.NaN)
        val gen  = generalizationData.lift(i).map(_._2).getOrElse(Double.NaN)
        val spec = specificityData.lift(i).map(_._2).getOrElse(Double.NaN)
        pw.println(f"$k,$comp%.6f,$gen%.4f,$spec%.4f")
      }
    } finally pw.close()

    println(s"\n  Validation CSV → ${csvFile.getAbsolutePath}")
  }

  // ---------------------------------------------------------------------------
  // Run all three metrics.
  // ---------------------------------------------------------------------------
  def run(
    model:              SSMModel,
    reference:          TriangleMesh[_3D],
    registered:         IndexedSeq[(String, TriangleMesh[_3D])],
    outDir:             File,
    tag:                String = "",
    maxModes:           Int    = 20,
    specificitySamples: Int    = 50
  )(implicit rng: Random): Unit = {

    println(s"\n=== STAGE 4: SSM VALIDATION${if (tag.nonEmpty) s" [$tag]" else ""} ===")

    val compData = compactness(model)
    val genData  = generalization(reference, registered, maxModes)
    val training = registered.map(_._2)
    val specData = specificity(model, training, numSamples = specificitySamples, maxModes = maxModes)

    printAndSave(compData, genData, specData, outDir, tag)
  }
}
