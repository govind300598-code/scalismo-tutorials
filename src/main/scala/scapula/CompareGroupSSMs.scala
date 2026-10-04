package scapula

import scalismo.geometry.*
import scalismo.io.StatisticalModelIO

import java.io.{File, PrintWriter}

/**
 * Compares the two group-specific PCA models built by Stage3PCAModelByGroup:
 *   - point-wise distance between the two mean shapes (same reference topology, so directly comparable)
 *   - where that difference is largest (written to a CSV you can color-map in scalismo-ui, same way as the
 *     per-vertex registration-error heatmap)
 *   - basic eigenspectrum comparison (rank, PC1-3 variance share), already printed by Stage3PCAModelByGroup, is
 *     NOT determined by the mean difference -- the two are independent questions.
 *
 * Run after Stage3PCAModelByGroup.
 */
object CompareGroupSSMs {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val outDir = Config.outDir
    val fileA = new File(outDir, "scapula_pca_model_hillsachs.json")
    val fileB = new File(outDir, "scapula_pca_model_paired.json")
    require(fileA.exists(), s"$fileA not found -- run Stage3PCAModelByGroup first.")
    require(fileB.exists(), s"$fileB not found -- run Stage3PCAModelByGroup first.")

    val modelA = StatisticalModelIO.readStatisticalTriangleMeshModel3D(fileA).get
    val modelB = StatisticalModelIO.readStatisticalTriangleMeshModel3D(fileB).get
    println(s"Loaded hillsachs model (rank ${modelA.rank}) and paired model (rank ${modelB.rank}).")

    val meanA = modelA.mean
    val meanB = modelB.mean
    require(meanA.pointSet.numberOfPoints == meanB.pointSet.numberOfPoints,
      s"Mean shapes have different point counts (${meanA.pointSet.numberOfPoints} vs ${meanB.pointSet.numberOfPoints}) " +
        "-- they were not built from the same reference topology, so this comparison is meaningless.")

    val pointsA = meanA.pointSet.points.toIndexedSeq
    val pointsB = meanB.pointSet.points.toIndexedSeq
    val diffs = pointsA.indices.map(i => (pointsA(i) - pointsB(i)).norm)

    val sorted = diffs.sorted
    val n = diffs.length
    val mean = diffs.sum / n
    val rms = math.sqrt(diffs.map(d => d * d).sum / n)
    val median = sorted(n / 2)
    val p95 = sorted(math.min(n - 1, (0.95 * n).toInt))
    val max = sorted.last
    val maxIdx = diffs.indices.maxBy(diffs)

    println(f"\nMean-shape difference (Hill-Sachs mean vs. paired mean), $n points:")
    println(f"  mean=$mean%.3f mm  rms=$rms%.3f mm  median=$median%.3f mm  p95=$p95%.3f mm  max=$max%.3f mm")
    println(f"  largest-difference point: index=$maxIdx  hillsachs=(${pointsA(maxIdx).x}%.1f, ${pointsA(maxIdx).y}%.1f, " +
      f"${pointsA(maxIdx).z}%.1f)  paired=(${pointsB(maxIdx).x}%.1f, ${pointsB(maxIdx).y}%.1f, ${pointsB(maxIdx).z}%.1f)  " +
      f"diff=${diffs(maxIdx)}%.3f mm")

    val csv = new File(outDir, "group_mean_shape_difference.csv")
    val pw = new PrintWriter(csv)
    try {
      pw.println("point_id,x,y,z,diff_mm")
      pointsA.indices.foreach { i =>
        pw.println(f"$i,${pointsA(i).x}%.4f,${pointsA(i).y}%.4f,${pointsA(i).z}%.4f,${diffs(i)}%.4f")
      }
    } finally pw.close()
    println(s"\nWrote per-point difference to $csv -- color-map it in scalismo-ui the same way as your " +
      "registration-error heatmap (x,y,z are the hillsachs mean's coordinates; diff_mm is the color value) to see " +
      "WHERE on the bone the two groups differ, rather than just the summary numbers above.")

    println("\nNote: this compares MEAN shapes only. It says nothing about whether the shapes overlap in general " +
      "(use each model's sample()/instance() to check that separately), and it does not control for the ~30-year " +
      "age gap between the two cohorts -- any difference found here is 'associated with group', not proven to be " +
      "caused by Hill-Sachs pathology specifically.")
  }
}
