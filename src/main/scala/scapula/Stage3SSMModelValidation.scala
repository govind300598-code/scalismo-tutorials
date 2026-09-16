package scapula

import breeze.linalg.{sum, DenseMatrix, DenseVector}
import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * STAGE 3 -- SSM validation.
 *
 * Reports the three metrics standard in the statistical-shape-model literature for exactly this purpose (Cootes et
 * al.; Styner et al. 2006, "Framework for the Statistical Shape Analysis of Brain Structures Using SPHARM-PDM";
 * Davies et al. 2010, "Building 3-D Statistical Shape Models by Direct Optimization") and used in essentially every
 * published bone/organ SSM paper's evaluation table:
 *
 *   - Compactness(k):     cumulative fraction of total shape variance explained by the first k modes. Higher/steeper
 *                         is better -- a compact model needs fewer parameters to describe the population.
 *   - Generalization(k):  leave-one-out. Rebuild the model without specimen i, project specimen i onto the first k
 *                         modes of that reduced model, measure the RMS point-to-point reconstruction error against
 *                         the real (held-out) specimen. Averaged over all i. Lower is better -- it measures whether
 *                         the model can represent shapes it was not trained on.
 *   - Specificity(k):     draw random instances from the (full) model's first k modes; for each, the RMS point
 *                         distance to its closest real training specimen. Averaged over many draws. Lower is better
 *                         -- it penalises a model flexible enough to generate anatomically implausible shapes.
 *
 * All three are reported per mode count (the standard compactness/generalization/specificity curves), plus, for
 * each of Config.compactnessThresholds, the mode count needed to reach that compactness and the generalization /
 * specificity at that mode count -- the single numbers usually quoted in a paper's results paragraph (e.g. "at 95%
 * compactness (k=7 modes), generalization = 1.3 mm, specificity = 1.8 mm").
 *
 * Run after Stage2GPNonRigidRegistration; reads its VTK output (never STL -- see Stage2GPNonRigidRegistration's header comment).
 */
object Stage3SSMModelValidation {

  final case class ModeRow(modes: Int, compactnessPct: Double, generalizationMm: Double, specificityMm: Double)

  def loadRegistered(dir: File): (TriangleMesh[_3D], IndexedSeq[(String, TriangleMesh[_3D])]) = {
    val referenceFile = new File(dir, "reference.vtk")
    require(referenceFile.exists(), s"Missing ${referenceFile.getPath} -- run Stage2GPNonRigidRegistration first.")
    val reference = ScapulaData.loadMesh(referenceFile)

    val registeredDir = new File(dir, "registered")
    val files = Option(registeredDir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.toLowerCase.endsWith(".vtk"))
      .sortBy(_.getName)
    require(files.nonEmpty, s"No registered meshes found in ${registeredDir.getPath} -- run Stage2GPNonRigidRegistration first.")

    val meshes = files.toIndexedSeq.map(f => f.getName.stripSuffix(".vtk") -> ScapulaData.loadMesh(f))
    meshes.foreach { case (id, m) =>
      require(
        m.pointSet.numberOfPoints == reference.pointSet.numberOfPoints,
        s"$id has ${m.pointSet.numberOfPoints} points, reference has ${reference.pointSet.numberOfPoints} points " +
          "-- these meshes are not in correspondence."
      )
    }
    (reference, meshes)
  }

  def compactnessPct(eigenvalues: DenseVector[Double], k: Int): Double = {
    val total = sum(eigenvalues)
    val kk = math.min(k, eigenvalues.length)
    val partial = sum(eigenvalues(0 until kk))
    100.0 * partial / total
  }

  def generalization(vectors: IndexedSeq[DenseVector[Double]], numPoints: Int, maxModes: Int): IndexedSeq[Double] = {
    val n = vectors.length
    val perFoldErrors: IndexedSeq[IndexedSeq[Double]] = (0 until n).map { holdOut =>
      val trainVectors = vectors.indices.filterNot(_ == holdOut).map(vectors)
      val pca = SSMPCACore.fit(SSMPCACore.dataMatrixFromVectors(trainVectors))
      val target = vectors(holdOut)
      val availableModes = math.min(maxModes, pca.eigenvectors.cols)
      val errors = (1 to availableModes).map { k =>
        val recon = SSMPCACore.project(pca, target, k)
        SSMPCACore.rmsPointDistance(recon, target, numPoints)
      }
      // A leave-one-out fold trains on n-1 specimens, so its model can have fewer than maxModes informative modes
      // (rank <= n-2). Beyond that point it has no more variance left to add, so its reconstruction -- and hence its
      // error -- stops changing; hold the last real value flat rather than silently averaging over fewer folds.
      if (availableModes < maxModes) errors ++ IndexedSeq.fill(maxModes - availableModes)(errors.last) else errors
    }
    (0 until maxModes).map(k => perFoldErrors.map(_(k)).sum / n)
  }

  def specificity(
      pca: SSMPCACore.DualPCA,
      trainingVectors: IndexedSeq[DenseVector[Double]],
      numPoints: Int,
      maxModes: Int,
      numSamples: Int
  )(implicit rng: Random): IndexedSeq[Double] =
    (1 to maxModes).map { k =>
      val distances = (0 until numSamples).map { _ =>
        val s = SSMPCACore.sample(pca, k)
        trainingVectors.map(v => SSMPCACore.rmsPointDistance(s, v, numPoints)).min
      }
      distances.sum / numSamples
    }

  def modesForCompactness(rows: Seq[ModeRow], thresholdPct: Double): Option[ModeRow] =
    rows.find(_.compactnessPct >= thresholdPct)

  def evaluatePopulation(name: String, meshes: IndexedSeq[TriangleMesh[_3D]], outDir: File)(implicit rng: Random): Seq[ModeRow] = {
    val n = meshes.length
    require(n >= 4, s"Need at least 4 specimens to run leave-one-out validation ('$name' has $n)")
    val numPoints = meshes.head.pointSet.numberOfPoints
    val vectors = meshes.map(SSMPCACore.toVector)
    val fullPca = SSMPCACore.fit(SSMPCACore.dataMatrixFromVectors(vectors))
    val maxModes = math.min(Config.maxValidationModes, fullPca.eigenvectors.cols)

    println(s"\n[$name] n=$n specimens, $numPoints vertices/specimen, full-model rank=${fullPca.eigenvectors.cols}, " +
      s"evaluating modes 1..$maxModes")

    val gen = generalization(vectors, numPoints, maxModes)
    val spec = specificity(fullPca, vectors, numPoints, maxModes, Config.specificitySamples)
    val rows = (1 to maxModes).map(k => ModeRow(k, compactnessPct(fullPca.eigenvalues, k), gen(k - 1), spec(k - 1)))

    val csv = new File(outDir, s"ssm_validation_$name.csv")
    val pw = new PrintWriter(csv)
    try {
      pw.println("modes,compactness_pct,generalization_mm,specificity_mm")
      rows.foreach(r => pw.println(f"${r.modes},${r.compactnessPct}%.4f,${r.generalizationMm}%.4f,${r.specificityMm}%.4f"))
    } finally pw.close()

    println(f"  ${"modes"}%6s ${"compactness(%)"}%16s ${"generalization(mm)"}%20s ${"specificity(mm)"}%17s")
    rows.foreach(r => println(f"  ${r.modes}%6d ${r.compactnessPct}%16.2f ${r.generalizationMm}%20.3f ${r.specificityMm}%17.3f"))
    println(s"  wrote ${csv.getPath}")

    Config.compactnessThresholds.foreach { thr =>
      modesForCompactness(rows, thr) match {
        case Some(r) =>
          println(f"  at $thr%.0f%% compactness (k=${r.modes} modes): generalization=${r.generalizationMm}%.3f mm, " +
            f"specificity=${r.specificityMm}%.3f mm")
        case None =>
          println(f"  $thr%.0f%% compactness not reached within $maxModes modes (max reached: " +
            f"${rows.last.compactnessPct}%.1f%%) -- raise SCAPULA_MAX_VALIDATION_MODES")
      }
    }
    rows
  }

  def writePaperSummary(outDir: File, results: Seq[(String, Int, Seq[ModeRow])]): Unit = {
    val f = new File(outDir, "ssm_summary_for_paper.csv")
    val pw = new PrintWriter(f)
    try {
      val header = "population,n" + Config.compactnessThresholds
        .map(t => f"k_at_${t.toInt}pct,generalization_mm_at_${t.toInt}pct,specificity_mm_at_${t.toInt}pct")
        .mkString(",", ",", "")
      pw.println(header)
      results.foreach { case (name, n, rows) =>
        val cells = Config.compactnessThresholds.map { thr =>
          modesForCompactness(rows, thr) match {
            case Some(r) => f"${r.modes},${r.generalizationMm}%.4f,${r.specificityMm}%.4f"
            case None    => ",,"
          }
        }
        pw.println(s"$name,$n," + cells.mkString(","))
      }
    } finally pw.close()
    println(s"\nWrote publication summary table: ${f.getPath}")
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    Config.outDir.mkdirs()
    val (_, registered) = loadRegistered(Config.outDir)

    val fullRows = evaluatePopulation("full", registered.map(_._2), Config.outDir)
    val results = scala.collection.mutable.ArrayBuffer(("full_n24_both_sides", registered.length, fullRows))

    if (Config.buildIndependentModel) {
      val specimens = ScapulaData.specimens(Config.dataDir)
      val bySubject = specimens.groupBy(_.subject)
      val chosenIds = bySubject.values.flatMap(group => group.find(!_.isRight).orElse(group.find(_.isRight)).map(_.modelId)).toSet
      val independentMeshes = registered.filter { case (id, _) => chosenIds.contains(id) }.map(_._2)
      println(s"\n[independent] using ${independentMeshes.length} unilateral specimens (one side per subject).")
      println("  Left and mirrored-right scapulae from the same person are NOT statistically independent samples --")
      println("  the 'full' n=24 numbers above overstate effective sample size; this model is the honest one to")
      println("  report as the primary result in a paper, with 'full' as a sensitivity check.")
      val indRows = evaluatePopulation("independent", independentMeshes, Config.outDir)
      results += (("independent_n12_one_side_per_subject", independentMeshes.length, indRows))
    }

    writePaperSummary(Config.outDir, results.toSeq)
  }
}
