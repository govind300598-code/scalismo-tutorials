package scapula

import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Entry point for the complete iterative scapula SSM pipeline.
 *
 * Run with:
 *   sbt "runMain scapula.Main"
 *
 * Key environment variables (all have defaults):
 *   SCAPULA_DATA_DIR – folder containing STL files and the landmark CSV
 *                      default: /home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula
 *   SCAPULA_OUT_DIR  – root for all outputs (results/, data/, comparison/)
 *                      default: /home/g25upadh/Documents/100 plus scapula data/scapula_atlas_out
 *   SCAPULA_UI       – show Scalismo UI after pipeline (true/false, default false)
 *
 * Pipeline stages (no flags needed – they run in order):
 *   1. Load all specimens and landmarks
 *   2. Decimate originals to ~8k working meshes (one-time, idempotent)
 *   3. Pick initial reference (first left scapula with valid landmarks)
 *   4. SSM1: register all → build model → compute Mean1
 *   5. SSM2: use Mean1 as reference → repeat → Mean2
 *   6. SSM3: use Mean2 as reference → repeat → Mean3
 *   7. SSM4: use Mean3 as reference → repeat → Mean4
 *   8. Mean1↔Mean2, Mean2↔Mean3, Mean3↔Mean4 surface comparisons
 *   9. SSM metrics table (variance, generalization, specificity)
 *  10. Convergence assessment
 */
object Main {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir      = Config.dataDir
    val outDir       = Config.outDir
    val preDir       = new File(outDir, "data/preprocessing/8k")
    val resultsDir   = new File(outDir, "results")
    val compDir      = new File(outDir, "comparison")
    val ssmCompDir   = new File(compDir, "SSM_comparison")

    println("=" * 80)
    println("  ScapulaAtlasRefinement – Iterative SSM Pipeline")
    println("=" * 80)
    println(s"  Data dir  : ${dataDir.getAbsolutePath}")
    println(s"  Output dir: ${outDir.getAbsolutePath}")

    // ── 1. Load specimens and landmarks ───────────────────────────────────────
    val csvFile = ScapulaData.csvFile(dataDir)
    println(s"\nLandmark CSV: ${csvFile.getName}")
    val (landmarks, fromHeader, header) = ScapulaData.readLandmarkCsv(csvFile)
    println(s"  ${landmarks.size} landmark rows parsed (columns from header: $fromHeader)")

    val allSpecimens = ScapulaData.specimens(dataDir)
    println(s"  ${allSpecimens.length} STL files found")

    val validSpecimens = allSpecimens.filter(s => landmarks.contains(s.modelId))
    println(s"  ${validSpecimens.length} specimens have landmark rows")

    // ── 2. Generate 8k working meshes (idempotent) ────────────────────────────
    println(s"\n[DECIMATION] Generating ~${Config.modelResolution}-vertex working meshes → ${preDir.getPath}")
    Decimation.generateAll(validSpecimens, preDir, targetN = Config.modelResolution)

    // ── 3. Pick initial reference (first left scapula with landmarks) ─────────
    val initialRefSpec = validSpecimens.find(!_.isRight).getOrElse(
      throw new RuntimeException("No left scapula found in the dataset")
    )
    println(s"\n[REFERENCE] Initial reference: ${initialRefSpec.modelId}")
    val initialRef  = ScapulaData.loadMesh(initialRefSpec.file)
    val initialRefL = landmarks(initialRefSpec.modelId)

    // Decimate the initial reference to ~8k so all SSM outputs are at this resolution
    val refMesh8k = Decimation.decimateByStride(initialRef, Config.modelResolution)
    println(f"  Reference: ${initialRef.pointSet.numberOfPoints} → ${refMesh8k.pointSet.numberOfPoints} vertices")

    // ── 4–7. SSM1 → SSM4 ─────────────────────────────────────────────────────
    // The GP prior (σ=30 mm, scale=10 mm) is FIXED across all iterations.
    // Only the reference mesh changes between iterations.
    println(s"\n[GP PRIOR] Building shared GP prior (σ=${NonRigidReg.gpSigma} mm, scale=${NonRigidReg.gpScaleFactor} mm)")
    val lowRankGP = NonRigidReg.buildPrior(refMesh8k, relativeTolerance = Config.gpRelativeTolerance, maxRank = Config.gpMaxRank)

    val allMetrics = scala.collection.mutable.ArrayBuffer.empty[SSMBuilder.SSMMetrics]
    val allMeans   = scala.collection.mutable.ArrayBuffer.empty[TriangleMesh[_3D]]

    var currentRef  = refMesh8k
    var currentRefL = initialRefL

    for (iter <- 1 to 4) {
      val label    = s"SSM$iter"
      val iterDir  = new File(resultsDir, label)

      val (ssm, mean, registered) = IterativePipeline.runIteration(
        iterLabel  = label,
        reference  = currentRef,
        referenceL = currentRefL,
        specimens  = validSpecimens,
        landmarks  = landmarks,
        outDir     = iterDir,
        lowRankGP  = lowRankGP
      )

      allMeans += mean

      // Compute and collect SSM metrics
      val metrics = SSMBuilder.computeMetrics(ssm, registered, label)
      allMetrics += metrics

      // Save metrics row
      val mFile = new File(new File(iterDir, "metrics"), "ssm_metrics.csv")
      mFile.getParentFile.mkdirs()
      val pw = new java.io.PrintWriter(mFile)
      pw.println("label,rank,mode1_pct,top5_pct,top10_pct,generalization_mm,specificity_mm")
      pw.println(s"${metrics.label},${metrics.rank},${metrics.mode1Pct},${metrics.top5Pct}," +
        s"${metrics.top10Pct},${metrics.generalization},${metrics.specificity}")
      pw.close()

      // Next iteration uses the current mean as the new reference.
      // The mean already has the same topology as currentRef.
      // Landmarks for the new reference: project the original reference landmarks
      // onto the mean mesh (nearest-vertex projection).
      currentRef  = mean
      currentRefL = projectLandmarksToMesh(currentRefL, mean)
    }

    // ── 8. Mean comparisons (Mean1↔Mean2, Mean2↔Mean3, Mean3↔Mean4) ──────────
    println("\n[COMPARISON] Surface-to-surface mean comparisons")
    val meanComparisons = Comparison.runAllMeanComparisons(allMeans.toIndexedSeq, compDir)
    Comparison.assessConvergence(meanComparisons)

    // ── 9. SSM comparison table ───────────────────────────────────────────────
    println("\n[SSM TABLE]")
    SSMBuilder.writeMetricsTable(allMetrics.toIndexedSeq, new File(ssmCompDir, "ssm_comparison.md"))

    // Print the table to stdout too
    println("| Model | Mode 1 | Top 5 | Top 10 | Generalization | Specificity | Reconstruction |")
    println("|-------|-------:|------:|-------:|---------------:|------------:|---------------:|")
    allMetrics.foreach(m => println(m.tableRow))

    // ── 10. Launch UI if requested ────────────────────────────────────────────
    if (Config.showUi) {
      println("\n[UI] Launching Scalismo UI viewer...")
      VisualizationApp.main(Array.empty)
    }

    println("\n\nPipeline complete.")
    println(s"  Results → ${resultsDir.getAbsolutePath}")
    println(s"  Comparison → ${compDir.getAbsolutePath}")
  }

  /**
   * Project landmarks from the old reference onto the new mean mesh by finding
   * the nearest vertex on the new mesh surface.
   *
   * This carries the landmark set forward from iteration to iteration so that
   * landmark-based rigid alignment remains consistent.
   */
  private def projectLandmarksToMesh(
    lms: IndexedSeq[scalismo.geometry.Landmark[_3D]],
    mesh: TriangleMesh[_3D]
  ): IndexedSeq[scalismo.geometry.Landmark[_3D]] = {
    val ops = mesh.operations
    lms.map { lm =>
      val closest = ops.closestPointOnSurface(lm.point).point
      lm.copy(point = closest)
    }
  }
}
