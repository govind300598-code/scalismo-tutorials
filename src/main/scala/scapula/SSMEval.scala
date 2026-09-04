package scapula

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.*
import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.statisticalmodel.{MultivariateNormalDistribution, PointDistributionModel}
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import scala.util.Using

/**
 * Comprehensive SSM evaluation.
 *
 * Metrics:
 *  A. Per-specimen registration quality vs rigid-aligned target:
 *       symmetric surface (mean, RMS, HD95, Hausdorff), point-to-point (mean, RMSE, max),
 *       Chamfer L1 and squared.
 *  B. Compactness: cumulative variance explained by k modes.
 *  C. LOO generalization: leave-one-out reconstruction error.
 *  D. Specificity: distance from random model samples to nearest training shape.
 *  E. Reference stability: surface distance between pass-1 and pass-2 mean references.
 *  F. Bilateral consistency: within-subject (L vs R) distance in pass-2 space.
 */
object SSMEval {

  def evaluate(
      model:        PointDistributionModel[_3D, TriangleMesh],
      ssmReference: TriangleMesh[_3D],
      pass1Pairs:   Map[String, TriangleMesh[_3D]],
      pass2Pairs:   Map[String, TriangleMesh[_3D]],
      specimens:    IndexedSeq[ScapulaData.Specimen],
      stability1:   Metrics.SurfaceStats,
      stability2:   Metrics.SurfaceStats,
      outDir:       File
  )(implicit rng: Random): Unit = {

    val ids         = pass2Pairs.keys.toIndexedSeq.sorted
    val trainMeshes = ids.flatMap(pass2Pairs.get)

    // -----------------------------------------------------------------------
    // A. Per-specimen registration quality
    // -----------------------------------------------------------------------
    println("  [A] Per-specimen registration quality...")
    val perSpec: IndexedSeq[(String, Metrics.FullDistStats)] = ids.map { id =>
      val reg    = pass2Pairs(id)
      // Compare registered mesh against the Pass-1 rigid-aligned proxy as the "target"
      val target = pass1Pairs.getOrElse(id, reg)
      val stats  = Metrics.allStats(reg, target)
      println(f"    $id: ${stats.render}")
      id -> stats
    }

    Using.resource(new PrintWriter(new File(outDir, "per_specimen_metrics.csv"))) { w =>
      w.println(s"id,${perSpec.head._2.header}")
      perSpec.foreach { case (id, s) => w.println(s"$id,${s.csv}") }
    }

    val avgSurf      = perSpec.map(_._2.surfMean).sum  / perSpec.length
    val avgHd        = perSpec.map(_._2.surfHd).sum    / perSpec.length
    val avgHd95      = perSpec.map(_._2.surfHd95).sum  / perSpec.length
    val avgPtptRmse  = perSpec.map(_._2.ptptRmse).sum  / perSpec.length
    val avgChamferL1 = perSpec.map(_._2.chamferL1).sum / perSpec.length
    println(f"    Avg: surf_mean=$avgSurf%.3f HD95=$avgHd95%.3f HD=$avgHd%.3f ptpt_RMSE=$avgPtptRmse%.3f chamfer_L1=$avgChamferL1%.3f")
    println(s"    Saved per_specimen_metrics.csv")

    // -----------------------------------------------------------------------
    // B. Compactness
    // -----------------------------------------------------------------------
    println("  [B] SSM compactness...")
    val eigenvalues: IndexedSeq[Double] = model.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val totalVar                        = eigenvalues.sum
    val cumVarPct: IndexedSeq[Double]   = eigenvalues.scanLeft(0.0)(_ + _).tail.map(_ / totalVar * 100.0)

    Using.resource(new PrintWriter(new File(outDir, "compactness.csv"))) { w =>
      w.println("mode,eigenvalue,variance_pct,cumulative_pct")
      eigenvalues.zip(cumVarPct).zipWithIndex.foreach { case ((ev, cv), i) =>
        w.println(f"${i + 1},$ev%.6f,${ev / totalVar * 100.0}%.4f,$cv%.4f")
      }
    }

    def modesFor(pct: Double): Int = {
      val idx = cumVarPct.indexWhere(_ >= pct)
      if (idx < 0) eigenvalues.length else idx + 1
    }
    val (m90, m95, m99) = (modesFor(90.0), modesFor(95.0), modesFor(99.0))
    println(f"    90%%→$m90 modes   95%%→$m95 modes   99%%→$m99 modes  (${eigenvalues.length} modes total)")
    println(s"    Saved compactness.csv")

    // -----------------------------------------------------------------------
    // C. Leave-one-out generalization
    // -----------------------------------------------------------------------
    println(s"  [C] Leave-one-out generalization (${ids.length} iterations)...")

    val looResults: IndexedSeq[(String, Metrics.FullDistStats)] = ids.zipWithIndex.map {
      case (heldOutId, idx) =>
        println(f"    LOO ${idx + 1}/${ids.length}: $heldOutId")
        val holdOut      = pass2Pairs(heldOutId)
        val trainData    = ids.filterNot(_ == heldOutId).flatMap(pass2Pairs.get).toIndexedSeq
        val looModel     = FullPipeline.buildSSM(ssmReference, trainData)

        // Fit: observations = every corresponding vertex, noise variance = 1 mm²
        val noiseModel = MultivariateNormalDistribution(
          DenseVector.zeros[Double](3),
          DenseMatrix.eye[Double](3) * 1.0
        )
        val correspondences: IndexedSeq[(PointId, Point[_3D], MultivariateNormalDistribution)] =
          ssmReference.pointSet.pointsWithId.map { case (_, pid) =>
            (pid, holdOut.pointSet.point(pid), noiseModel)
          }.toIndexedSeq

        val fittedModel = looModel.posterior(correspondences)
        val fittedMesh  = fittedModel.mean

        val stats = Metrics.allStats(fittedMesh, holdOut)
        println(f"      ${stats.render}")
        heldOutId -> stats
    }

    Using.resource(new PrintWriter(new File(outDir, "loo_generalization.csv"))) { w =>
      w.println(s"id,${looResults.head._2.header}")
      looResults.foreach { case (id, s) => w.println(s"$id,${s.csv}") }
    }

    val genSurf     = looResults.map(_._2.surfMean).sum / looResults.length
    val genHd95     = looResults.map(_._2.surfHd95).sum / looResults.length
    val genPtptRmse = looResults.map(_._2.ptptRmse).sum / looResults.length
    println(f"    LOO avg: surf_mean=$genSurf%.3f HD95=$genHd95%.3f ptpt_RMSE=$genPtptRmse%.3f mm")
    println(s"    Saved loo_generalization.csv")

    // -----------------------------------------------------------------------
    // D. Specificity
    // -----------------------------------------------------------------------
    println("  [D] Specificity (200 random samples)...")
    val specScores: IndexedSeq[Double] = (0 until 200).map { _ =>
      val sample = model.sample()
      trainMeshes.map(t => Metrics.symmetric(sample, t).mean).min
    }
    val specMean = specScores.sum / specScores.length
    val specStd  = math.sqrt(specScores.map(d => (d - specMean) * (d - specMean)).sum / specScores.length)

    Using.resource(new PrintWriter(new File(outDir, "specificity.csv"))) { w =>
      w.println("sample_idx,dist_to_nearest_training_mm")
      specScores.zipWithIndex.foreach { case (d, i) => w.println(f"${i + 1},$d%.4f") }
    }
    println(f"    Specificity: mean=$specMean%.3f  std=$specStd%.3f mm")
    println(s"    Saved specificity.csv")

    // -----------------------------------------------------------------------
    // E. Reference stability
    // -----------------------------------------------------------------------
    println("  [E] Reference stability...")
    println(f"    Seed → Mean-1:   ${stability1.render}")
    println(f"    Mean-1 → Mean-2: ${stability2.render}")

    // -----------------------------------------------------------------------
    // F. Bilateral consistency (within-subject in Pass-2 space)
    // -----------------------------------------------------------------------
    println("  [F] Bilateral consistency (within-subject, Pass-2)...")
    val bySubject = ids.groupBy(ScapulaData.subjectKey).toIndexedSeq.sortBy(_._1)
    val bilateralRows: IndexedSeq[(String, Metrics.SurfaceStats, Double)] =
      bySubject.flatMap { case (subj, sIds) =>
        val leftOpt  = sIds.find(!_.endsWith("_R")).flatMap(pass2Pairs.get)
        val rightOpt = sIds.find(_.endsWith("_R")).flatMap(pass2Pairs.get)
        for (l <- leftOpt; r <- rightOpt) yield {
          val surf     = Metrics.symmetric(l, r)
          val ptpt     = Metrics.correspondingDistances(l, r)
          val ptptRmse = math.sqrt(ptpt.map(x => x * x).sum / ptpt.length)
          println(f"    $subj%-32s ptpt_RMSE=$ptptRmse%.3f  ${surf.render}")
          (subj, surf, ptptRmse)
        }
      }

    if (bilateralRows.nonEmpty) {
      Using.resource(new PrintWriter(new File(outDir, "bilateral_consistency.csv"))) { w =>
        w.println("subject,surf_mean,surf_rms,surf_hd95,surf_hd,ptpt_rmse")
        bilateralRows.foreach { case (s, surf, r) =>
          w.println(f"$s,${surf.mean}%.4f,${surf.rms}%.4f,${surf.hd95}%.4f,${surf.hd}%.4f,$r%.4f")
        }
      }
      val bilMean = bilateralRows.map(_._2.mean).sum / bilateralRows.length
      println(f"    Within-subject avg surf_mean=$bilMean%.3f mm")
      println(s"    Saved bilateral_consistency.csv")
    }

    // -----------------------------------------------------------------------
    // Summary text report
    // -----------------------------------------------------------------------
    Using.resource(new PrintWriter(new File(outDir, "ssm_evaluation_summary.txt"))) { w =>
      def line(s: String = ""): Unit = w.println(s)
      def sep():                Unit = line("=" * 80)
      sep()
      line("SCAPULA SSM EVALUATION SUMMARY")
      sep()
      line(s"Specimens:              ${ids.length}")
      line(s"Reference vertices:     ${ssmReference.pointSet.numberOfPoints}")
      line(s"SSM modes (rank):       ${model.rank}")
      line()
      line("--- COMPACTNESS ---")
      line(f"90%% variance: $m90 modes")
      line(f"95%% variance: $m95 modes")
      line(f"99%% variance: $m99 modes")
      line()
      line("--- LOO GENERALIZATION ---")
      line(f"Avg surface mean:   $genSurf%.3f mm")
      line(f"Avg HD95:           $genHd95%.3f mm")
      line(f"Avg pt-pt RMSE:     $genPtptRmse%.3f mm")
      line()
      line("--- SPECIFICITY ---")
      line(f"Mean:  $specMean%.3f mm")
      line(f"Std:   $specStd%.3f mm")
      line()
      line("--- REFERENCE STABILITY ---")
      line(f"Seed → Mean-1:   mean=${stability1.mean}%.3f  HD=${stability1.hd}%.3f mm")
      line(f"Mean-1 → Mean-2: mean=${stability2.mean}%.3f  HD=${stability2.hd}%.3f mm")
      if (stability2.mean < 1.0)
        line("  => Converged: Mean-2 is within 1 mm of Mean-1. Reference bias removed.")
      else
        line("  => Consider a third pass if Mean-2 differs significantly from Mean-1.")
      line()
      line("--- PER-SPECIMEN REGISTRATION QUALITY ---")
      val hdr = f"${"ID"}%-42s ${"s_mean"}%7s ${"s_rms"}%7s ${"HD95"}%7s ${"HD"}%7s ${"ptpt_rms"}%9s ${"cham_L1"}%8s"
      line(hdr); line("-" * 93)
      perSpec.foreach { case (id, s) =>
        line(f"$id%-42s ${s.surfMean}%7.3f ${s.surfRms}%7.3f ${s.surfHd95}%7.3f ${s.surfHd}%7.3f ${s.ptptRmse}%9.3f ${s.chamferL1}%8.4f")
      }
      sep()
    }
    println(s"  Summary saved: ${new File(outDir, "ssm_evaluation_summary.txt").getPath}")
  }
}
