package scapula

import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.utils.Random

object SSMValidation {

  /** Full surface-distance statistics returned per specimen. */
  final case class SpecimenStats(
    msd:    Double,   // mean surface distance  (= Chamfer / 2 each side)
    rmse:   Double,   // root-mean-square error
    hd95:   Double,   // 95th-percentile Hausdorff distance
    hd:     Double    // maximum Hausdorff distance
  )

  def compactness(ssm: PointDistributionModel[_3D, TriangleMesh]): IndexedSeq[(Int, Double)] = {
    val evs   = ssm.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val total = evs.sum
    var cumul = 0.0
    evs.zipWithIndex.map { case (ev, i) =>
      cumul += ev / total * 100.0
      (i + 1, cumul)
    }
  }

  /**
   * GENERALIZATION — LOO reconstruction error (full surface stats).
   * Returns MSD, RMSE, HD95, HD per left-out specimen.
   * Symmetric surface distances (both directions) are used.
   */
  def generalization(
    reference:  TriangleMesh[_3D],
    registered: IndexedSeq[TriangleMesh[_3D]]
  ): IndexedSeq[SpecimenStats] = {
    registered.zipWithIndex.map { case (left, i) =>
      val train     = registered.indices.filterNot(_ == i).map(registered)
      val looSSM    = SSMBuilder.buildSSM(reference, train)
      val projected = looSSM.project(left)
      val s         = Metrics.symmetric(left, projected)
      SpecimenStats(s.mean, s.rms, s.hd95, s.hd)
    }
  }

  /**
   * SPECIFICITY — how plausible are random SSM samples?
   * Returns per-sample distance to nearest training shape.
   */
  def specificity(
    ssm:        PointDistributionModel[_3D, TriangleMesh],
    registered: IndexedSeq[TriangleMesh[_3D]],
    nSamples:   Int = 50
  )(implicit rng: Random): IndexedSeq[Double] = {
    (1 to nSamples).map { _ =>
      val sample = ssm.sample()
      registered.map { train =>
        val d = Metrics.surfaceDistances(sample, train)
        d.sum / d.length
      }.min
    }.toIndexedSeq
  }

  /**
   * REGISTRATION QUALITY — symmetric surface metrics between each
   * registered mesh and the SSM mean shape.
   * MSD ≈ Chamfer distance (symmetric mean), RMSE, HD95, Hausdorff.
   */
  def registrationQuality(
    meanShape:  TriangleMesh[_3D],
    registered: IndexedSeq[TriangleMesh[_3D]]
  ): IndexedSeq[SpecimenStats] = {
    registered.map { reg =>
      val s = Metrics.symmetric(reg, meanShape)
      SpecimenStats(s.mean, s.rms, s.hd95, s.hd)
    }
  }

  private def statsSummary(stats: IndexedSeq[SpecimenStats]): String = {
    val msds  = stats.map(_.msd);  val rmses = stats.map(_.rmse)
    val hd95s = stats.map(_.hd95); val hds   = stats.map(_.hd)
    f"  MSD    mean=${msds.sum/msds.length}%6.3f  sd=${stdDev(msds)}%5.3f  max=${msds.max}%6.3f  mm\n" +
    f"  RMSE   mean=${rmses.sum/rmses.length}%6.3f  sd=${stdDev(rmses)}%5.3f  max=${rmses.max}%6.3f  mm\n" +
    f"  HD95   mean=${hd95s.sum/hd95s.length}%6.3f  sd=${stdDev(hd95s)}%5.3f  max=${hd95s.max}%6.3f  mm\n" +
    f"  HD     mean=${hds.sum/hds.length}%6.3f  sd=${stdDev(hds)}%5.3f  max=${hds.max}%6.3f  mm"
  }

  private def stdDev(xs: IndexedSeq[Double]): Double = {
    val m = xs.sum / xs.length
    math.sqrt(xs.map(x => (x - m) * (x - m)).sum / xs.length)
  }

  def printReport(
    ssm:        PointDistributionModel[_3D, TriangleMesh],
    reference:  TriangleMesh[_3D],
    registered: IndexedSeq[TriangleMesh[_3D]]
  )(implicit rng: Random): Unit = {
    val sep = "=" * 70
    println(s"\n$sep")
    println("SSM VALIDATION REPORT")
    println(sep)

    // ── Compactness ────────────────────────────────────────────────────────
    println("\n[COMPACTNESS] Cumulative variance explained by first k modes")
    println(f"  ${"Modes"}%-6s  ${"Cumul Var%"}%10s")
    val comp = compactness(ssm)
    Seq(1, 2, 3, 5, 10, 15, 20).filter(_ <= ssm.rank).foreach { k =>
      val (_, cv) = comp(k - 1)
      println(f"  $k%-6d  $cv%10.2f%%")
    }
    val modes90 = comp.find(_._2 >= 90.0).map(_._1).getOrElse(ssm.rank)
    val modes95 = comp.find(_._2 >= 95.0).map(_._1).getOrElse(ssm.rank)
    println(f"  Modes for 90%% variance: $modes90")
    println(f"  Modes for 95%% variance: $modes95")

    // ── Registration quality ───────────────────────────────────────────────
    println("\n[REGISTRATION QUALITY] Registered meshes vs SSM mean shape")
    println("  (Chamfer=MSD symmetric, RMSE, HD95, Hausdorff per specimen)")
    println(f"  ${"Spec"}%-5s  ${"MSD(mm)"}%9s  ${"RMSE(mm)"}%10s  ${"HD95(mm)"}%10s  ${"HD(mm)"}%8s")
    val regQ = registrationQuality(ssm.mean, registered)
    regQ.zipWithIndex.foreach { case (s, i) =>
      println(f"  ${i+1}%-5d  ${s.msd}%9.3f  ${s.rmse}%10.3f  ${s.hd95}%10.3f  ${s.hd}%8.3f")
    }
    println(statsSummary(regQ))

    // ── Generalization ─────────────────────────────────────────────────────
    println("\n[GENERALIZATION] Leave-one-out reconstruction error")
    println("  (lower = model generalises well to unseen shapes)")
    println(f"  ${"Spec"}%-5s  ${"MSD(mm)"}%9s  ${"RMSE(mm)"}%10s  ${"HD95(mm)"}%10s  ${"HD(mm)"}%8s")
    val genStats = try {
      generalization(reference, registered)
    } catch {
      case e: Exception =>
        println(s"  [warn] Generalization skipped: ${e.getMessage}")
        IndexedSeq.empty[SpecimenStats]
    }
    if (genStats.nonEmpty) {
      genStats.zipWithIndex.foreach { case (s, i) =>
        println(f"  ${i+1}%-5d  ${s.msd}%9.3f  ${s.rmse}%10.3f  ${s.hd95}%10.3f  ${s.hd}%8.3f")
      }
      println(statsSummary(genStats))
    }

    // ── Specificity ────────────────────────────────────────────────────────
    println("\n[SPECIFICITY] Distance from random SSM samples to nearest training shape")
    println("  (lower = model only produces plausible bone shapes)")
    val specDists = specificity(ssm, registered)
    val specMean  = specDists.sum / specDists.length
    val specSD    = stdDev(specDists)
    println(f"  Mean ± SD : $specMean%6.3f ± $specSD%5.3f mm  (${specDists.length} samples)")
    println(f"  Min / Max : ${specDists.min}%6.3f / ${specDists.max}%6.3f mm")

    println(s"\n$sep\n")
  }
}
