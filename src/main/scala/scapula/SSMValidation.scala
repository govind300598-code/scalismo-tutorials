package scapula

import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.utils.Random

object SSMValidation {

  /**
   * COMPACTNESS — cumulative variance explained by the first k modes.
   * A good SSM captures 90%+ of variance in very few modes.
   */
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
   * GENERALIZATION — leave-one-out reconstruction error.
   *
   * For each registered shape i:
   *   1. Build a PCA model from the remaining N-1 shapes (using the same reference).
   *   2. Project shape i onto that model (find the closest point in model space).
   *   3. Measure mean surface distance between the projection and shape i.
   *
   * Lower = better. A good SSM generalizes well to unseen shapes.
   */
  def generalization(
    reference:   TriangleMesh[_3D],
    registered:  IndexedSeq[TriangleMesh[_3D]]
  ): IndexedSeq[Double] = {
    val n = registered.length
    registered.zipWithIndex.map { case (left, i) =>
      val train = registered.indices.filterNot(_ == i).map(registered)
      val looSSM = SSMBuilder.buildSSM(reference, train)
      // Project left-out shape: find best-fit coefficients
      val projected = looSSM.project(left)
      Metrics.surfaceDistances(left, projected).sum / left.pointSet.numberOfPoints
    }
  }

  /**
   * SPECIFICITY — how anatomically plausible are random samples?
   *
   * For each random sample from the SSM, find its nearest training shape
   * (minimum mean surface distance). Average over nSamples.
   *
   * Lower = better. A high value means the model produces shapes that don't
   * look like any real bone in the training set.
   */
  def specificity(
    ssm:        PointDistributionModel[_3D, TriangleMesh],
    registered: IndexedSeq[TriangleMesh[_3D]],
    nSamples:   Int = 50
  )(implicit rng: Random): Double = {
    (1 to nSamples).map { _ =>
      val sample = ssm.sample()
      registered.map { train =>
        val d = Metrics.surfaceDistances(sample, train)
        d.sum / d.length
      }.min
    }.sum / nSamples
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

    // ── Generalization ─────────────────────────────────────────────────────
    println("\n[GENERALIZATION] Leave-one-out reconstruction error (mm)")
    println("  (lower = model generalises well to unseen shapes)")
    val genErrors = generalization(reference, registered)
    genErrors.zipWithIndex.foreach { case (err, i) =>
      println(f"  specimen ${i + 1}%2d : $err%6.3f mm")
    }
    val meanGen = genErrors.sum / genErrors.length
    val maxGen  = genErrors.max
    println(f"  Mean : $meanGen%6.3f mm")
    println(f"  Max  : $maxGen%6.3f mm")

    // ── Specificity ────────────────────────────────────────────────────────
    println("\n[SPECIFICITY] Mean distance from random samples to nearest training shape (mm)")
    println("  (lower = model only produces plausible bone shapes)")
    val spec = specificity(ssm, registered)
    println(f"  Specificity (50 samples): $spec%6.3f mm")

    println(s"\n$sep\n")
  }
}
