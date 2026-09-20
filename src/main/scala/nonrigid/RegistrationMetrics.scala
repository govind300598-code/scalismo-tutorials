package nonrigid

import scalismo.geometry.{Landmark, _3D}
import scalismo.mesh.{MeshMetrics, TriangleMesh}

/**
 * Registration-quality and correspondence-quality metrics.
 *
 * All distance values are in the same unit as the mesh coordinates (mm for clinical bone data).
 *
 * Metric glossary
 * ---------------
 * directedMean   : mean{ d(v, surface(target)) | v in fitted }             — how close is the
 *                  deformed reference to the target surface (fitted → target, directed)
 * directedRMSE   : RMS of the same directed distances
 * symmetricMean  : mean over the union of (fitted→target) and (target→fitted) distances
 * symmetricRMSE  : RMS  over the same union
 * hd95           : 95th percentile of the symmetric distance pool — robust Hausdorff
 * hd             : full (100th-percentile) Hausdorff, via scalismo MeshMetrics.hausdorffDistance
 * chamfer        : directed mean(fitted→target) + directed mean(target→fitted)
 *                  (standard Chamfer distance; equals symmetricMean only when |fitted|=|target|)
 * lmResidual     : mean Euclidean distance between deformed reference landmarks and the
 *                  corresponding aligned target landmarks — direct correspondence-quality check;
 *                  Double.NaN when landmark data are not available
 */
object RegistrationMetrics {

  final case class AllMetrics(
    directedMean:  Double,
    directedRMSE:  Double,
    symmetricMean: Double,
    symmetricRMSE: Double,
    hd95:          Double,
    hd:            Double,
    chamfer:       Double,
    lmResidual:    Double
  ) {
    def render: String = {
      val lmPart = if (lmResidual.isNaN) "" else f"  LMresid=$lmResidual%5.2f"
      f"dir: mean=$directedMean%5.2f rms=$directedRMSE%5.2f" +
      f"  sym: mean=$symmetricMean%5.2f rms=$symmetricRMSE%5.2f" +
      f"  HD95=$hd95%5.2f  HD=$hd%6.2f  Chamfer=$chamfer%5.2f$lmPart  (mm)"
    }
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  /** Directed: for every vertex of `from`, the distance to the closest surface point of `to`. */
  private def directedDist(from: TriangleMesh[_3D], to: TriangleMesh[_3D]): IndexedSeq[Double] = {
    val ops = to.operations
    from.pointSet.points.map(p => (p - ops.closestPointOnSurface(p).point).norm).toIndexedSeq
  }

  private def mean(v: IndexedSeq[Double]): Double = v.sum / v.length

  private def rms(v: IndexedSeq[Double]): Double =
    math.sqrt(v.map(x => x * x).sum / v.length)

  /** p in [0,1]. */
  private def percentile(v: IndexedSeq[Double], p: Double): Double = {
    val s = v.sorted
    val i = math.min(s.length - 1, math.max(0, math.ceil(p * s.length).toInt - 1))
    s(i)
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Compute all metrics.
   *
   * @param fitted      deformed reference mesh produced by non-rigid registration
   * @param target      rigidly aligned target mesh
   * @param fittedLms   optional: reference landmarks carried through the final deformation field
   * @param targetLms   optional: rigidly aligned target landmarks (same IDs as fittedLms)
   */
  def compute(
    fitted:    TriangleMesh[_3D],
    target:    TriangleMesh[_3D],
    fittedLms: Option[IndexedSeq[Landmark[_3D]]] = None,
    targetLms: Option[IndexedSeq[Landmark[_3D]]] = None
  ): AllMetrics = {

    val dFT = directedDist(fitted, target)   // fitted → target
    val dTF = directedDist(target, fitted)   // target → fitted
    val sym = dFT ++ dTF

    // Chamfer: MeshMetrics.avgDistance is directed (from → to) and uses the same
    // closest-point logic, so we use it for consistency with scalismo internals.
    val chamfer = MeshMetrics.avgDistance(fitted, target) +
                  MeshMetrics.avgDistance(target, fitted)

    val lmRes: Double = (fittedLms, targetLms) match {
      case (Some(fl), Some(tl)) if fl.nonEmpty && fl.length == tl.length =>
        val byId = tl.map(l => l.id -> l.point).toMap
        val dists = fl.flatMap(l => byId.get(l.id).map(q => (l.point - q).norm))
        if (dists.isEmpty) Double.NaN else mean(dists.toIndexedSeq)
      case _ => Double.NaN
    }

    AllMetrics(
      directedMean  = mean(dFT),
      directedRMSE  = rms(dFT),
      symmetricMean = mean(sym),
      symmetricRMSE = rms(sym),
      hd95          = percentile(sym, 0.95),
      hd            = MeshMetrics.hausdorffDistance(fitted, target),
      chamfer       = chamfer,
      lmResidual    = lmRes
    )
  }

  /** Convenience: print a two-row table (before / after non-rigid). */
  def printComparison(
    label:     String,
    before:    AllMetrics,
    after:     AllMetrics
  ): Unit = {
    println(s"  $label")
    println(s"    before non-rigid : ${before.render}")
    println(s"    after  non-rigid : ${after.render}")
  }
}
