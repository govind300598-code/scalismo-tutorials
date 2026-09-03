package scapula

import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.io.MeshIO

import java.io.{File, PrintWriter}

/** Surface-to-surface comparisons between consecutive population means. */
object Comparison {

  def ensureDir(d: File): File = { d.mkdirs(); d }

  final case class MeanComparison(
    label: String,
    meanDist: Double,
    rmsDist: Double,
    hd95: Double,
    hd: Double,
    chamfer: Double,
    ptpRms: Option[Double]
  ) {
    def render: String =
      f"$label: mean=$meanDist%5.3f  rms=$rmsDist%5.3f  HD95=$hd95%5.3f  HD=$hd%6.3f  Chamfer=$chamfer%5.3f" +
        ptpRms.map(r => f"  PtP-RMS=$r%5.3f").getOrElse("") + "  (mm)"

    def tableRow: String =
      f"| $label%-15s | $meanDist%13.3f | $rmsDist%3.3f | $hd95%4.3f | $chamfer%7.3f |"
  }

  /**
   * Compare two meshes surface-to-surface.
   *
   * If `a` and `b` have the same number of points AND share a reference topology
   * (they are in point-to-point correspondence), `ptpRms` is also computed.
   */
  def compare(a: TriangleMesh[_3D], b: TriangleMesh[_3D], label: String): MeanComparison = {
    val stats = Metrics.symmetric(a, b)

    // Symmetric Chamfer distance = average of (mean a→b) and (mean b→a)
    val dab     = Metrics.surfaceDistances(a, b)
    val dba     = Metrics.surfaceDistances(b, a)
    val chamfer = (dab.sum / dab.length + dba.sum / dba.length) / 2.0

    val ptpRms =
      if (a.pointSet.numberOfPoints == b.pointSet.numberOfPoints) {
        val d = Metrics.correspondingDistances(a, b)
        Some(math.sqrt(d.map(x => x * x).sum / d.length))
      } else None

    MeanComparison(label, stats.mean, stats.rms, stats.hd95, stats.hd, chamfer, ptpRms)
  }

  /**
   * Save a per-vertex surface-distance scalar field as a VTK file
   * (distance from each vertex of `from` to the nearest point on `to`).
   * This can be loaded in ParaView or Scalismo-UI for colour-mapped visualisation.
   */
  def saveSurfaceDistanceVtk(
    from: TriangleMesh[_3D],
    to: TriangleMesh[_3D],
    file: File
  ): Unit = {
    file.getParentFile.mkdirs()
    val dists = Metrics.surfaceDistances(from, to)
    val nPts  = from.pointSet.numberOfPoints
    val nTris = from.triangulation.triangles.length

    val pw = new PrintWriter(file)
    pw.println("# vtk DataFile Version 3.0")
    pw.println("Surface distance map")
    pw.println("ASCII")
    pw.println("DATASET POLYDATA")
    pw.println(s"POINTS $nPts float")
    from.pointSet.points.foreach { p =>
      pw.println(f"${p.x}%.6f ${p.y}%.6f ${p.z}%.6f")
    }
    pw.println(s"POLYGONS $nTris ${nTris * 4}")
    from.triangulation.triangles.foreach { t =>
      pw.println(s"3 ${t.ptId1.id} ${t.ptId2.id} ${t.ptId3.id}")
    }
    pw.println(s"POINT_DATA $nPts")
    pw.println("SCALARS surface_distance float 1")
    pw.println("LOOKUP_TABLE default")
    dists.foreach(d => pw.println(f"$d%.6f"))
    pw.close()
    println(s"  VTK distance map → ${file.getPath}")
  }

  /**
   * Run all Mean1↔Mean2, Mean2↔Mean3, Mean3↔Mean4 comparisons.
   * Saves metrics CSV and VTK distance maps.
   */
  def runAllMeanComparisons(
    means: IndexedSeq[TriangleMesh[_3D]], // means(0)=Mean1, ..., means(3)=Mean4
    compDir: File
  ): IndexedSeq[MeanComparison] = {
    require(means.length == 4, "Expected exactly 4 means (Mean1..Mean4)")

    val pairs = IndexedSeq(
      (means(0), means(1), "Mean1_vs_Mean2"),
      (means(1), means(2), "Mean2_vs_Mean3"),
      (means(2), means(3), "Mean3_vs_Mean4")
    )

    val results = pairs.map { case (a, b, label) =>
      val dir = ensureDir(new File(compDir, label))
      val result = compare(a, b, label)
      println(s"  ${result.render}")

      // Save VTK distance maps (bidirectional)
      val figDir = ensureDir(new File(dir, "figures"))
      saveSurfaceDistanceVtk(a, b, new File(figDir, s"${label}_AtoB.vtk"))
      saveSurfaceDistanceVtk(b, a, new File(figDir, s"${label}_BtoA.vtk"))

      // Save CSV row
      val csv = new File(dir, s"${label}_metrics.csv")
      val pw  = new PrintWriter(csv)
      pw.println("label,mean_mm,rms_mm,hd95_mm,hd_mm,chamfer_mm,ptp_rms_mm")
      pw.println(
        s"${result.label},${result.meanDist},${result.rmsDist},${result.hd95},${result.hd},${result.chamfer}," +
          result.ptpRms.map(_.toString).getOrElse("NA")
      )
      pw.close()

      result
    }

    // Write convergence summary table
    writeConvergenceTable(results, new File(compDir, "convergence_summary.csv"))
    results
  }

  def writeConvergenceTable(comparisons: IndexedSeq[MeanComparison], file: File): Unit = {
    file.getParentFile.mkdirs()
    val pw = new PrintWriter(file)
    pw.println("comparison,mean_mm,rms_mm,hd95_mm,hd_mm,chamfer_mm")
    comparisons.foreach { c =>
      pw.println(s"${c.label},${c.meanDist},${c.rmsDist},${c.hd95},${c.hd},${c.chamfer}")
    }
    pw.close()
    println(s"  Convergence table → ${file.getPath}")
  }

  /** Print the convergence table and assess whether distances are decreasing. */
  def assessConvergence(comparisons: IndexedSeq[MeanComparison]): Unit = {
    println("\n=== TEMPLATE CONVERGENCE ===")
    println("| Comparison      | Mean distance | RMS   | HD95  | Chamfer |")
    println("|-----------------|-------------:|------:|------:|--------:|")
    comparisons.foreach { c => println(c.tableRow) }
    println()

    val means = comparisons.map(_.meanDist)
    val decreasing = means.sliding(2).forall { w => w(1) < w(0) }
    if (means.length >= 2) {
      if (decreasing)
        println("CONVERGENCE: Mean template distance is DECREASING across iterations (good).")
      else
        println("WARNING: Template distance is NOT monotonically decreasing – inspect manually.")

      val last = means.last
      val first = means.head
      if (last < first * 0.5)
        println(f"Strong convergence: final change ($last%.3f mm) < 50%% of initial ($first%.3f mm).")
      else
        println(f"Weak convergence: final change ($last%.3f mm) still large relative to initial ($first%.3f mm).")
    }
  }
}
