package scapula

import scalismo.geometry.*
import scalismo.io.StatisticalModelIO
import scalismo.mesh.TriangleMesh
import scalismo.numerics.PivotedCholesky
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection

import java.io.File

/**
 * Stage 3 — Build a Statistical Shape Model via PCA on registered displacement fields.
 *
 * All registered meshes must share the same topology as the reference (same vertex count
 * and connectivity — vertex i on every mesh corresponds to vertex i on every other mesh),
 * which is guaranteed when the meshes come from Stage 2 non-rigid registration.
 *
 * Uses PointDistributionModel (the preferred Scalismo 0.92+ API) so the model can be
 * saved/loaded via StatisticalModelIO.
 */
object Stage3SSM {

  type SSMModel = PointDistributionModel[_3D, TriangleMesh]

  final case class SSMResult(
    model:              SSMModel,
    modelFile:          File,
    numModes:           Int,
    cumulativeVariance: IndexedSeq[Double]
  )

  def run(
    reference:  TriangleMesh[_3D],
    registered: IndexedSeq[(String, TriangleMesh[_3D])],
    outDir:     File,
    tag:        String = ""
  ): SSMResult = {

    val nShapes = registered.length
    println(s"\n  Building SSM from $nShapes shapes…")

    val nRef = reference.pointSet.numberOfPoints
    val bad  = registered.filter(_._2.pointSet.numberOfPoints != nRef)
    if (bad.nonEmpty)
      throw new RuntimeException(
        s"${bad.length} registered meshes do not have $nRef points: ${bad.map(_._1).take(5).mkString(", ")}"
      )

    val meshes = registered.map(_._2)
    val dc     = DataCollection.fromTriangleMesh3DSequence(reference, meshes)
    val model  = PointDistributionModel.createUsingPCA(dc, PivotedCholesky.RelativeTolerance(0.01))

    val eigenvalues = model.gp.klBasis.map(_.eigenvalue).toIndexedSeq
    val totalVar    = eigenvalues.sum
    val cumVar      = eigenvalues.scanLeft(0.0)(_ + _).tail.map(_ / totalVar).toIndexedSeq
    val numModes    = eigenvalues.length

    println(f"  Modes: $numModes   Total variance: $totalVar%.1f mm²")
    println("  Cumulative variance:")
    Seq(0.50, 0.75, 0.90, 0.95, 0.99).foreach { t =>
      val k = cumVar.indexWhere(_ >= t)
      if (k >= 0) println(f"    ≥${(t * 100).toInt}%% → ${k + 1} modes")
    }
    val show = math.min(10, numModes)
    println(s"  First $show modes:")
    eigenvalues.take(show).zipWithIndex.foreach { case (ev, i) =>
      println(f"    mode ${i + 1}%2d  stddev=${math.sqrt(ev)}%6.2f mm  cumVar=${cumVar(i) * 100}%5.1f%%")
    }

    val suffix = if (tag.nonEmpty) s"_$tag" else ""
    val mFile  = new File(outDir, s"scapula_ssm$suffix.h5")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, mFile).get
    println(s"  Model saved → ${mFile.getAbsolutePath}")

    SSMResult(model, mFile, numModes, cumVar)
  }
}
