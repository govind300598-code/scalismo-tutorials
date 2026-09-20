package scapula

import breeze.linalg.{DenseMatrix, DenseVector, sum => bsum, min => bmin, max => bmax, svd}
import breeze.stats.stddev
import scalismo.mesh.TriangleMesh
import scalismo.geometry._3D

import java.io.{File, PrintWriter}

/**
 * Statistical Density Model (SDM) — built on the same mathematics as Cootes et al.'s
 * Active Appearance Model (AAM), applied to Hounsfield-Unit fields instead of grey-level
 * texture.
 *
 * Shape model:   s = s̄  + P_s · b_s
 * Density model: g = ḡ  + P_g · b_g
 * Joint model:   c = [W_s · b_s ; b_g],  then PCA → Q · b = c
 *
 * where W_s scales shape parameters so their variance matches the density variance.
 */
object StatisticalDensityModel {

  // ---------------------------------------------------------------------------
  // Data container
  // ---------------------------------------------------------------------------

  /**
   * @param ids        specimen identifiers, length N
   * @param huMatrix   N × P matrix; row i = HU values at P correspondence points
   * @param meanHU     mean HU field, length P
   * @param components K × P matrix; row k = k-th PC of HU variation
   * @param variances  explained variance per component, length K
   */
  final case class DensityModel(
    ids: IndexedSeq[String],
    huMatrix: DenseMatrix[Float],   // N × P
    meanHU: DenseVector[Float],     // P
    components: DenseMatrix[Float], // K × P
    variances: DenseVector[Float]   // K
  ) {
    val n: Int = ids.length
    val p: Int = meanHU.length
    val k: Int = variances.length

    def scores(i: Int, nComp: Int = k): DenseVector[Float] = {
      val row = huMatrix(i, ::).t - meanHU
      DenseVector((0 until math.min(nComp, k)).map { c =>
        components(c, ::).t.dot(row)
      }.toArray)
    }

    def reconstruct(s: DenseVector[Float]): DenseVector[Float] = {
      val nComp = s.length
      val field  = meanHU.copy
      (0 until nComp).foreach { c => field += components(c, ::).t *:* s(c) }
      field
    }

    def explainedVarianceRatio(nComp: Int): Double = {
      val total = variances.toArray.sum
      if (total == 0.0) 0.0
      else variances.toArray.take(nComp).sum / total
    }
  }

  // ---------------------------------------------------------------------------
  // Builder
  // ---------------------------------------------------------------------------

  def build(ids: IndexedSeq[String],
            fields: IndexedSeq[IndexedSeq[Float]],
            nComp: Int = Int.MaxValue): DensityModel = {
    require(ids.length == fields.length, "ids and fields must have the same length")
    require(fields.map(_.length).distinct.length == 1, "all HU fields must have the same length")
    val nSpecimens = ids.length
    val pPoints    = fields.head.length

    // Assemble N × P matrix
    val mat = DenseMatrix.zeros[Double](nSpecimens, pPoints)
    for ((f, i) <- fields.zipWithIndex; (v, j) <- f.zipWithIndex) mat(i, j) = v.toDouble

    // Mean-centre column by column
    val meanD = DenseVector.zeros[Double](pPoints)
    for (j <- 0 until pPoints) meanD(j) = bsum(mat(::, j)) / nSpecimens
    for (i <- 0 until nSpecimens) mat(i, ::) -= meanD.t

    // Thin SVD
    val svdResult = svd.reduced(mat)
    val k         = math.min(math.min(nComp, nSpecimens - 1), pPoints)

    val components = DenseMatrix.zeros[Float](k, pPoints)
    val variances  = DenseVector.zeros[Float](k)
    for (c <- 0 until k) {
      val row = svdResult.Vt(c, ::).t           // DenseVector[Double]
      components(c, ::) := row.map(_.toFloat).t  // assign as row
      variances(c) = (svdResult.S(c) * svdResult.S(c) / (nSpecimens - 1)).toFloat
    }

    // Raw HU matrix (before mean-subtraction) for score computation
    val rawMat = DenseMatrix.zeros[Float](nSpecimens, pPoints)
    for ((f, i) <- fields.zipWithIndex; (v, j) <- f.zipWithIndex) rawMat(i, j) = v

    DensityModel(
      ids        = ids,
      huMatrix   = rawMat,
      meanHU     = meanD.map(_.toFloat),
      components = components,
      variances  = variances
    )
  }

  // ---------------------------------------------------------------------------
  // Joint shape + density model  (AAM-style combined model)
  // ---------------------------------------------------------------------------

  /**
   * Concatenate per-specimen shape coordinate vectors (3P_shape) and HU vectors (P_density),
   * then run PCA on the joint matrix — capturing shape/density co-variation.
   *
   * Channels are scaled by their respective pooled standard deviation so neither
   * dominates the PCA (W_s scaling from Cootes et al.).
   */
  def buildJoint(ids: IndexedSeq[String],
                 shapeMeshes: IndexedSeq[TriangleMesh[_3D]],
                 huFields: IndexedSeq[IndexedSeq[Float]],
                 nComp: Int = Int.MaxValue): (DenseMatrix[Float], DenseVector[Float]) = {
    require(ids.length == shapeMeshes.length && ids.length == huFields.length)
    val n  = ids.length
    val p  = huFields.head.length
    val d  = 3 * shapeMeshes.head.pointSet.numberOfPoints + p

    val mat = DenseMatrix.zeros[Double](n, d)
    for (i <- 0 until n) {
      val pts = shapeMeshes(i).pointSet.points.toIndexedSeq
      var col = 0
      pts.foreach { pt =>
        mat(i, col) = pt.x; col += 1
        mat(i, col) = pt.y; col += 1
        mat(i, col) = pt.z; col += 1
      }
      huFields(i).foreach { v => mat(i, col) = v.toDouble; col += 1 }
    }

    // W_s scaling: each column divided by its standard deviation
    val std = DenseVector((0 until d).map { j =>
      val s = stddev(mat(::, j))
      if (s > 1e-8) s else 1.0
    }.toArray)
    for (j <- 0 until d) mat(::, j) /= std(j)

    // Mean-centre
    val mean = DenseVector.zeros[Double](d)
    for (j <- 0 until d) {
      mean(j) = bsum(mat(::, j)) / n
      mat(::, j) -= mean(j)
    }

    val svdR = svd.reduced(mat)
    val k    = math.min(math.min(nComp, n - 1), d)
    val comps = DenseMatrix.zeros[Float](k, d)
    val vars  = DenseVector.zeros[Float](k)
    for (c <- 0 until k) {
      val row = svdR.Vt(c, ::).t
      comps(c, ::) := row.map(_.toFloat).t
      vars(c) = (svdR.S(c) * svdR.S(c) / (n - 1)).toFloat
    }
    (comps, vars)
  }

  // ---------------------------------------------------------------------------
  // Reporting
  // ---------------------------------------------------------------------------

  def printSummary(model: DensityModel, nComp: Int = 10): Unit = {
    println(s"\n[SDM] Statistical Density Model  (N=${model.n}  P=${model.p}  K=${model.k})")
    println(f"  Mean HU: min=${bmin(model.meanHU)}%6.1f  max=${bmax(model.meanHU)}%6.1f  " +
            f"avg=${bsum(model.meanHU) / model.p}%6.1f")
    println(s"  Component  variance    cumulative")
    var cum = 0.0
    val total = model.variances.toArray.sum
    (0 until math.min(nComp, model.k)).foreach { c =>
      cum += model.variances(c)
      println(f"    PC${c + 1}%2d    ${model.variances(c)}%8.2f    ${cum / total * 100}%5.1f%%")
    }
  }

  def exportCsv(model: DensityModel, outFile: File, nComp: Int = 5): Unit = {
    val pw = new PrintWriter(outFile)
    try {
      val nc = math.min(nComp, model.k)
      pw.println("vertex_idx,mean_HU," + (1 to nc).map(c => s"PC$c").mkString(","))
      for (j <- 0 until model.p) {
        val row = Seq(j.toString, f"${model.meanHU(j)}%.2f") ++
          (0 until nc).map(c => f"${model.components(c, j)}%.6f")
        pw.println(row.mkString(","))
      }
    } finally pw.close()
    println(s"  Density model CSV written to: ${outFile.getAbsolutePath}")
  }
}
