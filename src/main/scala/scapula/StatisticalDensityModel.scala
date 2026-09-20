package scapula

import breeze.linalg.{DenseMatrix, DenseVector, svd}
import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel

import java.io.{File, PrintWriter}

/**
 * Statistical Density Model (SDM).
 *
 * After non-rigid registration establishes point-to-point correspondence across N specimens,
 * each specimen contributes a vector of HU values of length P (the number of correspondence
 * points).  PCA on the N × P matrix yields the principal modes of HU variation — exactly the
 * same mathematics as an SSM, applied to intensity rather than geometry.
 *
 * The joint shape+density model stacks [shape_coords (3P), HU_values (P)] per specimen and
 * runs a single PCA, capturing the correlation between local geometry and local bone density.
 */
object StatisticalDensityModel {

  // ---------------------------------------------------------------------------
  // Data container
  // ---------------------------------------------------------------------------

  /**
   * @param ids        specimen identifiers, length N
   * @param huMatrix   N × P matrix; row i = HU values for specimen i at P correspondence points
   * @param meanHU     mean HU field, length P
   * @param components K × P matrix; row k = k-th principal component (unit vector)
   * @param variances  explained variance per component, length K
   */
  final case class DensityModel(
    ids: IndexedSeq[String],
    huMatrix: DenseMatrix[Float],  // N × P
    meanHU: DenseVector[Float],    // P
    components: DenseMatrix[Float], // K × P
    variances: DenseVector[Float]  // K
  ) {
    val n: Int = ids.length
    val p: Int = meanHU.length
    val k: Int = variances.length

    /** Project specimen i back to scores on the first `nComp` components. */
    def scores(i: Int, nComp: Int = k): DenseVector[Float] = {
      val row = huMatrix(i, ::).t - meanHU
      DenseVector((0 until math.min(nComp, k)).map { c =>
        components(c, ::).t.dot(row)
      }.toArray)
    }

    /** Reconstruct an HU field from latent scores. */
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

  /**
   * Build a density model from a collection of aligned HU fields.
   *
   * @param ids     specimen identifiers
   * @param fields  per-specimen HU vectors, each of length P (all must be identical length)
   * @param nComp   maximum number of principal components to retain (default: keep all < N)
   */
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

    // Mean-centre
    val meanD = DenseVector.zeros[Double](pPoints)
    for (j <- 0 until pPoints) meanD(j) = mat(::, j).sum / nSpecimens
    for (i <- 0 until nSpecimens) mat(i, ::) -= meanD.t

    // Thin SVD: work in the smaller (N × N) space when N < P
    val svdResult = svd.reduced(mat)
    val k         = math.min(math.min(nComp, nSpecimens - 1), pPoints)

    val components = DenseMatrix.zeros[Float](k, pPoints)
    val variances  = DenseVector.zeros[Float](k)
    for (c <- 0 until k) {
      components(c, ::) := svdResult.Vt(c, ::).map(_.toFloat)
      variances(c)       = (svdResult.S(c) * svdResult.S(c) / (nSpecimens - 1)).toFloat
    }

    // Re-assemble raw HU matrix (before mean-subtraction) for score computation
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
  // Joint shape + density model
  // ---------------------------------------------------------------------------

  /**
   * Concatenate per-specimen shape coordinate vectors and HU vectors, then do PCA.
   *
   * The shape and HU channels are optionally scaled so their variances are commensurate
   * (by default shape coords are divided by their std and HU by theirs).
   *
   * Returns (components K × (3P + P), variances K).
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

    // Global std scaling so neither channel dominates
    val std = breeze.stats.stddev(mat(::, *)).t
    for (j <- 0 until d) if (std(j) > 1e-8) mat(::, j) /= std(j)

    // Mean-centre
    val mean = DenseVector.zeros[Double](d)
    for (j <- 0 until d) { mean(j) = mat(::, j).sum / n; mat(::, j) -= mean(j) }

    val svdR  = svd.reduced(mat)
    val k     = math.min(math.min(nComp, n - 1), d)
    val comps = DenseMatrix.zeros[Float](k, d)
    val vars  = DenseVector.zeros[Float](k)
    for (c <- 0 until k) {
      comps(c, ::) := svdR.Vt(c, ::).map(_.toFloat)
      vars(c)       = (svdR.S(c) * svdR.S(c) / (n - 1)).toFloat
    }
    (comps, vars)
  }

  // ---------------------------------------------------------------------------
  // Reporting
  // ---------------------------------------------------------------------------

  def printSummary(model: DensityModel, nComp: Int = 10): Unit = {
    println(s"\n[SDM] Statistical Density Model  (N=${model.n}  P=${model.p}  K=${model.k})")
    println(f"  Mean HU: min=${model.meanHU.min}%6.1f  max=${model.meanHU.max}%6.1f  " +
            f"avg=${model.meanHU.sum / model.p}%6.1f")
    println(s"  Component  variance    cumulative")
    var cum = 0.0
    val total = model.variances.toArray.sum
    (0 until math.min(nComp, model.k)).foreach { c =>
      cum += model.variances(c)
      println(f"    PC${c + 1}%2d    ${model.variances(c)}%8.2f    ${cum / total * 100}%5.1f%%")
    }
  }

  /** Write mean HU field and first few components to a CSV for external inspection. */
  def exportCsv(model: DensityModel, outFile: File, nComp: Int = 5): Unit = {
    val pw = new PrintWriter(outFile)
    try {
      val header = "vertex_idx,mean_HU," + (1 to math.min(nComp, model.k)).map(c => s"PC$c").mkString(",")
      pw.println(header)
      for (j <- 0 until model.p) {
        val row = Seq(j.toString, f"${model.meanHU(j)}%.2f") ++
          (0 until math.min(nComp, model.k)).map(c => f"${model.components(c, j)}%.6f")
        pw.println(row.mkString(","))
      }
    } finally pw.close()
    println(s"  Density model CSV written to: ${outFile.getAbsolutePath}")
  }
}
