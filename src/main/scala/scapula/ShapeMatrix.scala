package scapula

import breeze.linalg.{sum, DenseMatrix, DenseVector, eigSym}
import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.utils.Random

/**
 * Point-distribution-model PCA, implemented directly on top of breeze rather than scalismo's own
 * StatisticalMeshModel, so that Stage 3's validation numbers rest on formulas that can be checked by hand instead of
 * on a specific library API surface.
 *
 * Every mesh handed to this object is assumed to already be in full point-to-point correspondence with the
 * `reference` used in Stage 2 (same vertex count, same vertex order) -- that is exactly what Stage 2's registration
 * guarantees, and is why meshes are exchanged between the stages as VTK, never STL.
 */
object ShapeMatrix {

  def toVector(mesh: TriangleMesh[_3D]): DenseVector[Double] = {
    val n = mesh.pointSet.numberOfPoints
    val v = DenseVector.zeros[Double](n * 3)
    var i = 0
    mesh.pointSet.points.foreach { p =>
      v(3 * i) = p.x
      v(3 * i + 1) = p.y
      v(3 * i + 2) = p.z
      i += 1
    }
    v
  }

  def toMesh(v: DenseVector[Double], reference: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val n = reference.pointSet.numberOfPoints
    require(v.length == n * 3, s"vector has length ${v.length}, expected ${n * 3} for a ${n}-vertex reference")
    val pts = (0 until n).map(i => Point3D(v(3 * i), v(3 * i + 1), v(3 * i + 2)))
    scalismo.mesh.TriangleMesh3D(pts, reference.triangulation)
  }

  /** Rows = specimens, columns = flattened (x0,y0,z0,x1,y1,z1,...) point coordinates. */
  def dataMatrix(meshes: IndexedSeq[TriangleMesh[_3D]]): DenseMatrix[Double] = dataMatrixFromVectors(meshes.map(toVector))

  def dataMatrixFromVectors(vectors: IndexedSeq[DenseVector[Double]]): DenseMatrix[Double] = {
    require(vectors.nonEmpty)
    val d = vectors.head.length
    val m = DenseMatrix.zeros[Double](vectors.length, d)
    vectors.zipWithIndex.foreach { case (v, i) => m(i, ::) := v.t }
    m
  }

  /**
   * `eigenvectors` is d x k (k = rank, at most n-1), columns are unit-norm modes of variation sorted by descending
   * eigenvalue; `eigenvalues(i)` is the variance (mm^2) explained by mode i.
   */
  final case class DualPCA(mean: DenseVector[Double], eigenvalues: DenseVector[Double], eigenvectors: DenseMatrix[Double])

  /**
   * PCA via the Gram-matrix ("snapshot") trick. With n specimens in a d ~ 15000-dimensional space (5000 vertices x 3),
   * the d x d covariance matrix is never formed; instead the n x n Gram matrix of centred samples is eigen-decomposed
   * (cheap, n is at most ~24 here), and each of its eigenvectors v_i is lifted back to a shape-space mode via
   *   u_i = Xc^T v_i / sqrt(lambda_i * (n-1))
   * which is a unit vector because ||Xc^T v_i||^2 = v_i^T Xc Xc^T v_i = (n-1) lambda_i (v_i^T v_i) = (n-1) lambda_i,
   * and shares eigenvalue lambda_i with the true covariance matrix C = Xc^T Xc / (n-1): for K v = lambda v (K = Xc Xc^T
   * / (n-1)) we get C (Xc^T v) = Xc^T (K v) = lambda (Xc^T v). This is the same identity scalismo's own PCA relies on;
   * it is re-derived here explicitly so the eigenvalues used for compactness/specificity below are self-consistent.
   */
  def fit(data: DenseMatrix[Double]): DualPCA = {
    val n = data.rows
    val d = data.cols
    require(n >= 2, "PCA needs at least 2 samples")

    val mean = DenseVector.zeros[Double](d)
    for (j <- 0 until d) mean(j) = sum(data(::, j)) / n.toDouble

    val centered = DenseMatrix.zeros[Double](n, d)
    for (i <- 0 until n) centered(i, ::) := (data(i, ::).t - mean).t

    val gram = (centered * centered.t) / (n - 1).toDouble // n x n, symmetric PSD
    val es = eigSym(gram)
    val evalsAsc = es.eigenvalues
    val evecsAsc = es.eigenvectors

    val order = Array.range(0, n).sortBy(i => -evalsAsc(i))
    val keep = order.filter(i => evalsAsc(i) > 1e-10)

    val eigenvalues = DenseVector(keep.map(evalsAsc(_)))
    val eigenvectors = DenseMatrix.zeros[Double](d, keep.length)
    keep.zipWithIndex.foreach { case (srcIdx, dstIdx) =>
      val v = evecsAsc(::, srcIdx)
      val u = centered.t * v
      val norm = math.sqrt(evalsAsc(srcIdx) * (n - 1))
      eigenvectors(::, dstIdx) := u / norm
    }
    DualPCA(mean, eigenvalues, eigenvectors)
  }

  /** Reconstruct `vector` using only the first `numModes` modes (least-squares projection onto the truncated basis). */
  def project(pca: DualPCA, vector: DenseVector[Double], numModes: Int): DenseVector[Double] = {
    val k = math.min(numModes, pca.eigenvectors.cols)
    val centered = vector - pca.mean
    var recon = pca.mean.copy
    for (i <- 0 until k) {
      val coeff = pca.eigenvectors(::, i).dot(centered)
      recon = recon + pca.eigenvectors(::, i) * coeff
    }
    recon
  }

  /** Draw one random instance from the model restricted to its first `numModes` modes: mean + sum(N(0,1)*sqrt(lambda_i)*u_i). */
  def sample(pca: DualPCA, numModes: Int)(implicit rng: Random): DenseVector[Double] = {
    val k = math.min(numModes, pca.eigenvectors.cols)
    var s = pca.mean.copy
    for (i <- 0 until k) {
      val coeff = rng.scalaRandom.nextGaussian() * math.sqrt(pca.eigenvalues(i))
      s = s + pca.eigenvectors(::, i) * coeff
    }
    s
  }

  /** RMS Euclidean point-to-point distance between two same-length flattened shape vectors. */
  def rmsPointDistance(a: DenseVector[Double], b: DenseVector[Double], numPoints: Int): Double = {
    val diff = a - b
    var acc = 0.0
    var p = 0
    while (p < numPoints) {
      val dx = diff(3 * p); val dy = diff(3 * p + 1); val dz = diff(3 * p + 2)
      acc += dx * dx + dy * dy + dz * dz
      p += 1
    }
    math.sqrt(acc / numPoints)
  }
}
