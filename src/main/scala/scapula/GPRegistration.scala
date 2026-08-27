package scapula

import scalismo.common.*
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.*
import scalismo.mesh.*
import scalismo.statisticalmodel.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.utils.Random
import java.io.File
import scala.util.Try

/**
 * GP-ICP non-rigid registration using a zero-mean isotropic Gaussian kernel prior.
 *
 * Algorithm:
 *   1. Build a low-rank GP on the reference mesh (pivoted Cholesky approximation).
 *   2. Iterate: find trimmed closest-point correspondences on the target surface,
 *      then replace the running model with its posterior given those observations.
 *   3. Return the posterior mean as the registered (deformed reference) mesh.
 *
 * This registers the REFERENCE to the TARGET (output is in reference coordinate frame,
 * which is what you need for SSM building). Always rigidly-align the target to the
 * reference first (RigidAlign.landmarkThenIcp) before calling register().
 */
object GPRegistration {

  /**
   * Build a PointDistributionModel from a zero-mean isotropic Gaussian kernel GP.
   *
   * @param reference  Reference mesh (all registered shapes and the SSM share this topology)
   * @param sigma      Kernel bandwidth (mm). Controls how far deformation correlation reaches.
   * @param scale      Kernel amplitude (mm). Controls maximum deformation magnitude.
   */
  def buildModel(
    reference:         TriangleMesh[_3D],
    sigma:             Double,
    scale:             Double,
    relativeTolerance: Double = Config.gpRelativeTolerance,
    maxRank:           Int    = Config.gpMaxRank
  ): PointDistributionModel[_3D, TriangleMesh] = {
    val zeroMean = Field[_3D, EuclideanVector[_3D]](RealSpace[_3D], _ => EuclideanVector3D(0, 0, 0))
    val kernel   = DiagonalKernel[_3D](GaussianKernel[_3D](sigma) * (scale * scale), 3)
    val gp       = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)
    val lowRank  = LowRankGaussianProcess.approximateGPCholesky(
      reference, gp, relativeTolerance,
      NearestNeighborInterpolator3D[EuclideanVector[_3D]]()
    )
    val truncated = if (lowRank.rank > maxRank) lowRank.truncate(maxRank) else lowRank
    println(s"  GP rank: ${truncated.rank}  (sigma=$sigma  scale=$scale)")
    PointDistributionModel[_3D, TriangleMesh](truncated)
  }

  /**
   * Iterative GP posterior (GP-ICP).
   *
   * @param target     Rigidly-aligned target mesh (in reference coordinate frame)
   * @param model      Prior GP model built on the same reference
   * @param iterations Number of ICP iterations
   * @param numCorr    Correspondences sampled per iteration (spatially uniform)
   * @param noise      Gaussian noise variance on observations (mm^2). Tune lower for tighter fit.
   * @param trimFrac   Fraction of worst correspondences to discard (handles missing anatomy)
   */
  def register(
    target:     TriangleMesh[_3D],
    model:      PointDistributionModel[_3D, TriangleMesh],
    iterations: Int    = Config.icpIterations,
    numCorr:    Int    = 1000,
    noise:      Double = 1.0,
    trimFrac:   Double = 0.10
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val targetOps = target.operations
    var m         = model

    for (it <- 0 until iterations) {
      val current = m.mean
      val ids     = RigidAlign.uniformIds(current, numCorr)

      val pairs = ids.map { id =>
        val p       = current.pointSet.point(id)
        val closest = targetOps.closestPointOnSurface(p).point
        (id, p, closest)
      }

      // Drop the worst trimFrac correspondences (outliers, non-homologous anatomy).
      val keep    = math.max(20, (pairs.length * (1.0 - trimFrac)).toInt)
      val trimmed = pairs.sortBy { case (_, p, c) => (p - c).norm }.take(keep)
      val obs     = trimmed.map { case (id, _, c) => (id, c) }

      m = m.posterior(obs, noise)

      if ((it + 1) % 10 == 0) {
        val d = Metrics.symmetric(m.mean, target)
        println(f"    iter ${it + 1}%3d: ${d.render}")
      }
    }
    m.mean
  }

  /** Save PDM to HDF5 (scalismo .h5 format). Prints a warning on failure rather than throwing. */
  def saveModel(model: PointDistributionModel[_3D, TriangleMesh], file: File): Unit = {
    file.getParentFile.mkdirs()
    Try(StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, file).get)
      .failed.foreach(e => println(s"  Warning: could not save model to ${file.getName}: $e"))
  }

  /** Save a TriangleMesh to VTK. Throws on IO failure. */
  def saveRegistered(mesh: TriangleMesh[_3D], file: File): Unit = {
    file.getParentFile.mkdirs()
    MeshIO.writeMesh(mesh, file).get
    println(s"  Saved: ${file.getPath}")
  }

  /**
   * Save the shape at ±stdev standard deviations along a single mode.
   * Useful for sanity-checking the GP model by visualising what each mode does.
   */
  def saveModeShape(
    model:   PointDistributionModel[_3D, TriangleMesh],
    modeIdx: Int,
    stdev:   Double,
    file:    File
  ): Unit = {
    // In scalismo PDM, instance(coefficients) where coefficients(k) = s gives
    // the shape at s standard deviations along mode k (already eigenvalue-normalised).
    val coeffs = breeze.linalg.DenseVector.zeros[Double](model.rank)
    if (modeIdx < model.rank) coeffs(modeIdx) = stdev
    saveRegistered(model.instance(coeffs), file)
  }

  /** Decimate a mesh to at most targetVertices, falling back to the original on error. */
  def decimateIfNeeded(mesh: TriangleMesh[_3D], targetVertices: Int): TriangleMesh[_3D] =
    if (mesh.pointSet.numberOfPoints <= targetVertices) mesh
    else
      Try(mesh.operations.decimate(targetVertices)).getOrElse {
        println(s"  Decimation not available; using ${mesh.pointSet.numberOfPoints} vertices")
        mesh
      }
}
