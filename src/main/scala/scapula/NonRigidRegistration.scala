package scapula

import scalismo.common.{Field, PointId}
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.*
import scalismo.mesh.*
import scalismo.numerics.UniformMeshSampler3D
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.*
import scalismo.utils.Random

/**
 * Non-rigid registration via GP ICP.
 *
 * A Gaussian deformation prior is placed over the reference mesh. Each target is registered by
 * iteratively finding closest-point correspondences and computing the GP posterior (which is a
 * globally-smooth deformation field, so the result can never fold or self-intersect for
 * reasonable kernel parameters).
 *
 * Pass 1  registers every specimen to an arbitrary reference (the first left scapula).
 * Pass N  (N > 1)  rebuilds the reference as the Procrustes mean of the previous pass,
 *         decimated back to modelResolution vertices, then re-registers. This removes the
 *         reference-bias artefact that Pass 1 always introduces.
 */
object NonRigidRegistration {

  /**
   * Build the GP deformation prior for a reference mesh.
   *
   * A diagonal Gaussian kernel is used: each component of the 3-D deformation field has the
   * same scalar covariance, which is a sum of two Gaussians at different scales so the model
   * can capture both global pose residuals and fine local shape variation.
   */
  def buildModel(reference: TriangleMesh[_3D],
                 relativeTolerance: Double = Config.gpRelativeTolerance,
                 maxRank: Int = Config.gpMaxRank
  )(implicit rng: Random): PointDistributionModel[_3D, TriangleMesh] = {

    val scalarKernel: PDKernel[_3D] =
      GaussianKernel[_3D](80.0) * 25.0 +
        GaussianKernel[_3D](20.0) * 5.0

    val k: MatrixValuedPDKernel[_3D] = DiagonalKernel3D(scalarKernel, 3)

    val zeroMean = Field(EuclideanSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, k)

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference.pointSet,
      gp,
      relativeTolerance,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]](),
      maxNumberOfEigenpairs = maxRank
    )

    PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
  }

  /**
   * Fit the model to one target mesh using iterated closest-point posterior updates.
   *
   * sigma2   — observation noise variance (mm²). A larger value gives a smoother, more
   *            globally-regularised fit; smaller follows local surface detail more closely.
   * trimFraction — fraction of worst correspondences dropped per iteration.
   */
  def fitToTarget(model: PointDistributionModel[_3D, TriangleMesh],
                  target: TriangleMesh[_3D],
                  iterations: Int = Config.icpIterations,
                  numSamplePoints: Int = 500,
                  sigma2: Double = 9.0,
                  trimFraction: Double = 0.15
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val targetOps = target.operations
    val sampleIds = RigidAlign.uniformIds(model.reference, numSamplePoints)
    val noiseDistribution = MultivariateNormalDistribution(
      EuclideanVector.zeros[_3D],
      SquareMatrix.eye[_3D] * sigma2
    )

    var currentModel = model

    for (_ <- 0 until iterations) {
      val mean = currentModel.mean

      val pairs = sampleIds.map { id =>
        val moved   = mean.pointSet.point(id)
        val closest = targetOps.closestPointOnSurface(moved).point
        (id, moved, closest)
      }

      val keep = math.max(10, (pairs.length * (1.0 - trimFraction)).toInt)
      val trimmed = pairs.sortBy { case (_, m, t) => (m - t).norm2 }.take(keep)

      val observations = trimmed.map { case (id, _, closestPt) =>
        (id, closestPt, noiseDistribution)
      }

      currentModel = currentModel.posterior(observations)
    }

    currentModel.mean
  }

  /**
   * Register all specimens (left + mirrored-right) to `reference`, returning the registered
   * meshes with their original model IDs.
   *
   * After rigid pre-alignment each mesh is fitted with the GP ICP above.
   */
  def registerAll(reference: TriangleMesh[_3D],
                  referenceLms: IndexedSeq[Landmark[_3D]],
                  specimens: IndexedSeq[ScapulaData.Specimen],
                  landmarks: Map[String, IndexedSeq[Landmark[_3D]]],
                  model: PointDistributionModel[_3D, TriangleMesh],
                  onProgress: (String, TriangleMesh[_3D]) => Unit = (_, _) => ()
  )(implicit rng: Random): IndexedSeq[(String, TriangleMesh[_3D])] = {

    val tasks: IndexedSeq[(String, () => TriangleMesh[_3D])] =
      specimens.flatMap { s =>
        landmarks.get(s.modelId).map { lms =>
          val (mesh, _) = if (s.isRight) {
            val m = ScapulaData.mirrorMesh(ScapulaData.loadMesh(s.file))
            val l = ScapulaData.mirrorLandmarks(lms)
            RigidAlign.landmarkThenIcp(m, l, reference, referenceLms)
          } else {
            RigidAlign.landmarkThenIcp(ScapulaData.loadMesh(s.file), lms, reference, referenceLms)
          }

          val label = if (s.isRight) s.modelId + "_mirrored" else s.modelId

          label -> (() => fitToTarget(model, mesh))
        }
      }

    tasks.map { case (label, compute) =>
      val registered = compute()
      onProgress(label, registered)
      label -> registered
    }
  }

  /**
   * Rebuild a reference as the mean of the registered shapes from a previous pass, then
   * decimate to modelResolution vertices. Returns the new reference mesh.
   */
  def rebuildReference(registered: IndexedSeq[(String, TriangleMesh[_3D])],
                       modelResolution: Int = Config.modelResolution
  ): TriangleMesh[_3D] = {
    require(registered.nonEmpty)
    val n = registered.head._2.pointSet.numberOfPoints
    require(registered.forall(_._2.pointSet.numberOfPoints == n),
      "All registered meshes must be in correspondence (same point count)")

    val meanPts = (0 until n).map { i =>
      val sum = registered.foldLeft(EuclideanVector.zeros[_3D]) { case (acc, (_, m)) =>
        acc + m.pointSet.point(i).toVector
      }
      Point3D(sum.x / registered.size, sum.y / registered.size, sum.z / registered.size)
    }

    val meanMesh = TriangleMesh3D(meanPts, registered.head._2.triangulation)

    // Decimate to desired vertex count (approximate).
    meanMesh.operations.decimate(targetedNumberOfVertices = modelResolution)
  }
}
