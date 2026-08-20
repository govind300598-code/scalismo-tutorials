package scapula

import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D, TriangleList}
import scalismo.numerics.{PivotedCholesky, UniformMeshSampler3D}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.utils.Random

import java.io.File

/**
 * Stage 2 — Non-rigid registration.
 *
 * For each dataset directory this stage:
 *   1. Picks a reference (first left specimen, decimated to modelResolution vertices).
 *   2. Rigidly aligns every specimen (landmark Procrustes + trimmed ICP).
 *   3. Runs GP non-rigid ICP against the reference to establish dense correspondence.
 *   4. Optionally rebuilds the reference as the mean of pass-1 registrations and repeats
 *      (Config.refinePasses — removes reference-specimen bias).
 *   5. Writes registered meshes and the final reference to <outDir>/registered/.
 *   6. Returns the reference and the registered meshes (for Stage 3).
 */
object Stage2NonRigid {

  final case class NonRigidResult(
    reference:    TriangleMesh[_3D],
    registered:   IndexedSeq[(String, TriangleMesh[_3D])],
    registeredDir: File
  )

  // ---------------------------------------------------------------------------
  // Multi-scale Gaussian kernel. Scapula bounding box ≈ 130×100×50 mm.
  // Large scale captures global shape; medium captures acromion/glenoid;
  // small captures fine surface ridges.
  // ---------------------------------------------------------------------------
  private def buildKernel(): DiagonalKernel[_3D] = {
    val k = GaussianKernel[_3D](sigma = 80.0) * 50.0 +
            GaussianKernel[_3D](sigma = 30.0) * 20.0 +
            GaussianKernel[_3D](sigma = 10.0) *  5.0
    DiagonalKernel[_3D](k, 3)
  }

  private def buildGpmm(reference: TriangleMesh[_3D])(implicit rng: Random): StatisticalMeshModel = {
    val gp      = GaussianProcess[_3D, EuclideanVector[_3D]](buildKernel())
    val lowRank = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      Config.gpRelativeTolerance,
      NearestNeighborInterpolator3D()
    )
    println(f"  GPMM rank: ${lowRank.rank}")
    StatisticalMeshModel(reference, lowRank)
  }

  // ---------------------------------------------------------------------------
  // Non-rigid ICP: iteratively find closest-point correspondences on the target
  // and update the GP posterior. Noise annealing: large sigma2 early (global
  // deformation) → small sigma2 late (local detail).
  // ---------------------------------------------------------------------------
  private def nonRigidICP(
    reference:  TriangleMesh[_3D],
    gpmm:       StatisticalMeshModel,
    target:     TriangleMesh[_3D],
    iterations: Int
  )(implicit rng: Random): TriangleMesh[_3D] = {

    val targetOps  = target.operations
    val numSamples = math.min(1500, reference.pointSet.numberOfPoints)

    val sampleIds = UniformMeshSampler3D(reference, numSamples)
      .sample()
      .map { case (pt, _) => reference.pointSet.findClosestPoint(pt).id }
      .distinct
      .take(numSamples)

    var model = gpmm

    for (i <- 0 until iterations) {
      val fraction = i.toDouble / math.max(iterations - 1, 1)
      val sigma2   = math.pow(6.0 - 4.5 * fraction, 2)   // 36 → 2.25 mm²

      val currentMean = model.mean
      val observations: IndexedSeq[(scalismo.common.PointId, Point[_3D])] =
        sampleIds.map { ptId =>
          val p = currentMean.pointSet.point(ptId)
          val q = targetOps.closestPointOnSurface(p).point
          (ptId, q)
        }

      model = model.posterior(observations, sigma2)
    }

    model.mean
  }

  // ---------------------------------------------------------------------------
  // Register one specimen: rigid (landmark + ICP) then non-rigid.
  // ---------------------------------------------------------------------------
  private def registerOne(
    spec:      ScapulaData.Specimen,
    landmarks: Map[String, IndexedSeq[Landmark[_3D]]],
    reference: TriangleMesh[_3D],
    refLms:    IndexedSeq[Landmark[_3D]],
    gpmm:      StatisticalMeshModel
  )(implicit rng: Random): (TriangleMesh[_3D], Metrics.SurfaceStats, Metrics.SurfaceStats) = {

    val rawLms  = landmarks(spec.modelId)
    val rawMesh = ScapulaData.loadMesh(spec.file)

    val (workMesh, workLms) =
      if (spec.isRight)
        (ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
      else (rawMesh, rawLms)

    val (rigidMesh, _) = RigidAlign.landmarkThenIcp(workMesh, workLms, reference, refLms,
                                                     icpIterations = Config.icpIterations)
    val metricsRigid = Metrics.symmetric(rigidMesh, reference)

    val registered   = nonRigidICP(reference, gpmm, rigidMesh, Config.icpIterations)
    val metricsNonRig = Metrics.symmetric(registered, reference)

    (registered, metricsRigid, metricsNonRig)
  }

  // ---------------------------------------------------------------------------
  // Public entry point.
  // ---------------------------------------------------------------------------
  def run(dataDir: File, outDir: File)(implicit rng: Random): NonRigidResult = {
    val regDir = new File(outDir, "registered")
    regDir.mkdirs()

    val csv                            = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _)     = ScapulaData.readLandmarkCsv(csv)
    println(s"  CSV: ${csv.getName} | ${landmarks.size} entries | header-resolved=$fromHeader")

    val allSpecimens = ScapulaData.specimens(dataDir).filter(s => landmarks.contains(s.modelId))
    if (allSpecimens.isEmpty)
      throw new RuntimeException(s"No STL files with landmark rows found in ${dataDir.getPath}")

    val refSpec = allSpecimens.find(!_.isRight).getOrElse(allSpecimens.head)
    val refLms  = landmarks(refSpec.modelId)
    val refRaw  = ScapulaData.loadMesh(refSpec.file)
    var reference = refRaw.operations.decimate(Config.modelResolution)
    println(f"  Reference: ${refSpec.modelId} | raw=${refRaw.pointSet.numberOfPoints} | " +
            f"decimated=${reference.pointSet.numberOfPoints}")
    MeshIO.writeMesh(reference, new File(regDir, "reference.stl")).get

    val targets = allSpecimens.filterNot(_.modelId == refSpec.modelId)

    var registeredMeshes: IndexedSeq[(String, TriangleMesh[_3D])] = IndexedSeq.empty

    for (pass <- 1 to Config.refinePasses) {
      println(s"\n  [Pass $pass/${Config.refinePasses}] Building GPMM…")
      val gpmm = buildGpmm(reference)

      println(s"  Registering ${targets.length} specimens…")
      registeredMeshes = targets.zipWithIndex.map { case (spec, idx) =>
        val (reg, mR, mNR) = registerOne(spec, landmarks, reference, refLms, gpmm)
        println(f"    [${idx + 1}/${targets.length}] ${spec.modelId}%-34s | " +
                f"rigid ${mR.render} | non-rigid ${mNR.render}")
        MeshIO.writeMesh(reg, new File(regDir, s"registered_${spec.modelId}.stl")).get
        (spec.modelId, reg)
      }

      // Include the reference itself (no registration needed).
      registeredMeshes = registeredMeshes :+ (refSpec.modelId -> reference)

      if (pass < Config.refinePasses) {
        println(s"  Rebuilding mean reference for pass ${pass + 1}…")
        val allReg = registeredMeshes
        val meanPts = reference.pointSet.pointIds.toIndexedSeq.map { ptId =>
          val vecs = allReg.map(_._2.pointSet.point(ptId).toVector)
          val sum  = vecs.foldLeft(EuclideanVector.zeros[_3D])(_ + _)
          (sum * (1.0 / vecs.length)).toPoint
        }
        reference = TriangleMesh3D(meanPts, reference.triangulation)
        MeshIO.writeMesh(reference, new File(regDir, s"reference_pass${pass + 1}.stl")).get
      }
    }

    println(s"\n  Registered ${registeredMeshes.length} meshes → ${regDir.getAbsolutePath}")
    NonRigidResult(reference, registeredMeshes, regDir)
  }
}
