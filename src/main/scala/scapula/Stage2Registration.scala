package scapula

import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.geometry.*
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import scala.util.Using

/**
 * STAGE 2 -- NON-RIGID REGISTRATION.
 *
 * Registers every specimen to a common reference using GP-ICP (Gaussian Process model fitting via
 * iterative closest points). Multi-pass: the reference is rebuilt as the mean shape after each pass,
 * removing reference-specimen bias from the final model.
 *
 * Config knobs (via environment variables, see ScapulaData.Config):
 *   SCAPULA_MODEL_RES       decimated vertex count for reference (default 5000)
 *   SCAPULA_ICP_ITERS       GP-ICP iterations per specimen (default 40)
 *   SCAPULA_REFINE_PASSES   multi-pass count (default 2)
 *   SCAPULA_GP_TOL          Cholesky relative tolerance (default 0.01)
 *   SCAPULA_GP_MAX_RANK     hard rank cap on GP basis (default 250)
 */
object Stage2Registration {

  // ── types ─────────────────────────────────────────────────────────────────

  final case class RegistrationResult(
    specimen: ScapulaData.Specimen,
    registered: TriangleMesh[_3D],
    stats: Metrics.SurfaceStats
  )

  // ── GP model ──────────────────────────────────────────────────────────────

  /**
   * Build a low-rank GP deformation model on `reference`. The kernel is a sum of three Gaussians at
   * different scales (coarse global pose, medium regional, fine local), which lets the non-rigid ICP
   * converge from large misalignments without over-fitting noise at the end.
   */
  def buildGpModel(reference: TriangleMesh[_3D])(implicit rng: Random): PointDistributionModel[_3D, TriangleMesh] = {
    val kernel =
      DiagonalKernel[_3D](GaussianKernel[_3D](80.0) * 50.0, 3) +
      DiagonalKernel[_3D](GaussianKernel[_3D](40.0) * 20.0, 3) +
      DiagonalKernel[_3D](GaussianKernel[_3D](10.0) *  5.0, 3)

    val zeroMean = scalismo.common.Field(
      scalismo.common.EuclideanSpace[_3D],
      (_: Point[_3D]) => EuclideanVector.zeros[_3D]
    )
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, kernel)

    val lowRankGp = LowRankGaussianProcess.approximateGPCholesky(
      reference.pointSet,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )

    println(s"  GP rank: ${lowRankGp.rank}  (set SCAPULA_GP_TOL higher to reduce)")
    PointDistributionModel[_3D, TriangleMesh](reference, lowRankGp)
  }

  // ── GP-ICP ────────────────────────────────────────────────────────────────

  /**
   * Fit `model` to `target` via ICP-style posterior updates. At each iteration we find the closest
   * point on `target` for a uniform sample of reference points, treat those as noisy observations of
   * the deformation field, and take the GP posterior mean as the new current shape.
   */
  def gpIcp(
    model: PointDistributionModel[_3D, TriangleMesh],
    target: TriangleMesh[_3D],
    iterations: Int,
    numCorrespondences: Int = 500,
    noiseVariance: Double = 1.0
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val targetOps = target.operations
    val refIds = RigidAlign.uniformIds(model.reference, numCorrespondences)
    var current = model.mean

    for (_ <- 0 until iterations) {
      val observations = refIds.map { id =>
        val currPt = current.pointSet.point(id)
        val targetPt = targetOps.closestPointOnSurface(currPt).point
        (id, targetPt)
      }
      current = model.posterior(observations, noiseVariance).mean
    }
    current
  }

  // ── one registration pass ─────────────────────────────────────────────────

  /**
   * Register every specimen to `reference` and return the registered meshes together with per-specimen
   * surface-distance statistics (registered vs target). Writes progress to stdout.
   */
  def registerAll(
    specimens: IndexedSeq[ScapulaData.Specimen],
    landmarks: Map[String, IndexedSeq[Landmark[_3D]]],
    reference: TriangleMesh[_3D],
    refLms: IndexedSeq[Landmark[_3D]]
  )(implicit rng: Random): IndexedSeq[RegistrationResult] = {

    val model = buildGpModel(reference)
    println(s"  GP model rank: ${model.gp.rank}  (reference vertices: ${reference.pointSet.numberOfPoints})")

    specimens.zipWithIndex.flatMap { case (spec, idx) =>
      landmarks.get(spec.modelId).map { lms =>
        val rawMesh = ScapulaData.loadMesh(spec.file)
        val (mesh, lmsAligned) =
          if (spec.isRight)
            (ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(lms))
          else
            (rawMesh, lms)

        // 1. Rigid pre-alignment (landmark + ICP) to bring the specimen into the reference frame.
        val (rigidAligned, _) = RigidAlign.landmarkThenIcp(mesh, lmsAligned, reference, refLms)

        // 2. Non-rigid GP-ICP.
        val fitted = gpIcp(model, rigidAligned, Config.icpIterations)

        val stats = Metrics.symmetric(fitted, rigidAligned)
        val flag =
          if (ScapulaData.signedVolume(rawMesh) < 0) "  <-- ORIENTATION PROBLEM"
          else if (stats.hd > 20.0) "  <-- high HD (check acromion/coracoid)"
          else ""
        println(f"  [${idx + 1}%2d/${specimens.length}] ${spec.modelId}%-30s ${stats.render}$flag")
        RegistrationResult(spec, fitted, stats)
      }
    }
  }

  // ── mean shape ───────────────────────────────────────────────────────────

  def meanShape(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    require(meshes.nonEmpty)
    val n = meshes.head.pointSet.numberOfPoints
    val summedPts = Array.fill(n)(EuclideanVector.zeros[_3D])
    meshes.foreach { m =>
      m.pointSet.pointIds.foreach { id =>
        summedPts(id.id) = summedPts(id.id) + m.pointSet.point(id).toVector
      }
    }
    val meanPts = summedPts.map(v => Point3D(v.x / meshes.length, v.y / meshes.length, v.z / meshes.length))
    TriangleMesh3D(meanPts.toIndexedSeq, meshes.head.triangulation)
  }

  // ── main ──────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val outDir = Config.outDir
    outDir.mkdirs()

    println(s"Data directory : ${dir.getAbsolutePath}")
    println(s"Output directory: ${outDir.getAbsolutePath}")

    val csv = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val allSpecimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))

    println(s"${allSpecimens.length} specimens with landmarks found")

    // ── pick and decimate reference ──────────────────────────────────────
    val refSpec = allSpecimens.head  // alphabetically first left-side specimen
    val refRaw = ScapulaData.loadMesh(refSpec.file)
    val refLms = landmarks(refSpec.modelId)
    val reference0 = refRaw.operations.decimate(Config.modelResolution)
    println(s"Reference: ${refSpec.modelId}  (${reference0.pointSet.numberOfPoints} vertices after decimation)")

    // ── multi-pass registration ───────────────────────────────────────────
    var reference = reference0
    var results = IndexedSeq.empty[RegistrationResult]

    for (pass <- 1 to Config.refinePasses) {
      println()
      println(s"=== Registration pass $pass / ${Config.refinePasses} ===")
      results = registerAll(allSpecimens, landmarks, reference, refLms)

      if (pass < Config.refinePasses) {
        reference = meanShape(results.map(_.registered))
        println(s"  Reference updated to mean of ${results.length} registered shapes.")
      }
    }

    // ── save registered meshes ────────────────────────────────────────────
    val regDir = new File(outDir, "registered")
    regDir.mkdirs()
    results.foreach { r =>
      val f = new File(regDir, s"${r.specimen.modelId}.stl")
      MeshIO.writeMesh(r.registered, f).get
    }
    println(s"\nSaved ${results.length} registered meshes to ${regDir.getAbsolutePath}")

    // ── save metrics CSV ──────────────────────────────────────────────────
    val csvOut = new File(outDir, "registration_metrics.csv")
    Using.resource(new PrintWriter(csvOut)) { pw =>
      pw.println("modelId,isRight,subject,mean_mm,rms_mm,hd95_mm,hd_mm")
      results.foreach { r =>
        val s = r.stats
        pw.println(f"${r.specimen.modelId},${r.specimen.isRight},${r.specimen.subject},${s.mean}%.4f,${s.rms}%.4f,${s.hd95}%.4f,${s.hd}%.4f")
      }
    }
    println(s"Metrics written to ${csvOut.getAbsolutePath}")
  }
}
