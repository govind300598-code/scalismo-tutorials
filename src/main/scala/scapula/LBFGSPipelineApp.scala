package scapula

import scalismo.common.{Field, RealSpace}
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, LBFGSOptimizer}
import scalismo.registration.{GaussianProcessTransformationSpace, L2Regularizer, MeanSquaresMetric, Registration}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random
import breeze.linalg.DenseVector

/**
 * Standalone LBFGS non-rigid registration pipeline for scapulae.
 *
 * Follows Dennis Madsen's approach (Scalismo forum, Jun 2024):
 *   kernel:  multi-scale Gaussian (75 mm + 30 mm + 15 mm) with amplitude 15/10/5
 *   approx:  Cholesky low-rank GP, NearestNeighborInterpolator3D, tolerance 0.01
 *   fitting: coarse-to-fine LBFGS on MeanSquares distance-image metric
 *
 * Run with:  sbt "runMain scapula.LBFGSPipelineApp"
 */
object LBFGSPipelineApp {

  // ── Registration parameters (coarse → fine) ─────────────────────────────────
  case class RegParams(regularizationWeight: Double, iterations: Int, sampledPoints: Int)

  val schedule: Seq[RegParams] = Seq(
    RegParams(regularizationWeight = 1e-1, iterations = 50,  sampledPoints = 500),
    RegParams(regularizationWeight = 1e-2, iterations = 50,  sampledPoints = 1000),
    RegParams(regularizationWeight = 1e-4, iterations = 100, sampledPoints = 2000),
    RegParams(regularizationWeight = 1e-6, iterations = 100, sampledPoints = 5000)
  )

  // ── Build multi-scale GP model ───────────────────────────────────────────────
  def buildGPModel(
    reference: TriangleMesh[_3D]
  ): (LowRankGaussianProcess[_3D, EuclideanVector[_3D]], PointDistributionModel[_3D, TriangleMesh]) = {

    // Scapula longest dimension ~150 mm
    // Kernel scales: 75 mm (global), 30 mm (regional), 15 mm (fine detail)
    // Amplitudes:    15,             10,                5
    val k = DiagonalKernel3D(
      GaussianKernel[_3D](75.0) * 15.0 +
      GaussianKernel[_3D](30.0) * 10.0 +
      GaussianKernel[_3D](15.0) *  5.0,
      outputDim = 3
    )

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
      Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D]),
      k
    )

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = 0.01,
      interpolator = NearestNeighborInterpolator3D[EuclideanVector[_3D]]()
    )

    val gpmm = PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP)
    println(s"  GP model rank = ${gpmm.rank}")
    (lowRankGP, gpmm)
  }

  // ── One LBFGS stage ──────────────────────────────────────────────────────────
  def lbfgsStage(
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D],
    coefficients: DenseVector[Double],
    p: RegParams
  ): DenseVector[Double] = {

    val space      = GaussianProcessTransformationSpace(lowRankGP)
    val metric     = MeanSquaresMetric(
      reference.operations.toDistanceImage,
      target.operations.toDistanceImage,
      space,
      FixedPointsUniformMeshSampler3D(reference, p.sampledPoints)
    )
    val result = Registration(metric, L2Regularizer(space), p.regularizationWeight, LBFGSOptimizer(p.iterations))
      .iterator(coefficients)
      .toSeq.last
    result.parameters
  }

  // ── Register one target to reference ────────────────────────────────────────
  def register(
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    gpmm: PointDistributionModel[_3D, TriangleMesh],
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D]
  ): TriangleMesh[_3D] = {

    val finalCoeffs = schedule.foldLeft(DenseVector.zeros[Double](lowRankGP.rank)) { (c, p) =>
      lbfgsStage(lowRankGP, reference, target, c, p)
    }
    gpmm.instance(finalCoeffs)
  }

  // ── Main ─────────────────────────────────────────────────────────────────────
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (lmMap, _, _) = ScapulaData.readLandmarkCsv(csv)

    val allSpecimens = ScapulaData.specimens(dir).filter(s => lmMap.contains(s.modelId))
    println(s"[info] ${allSpecimens.length} specimens with landmarks")

    case class Spec(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[scalismo.geometry.Landmark[_3D]])

    val specimens: IndexedSeq[Spec] = allSpecimens.map { s =>
      val raw = ScapulaData.loadMesh(s.file)
      val lms = lmMap(s.modelId)
      if (s.isRight)
        Spec(s.modelId + "_mirrored", ScapulaData.mirrorMesh(raw), ScapulaData.mirrorLandmarks(lms))
      else
        Spec(s.modelId, raw, lms)
    }

    val ui = ScalismoUI("Scapula LBFGS Pipeline")
    Thread.sleep(2000)

    // ── G01: Raw input ────────────────────────────────────────────────────────
    val g01 = ui.createGroup("G01_RawInput")
    specimens.foreach(s => try { ui.show(g01, s.mesh, s.id) } catch { case _: Exception => })

    // ── G02: Reference ────────────────────────────────────────────────────────
    val ref = specimens.head
    val refDecimated = ref.mesh.operations.decimate(Config.modelResolution)
    println(s"[info] Reference: ${ref.id}  decimated → ${refDecimated.pointSet.numberOfPoints} vertices")
    val g02 = ui.createGroup("G02_Reference")
    ui.show(g02, refDecimated, ref.id)

    // ── G03: Landmark aligned ─────────────────────────────────────────────────
    println("[G03] Landmark Procrustes alignment")
    val lmAligned: IndexedSeq[Spec] = specimens.map { s =>
      val t = ScapulaData.rigidFromLandmarks(s.lms, ref.lms)
      s.copy(mesh = s.mesh.transform(t), lms = s.lms.map(lm => lm.copy(point = t(lm.point))))
    }
    val g03 = ui.createGroup("G03_LandmarkAligned")
    lmAligned.foreach(s => try { ui.show(g03, s.mesh, s.id) } catch { case _: Exception => })

    // ── G04: Rigid ICP ────────────────────────────────────────────────────────
    println("[G04] Rigid ICP alignment")
    val rigidAligned: IndexedSeq[Spec] = lmAligned.zipWithIndex.map { case (s, i) =>
      println(s"  [G04] ${i + 1}/${lmAligned.length}  (${s.id})")
      s.copy(mesh = RigidAlign.rigidIcp(s.mesh, refDecimated, Config.icpIterations))
    }
    val g04 = ui.createGroup("G04_RigidAligned")
    rigidAligned.foreach(s => try { ui.show(g04, s.mesh, s.id) } catch { case _: Exception => })
    println(s"[info] ${rigidAligned.length} rigid-aligned meshes")

    // ── G05: LBFGS non-rigid registration ─────────────────────────────────────
    println("[G05] Building multi-scale GP model (Cholesky approx)...")
    val (lowRankGP, gpmm) = buildGPModel(refDecimated)

    val tag = "lbfgs_nrr"
    val registered: IndexedSeq[TriangleMesh[_3D]] =
      SSMBuilder.loadMeshes(tag).getOrElse {
        val r = rigidAligned.zipWithIndex.map { case (s, i) =>
          println(s"  [G05] LBFGS ${i + 1}/${rigidAligned.length}  (${s.id})")
          register(lowRankGP, gpmm, refDecimated, s.mesh)
        }
        SSMBuilder.saveMeshes(r, tag)
        r
      }
    println(s"[info] ${registered.length} LBFGS-registered meshes")

    val g05 = ui.createGroup("G05_LBFGS_Registered")
    registered.zip(rigidAligned).foreach { case (m, s) =>
      try { ui.show(g05, m, s.id) } catch { case _: Exception => }
    }

    // ── G06: SSM ─────────────────────────────────────────────────────────────
    println("[G06] Building SSM via PCA...")
    val ssmName = "lbfgs_ssm"
    val ssm = SSMBuilder.loadSSM(ssmName).getOrElse {
      val m = SSMBuilder.buildSSM(refDecimated, registered)
      SSMBuilder.saveSSM(m, ssmName)
      m
    }
    println(s"[info] SSM rank = ${ssm.rank}")

    val g06 = ui.createGroup("G06_SSM_Interactive (drag sliders on right panel)")
    ui.show(g06, ssm, "SSM")

    println("\n[info] Done. Inspect each group in the left panel.")
    println("[info] G05 = all LBFGS-registered bones overlaid (should show fine patches)")
    println("[info] G06 = interactive SSM — select group, drag Mode sliders")
  }
}
