package lbfgspipeline

import scalismo.common.{Field, RealSpace}
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.geometry.{EuclideanVector, Landmark, Point, Point3D, _3D}
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D, TriangleCell, TriangleList}
import scalismo.numerics.{FixedPointsUniformMeshSampler3D, UniformMeshSampler3D}
import scalismo.registration.{GaussianProcessTransformationSpace, L2Regularizer, MeanSquaresMetric, Registration}
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.common.{DiscreteField, PointId}
import scalismo.io.{StatisticalModelIO}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random
import breeze.linalg.DenseVector
import scala.io.Source
import scala.util.Using
import java.io.File

/**
 * Standalone LBFGS non-rigid registration pipeline.
 * Package:  lbfgspipeline
 * Folder:   src/main/scala/lbfgspipeline/
 * File:     ScapulaLBFGS.scala
 *
 * Follows Dennis Madsen's forum approach (Jun 2024):
 *   - Multi-scale Gaussian kernel (75 + 30 + 15 mm, amplitude 15/10/5)
 *   - Cholesky low-rank GP approximation
 *   - Coarse-to-fine LBFGS on MeanSquares distance-image metric
 *
 * Run: sbt "runMain lbfgspipeline.ScapulaLBFGS"
 */
object ScapulaLBFGS {

  // ── Config (overridable via environment variables) ───────────────────────────
  private def env(key: String, default: String): String = sys.env.getOrElse(key, default)

  val dataDir: File    = new File(env("SCAPULA_DATA_DIR", "/home/g25upadh/Documents/database_v1.11/paired_scapulae_STLs"))
  val outDir:  File    = new File(env("SCAPULA_OUT_DIR",  "/home/g25upadh/Documents/database_v1.11/scapula_ssm_out/lbfgs_pipeline"))
  val modelRes: Int    = env("SCAPULA_MODEL_RES",  "5000").toInt
  val icpIter: Int     = env("SCAPULA_ICP_ITERS",  "40").toInt
  val seed: Long       = env("SCAPULA_SEED",        "42").toLong

  // ── Registration schedule (coarse → fine) ───────────────────────────────────
  case class Stage(regularizationWeight: Double, iterations: Int, sampledPoints: Int)

  val schedule: Seq[Stage] = Seq(
    Stage(regularizationWeight = 1e-1, iterations = 50,  sampledPoints = 500),
    Stage(regularizationWeight = 1e-2, iterations = 50,  sampledPoints = 1000),
    Stage(regularizationWeight = 1e-4, iterations = 100, sampledPoints = 2000),
    Stage(regularizationWeight = 1e-6, iterations = 100, sampledPoints = 5000)
  )

  // ── Landmark names ───────────────────────────────────────────────────────────
  val landmarkNames: IndexedSeq[String] = IndexedSeq("GC", "TS", "IA", "PLA", "AC")

  // ── Specimen ─────────────────────────────────────────────────────────────────
  case class Specimen(id: String, mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]])

  // ── Data loading ─────────────────────────────────────────────────────────────

  def loadMesh(f: File): TriangleMesh[_3D] =
    MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read mesh: ${f.getPath}"))

  def mirrorPoint(p: Point[_3D]): Point[_3D] = Point3D(-p.x, p.y, p.z)

  def mirrorMesh(m: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pts = m.pointSet.points.map(mirrorPoint).toIndexedSeq
    val flipped = m.triangulation.triangles.map(t => TriangleCell(t.ptId1, t.ptId3, t.ptId2))
    TriangleMesh3D(pts, TriangleList(flipped))
  }

  def mirrorLm(lms: IndexedSeq[Landmark[_3D]]): IndexedSeq[Landmark[_3D]] =
    lms.map(lm => lm.copy(point = mirrorPoint(lm.point)))

  def readCsv(dir: File): Map[String, IndexedSeq[Landmark[_3D]]] = {
    val csv = Option(dir.listFiles()).getOrElse(Array.empty)
      .filter(f => f.getName.toLowerCase.endsWith(".csv") &&
                   f.getName.toLowerCase.contains("scapula") &&
                   f.getName.toLowerCase.contains("model_data"))
      .sortBy(_.getName).headOption
      .getOrElse(throw new RuntimeException(s"No landmark CSV found in ${dir.getPath}"))
    val lines = Using.resource(Source.fromFile(csv))(_.getLines().toIndexedSeq)
    val header = lines.head.split(",", -1).toIndexedSeq.map(_.trim.toLowerCase.replaceAll("[^a-z0-9]",""))
    def col(lm: String, ax: Char): Int = {
      val l = lm.toLowerCase
      val i = header.indexWhere(h => h == s"$l$ax" || (h.startsWith(l) && h.endsWith(ax.toString) && h.length <= l.length + 2))
      if (i == -1) throw new RuntimeException(s"Cannot find column for $lm$ax in CSV")
      i
    }
    val cols = landmarkNames.map(lm => lm -> (col(lm,'x'), col(lm,'y'), col(lm,'z'))).toMap
    lines.tail.filter(_.trim.nonEmpty).map { line =>
      val c = line.split(",", -1)
      val id = c(0).trim
      val lms = landmarkNames.map { nm =>
        val (xi, yi, zi) = cols(nm)
        Landmark(nm, Point3D(c(xi).trim.toDouble, c(yi).trim.toDouble, c(zi).trim.toDouble))
      }
      id -> lms
    }.toMap
  }

  def loadSpecimens(dir: File, lmMap: Map[String, IndexedSeq[Landmark[_3D]]]): IndexedSeq[Specimen] =
    Option(dir.listFiles()).getOrElse(Array.empty)
      .filter(_.getName.toLowerCase.endsWith(".stl"))
      .sortBy(_.getName)
      .flatMap { f =>
        val id = f.getName.stripSuffix(".stl")
        lmMap.get(id).map { lms =>
          val isRight = id.endsWith("_R")
          val raw = loadMesh(f)
          if (isRight) Specimen(id + "_mirrored", mirrorMesh(raw), mirrorLm(lms))
          else         Specimen(id, raw, lms)
        }
      }.toIndexedSeq

  // ── Rigid alignment ──────────────────────────────────────────────────────────

  def rigidFromLandmarks(from: IndexedSeq[Landmark[_3D]], to: IndexedSeq[Landmark[_3D]]) =
    LandmarkRegistration.rigid3DLandmarkRegistration(from, to, center = Point3D(0, 0, 0))

  def rigidIcp(moving: TriangleMesh[_3D], target: TriangleMesh[_3D], iters: Int)
              (implicit rng: Random): TriangleMesh[_3D] = {
    val ops = target.operations
    val ids = UniformMeshSampler3D(moving, 2000).sample().map { case (pt, _) =>
      moving.pointSet.findClosestPoint(pt).id
    }.distinct
    var cur = moving
    for (_ <- 0 until iters) {
      val pairs = ids.map { id =>
        val p = cur.pointSet.point(id)
        (p, ops.closestPointOnSurface(p).point)
      }
      val keep = math.max(10, (pairs.length * 0.85).toInt)
      val trimmed = pairs.sortBy { case (m, t) => (m - t).norm }.take(keep)
      val t = LandmarkRegistration.rigid3DLandmarkRegistration(trimmed, center = Point3D(0, 0, 0))
      cur = cur.transform(t)
    }
    cur
  }

  // ── LBFGS non-rigid registration ─────────────────────────────────────────────

  def buildGPModel(reference: TriangleMesh[_3D])
  : (LowRankGaussianProcess[_3D, EuclideanVector[_3D]], PointDistributionModel[_3D, TriangleMesh]) = {

    val kernel = DiagonalKernel3D(
      GaussianKernel[_3D](75.0) * 15.0 +
      GaussianKernel[_3D](30.0) * 10.0 +
      GaussianKernel[_3D](15.0) *  5.0,
      outputDim = 3
    )
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
      Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D]),
      kernel
    )
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference, gp,
      relativeTolerance = 0.01,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )
    (lowRankGP, PointDistributionModel[_3D, TriangleMesh](reference, lowRankGP))
  }

  def lbfgsStage(
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D],
    coeffs: DenseVector[Double],
    s: Stage
  )(implicit rng: Random): DenseVector[Double] = {
    val space  = GaussianProcessTransformationSpace(lowRankGP)
    val metric = MeanSquaresMetric(
      reference.operations.toDistanceImage,
      target.operations.toDistanceImage,
      space,
      FixedPointsUniformMeshSampler3D(reference, s.sampledPoints)
    )
    Registration(metric, L2Regularizer(space), s.regularizationWeight, LBFGSOptimizer(s.iterations))
      .iterator(coeffs).toSeq.last.parameters
  }

  def register(
    lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
    gpmm: PointDistributionModel[_3D, TriangleMesh],
    reference: TriangleMesh[_3D],
    target: TriangleMesh[_3D]
  )(implicit rng: Random): TriangleMesh[_3D] = {
    val finalCoeffs = schedule.foldLeft(DenseVector.zeros[Double](lowRankGP.rank)) { (c, s) =>
      lbfgsStage(lowRankGP, reference, target, c, s)
    }
    gpmm.instance(finalCoeffs)
  }

  // ── Mean mesh ────────────────────────────────────────────────────────────────

  def meanMesh(meshes: IndexedSeq[TriangleMesh[_3D]]): TriangleMesh[_3D] = {
    val n = meshes.length
    val pts = (0 until meshes.head.pointSet.numberOfPoints).map { i =>
      val id  = PointId(i)
      val sum = meshes.foldLeft(EuclideanVector.zeros[_3D])((acc, m) => acc + m.pointSet.point(id).toVector)
      (sum * (1.0 / n)).toPoint
    }
    TriangleMesh3D(pts, meshes.head.triangulation)
  }

  // ── Cache helpers ─────────────────────────────────────────────────────────────

  def saveAll(meshes: IndexedSeq[TriangleMesh[_3D]], tag: String): Unit = {
    val dir = new File(outDir, tag); dir.mkdirs()
    meshes.zipWithIndex.foreach { case (m, i) =>
      MeshIO.writeMesh(m, new File(dir, f"mesh_$i%04d.vtk"))
        .recover { case e => println(s"[warn] save failed mesh $i: ${e.getMessage}") }
    }
  }

  def loadAll(tag: String): Option[IndexedSeq[TriangleMesh[_3D]]] = {
    val dir = new File(outDir, tag)
    if (!dir.exists()) return None
    val files = Option(dir.listFiles()).getOrElse(Array.empty).filter(_.getName.endsWith(".vtk")).sortBy(_.getName)
    if (files.isEmpty) return None
    val meshes = files.flatMap(f => MeshIO.readMesh(f).toOption)
    if (meshes.length == files.length) Some(meshes.toIndexedSeq) else None
  }

  // ── SSM ──────────────────────────────────────────────────────────────────────

  def buildSSM(reference: TriangleMesh[_3D], meshes: IndexedSeq[TriangleMesh[_3D]]) = {
    val defFields = meshes.map { mesh =>
      DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]](
        reference,
        reference.pointSet.pointsWithId.map { case (refPt, id) => mesh.pointSet.point(id) - refPt }.toIndexedSeq
      )
    }
    PointDistributionModel.createUsingPCA(DataCollection(defFields))
  }

  // ── Main ─────────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(seed)

    println(s"[LBFGS] Data dir : ${dataDir.getAbsolutePath}")
    println(s"[LBFGS] Out dir  : ${outDir.getAbsolutePath}")

    val lmMap    = readCsv(dataDir)
    val specimens = loadSpecimens(dataDir, lmMap)
    println(s"[LBFGS] ${specimens.length} specimens loaded")

    val ui = ScalismoUI("Scapula LBFGS Pipeline")
    Thread.sleep(2000)

    // G01 — raw input
    val g01 = ui.createGroup("G01_RawInput")
    specimens.foreach(s => try { ui.show(g01, s.mesh, s.id) } catch { case _: Exception => })

    // G02 — reference (decimated)
    val ref     = specimens.head
    val refMesh = ref.mesh.operations.decimate(modelRes)
    println(s"[LBFGS] Reference: ${ref.id}  ${refMesh.pointSet.numberOfPoints} vertices")
    val g02 = ui.createGroup("G02_Reference")
    ui.show(g02, refMesh, ref.id)
    ui.show(g02, ref.lms.toList, ref.id + "_lms")

    // G03 — landmark aligned
    println("[LBFGS] Landmark alignment (Procrustes)")
    val lmAligned = specimens.map { s =>
      val t = rigidFromLandmarks(s.lms, ref.lms)
      s.copy(mesh = s.mesh.transform(t), lms = s.lms.map(lm => lm.copy(point = t(lm.point))))
    }
    val g03 = ui.createGroup("G03_LandmarkAligned")
    lmAligned.foreach(s => try { ui.show(g03, s.mesh, s.id) } catch { case _: Exception => })

    // G04 — rigid ICP
    println("[LBFGS] Rigid ICP alignment")
    val rigidAligned = lmAligned.zipWithIndex.map { case (s, i) =>
      println(s"  [G04] ${i + 1}/${lmAligned.length}  ${s.id}")
      s.copy(mesh = rigidIcp(s.mesh, refMesh, icpIter))
    }
    val g04 = ui.createGroup("G04_RigidAligned")
    rigidAligned.foreach(s => try { ui.show(g04, s.mesh, s.id) } catch { case _: Exception => })

    // G05 — LBFGS non-rigid
    println("[LBFGS] Building GP model (Cholesky, multi-scale kernel 75/30/15 mm)")
    val (lowRankGP, gpmm) = buildGPModel(refMesh)
    println(s"[LBFGS] GP model rank = ${gpmm.rank}")

    val registered: IndexedSeq[TriangleMesh[_3D]] = loadAll("registered").getOrElse {
      val r = rigidAligned.zipWithIndex.map { case (s, i) =>
        println(s"  [G05] LBFGS ${i + 1}/${rigidAligned.length}  ${s.id}")
        register(lowRankGP, gpmm, refMesh, s.mesh)
      }
      saveAll(r, "registered")
      r
    }
    println(s"[LBFGS] ${registered.length} LBFGS-registered")

    val g05 = ui.createGroup("G05_LBFGS_Registered (fine patches here)")
    registered.zip(rigidAligned).foreach { case (m, s) =>
      try { ui.show(g05, m, s.id) } catch { case _: Exception => }
    }

    // G06 — SSM
    println("[LBFGS] Building SSM via PCA")
    val ssmFile = new File(outDir, "lbfgs_ssm.h5")
    val ssm = if (ssmFile.exists())
      StatisticalModelIO.readStatisticalTriangleMeshModel3D(ssmFile).getOrElse(buildSSM(refMesh, registered))
    else {
      val m = buildSSM(refMesh, registered)
      outDir.mkdirs()
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(m, ssmFile)
        .recover { case e => println(s"[warn] SSM save failed: ${e.getMessage}") }
      m
    }
    println(s"[LBFGS] SSM rank = ${ssm.rank}")

    val g06 = ui.createGroup("G06_SSM_Interactive (drag Mode sliders on right)")
    ui.show(g06, ssm, "SSM")

    println("\n[LBFGS] Done. Check G05 for fine patches, G06 for interactive SSM.")
  }
}
