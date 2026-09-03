package ssmpipeline

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.io.MeshIO
import scalismo.registration.LandmarkRegistration
import scalismo.transformations.TranslationAfterRotation

import java.io.File
import scala.io.Source
import scala.util.Using

object Config {
  private def env(key: String, default: String): String = sys.env.getOrElse(key, default)

  val dataDir: File = new File(env("SCAPULA_DATA_DIR", "/home/g25upadh/Documents/database_v1.11/paired_scapulae_STLs"))
  val outDir: File  = new File(env("SCAPULA_OUT_DIR",  "/home/g25upadh/Documents/database_v1.11/scapula_ssm_out"))

  val modelResolution: Int     = env("SCAPULA_MODEL_RES",        "5000").toInt
  val icpIterations: Int       = env("SCAPULA_ICP_ITERS",        "40").toInt
  val refinePasses: Int        = env("SCAPULA_REFINE_PASSES",    "2").toInt
  val gpRelativeTolerance: Double = env("SCAPULA_GP_TOL",        "0.01").toDouble
  val gpMaxRank: Int           = env("SCAPULA_GP_MAX_RANK",      "250").toInt
  val buildIndependentModel: Boolean = env("SCAPULA_INDEPENDENT_MODEL", "true").toBoolean
  val showUi: Boolean          = env("SCAPULA_UI",               "true").toBoolean
  val seed: Long               = env("SCAPULA_SEED",             "42").toLong
}

object ScapulaData {

  val landmarkNames: IndexedSeq[String] = IndexedSeq("GC", "TS", "IA", "PLA", "AC")

  private val fallbackStartIdx: Map[String, Int] =
    Map("GC" -> 11, "TS" -> 14, "IA" -> 17, "PLA" -> 20, "AC" -> 23)

  final case class Specimen(modelId: String, file: File, isRight: Boolean, subject: String)

  def csvFile(dir: File): File = {
    val files = Option(dir.listFiles()).getOrElse(Array.empty[File])
    files
      .filter(_.getName.toLowerCase.endsWith(".csv"))
      .filter(f => f.getName.toLowerCase.contains("scapula") && f.getName.toLowerCase.contains("model_data"))
      .filterNot(_.getName.toLowerCase.startsWith("single"))
      .sortBy(_.getName)
      .headOption
      .getOrElse(
        throw new RuntimeException(
          s"No landmark CSV found in ${dir.getPath}. Present: ${files.map(_.getName).mkString(", ")}"
        )
      )
  }

  private def normaliseHeader(h: String): String = h.trim.toLowerCase.replaceAll("[^a-z0-9]", "")

  def resolveColumns(header: IndexedSeq[String]): (Map[String, (Int, Int, Int)], Boolean) = {
    val norm = header.map(normaliseHeader)

    def findAxis(lm: String, axis: Char): Option[Int] = {
      val l = lm.toLowerCase
      norm.indexWhere { h =>
        h == s"$l$axis" || h == s"${l}_$axis".replace("_", "") || (h.startsWith(l) && h.endsWith(axis.toString) &&
          h.length <= l.length + 2)
      } match {
        case -1 => None
        case i  => Some(i)
      }
    }

    val byName = landmarkNames.flatMap { lm =>
      for {
        xi <- findAxis(lm, 'x')
        yi <- findAxis(lm, 'y')
        zi <- findAxis(lm, 'z')
      } yield lm -> (xi, yi, zi)
    }.toMap

    if (byName.size == landmarkNames.size) (byName, true)
    else (fallbackStartIdx.map { case (lm, i) => lm -> (i, i + 1, i + 2) }, false)
  }

  def readLandmarkCsv(file: File): (Map[String, IndexedSeq[Landmark[_3D]]], Boolean, IndexedSeq[String]) = {
    val lines = Using.resource(Source.fromFile(file))(_.getLines().toIndexedSeq)
    require(lines.nonEmpty, s"Landmark CSV ${file.getName} is empty")

    val header = lines.head.split(",", -1).toIndexedSeq.map(_.trim)
    val (cols, fromHeader) = resolveColumns(header)

    val entries = lines.tail.filter(_.trim.nonEmpty).map { line =>
      val c = line.split(",", -1)
      val modelId = c(0).trim
      val lms = landmarkNames.map { nm =>
        val (xi, yi, zi) = cols(nm)
        require(
          c.length > math.max(xi, math.max(yi, zi)),
          s"Row '$modelId' has ${c.length} columns but landmark $nm needs column ${math.max(xi, math.max(yi, zi))}"
        )
        Landmark(nm, Point3D(c(xi).trim.toDouble, c(yi).trim.toDouble, c(zi).trim.toDouble))
      }
      modelId -> lms
    }.toMap

    (entries, fromHeader, header)
  }

  def specimens(dir: File): IndexedSeq[Specimen] = {
    Option(dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.toLowerCase.endsWith(".stl"))
      .sortBy(_.getName)
      .toIndexedSeq
      .map { f =>
        val id = f.getName.stripSuffix(".stl")
        Specimen(id, f, id.endsWith("_R"), subjectKey(id))
      }
  }

  def subjectKey(modelId: String): String = modelId.stripSuffix("_L").stripSuffix("_R")

  def mirrorPoint(p: Point[_3D]): Point[_3D] = Point3D(-p.x, p.y, p.z)

  def mirrorMesh(mesh: TriangleMesh[_3D]): TriangleMesh[_3D] = {
    val pts     = mesh.pointSet.points.map(mirrorPoint).toIndexedSeq
    val flipped = mesh.triangulation.triangles.map(t => TriangleCell(t.ptId1, t.ptId3, t.ptId2))
    TriangleMesh3D(pts, TriangleList(flipped))
  }

  def mirrorLandmarks(lms: IndexedSeq[Landmark[_3D]]): IndexedSeq[Landmark[_3D]] =
    lms.map(lm => lm.copy(point = mirrorPoint(lm.point)))

  def signedVolume(mesh: TriangleMesh[_3D]): Double = {
    var acc = 0.0
    mesh.triangulation.triangles.foreach { t =>
      val a = mesh.pointSet.point(t.ptId1).toVector
      val b = mesh.pointSet.point(t.ptId2).toVector
      val c = mesh.pointSet.point(t.ptId3).toVector
      acc += a.dot(b.crossproduct(c)) / 6.0
    }
    acc
  }

  def rigidFromLandmarks(from: IndexedSeq[Landmark[_3D]],
                         to: IndexedSeq[Landmark[_3D]]
  ): TranslationAfterRotation[_3D] =
    LandmarkRegistration.rigid3DLandmarkRegistration(from, to, center = Point3D(0, 0, 0))

  def perLandmarkDistances(a: IndexedSeq[Landmark[_3D]], b: IndexedSeq[Landmark[_3D]]): IndexedSeq[(String, Double)] = {
    val bById = b.map(l => l.id -> l.point).toMap
    a.flatMap(la => bById.get(la.id).map(pb => la.id -> (la.point - pb).norm))
  }

  def loadMesh(f: File): TriangleMesh[_3D] =
    MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Could not read mesh ${f.getPath}"))
}

object Metrics {

  def surfaceDistances(from: TriangleMesh[_3D], to: TriangleMesh[_3D]): IndexedSeq[Double] = {
    val ops = to.operations
    from.pointSet.points.map(p => (p - ops.closestPointOnSurface(p).point).norm).toIndexedSeq
  }

  def percentile(values: IndexedSeq[Double], p: Double): Double = {
    require(values.nonEmpty)
    val sorted = values.sorted
    val idx    = math.min(sorted.length - 1, math.max(0, math.ceil(p * sorted.length).toInt - 1))
    sorted(idx)
  }

  final case class SurfaceStats(mean: Double, rms: Double, hd95: Double, hd: Double) {
    def render: String = f"mean=$mean%5.2f  rms=$rms%5.2f  HD95=$hd95%5.2f  HD=$hd%6.2f"
  }

  def symmetric(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): SurfaceStats = {
    val d = surfaceDistances(a, b) ++ surfaceDistances(b, a)
    SurfaceStats(
      mean = d.sum / d.length,
      rms  = math.sqrt(d.map(x => x * x).sum / d.length),
      hd95 = percentile(d, 0.95),
      hd   = d.max
    )
  }

  def correspondingDistances(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): IndexedSeq[Double] = {
    require(a.pointSet.numberOfPoints == b.pointSet.numberOfPoints, "meshes are not in correspondence")
    a.pointSet.points.zip(b.pointSet.points).map { case (p, q) => (p - q).norm }.toIndexedSeq
  }
}
