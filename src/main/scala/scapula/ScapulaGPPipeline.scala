package scapula

// ══════════════════════════════════════════════════════════════════════════════
//  HOW TO RUN (copy-paste into your terminal):
//
//    cd /home/g25upadh/Downloads/scalismo        ← your project folder
//    sbt "runMain scapula.ScapulaGPPipeline"
//
//  First run downloads dependencies (~2 min). Subsequent runs are fast.
//  The viewer window opens at the end — close it to exit.
// ══════════════════════════════════════════════════════════════════════════════

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.io.{MeshIO, StatismoIO}
import scalismo.registration.LandmarkRegistration
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random
import java.io.File
import scala.io.Source
import scala.util.Using

object ScapulaGPPipeline:

  // ╔══════════════════════════════════════════════════════════════════════════╗
  // ║  SET THESE PATHS BEFORE RUNNING                                         ║
  // ╚══════════════════════════════════════════════════════════════════════════╝

  // Folder containing the .stl files and the landmark CSV file
  val dataDir      = "/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"

  // Model ID of the reference — must match the first column of the CSV exactly
  val refId        = "paired_scapula_001_M_64_L"

  // Model ID of the target — the reference is rigidly aligned toward this
  val tgtId        = "paired_scapula_001_M_64_R"

  // Where the .h5 model file will be saved (parent folder must exist)
  val outputH5     = "/home/g25upadh/Documents/scapula_gp.h5"

  // Gaussian kernel width in mm — larger means smoother, longer-range modes
  val sigma        = 50.0

  // Gaussian kernel amplitude — larger means bigger displacements per mode
  val scale        = 100.0

  // Number of deformation modes; exactly 5 sliders will appear in the viewer
  val gpRank       = 5

  // ══════════════════════════════════════════════════════════════════════════
  //  PIPELINE — nothing below needs changing
  // ══════════════════════════════════════════════════════════════════════════

  def main(args: Array[String]): Unit =
    scalismo.initialize()
    implicit val rng: Random = Random(42L)

    // ── Step 1: Load meshes and landmarks from CSV ─────────────────────────
    println("\n[1/5] Loading meshes and landmarks...")

    val dir = new File(dataDir)
    require(dir.exists() && dir.isDirectory, s"Data folder not found:\n  $dataDir")

    val refFile = new File(dir, s"$refId.stl")
    val tgtFile = new File(dir, s"$tgtId.stl")
    require(refFile.exists(), s"STL not found: ${refFile.getPath}")
    require(tgtFile.exists(), s"STL not found: ${tgtFile.getPath}")

    val refMesh = MeshIO.readMesh(refFile)
      .getOrElse(sys.error(s"Cannot load mesh: ${refFile.getPath}"))
    val tgtMesh = MeshIO.readMesh(tgtFile)
      .getOrElse(sys.error(s"Cannot load mesh: ${tgtFile.getPath}"))

    // Find the landmark CSV automatically (any *scapula*model_data*.csv in the folder)
    val csvOpt = Option(dir.listFiles())
      .getOrElse(Array.empty)
      .filter(f => f.getName.toLowerCase.endsWith(".csv")
                && f.getName.toLowerCase.contains("scapula")
                && f.getName.toLowerCase.contains("model_data"))
      .sortBy(_.getName)
      .headOption
    val csvFile = csvOpt.getOrElse(sys.error(
      s"No *scapula*model_data*.csv found in $dataDir"))

    val refLms = landmarksFromCsv(csvFile, refId)
    val tgtLms = landmarksFromCsv(csvFile, tgtId)

    println(s"  CSV            : ${csvFile.getName}")
    println(s"  Reference mesh : ${refMesh.pointSet.numberOfPoints} vertices")
    println(s"  Target mesh    : ${tgtMesh.pointSet.numberOfPoints} vertices")
    println(s"  Landmarks      : ${refLms.size}  (${refLms.map(_.id).mkString(", ")})")

    // ── Step 2: Rigid alignment using landmarks ────────────────────────────
    println("\n[2/5] Rigid landmark alignment...")

    // Build (moving, fixed) point pairs matched by landmark ID
    val commonIds = refLms.map(_.id).toSet.intersect(tgtLms.map(_.id).toSet)
    require(commonIds.nonEmpty, "No shared landmark IDs between reference and target.")

    val pairs: IndexedSeq[(Point[_3D], Point[_3D])] =
      commonIds.toIndexedSeq.sorted.map { id =>
        refLms.find(_.id == id).get.point -> tgtLms.find(_.id == id).get.point
      }

    val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(
      pairs,
      center = Point3D(0.0, 0.0, 0.0)
    )
    val alignedRef = refMesh.transform(rigidTrans)
    println(s"  Done — ${commonIds.size} landmark pairs used.")

    // ── Step 3: GP model — one Gaussian kernel, rank 5 ────────────────────
    println(s"\n[3/5] Building GP model  (sigma=$sigma, scale=$scale, rank=$gpRank)...")

    val zeroMean: Field[_3D, EuclideanVector[_3D]] =
      Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector3D(0.0, 0.0, 0.0))

    val matrixKernel = DiagonalKernel(GaussianKernel[_3D](sigma) * scale, 3)

    val gp = GaussianProcess(zeroMean, matrixKernel)

    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(
      gp,
      alignedRef,
      numBasisFunctions = gpRank,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )

    val model = PointDistributionModel(alignedRef, lowRankGP)
    println(s"  Done — model rank = ${model.rank}")

    // ── Step 4: Save as .h5 ───────────────────────────────────────────────
    println(s"\n[4/5] Saving model  →  $outputH5")
    val h5 = new File(outputH5)
    h5.getParentFile.mkdirs()
    StatismoIO.writeStatismoMeshModel(model, h5) match
      case scala.util.Success(_)  => println("  Saved.")
      case scala.util.Failure(ex) => sys.error(s"Save failed: ${ex.getMessage}")

    // ── Step 5: Viewer — one slider per mode ──────────────────────────────
    println(s"\n[5/5] Opening viewer  (${model.rank} deformation modes)...")
    println("  Drag a slider to deform the shape along that mode.")
    println("  Close the viewer window to exit.\n")

    val ui = ScalismoUI()
    ui.show(ui.createGroup("Scapula GP"), model, "gp_model")

  // ── CSV helper — reads landmarks for one specimen from the project CSV ────
  //
  //  The CSV must have:
  //    • Column 0 : model ID  (e.g. paired_scapula_001_M_64_L)
  //    • A header row whose column names contain each landmark name followed
  //      by x / y / z  (e.g. "GC_x", "GCx", "gc x" all normalise the same)
  //    • Landmark names searched: GC, TS, IA, PLA, AC
  //    • Fallback column offsets used when header matching fails:
  //      GC→11, TS→14, IA→17, PLA→20, AC→23  (adjust if your CSV differs)
  private def landmarksFromCsv(csv: File, modelId: String): IndexedSeq[Landmark[_3D]] =
    Using.resource(Source.fromFile(csv)) { src =>
      val lines  = src.getLines().toIndexedSeq.filter(_.trim.nonEmpty)
      require(lines.length > 1, s"CSV has no data rows: ${csv.getName}")

      val rawHeader = lines.head.split(",", -1).toIndexedSeq
      // Normalise: strip everything except letters and digits, lowercase
      val normHeader = rawHeader.map(_.trim.toLowerCase.replaceAll("[^a-z0-9]", ""))

      val lmNames     = Seq("GC", "TS", "IA", "PLA", "AC")
      val fallbackCol = Map("GC" -> 11, "TS" -> 14, "IA" -> 17, "PLA" -> 20, "AC" -> 23)

      // For each landmark find the x,y,z column indices
      val colTriples: Seq[(String, Int, Int, Int)] = lmNames.map { lm =>
        val l = lm.toLowerCase
        def findCol(axis: Char): Int =
          normHeader.indexWhere { h =>
            h == s"$l$axis" ||
            (h.startsWith(l) && h.endsWith(axis.toString) && h.length <= l.length + 2)
          }
        val xi = findCol('x')
        val yi = findCol('y')
        val zi = findCol('z')
        if (xi >= 0 && yi >= 0 && zi >= 0) (lm, xi, yi, zi)
        else {
          val fb = fallbackCol(lm)
          println(s"  [CSV] Header match failed for '$lm' — using fallback columns ($fb, ${fb+1}, ${fb+2}).")
          (lm, fb, fb + 1, fb + 2)
        }
      }

      val row = lines.tail
        .find(_.split(",", -1)(0).trim == modelId)
        .getOrElse(sys.error(
          s"Model ID '$modelId' not found in ${csv.getName}.\n" +
          s"  First 3 IDs in file: ${lines.tail.take(3).map(_.split(",", -1)(0).trim).mkString(", ")}"))

      val cells = row.split(",", -1).map(_.trim)

      colTriples.map { case (lm, xi, yi, zi) =>
        require(cells.length > math.max(xi, math.max(yi, zi)),
          s"Row for '$modelId' has too few columns for landmark $lm (need col ${math.max(xi, math.max(yi, zi))} but only ${cells.length} columns).")
        Landmark(lm, Point3D(cells(xi).toDouble, cells(yi).toDouble, cells(zi).toDouble))
      }.toIndexedSeq
    }
