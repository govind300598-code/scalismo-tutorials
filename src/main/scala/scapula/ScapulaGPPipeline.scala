package scapula

// ══════════════════════════════════════════════════════════════════════════════
//  ScapulaGPPipeline — 5-step scapula GP pipeline
//
//  Terminal command (run from the project root folder):
//    sbt "runMain scapula.ScapulaGPPipeline"
//
//  Prerequisites:
//    1. Java 11+  →  java -version
//    2. sbt       →  sbt --version
//    3. The data directory, CSV, and STL files must already exist at the
//       paths set in the PARAMETERS block below.
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

object ScapulaGPPipeline:

  // ╔══════════════════════════════════════════════════════════════════════════╗
  // ║  PARAMETERS — change ONLY these values, then run with sbt               ║
  // ╚══════════════════════════════════════════════════════════════════════════╝

  // Folder that contains the .stl files AND the landmark CSV
  val dataDir: String = "/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"

  // Model ID of the reference specimen (must match the first column in the CSV)
  val refId: String   = "paired_scapula_001_M_64_L"

  // Model ID of the target specimen (must match the first column in the CSV)
  val tgtId: String   = "paired_scapula_001_M_64_R"

  // Where to write the saved GP model; the parent directory must exist
  val outputH5: String = "/home/g25upadh/Documents/scapula_gp.h5"

  // Gaussian kernel width in mm: larger = smoother, longer-range deformations
  val sigma: Double = 50.0

  // Gaussian kernel amplitude: larger = bigger displacements in each mode
  val scale: Double = 100.0

  // Number of deformation modes (basis functions) shown as viewer sliders
  val gpRank: Int = 5

  // ══════════════════════════════════════════════════════════════════════════════
  //  PIPELINE  — nothing below this line needs editing
  // ══════════════════════════════════════════════════════════════════════════════

  def main(args: Array[String]): Unit =

    scalismo.initialize()
    implicit val rng: Random = Random(42L)

    val dir = new File(dataDir)
    require(dir.exists(), s"Data directory not found: $dataDir")

    // ── Step 1: Load meshes and landmarks from CSV ─────────────────────────────
    println("\n[Step 1] Loading meshes and landmarks...")

    val refFile = new File(dir, s"$refId.stl")
    val tgtFile = new File(dir, s"$tgtId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getPath}")
    require(tgtFile.exists(), s"Target STL not found: ${tgtFile.getPath}")

    val refMesh = MeshIO.readMesh(refFile)
      .getOrElse(sys.error(s"Could not read reference mesh: ${refFile.getPath}"))
    val tgtMesh = MeshIO.readMesh(tgtFile)
      .getOrElse(sys.error(s"Could not read target mesh: ${tgtFile.getPath}"))

    // Read all landmarks from the CSV that lives in the same folder
    val csvFile = ScapulaData.csvFile(dir)
    val (allLandmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csvFile)
    if (!fromHeader)
      println("  WARNING: landmark columns resolved by fallback offsets, not header names.")

    val refLms = allLandmarks.getOrElse(refId,
      sys.error(s"No CSV row found for reference id '$refId' in ${csvFile.getName}"))
    val tgtLms = allLandmarks.getOrElse(tgtId,
      sys.error(s"No CSV row found for target id '$tgtId' in ${csvFile.getName}"))

    println(s"  Landmark CSV   : ${csvFile.getName}")
    println(s"  Reference mesh : ${refMesh.pointSet.numberOfPoints} vertices  (${refFile.getName})")
    println(s"  Target mesh    : ${tgtMesh.pointSet.numberOfPoints} vertices  (${tgtFile.getName})")
    println(s"  Landmarks      : ${refLms.size}  (${refLms.map(_.id).mkString(", ")})")

    // ── Step 2: Rigid registration using landmarks ─────────────────────────────
    println("\n[Step 2] Rigid landmark alignment...")

    // Pair landmarks by matching IDs so order in the CSV does not matter
    val sharedIds  = refLms.map(_.id).toSet & tgtLms.map(_.id).toSet
    require(sharedIds.nonEmpty, "No common landmark IDs between reference and target.")

    val pairs: IndexedSeq[(Point[_3D], Point[_3D])] =
      sharedIds.toIndexedSeq.sorted.map { id =>
        val rp = refLms.find(_.id == id).get.point
        val tp = tgtLms.find(_.id == id).get.point
        (rp, tp)
      }

    val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(pairs, center = Point3D(0, 0, 0))
    val alignedRef  = refMesh.transform(rigidTrans)
    println(s"  Done — ${sharedIds.size} landmark pairs used.")

    // ── Step 3: Single Gaussian kernel GP model, rank 5 ───────────────────────
    println(s"\n[Step 3] Building GP model  (sigma=$sigma mm, scale=$scale, rank=$gpRank)...")

    // Zero-mean field — no preferred deformation direction
    val zeroMean: Field[_3D, EuclideanVector[_3D]] =
      Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector3D(0.0, 0.0, 0.0))

    // One Gaussian scalar kernel, promoted to a 3-D diagonal matrix kernel
    val scalarKernel = GaussianKernel[_3D](sigma) * scale
    val matrixKernel = DiagonalKernel(scalarKernel, 3)

    val gp = GaussianProcess(zeroMean, matrixKernel)

    // Nystrom approximation on the aligned reference mesh
    val interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    val lowRankGP    = LowRankGaussianProcess.approximateGPNystrom(
      gp,
      alignedRef,
      numBasisFunctions = gpRank,
      interpolator = interpolator
    )

    val model = PointDistributionModel(alignedRef, lowRankGP)
    println(s"  GP model built.  Rank = ${model.rank}")

    // ── Step 4: Save GP model as .h5 ──────────────────────────────────────────
    println(s"\n[Step 4] Saving model  →  $outputH5")

    val h5File = new File(outputH5)
    h5File.getParentFile.mkdirs()   // create parent directory if it does not exist

    StatismoIO.writeStatismoMeshModel(model, h5File) match
      case scala.util.Success(_)  => println("  Saved OK.")
      case scala.util.Failure(ex) => sys.error(s"Failed to save .h5: ${ex.getMessage}")

    // ── Step 5: Open viewer and show all 5 deformation modes ──────────────────
    println("\n[Step 5] Opening Scalismo viewer...")
    println("  One slider appears per mode (0 – 4).")
    println("  Drag a slider to deform the mesh along that mode, then take a screenshot.")
    println("  Close the viewer window to exit the program.\n")

    val ui    = ScalismoUI()
    val group = ui.createGroup("Scapula GP")
    ui.show(group, model, "gp_model")

    // The JVM stays alive until the viewer window is closed.
