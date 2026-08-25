package scapula

// ══════════════════════════════════════════════════════════════════════════════
//  ScapulaGPPipeline — full 5-step GP pipeline for a single pair of scapulae
//
//  Run from the project root:
//    sbt "runMain scapula.ScapulaGPPipeline"
//
//  Landmarks must be in Scalismo JSON format. To convert a raw CSV row, see
//  ScapulaData.readLandmarkCsv in this project, or use the Scalismo viewer
//  to place and save landmarks interactively.
// ══════════════════════════════════════════════════════════════════════════════

import scalismo.geometry.*
import scalismo.common.*
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.io.{LandmarkIO, MeshIO, StatismoIO}
import scalismo.registration.LandmarkRegistration
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random
import java.io.File

object ScapulaGPPipeline:

  // ╔══════════════════════════════════════════════════════════════════════════╗
  // ║  PARAMETERS — edit ONLY this block before running                       ║
  // ╚══════════════════════════════════════════════════════════════════════════╝

  // Reference scapula mesh (.stl or .vtk); the GP deformation model is built on this shape
  val refMeshPath: String  = "/home/user/Documents/100 plus scapula data/paired_scapulae_STLs_scapula/paired_scapula_001_M_64_L.stl"

  // Target scapula mesh; the reference is rigidly moved toward this bone
  val tgtMeshPath: String  = "/home/user/Documents/100 plus scapula data/paired_scapulae_STLs_scapula/paired_scapula_001_M_64_R.stl"

  // Scalismo JSON landmark file for the reference (must share landmark IDs with target)
  val refLmPath: String    = "/home/user/Documents/ref_landmarks.json"

  // Scalismo JSON landmark file for the target
  val tgtLmPath: String    = "/home/user/Documents/tgt_landmarks.json"

  // Output h5 file path; directory must already exist
  val outputH5Path: String = "/home/user/Documents/scapula_gp.h5"

  // Gaussian kernel width in mm: larger value → smoother, longer-range deformation modes
  val sigma: Double = 50.0

  // Gaussian kernel amplitude: larger value → bigger displacements in every mode
  val scale: Double = 100.0

  // Number of GP basis functions (deformation modes); fixed at 5 as required
  val gpRank: Int   = 5

  // ══════════════════════════════════════════════════════════════════════════════
  //  PIPELINE — no changes needed below this line
  // ══════════════════════════════════════════════════════════════════════════════

  def main(args: Array[String]): Unit =

    scalismo.initialize()
    implicit val rng: Random = Random(42L)

    // ── Step 1: Load meshes and landmarks ──────────────────────────────────────
    println("\n[Step 1] Loading meshes and landmarks...")

    val refMesh = MeshIO.readMesh(new File(refMeshPath))
      .getOrElse(sys.error(s"Cannot read reference mesh: $refMeshPath"))

    val tgtMesh = MeshIO.readMesh(new File(tgtMeshPath))
      .getOrElse(sys.error(s"Cannot read target mesh: $tgtMeshPath"))

    // readLandmarksJson reads the Scalismo JSON landmark format
    val refLms = LandmarkIO.readLandmarksJson[_3D](new File(refLmPath))
      .getOrElse(sys.error(s"Cannot read reference landmarks: $refLmPath"))

    val tgtLms = LandmarkIO.readLandmarksJson[_3D](new File(tgtLmPath))
      .getOrElse(sys.error(s"Cannot read target landmarks: $tgtLmPath"))

    // Sort by landmark ID so both sequences are paired in the same order
    val refSorted = refLms.sortBy(_.id)
    val tgtSorted = tgtLms.sortBy(_.id)

    require(
      refSorted.map(_.id) == tgtSorted.map(_.id),
      s"Landmark IDs do not match:\n  ref = ${refSorted.map(_.id)}\n  tgt = ${tgtSorted.map(_.id)}"
    )

    println(s"  Reference mesh : ${refMesh.pointSet.numberOfPoints} vertices")
    println(s"  Target mesh    : ${tgtMesh.pointSet.numberOfPoints} vertices")
    println(s"  Landmarks      : ${refSorted.size}  (${refSorted.map(_.id).mkString(", ")})")

    // ── Step 2: Rigid registration using landmarks ─────────────────────────────
    println("\n[Step 2] Rigid alignment via landmarks...")

    val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(
      refSorted,
      tgtSorted,
      center = Point3D(0, 0, 0)
    )

    val alignedRef = refMesh.transform(rigidTrans)
    println("  Reference mesh rigidly aligned to the target coordinate frame.")

    // ── Step 3: GP model with a single Gaussian kernel, rank 5 ────────────────
    println(s"\n[Step 3] Building GP model  (sigma=$sigma mm  scale=$scale  rank=$gpRank)...")

    // Zero mean: no preferred deformation direction
    val zeroMean = Field(EuclideanSpace3D, (_: Point[_3D]) => EuclideanVector3D(0, 0, 0))

    // Single Gaussian scalar kernel scaled by 'scale'
    val scalarKernel = GaussianKernel[_3D](sigma) * scale

    // Promote the scalar kernel to a 3×3 diagonal matrix kernel for 3-D displacements
    val matrixKernel = DiagonalKernel(scalarKernel, 3)

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, matrixKernel)

    // Nystrom approximation — samples gpRank basis functions on the aligned reference mesh
    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(
      gp,
      alignedRef,
      numBasisFunctions = gpRank,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
    )

    val model = PointDistributionModel(alignedRef, lowRankGP)
    println(s"  GP model built.  Actual rank = ${model.rank}")

    // ── Step 4: Save as .h5 ────────────────────────────────────────────────────
    println(s"\n[Step 4] Saving model  →  $outputH5Path")

    StatismoIO.writeStatismoMeshModel(model, new File(outputH5Path))
      .fold(
        err => sys.error(s"Failed to write .h5 file: ${err.getMessage}"),
        _   => println("  Model saved successfully.")
      )

    // ── Step 5: Visualise all 5 deformation modes ──────────────────────────────
    println("\n[Step 5] Launching Scalismo viewer...")
    println("  The viewer opens with a slider for each of the 5 modes.")
    println("  Drag a slider to see that mode's shape deformation.")
    println("  Take a screenshot for each mode, then close the viewer window.\n")

    val ui    = ScalismoUI()
    val group = ui.createGroup("Scapula GP Model")
    ui.show(group, model, "gp_model")

    // The application stays alive while the viewer window is open.
    // Closing the viewer window (or pressing Ctrl-C in the terminal) exits.
