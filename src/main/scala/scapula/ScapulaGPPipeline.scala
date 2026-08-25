package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.io.{ MeshIO, StatisticalModelIO }
import scalismo.statisticalmodel.{ GaussianProcess, LowRankGaussianProcess, PointDistributionModel }
import scalismo.kernels.{ DiagonalKernel, GaussianKernel }
import scalismo.numerics.UniformMeshSampler3D
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Full 5-step GP pipeline for scapula shape modelling.
 *
 * Run from the project root with:
 *   sbt "runMain scapula.ScapulaGPPipeline"
 *
 * All tunable values are collected below so you never have to search the code body.
 */
object ScapulaGPPipeline:

  // ── Paths ────────────────────────────────────────────────────────────────────
  /** Folder containing the paired STLs and the landmark CSV. */
  val dataDir: String = "/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"

  /** Reference specimen id – the STL is <refId>.stl, and the CSV row key is <refId>. */
  val refId: String = "paired_scapula_001_M_64_L"

  /** Target specimen id – rigid alignment brings the reference into this space. */
  val targetId: String = "paired_scapula_002_M_56_L"

  /** Where to write the GP model; the directory must already exist. */
  val outputH5: String = "/tmp/scapula_gp_model.h5"

  // ── Kernel parameters ────────────────────────────────────────────────────────
  /** Gaussian kernel width in mm. Larger → broader, smoother deformations. */
  val sigma: Double = 50.0

  /** Kernel amplitude (variance). Larger → bigger displacement magnitudes in every mode. */
  val scaleFactor: Double = 100.0

  // ── Model rank ───────────────────────────────────────────────────────────────
  /** Number of GP basis functions = number of deformation modes shown in the UI slider. */
  val rank: Int = 5

  /** Points sampled from the mesh for the Nystrom approximation. More → more accurate, slower. */
  val nystromSamplePoints: Int = 300

  // ─────────────────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit =
    scalismo.initialize()
    implicit val rng: Random = Random(42L)

    // ── Step 1 – Load meshes and landmarks ────────────────────────────────────
    println("Step 1 – Loading meshes …")

    val refMesh    = MeshIO.readMesh(new File(s"$dataDir/$refId.stl")).get
    val targetMesh = MeshIO.readMesh(new File(s"$dataDir/$targetId.stl")).get

    println(s"  Reference : ${refMesh.pointSet.numberOfPoints} vertices")
    println(s"  Target    : ${targetMesh.pointSet.numberOfPoints} vertices")

    println("  Loading landmark CSV …")
    val csvFile              = ScapulaData.csvFile(new File(dataDir))
    val (allLms, _, _)       = ScapulaData.readLandmarkCsv(csvFile)
    val refLms               = allLms(refId)
    val targetLms            = allLms(targetId)

    println(s"  Loaded ${refLms.length} landmarks for reference  (${refLms.map(_.id).mkString(", ")})")
    println(s"  Loaded ${targetLms.length} landmarks for target")

    // ── Step 2 – Rigid registration using landmarks ───────────────────────────
    println("Step 2 – Rigid landmark registration …")

    val rigidTrans     = ScapulaData.rigidFromLandmarks(refLms, targetLms)
    val rigidlyAligned = refMesh.transform(rigidTrans)

    println("  Reference mesh rigidly aligned into target coordinate frame")

    // ── Step 3 – Single Gaussian kernel GP model (rank = 5) ───────────────────
    println(s"Step 3 – Building GP model  [sigma=$sigma mm | scale=$scaleFactor | rank=$rank] …")

    //  One scalar Gaussian kernel, scaled by amplitude, then extended to 3-D output
    //  via a diagonal matrix kernel.  This is intentionally ONE kernel: no combinations.
    val scalarKernel = GaussianKernel[_3D](sigma) * scaleFactor
    val kernel       = DiagonalKernel[_3D](scalarKernel, outputDim = 3)

    //  Zero-mean Gaussian process over 3-D displacement fields
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

    //  Nystrom approximation on the aligned mesh → exactly `rank` basis functions
    val sampler   = UniformMeshSampler3D(rigidlyAligned, nystromSamplePoints)
    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(gp, sampler, numBasisFunctions = rank)

    //  Wrap as a PointDistributionModel so the UI and IO layers understand it
    val pdm = PointDistributionModel[_3D, TriangleMesh](rigidlyAligned, lowRankGP)

    println(s"  GP model built — rank = ${pdm.rank}")

    // ── Step 4 – Save as .h5 ──────────────────────────────────────────────────
    println(s"Step 4 – Saving model to $outputH5 …")

    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(pdm, new File(outputH5)).get

    println("  Saved — you can load this file with StatisticalModelIO.readStatisticalTriangleMeshModel3D")

    // ── Step 5 – Visualise all 5 deformation modes ────────────────────────────
    println("Step 5 – Opening Scalismo UI …")
    println(s"  $rank modes will appear.  Drag each slider to explore that mode.")
    println("  Close the viewer window to exit the programme.")

    val ui         = ScalismoUI()
    val modelGroup = ui.createGroup("Scapula GP Model")
    ui.show(modelGroup, pdm, "GP model")

    //  The Swing event thread keeps the JVM alive until the window is closed.
