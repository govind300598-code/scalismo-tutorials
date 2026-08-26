package scapula

import breeze.linalg.DenseVector
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Single-reference, single Gaussian-kernel experiment.
 *
 * Supervisor requirements:
 *   - ONE Gaussian kernel, ONE reference scapula, GP rank = 5.
 *   - Vary σ (50 / 65 / 80 mm) with scale fixed at 20 mm.
 *   - Vary scale (10 / 20 / 30 mm) with σ fixed at 65 mm.
 *   - Baseline (σ=65, scale=20) is built once and shared between both sets.
 *   - Save each model as .h5; save deformed meshes at ±3 std per mode.
 *   - Print a comparison table.
 *
 * Run:
 *   SCAPULA_DATA_DIR=/path/to/paired_scapulae_STLs \
 *   SCAPULA_OUT_DIR=/path/to/output \
 *   sbt "runMain scapula.GaussianKernelExperiment"
 *
 * Set SCAPULA_UI=false for a headless run (no GUI window).
 */
object GaussianKernelExperiment {

  // ── Parameters ─────────────────────────────────────────────────────────────

  val referenceId: String = "paired_scapula_001_M_64_L"

  /** Fixed rank for all experiments as requested. */
  val gpRank: Int = 5

  /**
   * (sigma_mm, scale_mm) pairs to test.
   * Convention: kernel = DiagonalKernel(GaussianKernel(sigma) * scale, 3),
   * matching RegisterAllFiveScapulae exactly.
   * This means 'scale' acts as the variance; std = sqrt(scale) mm.
   */
  val experiments: IndexedSeq[(Double, Double)] = IndexedSeq(
    (50.0, 20.0),  // Set A-1: narrow kernel
    (65.0, 20.0),  // Set A-2 / B-2: baseline
    (80.0, 20.0),  // Set A-3: wide kernel
    (65.0, 10.0),  // Set B-1: small amplitude
    (65.0, 30.0)   // Set B-3: large amplitude
  )

  // ── Helpers ────────────────────────────────────────────────────────────────

  def tag(sigma: Double, scale: Double): String =
    f"sigma${sigma.toInt}_scale${scale.toInt}_rank$gpRank"

  /**
   * Deform the reference mesh by `coeff` standard deviations along one mode.
   * All other mode coefficients are zero.
   */
  def sampledMesh(model: PointDistributionModel[_3D, TriangleMesh],
                  modeIdx: Int,
                  coeff: Double): TriangleMesh[_3D] = {
    val v = DenseVector.zeros[Double](model.rank)
    v(modeIdx) = coeff
    model.instance(v)
  }

  /**
   * Maximum pointwise displacement between two meshes in correspondence.
   * Assumes they were produced from the same reference (same topology).
   */
  def maxDisplacement(a: TriangleMesh[_3D], b: TriangleMesh[_3D]): Double =
    a.pointSet.points.zip(b.pointSet.points)
      .map { case (p, q) => (p - q).norm }
      .toIndexedSeq
      .max

  // ── Result record ──────────────────────────────────────────────────────────

  final case class ExperimentResult(
    sigma:        Double,
    scale:        Double,
    rank:         Int,
    h5File:       File,
    /** max pointwise displacement of reference vs +3σ deformation, per mode */
    modeMaxDisp:  IndexedSeq[Double]
  )

  // ── Main ───────────────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    // Paths
    val dataDir = Config.dataDir
    require(dataDir.exists(),
      s"Data directory not found: ${dataDir.getAbsolutePath}\n" +
        "Set SCAPULA_DATA_DIR to the folder containing the STL files.")

    val outDir = new File(Config.outDir, "GaussianKernelExperiment")
    outDir.mkdirs()
    println(s"Output directory : ${outDir.getAbsolutePath}")

    // Reference mesh
    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")

    val refRaw = ScapulaData.loadMesh(refFile)
    val refMesh: TriangleMesh[_3D] =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
        refRaw.operations.decimate(Config.modelResolution)
      else refRaw

    println(s"Reference        : $referenceId")
    println(f"Vertices         : ${refMesh.pointSet.numberOfPoints}")
    println(s"GP rank (fixed)  : $gpRank")
    println()

    // Optional UI
    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None

    // Run each experiment
    val results = scala.collection.mutable.Buffer.empty[ExperimentResult]

    for ((sigma, scale) <- experiments) {
      val t = tag(sigma, scale)
      println(s"=== $t ===")

      // 1. Build GP prior — same kernel construction as RegisterAllFiveScapulae
      //    k(x, y) = scale * exp( -||x - y||^2 / (2 * sigma^2) )   [per axis]
      val kernel = DiagonalKernel(GaussianKernel[_3D](sigma) * scale, 3)
      val gp = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

      // 2. Low-rank approximation: fixed rank = gpRank.
      //    We sample 10× points so the eigendecomposition has enough candidates,
      //    then keep the leading gpRank eigenvectors.
      val sampler = UniformMeshSampler3D(refMesh, gpRank * 10)
      val lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]] =
        LowRankGaussianProcess.approximateGP(gp, sampler, numBasisFunctions = gpRank)

      // 3. Wrap in a PointDistributionModel tied to the reference mesh
      val model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)
      println(s"  Actual rank : ${model.rank}")

      // 4. Save model as .h5
      val h5File = new File(outDir, s"$t.h5")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5File).get
      println(s"  Saved .h5   : ${h5File.getName}")

      // 5. Save deformed meshes at ±3 std for each mode
      val modeMaxDisp = scala.collection.mutable.Buffer.empty[Double]
      for (modeIdx <- 0 until model.rank) {
        val posMesh = sampledMesh(model, modeIdx, 3.0)
        val negMesh = sampledMesh(model, modeIdx, -3.0)

        val posFile = new File(outDir, s"${t}_mode${modeIdx + 1}_pos3std.vtk")
        val negFile = new File(outDir, s"${t}_mode${modeIdx + 1}_neg3std.vtk")
        MeshIO.writeMesh(posMesh, posFile).get
        MeshIO.writeMesh(negMesh, negFile).get

        val maxD = maxDisplacement(refMesh, posMesh)
        modeMaxDisp += maxD
        println(f"  Mode ${modeIdx + 1}: max deformation = $maxD%6.1f mm  -> saved ±3std VTKs")
      }

      // 6. Show in UI if enabled
      ui.foreach { scalismoUi =>
        val grp = scalismoUi.createGroup(t)
        scalismoUi.show(grp, model, s"model ($t)")
        scalismoUi.show(grp, model.mean, "mean")
        for (modeIdx <- 0 until model.rank) {
          scalismoUi.show(grp, sampledMesh(model, modeIdx, 3.0),  s"mode${modeIdx + 1} +3std")
          scalismoUi.show(grp, sampledMesh(model, modeIdx, -3.0), s"mode${modeIdx + 1} -3std")
        }
      }

      results += ExperimentResult(sigma, scale, model.rank, h5File, modeMaxDisp.toIndexedSeq)
      println()
    }

    // Write comparison table
    val csvFile = new File(outDir, "comparison_table.csv")
    writeTable(results.toIndexedSeq, csvFile)
    println(s"Comparison table: ${csvFile.getAbsolutePath}")
    println()
    printTable(results.toIndexedSeq)
    println()
    printInterpretation()

    ui.foreach(_ => println("\nUI open — close the Scalismo window to exit."))
  }

  // ── Reporting ──────────────────────────────────────────────────────────────

  def writeTable(results: IndexedSeq[ExperimentResult], file: File): Unit = {
    val pw = new PrintWriter(file)
    try {
      val modeHeaders = (1 to gpRank).map(i => s"mode${i}_max_disp_mm").mkString(",")
      pw.println(s"sigma_mm,scale_mm,rank,h5_file,$modeHeaders,experiment_set")
      results.foreach { r =>
        val maxes = r.modeMaxDisp.padTo(gpRank, 0.0).take(gpRank)
          .map(d => f"$d%.1f").mkString(",")
        val set = if (r.scale == 20.0 && r.sigma != 65.0) "vary_sigma"
                  else if (r.sigma == 65.0 && r.scale != 20.0) "vary_scale"
                  else "baseline"
        pw.println(f"${r.sigma},${r.scale},${r.rank},${r.h5File.getName},$maxes,$set")
      }
    } finally pw.close()
  }

  def printTable(results: IndexedSeq[ExperimentResult]): Unit = {
    val header = f"${"σ (mm)"}%-8s  ${"scale"}%-7s  ${"rank"}%-6s  " +
                 (1 to gpRank).map(i => f"mode$i%-10s").mkString("  ") +
                 "  h5 file"
    println(header)
    println("-" * (header.length + 20))
    results.foreach { r =>
      val modes = r.modeMaxDisp.padTo(gpRank, 0.0).take(gpRank)
        .map(d => f"${d}%.1f mm").map(s => f"$s%-10s").mkString("  ")
      println(f"${r.sigma}%-8.0f  ${r.scale}%-7.0f  ${r.rank}%-6d  $modes  ${r.h5File.getName}")
    }
  }

  def printInterpretation(): Unit = {
    println("=" * 70)
    println("HOW σ AND SCALE AFFECT DEFORMATION")
    println("=" * 70)
    println()
    println("σ (length-scale, mm)")
    println("  Governs spatial correlation: how far the influence of a")
    println("  deformation at one point reaches across the bone surface.")
    println("  Small σ  -> local, patch-like deformations;")
    println("              independent movement of acromion, coracoid, blade.")
    println("  Large σ  -> global, smooth deformations;")
    println("              whole-bone bending/scaling in each mode.")
    println()
    println("scale (kernel amplitude, acts as variance)")
    println("  Governs deformation MAGNITUDE. Prior std ≈ sqrt(scale) mm.")
    println("  scale=10 -> std ≈ 3.2 mm,  ±3std  ≈ ±9.5 mm")
    println("  scale=20 -> std ≈ 4.5 mm,  ±3std  ≈ ±13.4 mm  (baseline)")
    println("  scale=30 -> std ≈ 5.5 mm,  ±3std  ≈ ±16.4 mm")
    println()
    println("rank = 5 throughout — the model has exactly 5 deformation modes.")
    println("Higher-rank models would capture finer shape detail.")
    println("=" * 70)
  }
}
