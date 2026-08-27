package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.utils.Random
import java.io.{File, PrintWriter}
import scala.collection.mutable.ArrayBuffer

/**
 * STAGE 3 -- KERNEL PARAMETER SWEEP.
 *
 * Sweeps the Gaussian kernel bandwidth (sigma) and amplitude (scale), registers
 * the same 3 left-scapula targets for each parameter combination, and records
 * the mean / RMS / HD95 / Hausdorff surface distances in a CSV table.
 *
 * Output (in $SCAPULA_OUT_DIR/GaussianKernelExperiment/):
 *   comparison_table.csv            -- sigma vs scale results table
 *   <ref>_<target>_sig*_s*_registered.vtk -- registered meshes
 *
 * Run with:
 *   SCAPULA_DATA_DIR=/path/to/stls SCAPULA_OUT_DIR=/path/to/out sbt \
 *       "runMain scapula.GaussianKernelExperiment"
 *
 * Runtime: ~10–30 min depending on dataset size and hardware.
 * Set SCAPULA_ICP_ITERS=20 to speed up the sweep at some accuracy cost.
 */
object GaussianKernelExperiment {

  // Parameter grid -- extend or narrow as needed
  val sigmas:     IndexedSeq[Double] = IndexedSeq(30.0, 65.0, 100.0)
  val scales:     IndexedSeq[Double] = IndexedSeq(10.0, 20.0, 30.0)
  val numTargets: Int                = 3  // use 3 targets per combination

  final case class Row(
    sigma:    Double,
    scale:    Double,
    gpRank:   Int,
    meanDist: Double,
    rms:      Double,
    hd95:     Double,
    hd:       Double
  )

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    val outDir  = new File(Config.outDir, "GaussianKernelExperiment")
    outDir.mkdirs()

    println("=" * 80)
    println("STAGE 3 -- GAUSSIAN KERNEL PARAMETER SWEEP")
    println("=" * 80)
    println(s"Data dir  : ${dataDir.getAbsolutePath}")
    println(s"Out dir   : ${outDir.getAbsolutePath}")
    println(s"Sigmas    : ${sigmas.mkString(", ")} mm")
    println(s"Scales    : ${scales.mkString(", ")} mm")
    println(s"Targets   : $numTargets")
    println(s"ICP iters : ${Config.icpIterations}   GP tol: ${Config.gpRelativeTolerance}   max rank: ${Config.gpMaxRank}")

    // ---- load data ---------------------------------------------------------
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dataDir))
    if (!fromHeader)
      println("  WARNING: landmark columns resolved by fallback offsets. Verify Stage 1 [A1] output.")

    val leftSpecs = ScapulaData.specimens(dataDir)
      .filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(leftSpecs.size > numTargets,
      s"Need at least ${numTargets + 1} left specimens with landmarks, found ${leftSpecs.size}")

    // ---- reference ---------------------------------------------------------
    val refSpec = leftSpecs.head
    val rawRef  = ScapulaData.loadMesh(refSpec.file)
    val refMesh = GPRegistration.decimateIfNeeded(rawRef, Config.modelResolution)
    val refLms  = landmarks(refSpec.modelId)
    println(s"\nReference: ${refSpec.modelId}  (${refMesh.pointSet.numberOfPoints} vertices)")

    // ---- pre-align targets once (reused across all parameter combinations) -
    val targetSpecs = leftSpecs.tail.take(numTargets)
    println(s"\nPre-aligning $numTargets targets (rigid, done once) ...")
    val rigidTargets: IndexedSeq[(String, scalismo.mesh.TriangleMesh[scalismo.geometry._3D])] =
      targetSpecs.map { spec =>
        val m = ScapulaData.loadMesh(spec.file)
        val l = landmarks(spec.modelId)
        val (aligned, _) = RigidAlign.landmarkThenIcp(m, l, refMesh, refLms)
        println(s"  ${spec.modelId}")
        (spec.modelId, aligned)
      }

    // ---- parameter sweep ---------------------------------------------------
    val rows        = ArrayBuffer[Row]()
    val combinations = for { s <- sigmas; sc <- scales } yield (s, sc)

    combinations.zipWithIndex.foreach { case ((sigma, scale), idx) =>
      println(s"\n[${idx + 1}/${combinations.length}] sigma=$sigma  scale=$scale ...")

      val model = GPRegistration.buildModel(refMesh, sigma, scale)
      val rank  = model.rank

      val stats = rigidTargets.map { case (id, target) =>
        print(s"  $id ... ")
        val registered = GPRegistration.register(target, model)
        val d          = Metrics.symmetric(registered, target)
        println(d.render)

        // Save registered mesh for every combination (for inspection)
        val name = s"${refSpec.modelId}_${id}_sig${sigma.toInt}_s${scale.toInt}_registered.vtk"
        GPRegistration.saveRegistered(registered, new File(outDir, name))

        d
      }

      val n = stats.length.toDouble
      val row = Row(
        sigma    = sigma,
        scale    = scale,
        gpRank   = rank,
        meanDist = stats.map(_.mean).sum / n,
        rms      = stats.map(_.rms).sum  / n,
        hd95     = stats.map(_.hd95).sum / n,
        hd       = stats.map(_.hd).sum   / n
      )
      rows += row
      println(f"  => mean=${row.meanDist}%.2f  rms=${row.rms}%.2f  HD95=${row.hd95}%.2f  HD=${row.hd}%.2f  rank=$rank")
    }

    // ---- save CSV ----------------------------------------------------------
    val csvFile = new File(outDir, "comparison_table.csv")
    val pw      = new PrintWriter(csvFile)
    pw.println("sigma_mm,scale_mm,gp_rank,mean_mm,rms_mm,hd95_mm,hd_mm")
    rows.foreach { r =>
      pw.println(f"${r.sigma}%.1f,${r.scale}%.1f,${r.gpRank},${r.meanDist}%.4f,${r.rms}%.4f,${r.hd95}%.4f,${r.hd}%.4f")
    }
    pw.close()
    println(s"\nComparison table saved: ${csvFile.getAbsolutePath}")

    // ---- print summary table -----------------------------------------------
    println()
    println(f"${"sigma"}%7s  ${"scale"}%7s  ${"rank"}%6s  ${"mean"}%7s  ${"rms"}%7s  ${"HD95"}%7s  ${"HD"}%8s")
    println("-" * 65)
    rows.foreach { r =>
      println(f"${r.sigma}%7.0f  ${r.scale}%7.0f  ${r.gpRank}%6d  ${r.meanDist}%7.3f  ${r.rms}%7.3f  ${r.hd95}%7.3f  ${r.hd}%8.3f")
    }

    // ---- highlight best combination ----------------------------------------
    if (rows.nonEmpty) {
      val best = rows.minBy(_.hd95)
      println(s"\nBest HD95: sigma=${best.sigma} mm, scale=${best.scale} mm (HD95=${best.hd95}%.2f mm, mean=${best.meanDist}%.2f mm)")
    }

    println("\n" + "=" * 80)
    println("Done.  Results in: " + outDir.getAbsolutePath)
    println("=" * 80)
  }
}
