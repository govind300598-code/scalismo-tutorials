package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Entry point for the scapula SSM pipeline.
 *
 * Single-pass pipeline:
 *   1. Load specimens + landmark CSV
 *   2. Decimate originals to ~8k working meshes (idempotent — skips existing files)
 *   3. Pick reference specimen (SCAPULA_REF_IDX, default 2)
 *   4. Build Gaussian GP prior on reference (σ=30mm, scale=10mm, NearestNeighborInterpolator3D)
 *   5. For each specimen:
 *        a. Load 8k mesh; mirror right scapulae to left frame
 *        b. Landmark Procrustes + trimmed ICP → rigid_registered/
 *        c. GP-ICP non-rigid registration (single Gaussian kernel) → nonrigid_registered/
 *   6. DataCollection (GPA-style alignment is embedded in scalismo's PCA)
 *        + PCA → StatisticalMeshModel → model/scapula_ssm.h5
 *   7. Save mean, PCA mode meshes (±1σ/±2σ/±3σ, modes 1–3), metrics
 *   8. Launch Scalismo viewer if SCAPULA_UI=true
 *
 * Configure via environment variables (all have defaults):
 *   SCAPULA_DATA_DIR   folder containing STL files and the landmark CSV
 *                      default: /home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula
 *   SCAPULA_OUT_DIR    root output folder
 *                      default: /home/g25upadh/Documents/100 plus scapula data/scapula_atlas_out
 *   SCAPULA_REF_IDX    0-based index into sorted valid-specimen list (default 2)
 *   SCAPULA_MODEL_RES  target vertex count after decimation (default 8000)
 *   SCAPULA_ICP_ITERS  ICP + GP-ICP iteration count (default 40)
 *   SCAPULA_UI         "true" to open viewer after pipeline (default false)
 *
 * Run with:
 *   sbt "runMain scapula.Main"
 */
object Main {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir  = Config.dataDir
    val outDir   = Config.outDir
    val preDir   = new File(outDir, "data/8k")
    val rigidDir = new File(outDir, "rigid_registered")
    val nrDir    = new File(outDir, "nonrigid_registered")
    val modelDir = new File(outDir, "model")
    val meanDir  = new File(outDir, "mean")
    val modesDir = new File(outDir, "pca_modes")
    val metricsDir = new File(outDir, "metrics")
    val sdDir    = new File(outDir, "surface_distance")

    Seq(preDir, rigidDir, nrDir, modelDir, meanDir, modesDir, metricsDir, sdDir)
      .foreach(_.mkdirs())

    println("=" * 80)
    println("  Scapula SSM Pipeline  (single-pass)")
    println("=" * 80)
    println(s"  Data dir : ${dataDir.getAbsolutePath}")
    println(s"  Out dir  : ${outDir.getAbsolutePath}")

    // ── 1. Load specimens and landmarks ──────────────────────────────────────
    val csvFile = ScapulaData.csvFile(dataDir)
    println(s"\nLandmark CSV: ${csvFile.getName}")
    val (lmMap, fromHeader, _) = ScapulaData.readLandmarkCsv(csvFile)
    if (!fromHeader)
      println("  WARNING: landmark columns resolved by fallback offsets — verify CSV header")

    val allSpecs  = ScapulaData.specimens(dataDir)
    val validSpecs = allSpecs.filter(s => lmMap.contains(s.modelId))
    println(s"  ${validSpecs.length} / ${allSpecs.length} specimens have landmark rows")

    require(validSpecs.nonEmpty, "No specimens with landmarks found. Check SCAPULA_DATA_DIR.")

    // ── 2. Decimate to 8k (idempotent) ───────────────────────────────────────
    println(s"\n[DECIMATION] ~${Config.modelResolution}-vertex working meshes → ${preDir.getPath}")
    Decimation.generateAll(validSpecs, preDir, Config.modelResolution)

    // ── 3. Pick reference specimen ────────────────────────────────────────────
    val refIdx  = Config.refIdx.min(validSpecs.length - 1)
    val refSpec = validSpecs(refIdx)
    val refLms  = if (refSpec.isRight) ScapulaData.mirrorLandmarks(lmMap(refSpec.modelId))
                  else lmMap(refSpec.modelId)
    println(s"\n[REFERENCE] index=$refIdx  id=${refSpec.modelId}")

    val refRaw  = ScapulaData.loadMesh(new File(preDir, refSpec.modelId + ".stl"))
    val refMesh = Decimation.keepLargestComponent(refRaw)
    println(f"  ${refMesh.pointSet.numberOfPoints} vertices  (after keepLargestComponent)")

    // Save reference for the viewer
    MeshIO.writeMesh(refMesh, new File(modelDir, "reference.stl"))
      .fold(e => println(s"  WARNING: could not save reference: $e"), _ => ())

    // ── 4. Build Gaussian GP prior ────────────────────────────────────────────
    // Kernel: DiagonalKernel(GaussianKernel(σ=30mm) × scale², 3)
    // Interpolator: NearestNeighborInterpolator3D (see NonRigidReg for rationale)
    println(s"\n[GP PRIOR] σ=${NonRigidReg.gpSigma}mm  scale=${NonRigidReg.gpScaleFactor}mm  " +
            s"tol=${Config.gpRelativeTolerance}  maxRank=${Config.gpMaxRank}")
    val lowRankGP = NonRigidReg.buildPrior(refMesh,
      relativeTolerance = Config.gpRelativeTolerance,
      maxRank           = Config.gpMaxRank)

    // ── 5. Register all specimens ─────────────────────────────────────────────
    println(s"\n[REGISTRATION] ${validSpecs.length} specimens  " +
            s"(ICP iters=${Config.icpIterations}, GP-ICP iters=${Config.icpIterations})")

    val sdLog = new PrintWriter(new File(sdDir, "registration_distances.csv"))
    sdLog.println("modelId,mean_mm,rms_mm,hd95_mm,hd_mm")

    val registered = scala.collection.mutable.ArrayBuffer.empty[TriangleMesh[_3D]]

    validSpecs.foreach { spec =>
      val specLms = lmMap(spec.modelId)
      try {
        print(s"  ${spec.modelId} ")

        // Load 8k working mesh; mirror right scapulae to common left frame
        val raw8k  = ScapulaData.loadMesh(new File(preDir, spec.modelId + ".stl"))
        val mesh8k = Decimation.keepLargestComponent(raw8k)
        val (workMesh, workLms) =
          if (spec.isRight) (ScapulaData.mirrorMesh(mesh8k), ScapulaData.mirrorLandmarks(specLms))
          else (mesh8k, specLms)

        // a. Landmark Procrustes + trimmed ICP
        val (rigidMesh, _) = RigidAlign.landmarkThenIcp(
          workMesh, workLms,
          refMesh,  refLms,
          icpIterations = Config.icpIterations)
        MeshIO.writeMesh(rigidMesh, new File(rigidDir, spec.modelId + ".stl"))
          .getOrElse(throw new RuntimeException("rigid write failed"))
        print("[rigid] ")

        // b. GP-ICP non-rigid registration (single Gaussian kernel)
        val nrMesh = NonRigidReg.register(
          refMesh,   rigidMesh, lowRankGP,
          nIter              = Config.icpIterations,
          sigma2             = 1.0,
          numCorrespondences = 500)
        MeshIO.writeMesh(nrMesh, new File(nrDir, spec.modelId + ".stl"))
          .getOrElse(throw new RuntimeException("nonrigid write failed"))
        print("[nreg] ")

        // Registration quality: symmetric surface distance nrMesh ↔ rigidMesh
        val sd = Metrics.symmetric(nrMesh, rigidMesh)
        sdLog.println(s"${spec.modelId},${sd.mean},${sd.rms},${sd.hd95},${sd.hd}")
        println(f"done  reg-dist mean=${sd.mean}%5.2f mm  hd95=${sd.hd95}%5.2f mm")

        registered += nrMesh
      } catch {
        case e: Exception => println(s"  ERROR: ${e.getMessage}")
      }
    }
    sdLog.close()
    println(s"\n  ${registered.length} / ${validSpecs.length} successful registrations")
    require(registered.nonEmpty, "No successful registrations — cannot build SSM.")

    // ── 6. PCA → StatisticalMeshModel ────────────────────────────────────────
    // DataCollection.fromTriangleMesh3DSequence performs GPA-style alignment
    // (scalismo uses Procrustes alignment internally before PCA).
    println(s"\n[SSM]  PCA on ${registered.length} registered meshes in dense correspondence")
    val ssm = SSMBuilder.buildFromCorrespondences(registered.toIndexedSeq)
    println(s"  SSM rank = ${ssm.rank}")

    val modelFile = new File(modelDir, "scapula_ssm.h5")
    SSMBuilder.saveModel(ssm, modelFile)
    SSMBuilder.saveMean(ssm, new File(meanDir, "mean.stl"))

    // ── 7. Variance report + metrics ─────────────────────────────────────────
    println("\n[METRICS]")
    val report = SSMBuilder.varianceReport(ssm, "SSM")
    val pw = new PrintWriter(new File(metricsDir, "variance_report.txt"))
    pw.println(report); pw.close()

    val metrics = SSMBuilder.computeMetrics(ssm, registered.toIndexedSeq, "SSM")
    val mpw = new PrintWriter(new File(metricsDir, "ssm_metrics.csv"))
    mpw.println("label,rank,mode1_pct,top5_pct,top10_pct,generalization_mm,specificity_mm")
    mpw.println(s"SSM,${metrics.rank},${metrics.mode1Pct},${metrics.top5Pct}," +
      s"${metrics.top10Pct},${metrics.generalization},${metrics.specificity}")
    mpw.close()
    println(f"  Mode1=${metrics.mode1Pct}%.1f%%  Top5=${metrics.top5Pct}%.1f%%  " +
            f"Top10=${metrics.top10Pct}%.1f%%")
    println(f"  Generalization=${metrics.generalization}%.3f mm  " +
            f"Specificity=${metrics.specificity}%.3f mm")

    // ── 8. Save PCA mode meshes (modes 1–3, ±1σ/±2σ/±3σ) ────────────────────
    println("\n[PCA MODES]")
    saveModeDeformations(ssm, modesDir)

    // ── 9. Launch viewer ──────────────────────────────────────────────────────
    println("\n" + "=" * 80)
    println(s"  Pipeline complete.")
    println(s"  Model   → ${modelFile.getPath}")
    println(s"  Mean    → ${new File(meanDir, "mean.stl").getPath}")
    println(s"  Results → ${outDir.getAbsolutePath}")
    println("=" * 80)

    if (Config.showUi) {
      println("\n[UI] Launching Scalismo viewer…")
      VisualizationApp.main(Array.empty)
    } else {
      println("\nTo view results:  SCAPULA_UI=true sbt \"runMain scapula.Main\"")
      println("  or separately:  sbt \"runMain scapula.VisualizationApp\"")
    }
  }

  // ---------------------------------------------------------------------------
  // PCA mode deformation shapes  (±1σ / ±2σ / ±3σ for modes 1–3)
  // ---------------------------------------------------------------------------
  private def saveModeDeformations(ssm: StatisticalMeshModel, outDir: File): Unit = {
    val nModes = math.min(3, ssm.rank)
    for (modeIdx <- 0 until nModes) {
      val sigma = math.sqrt(ssm.gp.klBasis(modeIdx).eigenvalue)
      for (alpha <- Seq(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)) {
        val coeffs = DenseVector.zeros[Double](ssm.rank)
        coeffs(modeIdx) = alpha * sigma
        val shape = ssm.instance(coeffs)
        val tag   = alpha match {
          case 0.0            => "mean"
          case a if a < 0.0   => s"minus${(-a).toInt}sd"
          case a              => s"plus${a.toInt}sd"
        }
        MeshIO.writeMesh(shape, new File(outDir, s"mode${modeIdx + 1}_${tag}.stl"))
          .getOrElse(println(s"  WARNING: could not save mode${modeIdx+1}_$tag"))
      }
      println(f"  Mode ${modeIdx+1}: σ=${sigma}%.1f mm  → ±1σ/±2σ/±3σ saved")
    }
    println(s"  PCA modes → ${outDir.getPath}")
  }
}
