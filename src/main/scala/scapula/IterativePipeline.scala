package scapula

import scalismo.geometry._3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random

import java.io.{File, PrintWriter}
import java.time.Instant

/**
 * Main iterative SSM pipeline:
 *   Initial Reference → SSM1 → Mean1 → SSM2 → Mean2 → SSM3 → Mean3 → SSM4 → Mean4
 *
 * Per-iteration folder layout:
 *   results/SSM{n}/
 *     reference/          – reference mesh used for this iteration
 *     rigid_registered/   – landmark-rigid aligned meshes
 *     nonrigid_registered/– GPMM registered meshes (in dense correspondence)
 *     correspondences/    – per-subject vertex count confirmation
 *     mean/               – mean mesh of this iteration
 *     model/              – SSM .h5 file
 *     PCA_modes/          – mode deformation meshes (-3σ, 0, +3σ for modes 1-3)
 *     surface_distance/   – per-subject registration surface distance metrics
 *     metrics/            – PCA variance table, generalization, specificity
 *     logs/               – run log and reproducibility record
 */
object IterativePipeline {

  private def dir(base: File, name: String): File = {
    val d = new File(base, name); d.mkdirs(); d
  }

  // ---------------------------------------------------------------------------
  // Registration pipeline for one SSM iteration
  // ---------------------------------------------------------------------------

  /**
   * Run one full SSM iteration.
   *
   * @param iterLabel   e.g. "SSM1"
   * @param reference   reference mesh for this iteration
   * @param specimens   all scapula specimens to register
   * @param landmarks   landmark map (modelId → landmarks)
   * @param referenceL  landmarks for the reference mesh (same IDs as in `landmarks`)
   * @param outDir      base output dir, e.g. results/SSM1/
   * @param lowRankGP   the shared GP prior (σ=30, scale=10, fixed across iterations)
   * @return (ssm, meanMesh, registeredMeshes, allMetrics)
   */
  def runIteration(
    iterLabel: String,
    reference: TriangleMesh[_3D],
    referenceL: IndexedSeq[scalismo.geometry.Landmark[_3D]],
    specimens: IndexedSeq[ScapulaData.Specimen],
    landmarks: Map[String, IndexedSeq[scalismo.geometry.Landmark[_3D]]],
    outDir: File,
    lowRankGP: LowRankGaussianProcess[_3D, scalismo.geometry.EuclideanVector[_3D]]
  )(implicit rng: Random): (PointDistributionModel[_3D, TriangleMesh], TriangleMesh[_3D], IndexedSeq[TriangleMesh[_3D]]) = {

    val t0 = System.currentTimeMillis()
    println(s"\n" + "=" * 80)
    println(s"  $iterLabel  (${specimens.length} specimens)")
    println("=" * 80)

    val rigidDir    = dir(outDir, "rigid_registered")
    val nonRigidDir = dir(outDir, "nonrigid_registered")
    val sdDir       = dir(outDir, "surface_distance")
    val logDir      = dir(outDir, "logs")
    val refDir      = dir(outDir, "reference")

    // Save the reference used for this iteration
    MeshIO.writeMesh(reference, new File(refDir, s"${iterLabel}_reference.stl"))

    // Collect successfully registered meshes
    val registered = scala.collection.mutable.ArrayBuffer.empty[TriangleMesh[_3D]]
    val sdLog      = new PrintWriter(new File(sdDir, "registration_surface_distances.csv"))
    sdLog.println("modelId,mean_mm,rms_mm,hd95_mm,hd_mm")

    specimens.foreach { spec =>
      landmarks.get(spec.modelId) match {
        case None =>
          println(s"  [SKIP] ${spec.modelId}: no landmark row in CSV")

        case Some(specLms) =>
          try {
            print(s"  ${spec.modelId} ")

            // ── 1. Load working mesh (full-res STL used as registration TARGET) ──
            val target = ScapulaData.loadMesh(spec.file)

            // ── 2. Landmark-based rigid registration ──────────────────────────────
            // Mirror right scapulae so all are in a common left-scapula frame.
            val (workMesh, workLms) =
              if (spec.isRight) (ScapulaData.mirrorMesh(target), ScapulaData.mirrorLandmarks(specLms))
              else (target, specLms)

            val (rigidMesh, rigidLms) = RigidAlign.landmarkThenIcp(
              workMesh, workLms,
              reference, referenceL,
              icpIterations = Config.icpIterations
            )
            MeshIO.writeMesh(rigidMesh, new File(rigidDir, spec.modelId + ".stl"))
            print("[rigid] ")

            // ── 3. GPMM non-rigid registration ───────────────────────────────────
            val nrMesh = NonRigidReg.register(
              reference, rigidMesh, lowRankGP,
              nIter = Config.icpIterations,
              sigma2 = 1.0,
              numCorrespondences = 500
            )
            MeshIO.writeMesh(nrMesh, new File(nonRigidDir, spec.modelId + ".stl"))
            print("[nreg] ")

            // ── 4. Registration surface distance (nrMesh ↔ rigidMesh) ─────────────
            val sd = Metrics.symmetric(nrMesh, rigidMesh)
            sdLog.println(s"${spec.modelId},${sd.mean},${sd.rms},${sd.hd95},${sd.hd}")
            println(f"done  reg-dist mean=${sd.mean}%5.2f mm")

            registered += nrMesh
          } catch {
            case e: Exception =>
              println(s"  ERROR: ${e.getMessage}")
          }
      }
    }

    sdLog.close()
    println(s"\n  $iterLabel: ${registered.length}/${specimens.length} successful registrations")

    // ── 5. Build SSM ──────────────────────────────────────────────────────────
    require(registered.nonEmpty, s"$iterLabel: no successful registrations – cannot build SSM")
    val ssm = SSMBuilder.buildFromCorrespondences(registered.toIndexedSeq)
    SSMBuilder.saveModel(ssm, new File(dir(outDir, "model"), s"$iterLabel.h5"))
    SSMBuilder.saveMean(ssm, new File(dir(outDir, "mean"), s"${iterLabel}_mean.stl"))

    // ── 6. Variance report ────────────────────────────────────────────────────
    val report = SSMBuilder.varianceReport(ssm, iterLabel)
    val pw = new PrintWriter(new File(dir(outDir, "metrics"), "variance_report.txt"))
    pw.println(report); pw.close()

    // ── 7. Save PCA mode meshes (modes 1-3, ±3σ) ─────────────────────────────
    saveModeDeformations(ssm, dir(outDir, "PCA_modes"), iterLabel)

    // ── 8. Reproducibility log ────────────────────────────────────────────────
    saveReproducibilityLog(
      iterLabel, reference, lowRankGP, registered.length, specimens.length,
      System.currentTimeMillis() - t0, new File(logDir, "reproducibility.txt")
    )

    (ssm, ssm.mean, registered.toIndexedSeq)
  }

  // ---------------------------------------------------------------------------
  // PCA mode deformation shapes
  // ---------------------------------------------------------------------------

  def saveModeDeformations(
    ssm: PointDistributionModel[_3D, TriangleMesh],
    outDir: File,
    label: String
  ): Unit = {
    import breeze.linalg.DenseVector
    val nModes = math.min(3, ssm.rank)
    for (modeIdx <- 0 until nModes) {
      for (alpha <- Seq(-3.0, 0.0, 3.0)) {
        val coeffs = DenseVector.zeros[Double](ssm.rank)
        coeffs(modeIdx) = alpha * math.sqrt(ssm.variance(modeIdx))
        val shape = ssm.instance(coeffs)
        val tag   = if (alpha < 0) s"minus${(-alpha).toInt}sd" else if (alpha == 0.0) "mean" else s"plus${alpha.toInt}sd"
        MeshIO.writeMesh(shape, new File(outDir, s"${label}_mode${modeIdx + 1}_${tag}.stl"))
      }
    }
    println(s"  Saved PCA mode meshes (modes 1-$nModes, ±3σ) → ${outDir.getPath}")
  }

  // ---------------------------------------------------------------------------
  // Reproducibility log
  // ---------------------------------------------------------------------------

  private def saveReproducibilityLog(
    label: String,
    reference: TriangleMesh[_3D],
    gp: LowRankGaussianProcess[_3D, _],
    nSuccess: Int,
    nTotal: Int,
    elapsedMs: Long,
    file: File
  ): Unit = {
    file.getParentFile.mkdirs()
    val pw = new PrintWriter(file)
    pw.println(s"Iteration       : $label")
    pw.println(s"Timestamp       : ${Instant.now()}")
    pw.println(s"Reference pts   : ${reference.pointSet.numberOfPoints}")
    pw.println(s"Reference tris  : ${reference.triangulation.triangles.length}")
    pw.println(s"GP sigma        : ${NonRigidReg.gpSigma} mm")
    pw.println(s"GP scaleFactor  : ${NonRigidReg.gpScaleFactor} mm")
    pw.println(s"GP variance     : ${NonRigidReg.gpScaleFactor * NonRigidReg.gpScaleFactor} mm²")
    pw.println(s"GP rank         : ${gp.rank}")
    pw.println(s"Interpolator    : NearestNeighborInterpolator3D (scalismo)")
    pw.println(s"ICP iterations  : ${Config.icpIterations}")
    pw.println(s"NR ICP iters    : ${Config.icpIterations}")
    pw.println(s"NR sigma²       : 1.0")
    pw.println(s"Correspondences : 500 per iteration")
    pw.println(s"Decimation      : ~${Config.modelResolution} vertices (stride-based)")
    pw.println(s"Landmark set    : ${ScapulaData.landmarkNames.mkString(", ")}")
    pw.println(s"Rigid align     : LandmarkRegistration.rigid3D + trimmed ICP")
    pw.println(s"Successful      : $nSuccess / $nTotal")
    pw.println(f"Elapsed         : ${elapsedMs / 1000.0}%.1f s")
    pw.close()
  }
}
