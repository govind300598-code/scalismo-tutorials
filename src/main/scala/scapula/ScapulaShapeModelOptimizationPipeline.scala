package scapula

import breeze.linalg.DenseVector
import scalismo.common.*
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.kernels.*
import scalismo.numerics.*
import scalismo.registration.*
import scalismo.statisticalmodel.*
import scalismo.ui.api.*
import scalismo.utils.Random

import java.awt.Color
import java.io.{File, PrintWriter}

/**
 * Stage 2 – Non-rigid registration and statistical shape model construction.
 *
 * Kernel design follows Dennis Madsen's heuristic:
 *   σ_coarse ≈ longest_mesh_dimension / 2   (captures global shape variation)
 *   σ_fine   ≈ σ_coarse / 10                (captures local surface detail)
 *
 * For the left scapulae in this dataset (longest dimension ≈ 130–150 mm) the
 * recommended starting point is σ_coarse = 70, σ_fine = 7.  The kernel sweep
 * in Phase 1 confirms or refines this around those anchor values.
 */
object ScapulaShapeModelOptimizationPipeline {

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir    = Config.dataDir
    val outDir = Config.outDir
    outDir.mkdirs()

    // -----------------------------------------------------------------------
    // Load landmarks and specimens
    // -----------------------------------------------------------------------
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dir))
    if (!fromHeader)
      println("!! CSV columns resolved by fallback offsets – verify landmarks before trusting metrics.")

    val allSpecimens = ScapulaData.specimens(dir)
    val leftSpecimens = allSpecimens
      .filter(s => !s.isRight && landmarks.contains(s.modelId))
      .sortBy(_.modelId)

    require(leftSpecimens.nonEmpty, s"No left-side STL files with landmark entries found in ${dir.getPath}")

    // -----------------------------------------------------------------------
    // Reference: first left specimen (arbitrary; re-registered in later passes)
    // -----------------------------------------------------------------------
    val refSpec   = leftSpecimens.head
    val refRaw    = ScapulaData.loadMesh(refSpec.file)
    val refMesh   = if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
                      refRaw.operations.decimate(Config.modelResolution)
                    else refRaw
    val refLms    = landmarks(refSpec.modelId)

    println(s"Reference : ${refSpec.modelId}  (${refMesh.pointSet.numberOfPoints} vertices after decimation)")

    // -----------------------------------------------------------------------
    // Estimate mesh extent to guide the kernel sweep.
    //
    // Dennis Madsen: "For the sigma value, I like to use a value that is 1/2 or
    // 1/3 of the longest distance in the mesh."
    // -----------------------------------------------------------------------
    val pts = refMesh.pointSet.points.toIndexedSeq
    val xExt = pts.map(_.x).max - pts.map(_.x).min
    val yExt = pts.map(_.y).max - pts.map(_.y).min
    val zExt = pts.map(_.z).max - pts.map(_.z).min
    val longestAxis = Seq(xExt, yExt, zExt).max
    println(f"  Mesh extents: X=${xExt}%.1f  Y=${yExt}%.1f  Z=${zExt}%.1f mm  (longest axis = $longestAxis%.1f mm)")

    // Anchor values derived from the heuristic.  The user's instruction ("seventy
    // and seven") sets the coarse anchor at 70 mm and the fine anchor at 7 mm.
    val coarseAnchor = 70.0
    val fineAnchor   =  7.0

    // Sweep a ± neighbourhood around the anchor so the pipeline self-tunes.
    val coarseSigmas = Seq(50.0, 60.0, coarseAnchor, 80.0, 90.0)
    val fineSigmas   = Seq(fineAnchor, 10.0, 15.0)

    // -----------------------------------------------------------------------
    // Helper: build a low-rank GP from a mixed 3-scale Gaussian kernel.
    //
    // Three scales give:
    //   global  – large σ, large amplitude: translations / overall bending
    //   medium  – medium σ, medium amplitude: regional shape changes
    //   local   – small σ, small amplitude: surface micro-detail
    // -----------------------------------------------------------------------
    def buildLowRankGP(sigmaCoarse: Double, sigmaFine: Double,
                       tol: Double = Config.gpRelativeTolerance): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = {
      val sigmaMid   = math.sqrt(sigmaCoarse * sigmaFine)   // geometric mean of coarse and fine
      val kCoarse    = GaussianKernel[_3D](sigma = sigmaCoarse, scaleFactor = 10.0)
      val kMid       = GaussianKernel[_3D](sigma = sigmaMid,    scaleFactor =  5.0)
      val kFine      = GaussianKernel[_3D](sigma = sigmaFine,   scaleFactor =  3.0)
      val combined   = DiagonalKernel3D(kCoarse + kMid + kFine, 3)
      val zeroMean   = Field(EuclideanSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
      val gp         = GaussianProcess[_3D, EuclideanVector[_3D]](zeroMean, combined)
      LowRankGaussianProcess.approximateGPCholesky(
        refMesh, gp,
        relativeTolerance = tol,
        interpolator      = TriangleMeshInterpolator3D[EuclideanVector[_3D]]()
      )
    }

    // -----------------------------------------------------------------------
    // Helper: run GP-ICP non-rigid registration of `target` to `refMesh`.
    // -----------------------------------------------------------------------
    def nonRigidRegister(target: TriangleMesh[_3D],
                         lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]]): TriangleMesh[_3D] = {
      val transformSpace = GaussianProcessTransformationSpace[_3D](lowRankGP)
      val fixedImg       = refMesh.operations.toDistanceImage
      val movingImg      = target.operations.toDistanceImage
      val sampler        = FixedPointsUniformMeshSampler3D(refMesh, 2000)
      val metric         = MeanSquaresMetric(fixedImg, movingImg, transformSpace, sampler)
      val optimizer      = LBFGSOptimizer(maxNumberOfIterations = Config.icpIterations)
      val regularizer    = L2Regularizer[_3D](transformSpace)
      val reg            = Registration(metric, regularizer, regularizationWeight = 5e-5, optimizer)
      val finalParams    = reg.iterator(DenseVector.zeros[Double](lowRankGP.rank)).toSeq.last.parameters
      refMesh.transform(transformSpace.transformationForParameters(finalParams))
    }

    // -----------------------------------------------------------------------
    // Phase 1 – Kernel sweep on the first target to find optimal (σ_c, σ_f).
    // -----------------------------------------------------------------------
    println()
    println("=== Phase 1 – Dennis Madsen Heuristic Kernel Sweep ===")
    println(f"    Coarse anchor = ${coarseAnchor}%.0f mm   Fine anchor = ${fineAnchor}%.0f mm")

    val sweepSpec  = leftSpecimens.drop(1).headOption.getOrElse(leftSpecimens.head)
    val sweepRaw   = ScapulaData.loadMesh(sweepSpec.file)
    val sweepMesh  = if (sweepRaw.pointSet.numberOfPoints > Config.modelResolution)
                       sweepRaw.operations.decimate(Config.modelResolution)
                     else sweepRaw
    val sweepLms   = landmarks(sweepSpec.modelId)

    // Rigid-align the sweep target to the reference before evaluating kernels.
    val (sweepAligned, _) = RigidAlign.landmarkThenIcp(sweepMesh, sweepLms, refMesh, refLms)

    val sweepReportFile = new File(outDir, "scapula_kernel_sweep_report.csv")
    val sweepPw = new PrintWriter(sweepReportFile)
    sweepPw.println("Coarse_Sigma,Fine_Sigma,GP_Rank,Mean_mm,HD95_mm,RMSE_mm")

    var bestCoarseSigma = coarseAnchor
    var bestFineSigma   = fineAnchor
    var minSweepMean    = Double.MaxValue

    for (cSig <- coarseSigmas; fSig <- fineSigmas) {
      print(f"  Coarse σ=${cSig}%.0f  Fine σ=${fSig}%.0f  ...")
      val lowRankGP = buildLowRankGP(cSig, fSig, tol = 0.05)  // looser tol for speed during sweep
      print(f" rank=${lowRankGP.rank}  fitting...")
      val fitted    = nonRigidRegister(sweepAligned, lowRankGP)
      val stats     = Metrics.symmetric(fitted, sweepAligned)
      println(f"  ${stats.render}")

      sweepPw.println(f"$cSig%.1f,$fSig%.1f,${lowRankGP.rank},${stats.mean}%.4f,${stats.hd95}%.4f,${stats.rms}%.4f")
      sweepPw.flush()

      if (stats.mean < minSweepMean) {
        minSweepMean    = stats.mean
        bestCoarseSigma = cSig
        bestFineSigma   = fSig
      }
    }
    sweepPw.close()
    println(f"\n  Best kernel: Coarse σ=${bestCoarseSigma}%.0f  Fine σ=${bestFineSigma}%.0f  (mean dist=${minSweepMean}%.4f mm)")
    println(f"  Sweep report saved to: ${sweepReportFile.getAbsolutePath}")

    // -----------------------------------------------------------------------
    // Phase 2 – Build optimal GP and register all left-side specimens.
    // -----------------------------------------------------------------------
    println()
    println("=== Phase 2 – Building Statistical Shape Model & Executing Non-Rigid Registration ===")

    val optLowRankGP = buildLowRankGP(bestCoarseSigma, bestFineSigma, tol = Config.gpRelativeTolerance)
    val pdm          = PointDistributionModel(refMesh, optLowRankGP)
    println(s"  GP rank: ${optLowRankGP.rank}  (relative tolerance ${Config.gpRelativeTolerance})")
    println(s"  Registering ${leftSpecimens.size} specimens ...")

    val finalReport = new File(outDir, "final_scapula_evaluation_report.csv")
    val finalPw     = new PrintWriter(finalReport)
    finalPw.println("Specimen_ID,Mean_mm,HD95_mm,RMSE_mm,HD_mm")

    val registeredMeshes = leftSpecimens.zipWithIndex.map { case (spec, idx) =>
      println(s"  [${idx + 1}/${leftSpecimens.size}] ${spec.modelId}")
      val raw    = ScapulaData.loadMesh(spec.file)
      val mesh   = if (raw.pointSet.numberOfPoints > Config.modelResolution)
                     raw.operations.decimate(Config.modelResolution)
                   else raw
      val lms    = landmarks(spec.modelId)
      val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, refMesh, refLms)
      val fitted  = nonRigidRegister(aligned, optLowRankGP)
      val stats   = Metrics.symmetric(fitted, aligned)
      println(f"    ${stats.render}")

      val outFile = new File(outDir, s"fitted_${spec.modelId}.stl")
      MeshIO.writeMesh(fitted, outFile)

      finalPw.println(s"${spec.modelId},${stats.mean},${stats.hd95},${stats.rms},${stats.hd}")
      finalPw.flush()
      (spec, aligned, fitted, stats)
    }
    finalPw.close()

    // Summary
    val means = registeredMeshes.map(_._4.mean)
    val hd95s = registeredMeshes.map(_._4.hd95)
    println()
    println("=== Registration Summary ===")
    println(f"  Mean surface distance : avg=${means.sum / means.length}%.4f  min=${means.min}%.4f  max=${means.max}%.4f  mm")
    println(f"  HD95                  : avg=${hd95s.sum / hd95s.length}%.4f  min=${hd95s.min}%.4f  max=${hd95s.max}%.4f  mm")
    println(f"  Final report : ${finalReport.getAbsolutePath}")

    // Best / worst for UI labelling
    val best  = registeredMeshes.minBy(_._4.mean)
    val worst = registeredMeshes.maxBy(_._4.mean)

    // -----------------------------------------------------------------------
    // Phase 3 – Scalismo UI
    // -----------------------------------------------------------------------
    if (Config.showUi) {
      val ui     = ScalismoUI()
      val regGrp = ui.createGroup("Registration_Comparison_Results")

      val refView = ui.show(regGrp, refMesh, "Reference_Template")
      refView.color = Color.LIGHT_GRAY

      registeredMeshes.foreach { case (spec, aligned, fitted, stats) =>
        val label = if (spec == best._1) "Best Case"
                    else if (spec == worst._1) "Worst Case"
                    else "Standard Target"
        val targetView = ui.show(regGrp, aligned, s"Target_${spec.modelId} ($label)")
        targetView.color = Color.RED
        val fittedView = ui.show(regGrp, fitted, s"Fitted_${spec.modelId} ($label)")
        fittedView.color = Color.GRAY
      }

      val modelGrp = ui.createGroup("Statistical_Shape_Model_Modes")
      ui.show(modelGrp, pdm, "Scapula_PDM_Modes")

      println()
      println("Scalismo UI open. Red = target (rigid-aligned), Gray = fitted (non-rigid).")
      println("Press ENTER to exit.")
      scala.io.StdIn.readLine()
    }
  }
}
