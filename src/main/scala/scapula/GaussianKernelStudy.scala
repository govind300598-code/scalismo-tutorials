package scapula

import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.utils.Random
import java.io.{File, PrintWriter}

/**
 * Gaussian Kernel Parameter Study for Scapula Non-Rigid Registration.
 *
 * Phase 1 — vary σ (sigma) with amplitude and rank fixed.
 * Phase 2 — vary amplitude with the best σ from Phase 1 and rank fixed.
 *
 * GP rank is held at gpRank = 5 throughout to isolate kernel effects and
 * keep mode interpretation straightforward.
 *
 * Per-experiment outputs:
 *   • registered mesh saved as VTK
 *   • symmetric surface distance stats (mean, RMS, HD95, HD)
 *   • displacement from reference (mean and max)
 *   • qualitative deformation character label
 *
 * A text comparison table is printed and also saved alongside the meshes.
 */
object GaussianKernelStudy {

  val gpRank: Int   = 5
  val icpIters: Int = 40

  val fixedAmplitude: Double              = 20.0
  val sigmaValues: IndexedSeq[Double]     = IndexedSeq(25.0, 50.0, 75.0, 100.0, 150.0)

  val amplitudeValues: IndexedSeq[Double] = IndexedSeq(5.0, 10.0, 20.0, 40.0, 80.0)

  val referenceId: String = "paired_scapula_001_M_64_L"
  val targetId: String    = "paired_scapula_002_M_56_L"

  final case class ExpResult(
    tag: String,
    sigma: Double,
    amplitude: Double,
    actualRank: Int,
    stats: Metrics.SurfaceStats,
    meanDisp: Double,
    maxDisp: Double,
    deformChar: String
  )

  // --------------------------------------------------------------------------
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    if (!dataDir.exists())
      sys.error(
        s"Data directory not found: ${dataDir.getAbsolutePath}\n" +
          "Set SCAPULA_DATA_DIR to the folder containing the STL files and landmark CSV."
      )

    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("!! WARNING: landmark columns resolved by fallback offsets — verify CSV structure.")

    val refFile = new File(dataDir, s"$referenceId.stl")
    val tgtFile = new File(dataDir, s"$targetId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")
    require(tgtFile.exists(), s"Target STL not found: ${tgtFile.getAbsolutePath}")
    require(landmarks.contains(referenceId), s"$referenceId absent from landmark CSV")
    require(landmarks.contains(targetId),    s"$targetId absent from landmark CSV")

    val refRaw  = ScapulaData.loadMesh(refFile)
    val refMesh = if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
      refRaw.operations.decimate(Config.modelResolution) else refRaw
    val refLms  = landmarks(referenceId)

    val tgtRaw  = ScapulaData.loadMesh(tgtFile)
    val tgtMesh = if (tgtRaw.pointSet.numberOfPoints > Config.modelResolution)
      tgtRaw.operations.decimate(Config.modelResolution) else tgtRaw
    val tgtLms  = landmarks(targetId)

    println(s"Reference : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")
    println(s"Target    : $targetId  (${tgtMesh.pointSet.numberOfPoints} vertices)")
    println(f"GP rank   : $gpRank modes (Nystrom)   ICP iters : $icpIters")

    val (targetAligned, _) = RigidAlign.landmarkThenIcp(
      tgtMesh, tgtLms, refMesh, refLms, icpIterations = Config.icpIterations
    )
    println("Target rigidly aligned into reference space")

    val rigidBaseline = Metrics.symmetric(targetAligned, refMesh)
    println(f"Rigid-only baseline error: ${rigidBaseline.render}")

    val outDir = new File(dataDir.getParentFile, "Scapula_GP_KernelStudy")
    outDir.mkdirs()

    val ptStride = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds    = (0 until refMesh.pointSet.numberOfPoints by ptStride).map(PointId(_))

    // ---- Phase 1: vary sigma ----
    println("\n" + "=" * 70)
    println(f"PHASE 1 — vary sigma  (amplitude = $fixedAmplitude mm, rank = $gpRank)")
    println("=" * 70)
    val sigmaResults: IndexedSeq[ExpResult] = sigmaValues.map { sigma =>
      runExperiment(
        refMesh, targetAligned, sigma, fixedAmplitude, ptIds, outDir,
        tag = f"sigma${sigma.toInt}mm_amp${fixedAmplitude.toInt}mm"
      )
    }

    val bestSigmaIdx = sigmaResults.indices.minBy(i => sigmaResults(i).stats.mean)
    val bestSigma    = sigmaResults(bestSigmaIdx).sigma
    println(f"\nBest sigma: $bestSigma%.1f mm  " +
            f"(mean surface error = ${sigmaResults(bestSigmaIdx).stats.mean}%.2f mm)")

    // ---- Phase 2: vary amplitude ----
    println("\n" + "=" * 70)
    println(f"PHASE 2 — vary amplitude  (sigma = $bestSigma mm, rank = $gpRank)")
    println("=" * 70)
    val ampResults: IndexedSeq[ExpResult] = amplitudeValues.map { amp =>
      runExperiment(
        refMesh, targetAligned, bestSigma, amp, ptIds, outDir,
        tag = f"sigma${bestSigma.toInt}mm_amp${amp.toInt}mm"
      )
    }

    val report = buildReport(sigmaResults, ampResults, bestSigma)
    println("\n" + report)

    val reportFile = new File(outDir, "gaussian_kernel_study_report.txt")
    val pw = new PrintWriter(reportFile)
    try pw.println(report) finally pw.close()

    println(f"Report : ${reportFile.getAbsolutePath}")
    println(f"Meshes : ${outDir.getAbsolutePath}")
  }

  // --------------------------------------------------------------------------
  /** Run one non-rigid GP ICP experiment for a given (sigma, amplitude) pair. */
  def runExperiment(
    refMesh: TriangleMesh[_3D],
    targetAligned: TriangleMesh[_3D],
    sigma: Double,
    amplitude: Double,
    ptIds: IndexedSeq[PointId],
    outDir: File,
    tag: String
  )(implicit rng: Random): ExpResult = {

    print(f"  sigma=$sigma%6.1f mm  amp=$amplitude%5.1f mm  rank=$gpRank  ... ")

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](sigma) * amplitude, 3)
    )
    val lowRankGP = LowRankGaussianProcess.approximateGPNystrom(
      refMesh,
      gp,
      numBasisFunctions = gpRank,
      interpolator      = NearestNeighborInterpolator3D()
    )

    var model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)

    for (_ <- 0 until icpIters) {
      val mean = model.mean
      val corr: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
        val pt      = mean.pointSet.point(pid)
        val nearest = targetAligned.operations.closestPointOnSurface(pt).point
        if ((nearest - pt).norm < 15.0) Some((pid, nearest)) else None
      }
      if (corr.nonEmpty)
        model = model.posterior(corr, sigma2 = 1.0)
    }

    val regMesh = model.mean
    val stats   = Metrics.symmetric(regMesh, targetAligned)

    val disps    = refMesh.pointSet.points.zip(regMesh.pointSet.points)
                     .map { case (p, q) => (p - q).norm }.toIndexedSeq
    val meanDisp = disps.sum / disps.size
    val maxDisp  = disps.max
    val char     = characterize(sigma, amplitude, stats.mean)

    println(f"mean=${stats.mean}%5.2f  rms=${stats.rms}%5.2f  " +
            f"HD95=${stats.hd95}%5.2f  disp_mean=${meanDisp}%5.1f  disp_max=${maxDisp}%5.1f  [$char]")

    MeshIO.writeMesh(regMesh, new File(outDir, s"registered_$tag.vtk")).get

    ExpResult(tag, sigma, amplitude, lowRankGP.rank, stats, meanDisp, maxDisp, char)
  }

  // --------------------------------------------------------------------------
  /** Concise qualitative label for a kernel configuration and its result. */
  def characterize(sigma: Double, amplitude: Double, fitMean: Double): String = {
    val spatial = if (sigma < 40)       "very-local"
                  else if (sigma < 70)   "local-moderate"
                  else if (sigma < 110)  "moderate-global"
                  else                   "global"
    val mag     = if (amplitude < 8)    "tight-prior"
                  else if (amplitude < 30) "moderate-prior"
                  else                   "loose-prior"
    val qual    = if (fitMean < 1.5)    "excellent"
                  else if (fitMean < 3.0) "good"
                  else if (fitMean < 5.0) "fair"
                  else                   "poor"
    s"$spatial / $mag / fit=$qual"
  }

  // --------------------------------------------------------------------------
  def buildReport(
    sigRes: IndexedSeq[ExpResult],
    ampRes: IndexedSeq[ExpResult],
    bestSigma: Double
  ): String = {

    val bestAmpIdx = ampRes.indices.minBy(i => ampRes(i).stats.mean)
    val bestAmp    = ampRes(bestAmpIdx).amplitude
    val line80     = "=" * 80
    val dash130    = "-" * 130
    val sb         = new StringBuilder

    def section(title: String): Unit = {
      sb.append("\n").append(line80).append("\n")
      sb.append(title).append("\n")
      sb.append(line80).append("\n")
    }

    def tableHeader(): Unit =
      sb.append(
        f"  ${"sigma(mm)"}%-10s  ${"Rank"}%-5s  ${"Mean(mm)"}%-9s  " +
        f"${"RMS(mm)"}%-8s  ${"HD95(mm)"}%-9s  ${"HD(mm)"}%-8s  " +
        f"${"DispMean"}%-9s  ${"DispMax"}%-9s  Character\n"
      ).append("  ").append(dash130).append("\n")

    def tableRow(r: ExpResult, isBest: Boolean): Unit = {
      val mark = if (isBest) "  <-- BEST" else ""
      sb.append(
        f"  ${r.sigma}%-10.1f  ${r.actualRank}%-5d  ${r.stats.mean}%-9.2f  " +
        f"${r.stats.rms}%-8.2f  ${r.stats.hd95}%-9.2f  ${r.stats.hd}%-8.2f  " +
        f"${r.meanDisp}%-9.2f  ${r.maxDisp}%-9.2f  ${r.deformChar}$mark\n"
      )
    }

    // ---- Header ----
    section("GAUSSIAN KERNEL PARAMETER STUDY — SCAPULA REGISTRATION")
    sb.append(f"  Reference : $referenceId\n")
    sb.append(f"  Target    : $targetId\n")
    sb.append(f"  GP rank   : $gpRank modes (Nystrom approximation)\n")
    sb.append(f"  ICP iters : $icpIters\n")

    // ---- Phase 1 table ----
    section(f"PHASE 1 — Effect of sigma  (amplitude fixed = $fixedAmplitude%.1f mm, rank = $gpRank)")
    tableHeader()
    sigRes.foreach(r => tableRow(r, r.sigma == bestSigma))

    sb.append("\n  Observations — effect of sigma:\n")
    sb.append("  * sigma = 25 mm (very-local): deformation confined to ~25 mm radius patches.\n")
    sb.append("    With only 5 modes the patches are nearly independent — the surface can\n")
    sb.append("    develop folds or spikes. Unrealistic for smooth compact bone.\n")
    sb.append("  * sigma = 50 mm (local-moderate): adjacent features begin to correlate.\n")
    sb.append("    Acromion or coracoid tip may still move independently of the glenoid.\n")
    sb.append("  * sigma = 75–100 mm (moderate-global): deformations span anatomical features\n")
    sb.append("    together (blade, spine, glenoid, coracoid). Matches the spatial scale of\n")
    sb.append("    real inter-individual shape variation in a ~150 mm scapula.\n")
    sb.append("  * sigma = 150 mm (global): modes approach whole-bone shift/tilt.\n")
    sb.append("    With rank = 5 only 5 global modes remain; local anatomy cannot adapt.\n")
    sb.append(f"  --> Best sigma: $bestSigma%.0f mm  " +
              f"(${(bestSigma / 150.0 * 100).toInt}%% of ~150 mm scapula span)\n")

    // ---- Phase 2 table ----
    section(f"PHASE 2 — Effect of amplitude  (sigma fixed = $bestSigma%.1f mm, rank = $gpRank)")
    sb.append(
      f"  ${"amp(mm)"}%-10s  ${"Rank"}%-5s  ${"Mean(mm)"}%-9s  " +
      f"${"RMS(mm)"}%-8s  ${"HD95(mm)"}%-9s  ${"HD(mm)"}%-8s  " +
      f"${"DispMean"}%-9s  ${"DispMax"}%-9s  Character\n"
    ).append("  ").append(dash130).append("\n")
    ampRes.foreach(r => tableRow(r, r.amplitude == bestAmp))

    sb.append("\n  Observations — effect of amplitude:\n")
    sb.append("  * The GP kernel k(x,y) = amplitude * exp(-||x-y||^2 / (2*sigma^2)).\n")
    sb.append("    At any point, the prior std dev per vector component = sqrt(amplitude).\n")
    sb.append("  * Low amplitude (5–10): prior std dev = 2–3 mm per axis. Inter-individual\n")
    sb.append("    scapula differences often exceed this range, so the model cannot deform\n")
    sb.append("    enough to close the gap; surface error remains high.\n")
    sb.append("  * Moderate amplitude (15–30): prior std dev = 4–5.5 mm per axis; total\n")
    sb.append("    expected displacement ~7–10 mm. Covers realistic between-subject variation.\n")
    sb.append("  * High amplitude (40–80): prior std dev = 6–9 mm per axis (~11–15 mm total).\n")
    sb.append("    With only 5 modes the few global bases are pushed to extremes; unrealistic\n")
    sb.append("    bone shapes and posterior overfitting to surface noise can occur.\n")
    sb.append(f"  --> Best amplitude: $bestAmp%.0f mm  " +
              f"(prior std dev per axis = ${math.sqrt(bestAmp)}%.1f mm)\n")

    // ---- Summary ----
    section("SUMMARY — PARAMETER ROLES AND RECOMMENDED RANGE FOR SCAPULA")
    sb.append("\n")
    sb.append("  SIGMA (length-scale)\n")
    sb.append("  -------------------\n")
    sb.append("  Role: controls the spatial extent of deformation correlation.\n")
    sb.append("        k(x,y) = amp * exp(-||x-y||^2 / (2*sigma^2))\n")
    sb.append("  Small sigma: local wrinkles/spikes — unrealistic for smooth bone.\n")
    sb.append("  Large sigma: global shift/tilt only — misses local anatomy.\n")
    sb.append("  Recommended for scapula (~150 mm span): sigma in [50, 100] mm.\n")
    sb.append(f"  Best found in this study: $bestSigma%.0f mm  " +
              f"(${(bestSigma / 150.0 * 100).toInt}%% of bone span)\n")
    sb.append("\n")
    sb.append("  AMPLITUDE (kernel magnitude)\n")
    sb.append("  ----------------------------\n")
    sb.append("  Role: sets prior variance of deformation (mm^2 per component).\n")
    sb.append("        prior std dev per axis = sqrt(amplitude) mm\n")
    sb.append("  Low amplitude: model cannot bridge the shape gap to the target.\n")
    sb.append("  High amplitude: uninformative prior; risks implausible shapes with few modes.\n")
    sb.append("  Recommended for scapula: amplitude in [15, 30]  (std dev 3.9–5.5 mm/axis).\n")
    sb.append(f"  Best found in this study: $bestAmp%.0f mm  " +
              f"(std dev ${math.sqrt(bestAmp)}%.1f mm/axis)\n")
    sb.append("\n")
    sb.append("  GP RANK\n")
    sb.append("  -------\n")
    sb.append(f"  Fixed at $gpRank modes here to isolate kernel effects.\n")
    sb.append("  Low rank (5) captures only dominant global shape variation — good for\n")
    sb.append("  understanding the kernel but insufficient for a production SSM.\n")
    sb.append("  For production use: rank = 50–150 (gpRelativeTolerance ~ 0.01).\n")
    sb.append("  Higher rank = finer correspondence, slower posterior updates.\n")
    sb.append("\n")
    sb.append("  DEFORMATION QUALITY CHECKS\n")
    sb.append("  --------------------------\n")
    sb.append("  1. Visual: open the registered VTK in ParaView or scalismo-ui.\n")
    sb.append("     Inspect for folds, spikes, or discontinuities (small sigma artifacts).\n")
    sb.append("  2. DispMax: if max displacement >> DispMean the deformation is spiky.\n")
    sb.append("  3. HD vs HD95: a large gap (HD >> HD95) signals a few outlier vertices,\n")
    sb.append("     often at thin structures (acromion tip, coracoid) where sigma is too small.\n")
    sb.append("  4. Surface error: mean < 2 mm is excellent for a 150 mm bone with rank=5.\n")

    sb.toString()
  }
}
