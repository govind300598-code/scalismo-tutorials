package scapula

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.awt.Color
import java.io.{File, PrintWriter}

/**
 * GP kernel parameter sweep over a 3x3 grid of (sigma, scale).
 *
 * Experiments (one parameter varied at a time from baseline T5):
 *
 *  ID  sigma(mm)  scale(mm)  rank  output h5 prefix
 *  T1  40         10         5     scapula_ffdm_sig40_s10_r5
 *  T2  40         20         5     scapula_ffdm_sig40_s20_r5
 *  T3  40         30         5     scapula_ffdm_sig40_s30_r5
 *  T4  65         10         5     scapula_ffdm_sig65_s10_r5
 *  T5  65         20         5     scapula_ffdm_sig65_s20_r5  <- baseline
 *  T6  65         30         5     scapula_ffdm_sig65_s30_r5
 *  T7  90         10         5     scapula_ffdm_sig90_s10_r5
 *  T8  90         20         5     scapula_ffdm_sig90_s20_r5
 *  T9  90         30         5     scapula_ffdm_sig90_s30_r5
 *
 * Reference (smooth, no artifacts): paired_scapula_001_M_64_L
 * Five targets chosen for maximal anatomical diversity (age and sex):
 *   002_M_56, 004_F_67, 007_M_26, 010_F_43, 012_M_68
 *
 * For each (experiment, target) pair:
 *   - prior .h5   : prior GP shape model
 *   - posterior .h5: posterior after GP-ICP fitting
 *   - registered .vtk: posterior mean mesh (same connectivity as reference)
 *   - mode VTKs (+-3std) for the prior
 *
 * Summary: comparison_table.csv + console table (mean/rms/HD95 averaged over 5 targets).
 */
object GaussianKernelExperiment {

  val referenceId: String = "paired_scapula_001_M_64_L"

  // Five targets chosen for anatomical diversity (young/old, male/female)
  val targetIds: IndexedSeq[String] = IndexedSeq(
    "paired_scapula_002_M_56_L",
    "paired_scapula_004_F_67_L",
    "paired_scapula_007_M_26_L",
    "paired_scapula_010_F_43_L",
    "paired_scapula_012_M_68_L"
  )

  val gpRank: Int        = 5
  val icpSigma2: Double  = 1.0
  val icpMaxDist: Double = 15.0

  final case class Experiment(id: String, sigma: Double, scale: Double) {
    def tag: String = f"scapula_ffdm_sig${sigma.toInt}_s${scale.toInt}_r$gpRank"
  }

  val experiments: IndexedSeq[Experiment] = IndexedSeq(
    Experiment("T1", 40.0, 10.0),
    Experiment("T2", 40.0, 20.0),
    Experiment("T3", 40.0, 30.0),
    Experiment("T4", 65.0, 10.0),
    Experiment("T5", 65.0, 20.0),
    Experiment("T6", 65.0, 30.0),
    Experiment("T7", 90.0, 10.0),
    Experiment("T8", 90.0, 20.0),
    Experiment("T9", 90.0, 30.0)
  )

  final case class FitResult(
    expId: String, targetId: String,
    sigma: Double, scale: Double,
    stats: Metrics.SurfaceStats
  )

  def gpIcp(
    priorModel: PointDistributionModel[_3D, TriangleMesh],
    targetAligned: TriangleMesh[_3D],
    ptIds: IndexedSeq[PointId],
    iterations: Int
  ): PointDistributionModel[_3D, TriangleMesh] = {
    var model = priorModel
    for (iter <- 0 until iterations) {
      val meanMesh = model.mean
      val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
        val pt      = meanMesh.pointSet.point(pid)
        val nearest = targetAligned.operations.closestPointOnSurface(pt).point
        if ((nearest - pt).norm < icpMaxDist) Some((pid, nearest)) else None
      }
      if (correspondences.nonEmpty)
        model = model.posterior(correspondences, sigma2 = icpSigma2)
      if (iter == 0 || (iter + 1) % 10 == 0 || iter == iterations - 1)
        println(f"      iter ${iter + 1}%3d/$iterations: ${correspondences.size} correspondences")
    }
    model
  }

  def modeInstance(
    model: PointDistributionModel[_3D, TriangleMesh],
    modeIdx: Int,
    coeff: Double
  ): TriangleMesh[_3D] = {
    val v = DenseVector.zeros[Double](model.rank)
    v(modeIdx) = coeff
    model.instance(v)
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    require(dataDir.exists(), s"Data directory not found: ${dataDir.getAbsolutePath}")

    val outDir = new File(Config.outDir, "GaussianKernelExperiment")
    outDir.mkdirs()
    println(s"Output directory : ${outDir.getAbsolutePath}")
    println(s"GP rank (fixed)  : $gpRank  (Nystrom approximation)")
    println(s"ICP iterations   : ${Config.icpIterations}  sigma2=$icpSigma2  maxDist=$icpMaxDist mm")
    println()

    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("WARNING: landmark columns resolved by fallback offsets — verify CSV structure.")
    require(landmarks.contains(referenceId), s"'$referenceId' missing from landmark CSV.")

    // ------------------------------------------------------------------ Reference
    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")
    val refRaw  = ScapulaData.loadMesh(refFile)
    val refMesh = if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
                    refRaw.operations.decimate(Config.modelResolution) else refRaw
    val refLms  = landmarks(referenceId)
    println(s"Reference : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")

    // ------------------------------------------------------------------ Rigid-align targets (done once)
    println("Rigid-aligning all targets into reference space...")
    val alignedTargets: IndexedSeq[(String, TriangleMesh[_3D])] = targetIds.flatMap { tId =>
      val tFile = new File(dataDir, s"$tId.stl")
      if (!tFile.exists()) {
        println(s"  SKIP $tId — STL not found")
        None
      } else if (!landmarks.contains(tId)) {
        println(s"  SKIP $tId — not in landmark CSV")
        None
      } else {
        val raw  = ScapulaData.loadMesh(tFile)
        val mesh = if (raw.pointSet.numberOfPoints > Config.modelResolution)
                     raw.operations.decimate(Config.modelResolution) else raw
        val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, landmarks(tId), refMesh, refLms, icpIterations = 30)
        println(s"  OK  $tId  (${aligned.pointSet.numberOfPoints} vertices)")
        Some(tId -> aligned)
      }
    }
    println()

    // Subsample point ids for ICP correspondences
    val ptStride = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds    = (0 until refMesh.pointSet.numberOfPoints by ptStride).map(PointId(_))

    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None
    ui.foreach { u =>
      val g = u.createGroup("Reference")
      val v = u.show(g, refMesh, referenceId)
      v.color   = new Color(240, 190, 80)
      v.opacity = 0.65f
      alignedTargets.foreach { case (tId, m) =>
        val tv = u.show(g, m, tId.replaceAll("paired_scapula_", ""))
        tv.color   = new Color(200, 200, 200)
        tv.opacity = 0.25f
      }
    }

    val allResults = scala.collection.mutable.Buffer.empty[FitResult]

    // ------------------------------------------------------------------ Experiment loop
    for (exp <- experiments) {
      println(s"=== ${exp.id}  σ=${exp.sigma} mm  scale=${exp.scale} mm  rank=$gpRank ===")

      val kernel = DiagonalKernel(GaussianKernel[_3D](exp.sigma) * exp.scale, 3)
      val gp     = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

      // Nystrom approximation gives exactly gpRank components
      val sampler   = UniformMeshSampler3D(refMesh, gpRank * 30)
      val priorLRGP = LowRankGaussianProcess.approximateGPNystrom(refMesh, gp, gpRank, sampler)
      val priorModel = PointDistributionModel[_3D, TriangleMesh](refMesh, priorLRGP)
      println(s"  Prior rank : ${priorModel.rank}")

      // Save prior model
      val priorH5 = new File(outDir, s"${exp.tag}_prior.h5")
      StatisticalModelIO.writeStatisticalTriangleMeshModel3D(priorModel, priorH5).get
      println(s"  Saved prior : ${priorH5.getName}")

      // Save +-3std mode shapes for all prior modes
      for (modeIdx <- 0 until priorModel.rank) {
        MeshIO.writeMesh(
          modeInstance(priorModel, modeIdx,  3.0),
          new File(outDir, s"${exp.tag}_mode${modeIdx + 1}_pos3std.vtk")
        ).get
        MeshIO.writeMesh(
          modeInstance(priorModel, modeIdx, -3.0),
          new File(outDir, s"${exp.tag}_mode${modeIdx + 1}_neg3std.vtk")
        ).get
      }

      val uiGroup = ui.map(_.createGroup(exp.id))

      for ((tId, targetAligned) <- alignedTargets) {
        val shortId = tId.replaceAll("paired_scapula_", "").replaceAll("_L$", "")
        println(s"  -- $shortId --")

        val postModel  = gpIcp(priorModel, targetAligned, ptIds, Config.icpIterations)
        val registered = postModel.mean
        val stats      = Metrics.symmetric(registered, targetAligned)
        println(s"     Quality : ${stats.render}")
        if (stats.mean > 3.0)
          println(s"     !! mean > 3 mm — check landmark CSV row for $tId")

        val postH5 = new File(outDir, s"${exp.tag}_${shortId}_posterior.h5")
        val regVtk = new File(outDir, s"${exp.tag}_${shortId}_registered.vtk")
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(postModel, postH5).get
        MeshIO.writeMesh(registered, regVtk).get

        uiGroup.foreach { g =>
          val tv = ui.get.show(g, targetAligned, s"Target $shortId")
          tv.color   = new Color(200, 200, 200)
          tv.opacity = 0.30f
          val rv = ui.get.show(g, registered, s"Reg $shortId")
          rv.color   = new Color(50, 150, 220)
          rv.opacity = 0.85f
        }

        allResults += FitResult(exp.id, tId, exp.sigma, exp.scale, stats)
      }
      println()
    }

    // ------------------------------------------------------------------ CSV output
    val csvOut = new File(outDir, "comparison_table.csv")
    val pw = new PrintWriter(csvOut)
    try {
      pw.println("exp_id,sigma_mm,scale_mm,rank,target_short_id,mean_mm,rms_mm,hd95_mm,hd_mm")
      allResults.foreach { r =>
        val shortId = r.targetId.replaceAll("paired_scapula_", "").replaceAll("_L$", "")
        pw.println(
          f"${r.expId},${r.sigma},${r.scale},$gpRank,$shortId," +
          f"${r.stats.mean}%.3f,${r.stats.rms}%.3f,${r.stats.hd95}%.3f,${r.stats.hd}%.3f"
        )
      }
    } finally pw.close()
    println(s"Per-target CSV : ${csvOut.getAbsolutePath}")

    // ------------------------------------------------------------------ Summary table (mean over 5 targets)
    println()
    println(f"${"ID"}%-4s  ${"σ (mm)"}%-8s  ${"Sc (mm)"}%-8s  ${"mean mm"}%-9s  ${"rms mm"}%-9s  ${"HD95 mm"}%-9s")
    println("-" * 62)
    experiments.foreach { exp =>
      val rows = allResults.filter(_.expId == exp.id)
      if (rows.nonEmpty) {
        val n    = rows.size.toDouble
        val mean = rows.map(_.stats.mean).sum / n
        val rms  = rows.map(_.stats.rms).sum  / n
        val hd95 = rows.map(_.stats.hd95).sum / n
        println(f"${exp.id}%-4s  ${exp.sigma}%-8.0f  ${exp.scale}%-8.0f  $mean%7.3f    $rms%7.3f    $hd95%7.3f")
      }
    }
    println()
    println("sigma  : spatial reach — small = local patch, large = global bending.")
    println("scale  : amplitude, std_dev = sqrt(scale) mm.  Larger = more deformation allowed.")
    println()
    println(s"All outputs in : ${outDir.getAbsolutePath}")
    ui.foreach(_ => println("UI open — close the window to exit."))
  }
}
