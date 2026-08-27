package scapula

import scalismo.common.PointId
import scalismo.common.interpolation.NearestNeighborInterpolator3D
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, PointDistributionModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random
import java.awt.Color
import java.io.File

/**
 * Baseline GP-ICP registration: reference -> 5 diverse targets.
 *
 * Uses the T5 parameters (σ=65 mm, scale=20 mm) from GaussianKernelExperiment.
 * Each target is rigidly aligned into the reference coordinate space first,
 * then the reference is deformed to match it using GP-ICP posterior updates.
 *
 * Outputs (in <SCAPULA_OUT_DIR>/Scapula_GP_Registered/):
 *   registered_<id>.vtk  — deformed reference mesh (= registered surface)
 *   model_<id>.h5        — posterior shape model
 *
 * Run with:
 *   SCAPULA_DATA_DIR=<stl_dir> SCAPULA_OUT_DIR=<out_dir> sbt "runMain scapula.RegisterAllFiveScapulae"
 */
object RegisterAllFiveScapulae {

  val gpSigma:     Double = 65.0
  val gpAmplitude: Double = 20.0

  val referenceId: String = "paired_scapula_001_M_64_L"

  val targetIds: IndexedSeq[String] = IndexedSeq(
    "paired_scapula_002_M_56_L", // M56
    "paired_scapula_004_F_67_L", // F67
    "paired_scapula_007_M_26_L", // M26
    "paired_scapula_010_F_43_L", // F43
    "paired_scapula_012_M_68_L"  // M68
  )

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    require(dataDir.exists(), s"Data directory not found: ${dataDir.getAbsolutePath}")
    println(s"Data directory : ${dataDir.getAbsolutePath}")

    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("!! WARNING: landmark columns resolved by fallback offsets — verify CSV structure.")
    println(s"${landmarks.size} specimens in CSV")

    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")
    require(landmarks.contains(referenceId), s"'$referenceId' not in CSV.")

    val refRaw  = ScapulaData.loadMesh(refFile)
    val refMesh = if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
                    refRaw.operations.decimate(Config.modelResolution) else refRaw
    val refLms  = landmarks(referenceId)
    println(s"\nReference : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")
    println(s"GP kernel : Gaussian  σ=$gpSigma mm  scale=$gpAmplitude mm")

    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](gpSigma) * gpAmplitude, 3))
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      refMesh, gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )
    println(s"GP rank   : ${lowRankGP.rank}")

    val ptStride = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds    = (0 until refMesh.pointSet.numberOfPoints by ptStride).map(PointId(_))

    val outDir = new File(Config.outDir, "Scapula_GP_Registered")
    outDir.mkdirs()
    println(s"Output    : ${outDir.getAbsolutePath}\n")

    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None
    ui.foreach { u =>
      val g = u.createGroup("Reference")
      u.show(g, refMesh, referenceId).color = new Color(240, 190, 80)
    }

    val summary = scala.collection.mutable.Buffer.empty[(String, Metrics.SurfaceStats)]

    for ((tId, idx) <- targetIds.zipWithIndex) {
      println(s"=== [${idx + 1}/5]  $tId ===")
      val tFile = new File(dataDir, s"$tId.stl")
      if (!tFile.exists()) {
        println(s"  SKIP — STL not found: ${tFile.getAbsolutePath}")
      } else if (!landmarks.contains(tId)) {
        println(s"  SKIP — '$tId' absent from landmark CSV")
      } else {
        val tRaw  = ScapulaData.loadMesh(tFile)
        val tMesh = if (tRaw.pointSet.numberOfPoints > Config.modelResolution)
                      tRaw.operations.decimate(Config.modelResolution) else tRaw

        val (targetInRefSpace, _) = RigidAlign.landmarkThenIcp(
          tMesh, landmarks(tId), refMesh, refLms,
          icpIterations = Config.icpIterations)
        println(s"  1. Rigid alignment done")

        var model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)
        for (iter <- 0 until Config.icpIterations) {
          val meanMesh = model.mean
          val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
            val pt      = meanMesh.pointSet.point(pid)
            val nearest = targetInRefSpace.operations.closestPointOnSurface(pt).point
            if ((nearest - pt).norm < 15.0) Some((pid, nearest)) else None
          }
          if (correspondences.nonEmpty)
            model = model.posterior(correspondences, sigma2 = 1.0)
          if (iter == 0 || (iter + 1) % 10 == 0 || iter == Config.icpIterations - 1)
            println(s"    GP iter ${iter + 1}/${Config.icpIterations}: ${correspondences.size} correspondences")
        }

        val regMesh = model.mean
        val stats   = Metrics.symmetric(regMesh, targetInRefSpace)
        println(s"  2. GP ICP done   ${stats.render}")
        if (stats.mean > 3.0)
          println(s"  !! mean > 3 mm — check landmark CSV row for $tId")

        MeshIO.writeMesh(regMesh, new File(outDir, s"registered_$tId.vtk")).get
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(
          model, new File(outDir, s"model_$tId.h5")).get
        println(s"  Saved registered_$tId.vtk + model_$tId.h5")

        summary += (tId -> stats)

        ui.foreach { u =>
          val g = u.createGroup(s"[${idx + 1}] $tId")
          u.show(g, targetInRefSpace, "Target (aligned)").color = new Color(200, 200, 200)
          u.show(g, regMesh, "Registered").color          = new Color(50, 155, 220)
          u.show(g, model, "Shape model (modes)")
        }
      }
      println()
    }

    println("=" * 60)
    println("Summary (all meshes in reference coordinate space)")
    println("=" * 60)
    summary.foreach { case (id, s) =>
      println(f"  ${id.takeRight(20)}%-20s  ${s.render}")
    }
    println(s"\nResults -> ${outDir.getAbsolutePath}")
  }
}
