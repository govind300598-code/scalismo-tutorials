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
 * GP non-rigid registration of five target scapulae to a single reference.
 *
 * Baseline parameters (T5 from the kernel experiment):
 *   sigma = 65 mm,  scale = 20 mm,  GP approximated by pivoted Cholesky.
 *
 * Pipeline per target:
 *   1. Load target STL; decimate to model resolution.
 *   2. Rigid align (landmark Procrustes + trimmed ICP) into reference space.
 *   3. GP non-rigid ICP: iteratively find surface correspondences and update
 *      the GP posterior until the reference deforms onto the target.
 *   4. Save registered mesh (.vtk) and posterior shape model (.h5).
 *
 * Output folder: <SCAPULA_OUT_DIR>/Scapula_GP_Registered/
 */
object RegisterAllFiveScapulae {

  // Baseline GP kernel parameters (T5)
  val gpSigma: Double     = 65.0
  val gpScale: Double     = 20.0
  val icpMaxDist: Double  = 15.0
  val icpSigma2: Double   = 1.0

  val referenceId: String = "paired_scapula_001_M_64_L"

  // Five targets chosen for maximal anatomical diversity (age and sex)
  val targetIds: IndexedSeq[String] = IndexedSeq(
    "paired_scapula_002_M_56_L",
    "paired_scapula_004_F_67_L",
    "paired_scapula_007_M_26_L",
    "paired_scapula_010_F_43_L",
    "paired_scapula_012_M_68_L"
  )

  val outFolderName: String = "Scapula_GP_Registered"

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    require(dataDir.exists(),
      s"Data directory not found: ${dataDir.getAbsolutePath}\n" +
      "Set SCAPULA_DATA_DIR to the folder containing the STL files and landmark CSV.")
    println(s"Data directory : ${dataDir.getAbsolutePath}")

    val csv = ScapulaData.csvFile(dataDir)
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("  !! WARNING: landmark columns resolved by fallback offsets — verify CSV structure.")
    println(s"  ${landmarks.size} specimens in CSV")

    require(landmarks.contains(referenceId),
      s"'$referenceId' not in CSV. First 5 keys: ${landmarks.keys.take(5).mkString(", ")}")

    // ------------------------------------------------------------------ Reference
    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")
    val refRaw: TriangleMesh[_3D]  = ScapulaData.loadMesh(refFile)
    val refMesh: TriangleMesh[_3D] =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
        refRaw.operations.decimate(Config.modelResolution)
      else refRaw
    val refLms: IndexedSeq[Landmark[_3D]] = landmarks(referenceId)

    println(s"\nReference      : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")
    println("Reference landmarks:")
    refLms.foreach { l =>
      println(f"    ${l.id}%-4s  (${l.point.x}%9.2f, ${l.point.y}%9.2f, ${l.point.z}%9.2f) mm")
    }

    // ------------------------------------------------------------------ Build GP prior (once)
    println(s"\nGP kernel      : Gaussian  sigma=$gpSigma mm  scale=$gpScale mm")
    val kernel = DiagonalKernel(GaussianKernel[_3D](gpSigma) * gpScale, 3)
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](kernel)

    println("Approximating GP on reference mesh (pivoted Cholesky)...")
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      refMesh,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )
    println(s"  GP rank: ${lowRankGP.rank}")

    // Subsample point ids for correspondence search
    val ptStride = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds    = (0 until refMesh.pointSet.numberOfPoints by ptStride).map(PointId(_))

    // ------------------------------------------------------------------ Output
    val outDir = new File(Config.outDir, outFolderName)
    outDir.mkdirs()
    println(s"\nOutput folder  : ${outDir.getAbsolutePath}\n")

    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None
    ui.foreach { u =>
      val g    = u.createGroup("Reference")
      val view = u.show(g, refMesh, referenceId)
      view.color   = new Color(240, 190, 80)
      view.opacity = 0.70f
    }

    // ------------------------------------------------------------------ Target loop
    for ((tId, idx) <- targetIds.zipWithIndex) {
      println(s"=== [${idx + 1}/${targetIds.length}]  $tId ===")

      val tFile = new File(dataDir, s"$tId.stl")
      if (!tFile.exists()) {
        println(s"  SKIP — STL not found: ${tFile.getAbsolutePath}")
      } else if (!landmarks.contains(tId)) {
        println(s"  SKIP — '$tId' absent from landmark CSV")
      } else {

        val targetRaw: TriangleMesh[_3D]  = ScapulaData.loadMesh(tFile)
        val targetMesh: TriangleMesh[_3D] =
          if (targetRaw.pointSet.numberOfPoints > Config.modelResolution)
            targetRaw.operations.decimate(Config.modelResolution)
          else targetRaw
        val targetLms: IndexedSeq[Landmark[_3D]] = landmarks(tId)

        println(s"  Target vertices : ${targetMesh.pointSet.numberOfPoints}")

        // Sanity-check: are landmarks spatially spread out?
        val xs = targetLms.map(_.point.x)
        val ys = targetLms.map(_.point.y)
        val zs = targetLms.map(_.point.z)
        val spread = math.sqrt(
          math.pow(xs.max - xs.min, 2) +
          math.pow(ys.max - ys.min, 2) +
          math.pow(zs.max - zs.min, 2)
        )
        if (spread < 5.0)
          println(f"  !! WARNING: landmark spread = $spread%.1f mm — CSV row may be wrong.")

        // Step 1: rigid alignment of target into reference space
        val (targetInRefSpace: TriangleMesh[_3D], _) = RigidAlign.landmarkThenIcp(
          targetMesh, targetLms,
          refMesh,    refLms,
          icpIterations = Config.icpIterations
        )
        println("  1. Target rigidly aligned into reference space")

        // Step 2: GP non-rigid ICP — deform reference to match target
        var model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)

        for (iter <- 0 until Config.icpIterations) {
          val meanMesh = model.mean
          val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
            val pt      = meanMesh.pointSet.point(pid)
            val nearest = targetInRefSpace.operations.closestPointOnSurface(pt).point
            if ((nearest - pt).norm < icpMaxDist) Some((pid, nearest)) else None
          }
          if (correspondences.nonEmpty)
            model = model.posterior(correspondences, sigma2 = icpSigma2)
          if (iter == 0 || (iter + 1) % 10 == 0 || iter == Config.icpIterations - 1)
            println(s"    GP iter ${iter + 1}/${Config.icpIterations}: ${correspondences.size} correspondences")
        }

        val registeredMesh: TriangleMesh[_3D] = model.mean
        println("  2. GP non-rigid ICP complete")

        val distStats = Metrics.symmetric(registeredMesh, targetInRefSpace)
        println(s"  Quality : ${distStats.render}")
        if (distStats.mean > 3.0)
          println(s"  !! mean > 3 mm — check landmark CSV row for $tId")

        // Step 3: save outputs
        val shortId = tId.replaceAll("paired_scapula_", "").replaceAll("_L$", "")
        val vtkOut  = new File(outDir, s"registered_${shortId}.vtk")
        val h5Out   = new File(outDir, s"model_${shortId}.h5")
        MeshIO.writeMesh(registeredMesh, vtkOut).get
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5Out).get
        println(s"  Saved .vtk -> ${vtkOut.getName}")
        println(s"  Saved .h5  -> ${h5Out.getName}")

        ui.foreach { u =>
          val g     = u.createGroup(s"[${idx + 1}] $tId")
          val tView = u.show(g, targetInRefSpace, "Target (ref space)")
          tView.color   = new Color(210, 210, 210)
          tView.opacity = 0.40f
          val rView = u.show(g, registeredMesh, "Registered")
          rView.color   = new Color(50, 155, 220)
          rView.opacity = 0.90f
          u.show(g, model, "Shape model (modes)")
        }
      }
      println()
    }

    println("All done. Meshes are in the reference coordinate space.")
    println(s"Results -> ${outDir.getAbsolutePath}")
    ui.foreach(_ => println("UI open — close the window to exit."))
  }
}
