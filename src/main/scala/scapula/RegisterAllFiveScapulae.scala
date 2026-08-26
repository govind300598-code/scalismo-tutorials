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

object RegisterAllFiveScapulae {

  val gpSigma: Double     = 65.0
  val gpAmplitude: Double = 20.0

  val referenceId: String = "paired_scapula_001_M_64_L"

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
    if (!dataDir.exists())
      sys.error(
        s"Data directory not found: ${dataDir.getAbsolutePath}\n" +
          "Set SCAPULA_DATA_DIR to the folder containing the STL files and landmark CSV."
      )
    println(s"Data directory : ${dataDir.getAbsolutePath}")

    val csv = ScapulaData.csvFile(dataDir)
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("  !! WARNING: landmark columns resolved by fallback offsets — verify CSV structure.")
    println(s"  ${landmarks.size} specimens in CSV, landmarks: ${ScapulaData.landmarkNames.mkString(", ")}")

    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(),
      s"Reference STL not found: ${refFile.getAbsolutePath}")
    require(landmarks.contains(referenceId),
      s"'$referenceId' not in CSV. First 5 keys: ${landmarks.keys.take(5).mkString(", ")}")

    val refRaw: TriangleMesh[_3D] = ScapulaData.loadMesh(refFile)
    val refMesh: TriangleMesh[_3D] =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
        refRaw.operations.decimate(Config.modelResolution)
      else refRaw
    val refLms: IndexedSeq[Landmark[_3D]] = landmarks(referenceId)

    println(s"\nReference      : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")
    println("Reference landmarks (from CSV):")
    refLms.foreach { l =>
      println(f"    ${l.id}%-4s  (${l.point.x}%9.2f, ${l.point.y}%9.2f, ${l.point.z}%9.2f) mm")
    }

    println(s"\nGP kernel      : Gaussian  sigma=$gpSigma mm  amplitude=$gpAmplitude mm")
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](gpSigma) * gpAmplitude, 3)
    )

    println("Approximating GP on reference mesh (once for all targets)...")
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      refMesh,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )
    println(s"  GP rank: ${lowRankGP.rank}")

    val ptStride = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds    = (0 until refMesh.pointSet.numberOfPoints by ptStride).map(PointId(_))

    val outDir = new File(dataDir.getParentFile, outFolderName)
    outDir.mkdirs()
    println(s"\nOutput folder  : ${outDir.getAbsolutePath}\n")

    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None

    ui.foreach { scalismoUi =>
      val grp  = scalismoUi.createGroup("Reference")
      val view = scalismoUi.show(grp, refMesh, referenceId)
      view.color   = new Color(240, 190, 80)
      view.opacity = 0.70f
    }

    for ((tId, idx) <- targetIds.zipWithIndex) {
      println(s"=== [${idx + 1}/5]  $tId ===")

      val tFile = new File(dataDir, s"$tId.stl")
      if (!tFile.exists())
        println(s"  SKIP - STL not found: ${tFile.getAbsolutePath}")
      else if (!landmarks.contains(tId))
        println(s"  SKIP - '$tId' absent from landmark CSV")
      else {

        val targetRaw: TriangleMesh[_3D] = ScapulaData.loadMesh(tFile)
        val targetMesh: TriangleMesh[_3D] =
          if (targetRaw.pointSet.numberOfPoints > Config.modelResolution)
            targetRaw.operations.decimate(Config.modelResolution)
          else targetRaw
        val targetLms: IndexedSeq[Landmark[_3D]] = landmarks(tId)

        println(s"  Target vertices : ${targetMesh.pointSet.numberOfPoints}")
        println("  Target landmarks (from CSV):")
        targetLms.foreach { l =>
          println(f"      ${l.id}%-4s  (${l.point.x}%9.2f, ${l.point.y}%9.2f, ${l.point.z}%9.2f) mm")
        }

        val xs = targetLms.map(_.point.x)
        val ys = targetLms.map(_.point.y)
        val zs = targetLms.map(_.point.z)
        val spread = math.sqrt(
          math.pow(xs.max - xs.min, 2) +
            math.pow(ys.max - ys.min, 2) +
            math.pow(zs.max - zs.min, 2)
        )
        if (spread < 5.0)
          println(f"  !! WARNING: landmark spread = $spread%.1f mm — CSV row looks wrong.")

        val (targetInRefSpace: TriangleMesh[_3D], _) = RigidAlign.landmarkThenIcp(
          targetMesh, targetLms,
          refMesh,    refLms,
          icpIterations = Config.icpIterations
        )
        println(s"  1. Target rigidly aligned into reference space")

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

        val registeredMesh: TriangleMesh[_3D] = model.mean
        println(s"  2. GP non-rigid ICP done")

        val distStats = Metrics.symmetric(registeredMesh, targetInRefSpace)
        println(s"  Quality : ${distStats.render}")
        if (distStats.mean > 3.0)
          println(s"  !! mean > 3 mm - check landmark CSV row for $tId")

        val vtkOut = new File(outDir, s"registered_$tId.vtk")
        val h5Out  = new File(outDir, s"model_$tId.h5")
        MeshIO.writeMesh(registeredMesh, vtkOut).get
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5Out).get
        println(s"  Saved .vtk -> ${vtkOut.getName}")
        println(s"  Saved .h5  -> ${h5Out.getName}")

        ui.foreach { scalismoUi =>
          val grp   = scalismoUi.createGroup(s"[${idx + 1}] $tId")
          val tView = scalismoUi.show(grp, targetInRefSpace, "Target in ref space")
          tView.color   = new Color(210, 210, 210)
          tView.opacity = 0.40f
          val rView = scalismoUi.show(grp, registeredMesh, "Registered mesh")
          rView.color   = new Color(50, 155, 220)
          rView.opacity = 0.90f
          scalismoUi.show(grp, model, "Shape model (modes)")
        }
      }
      println()
    }

    println(s"All done. All meshes are in the reference coordinate space.")
    println(s"Results -> ${outDir.getAbsolutePath}")
  }
}
