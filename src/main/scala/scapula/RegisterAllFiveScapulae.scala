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
 * Registers 5 target scapulae to the reference using landmarks read from the CSV
 * file that lives in the data directory, then rigid ICP, then GP non-rigid ICP.
 *
 * Data directory: set SCAPULA_DATA_DIR env var, or relies on Config.dataDir default.
 *
 * ROOT CAUSE OF YEAR-LONG MISALIGNMENT: previous scripts hardcoded reference landmark
 * coordinates and gave every target a tiny (2–5 mm) shift of those same coordinates.
 * Scapulae from different subjects live in completely different coordinate frames, so
 * those "landmarks" pointed to empty space and Procrustes produced random orientations.
 * This script reads the actual per-specimen 3D coordinates from the landmark CSV,
 * which is the only correct source.
 */
object RegisterAllFiveScapulae {

  val referenceId = "paired_scapula_001_M_64_L"

  val targetIds: IndexedSeq[String] = IndexedSeq(
    "paired_scapula_002_M_56_L",
    "paired_scapula_004_F_67_L",
    "paired_scapula_007_M_26_L",
    "paired_scapula_010_F_43_L",
    "paired_scapula_012_M_68_L"
  )

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

    // ── Load actual per-specimen landmark coordinates from CSV ──────────────
    val csv = ScapulaData.csvFile(dataDir)
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")
    val (landmarks, fromHeader, header) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println(
        "  !! WARNING: landmark column names could not be matched to the header.\n" +
          "     Falling back to hardcoded column offsets — verify the CSV structure."
      )
    println(s"  ${landmarks.size} specimens parsed, ${ScapulaData.landmarkNames.size} landmarks each")

    // ── Reference mesh and its actual CSV landmarks ─────────────────────────
    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(), s"Reference STL not found: ${refFile.getAbsolutePath}")
    require(landmarks.contains(referenceId),
      s"Reference '$referenceId' not found in CSV. Keys (first 5): ${landmarks.keys.take(5).mkString(", ")}")

    val refRaw: TriangleMesh[_3D] = ScapulaData.loadMesh(refFile)
    val refMesh: TriangleMesh[_3D] =
      if (refRaw.pointSet.numberOfPoints > Config.modelResolution)
        refRaw.operations.decimate(Config.modelResolution)
      else refRaw
    val refLms: IndexedSeq[Landmark[_3D]] = landmarks(referenceId)

    println(s"\nReference      : $referenceId  (${refMesh.pointSet.numberOfPoints} vertices)")
    println("Reference landmarks from CSV:")
    refLms.foreach { l =>
      println(f"    ${l.id}%-4s  (${l.point.x}%9.3f, ${l.point.y}%9.3f, ${l.point.z}%9.3f) mm")
    }

    // ── GP prior: wide enough to cover any plausible scapula shape ──────────
    // sigma=80mm spans the whole bone; amplitude=25mm is conservative (posterior shrinks it).
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](80.0) * 25.0, 3)
    )

    val outDir = new File(dataDir.getParentFile, "FiveTargets_Registered")
    outDir.mkdirs()
    println(s"\nOutput directory: ${outDir.getAbsolutePath}\n")

    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None

    for ((tId, idx) <- targetIds.zipWithIndex) {
      println(s"── [${idx + 1}/5]  $tId ──")

      val tFile = new File(dataDir, s"$tId.stl")
      val skip = !tFile.exists() || !landmarks.contains(tId)
      if (!tFile.exists())
        println(s"  SKIP – STL not found: ${tFile.getAbsolutePath}")
      else if (!landmarks.contains(tId))
        println(s"  SKIP – '$tId' absent from landmark CSV")

      if (!skip) {
        val targetRaw: TriangleMesh[_3D] = ScapulaData.loadMesh(tFile)
        val targetMesh: TriangleMesh[_3D] =
          if (targetRaw.pointSet.numberOfPoints > Config.modelResolution)
            targetRaw.operations.decimate(Config.modelResolution)
          else targetRaw
        val targetLms: IndexedSeq[Landmark[_3D]] = landmarks(tId)

        println(s"  Target vertices: ${targetMesh.pointSet.numberOfPoints}")
        println("  Target landmarks from CSV:")
        targetLms.foreach { l =>
          println(f"      ${l.id}%-4s  (${l.point.x}%9.3f, ${l.point.y}%9.3f, ${l.point.z}%9.3f) mm")
        }

        // Check landmark spread — a tiny spread (< 5mm) means the CSV row is wrong
        val spread = {
          val pts = targetLms.map(_.point)
          val min = Point3D(pts.map(_.x).min, pts.map(_.y).min, pts.map(_.z).min)
          val max = Point3D(pts.map(_.x).max, pts.map(_.y).max, pts.map(_.z).max)
          (max - min).norm
        }
        if (spread < 5.0)
          println(s"  !! WARNING: landmark spread is only ${spread}%.1f mm — CSV row may be all-zero or bogus")

        // ── Step 1: Landmark Procrustes → trimmed rigid ICP ─────────────────
        // RigidAlign.landmarkThenIcp: rigid3DLandmarkRegistration(refLms, targetLms)
        // then trimmed ICP. Actual per-specimen coordinates from the CSV ensure
        // the bone lands in the right neighbourhood before ICP starts.
        val (alignedRef: TriangleMesh[_3D], _) = RigidAlign.landmarkThenIcp(
          refMesh, refLms, targetMesh, targetLms,
          icpIterations = Config.icpIterations
        )
        println(s"  1. Landmark Procrustes + rigid ICP done")

        // ── Step 2: GP approximation on the rigidly-aligned reference ────────
        val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
          alignedRef,
          gp,
          relativeTolerance = Config.gpRelativeTolerance,
          interpolator = NearestNeighborInterpolator3D()
        )
        println(s"  2. GP approximated (rank ${lowRankGP.rank})")

        // ── Step 3: GP non-rigid ICP ──────────────────────────────────────────
        var model = PointDistributionModel[_3D, TriangleMesh](alignedRef, lowRankGP)
        val step  = math.max(1, alignedRef.pointSet.numberOfPoints / 2500)
        val ptIds = (0 until alignedRef.pointSet.numberOfPoints by step).map(PointId(_))

        for (iter <- 0 until Config.icpIterations) {
          val mean = model.mean
          val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
            val pt      = mean.pointSet.point(pid)
            val nearest = targetMesh.operations.closestPointOnSurface(pt).point
            if ((nearest - pt).norm < 15.0) Some((pid, nearest)) else None
          }
          if (correspondences.nonEmpty)
            model = model.posterior(correspondences, sigma2 = 1.0)
          if (iter == 0 || iter == Config.icpIterations - 1 || (iter + 1) % 10 == 0)
            println(s"    GP iter ${iter + 1}/${Config.icpIterations}: ${correspondences.size} correspondences")
        }

        val registeredMesh: TriangleMesh[_3D] = model.mean
        println(s"  3. GP non-rigid ICP done")

        // Quick sanity: mean surface distance registered→target
        val distStats = Metrics.symmetric(registeredMesh, targetMesh)
        println(s"  Registration quality: ${distStats.render}")
        if (distStats.mean > 3.0)
          println(s"  !! mean > 3mm — check landmark CSV row for $tId")

        // ── Step 4: Save ───────────────────────────────────────────────────────
        val h5Out  = new File(outDir, s"model_$tId.h5")
        val vtkOut = new File(outDir, s"registered_$tId.vtk")
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5Out).get
        MeshIO.writeMesh(registeredMesh, vtkOut).get
        println(s"  Saved .vtk → ${vtkOut.getAbsolutePath}")

        // ── Step 5: Visualise ─────────────────────────────────────────────────
        ui.foreach { scalismoUi =>
          val grp                  = scalismoUi.createGroup(s"[${idx + 1}] $tId")
          val tView                = scalismoUi.show(grp, targetMesh,     "Target")
          tView.color              = new Color(210, 210, 210)
          tView.opacity            = 0.45f
          val rView                = scalismoUi.show(grp, registeredMesh, "Registered")
          rView.color              = new Color(50, 155, 220)
          rView.opacity            = 0.85f
        }
      }
      println()
    }

    println(s"All done. Results in: ${outDir.getAbsolutePath}")
  }
}
