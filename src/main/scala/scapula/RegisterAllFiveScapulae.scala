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
 * Registers 5 target scapulae to the reference and shows the posterior shape
 * model (modes) in the Scalismo Viewer alongside the target surface.
 *
 * Pipeline per target:
 *   1. Landmark Procrustes   — actual per-specimen CSV coordinates
 *   2. Trimmed rigid ICP     — refines pose without touching shape
 *   3. GP non-rigid ICP      — establishes dense correspondence
 *
 * Kernel choice (single Gaussian, best for scapula):
 *   sigma     = 65 mm  — half the scapula diagonal; captures smooth whole-bone
 *                         deformations (blade tilt, glenoid offset, acromion
 *                         angle) without introducing wiggly artefacts
 *   amplitude = 20 mm  — covers realistic inter-subject shape variation;
 *                         the posterior shrinks this where data is close
 *
 * Viewer shows per target:
 *   • "Target in ref space"  — grey, semi-transparent target mesh
 *   • "Registered mesh"      — blue, opaque registered mean
 *   • "Shape model (modes)"  — interactive sliders to explore PC modes 1-N
 *
 * Outputs saved to:  <data-parent>/Scapula_GP_Registered/
 *   registered_<id>.vtk   — registered surface mesh
 *   model_<id>.h5         — posterior PDM (for SSM or quality checking)
 *
 * Data directory: set SCAPULA_DATA_DIR env var, or uses Config.dataDir default.
 */
object RegisterAllFiveScapulae {

  // ── Kernel: single Gaussian, tuned for whole-scapula shape ─────────────────
  // A scapula is ~130 mm diagonal.  sigma = 65 mm (half diagonal) gives one
  // smooth spatial scale that spans the entire bone, letting the GP deform the
  // blade, acromion, glenoid and coracoid as a coherent unit rather than
  // independently.  Sigma < 40 mm makes the deformation field too local and
  // causes the blade to fold; sigma > 90 mm makes it too global and prevents
  // the glenoid from fitting independently of the blade.
  val gpSigma: Double     = 65.0   // mm
  val gpAmplitude: Double = 20.0   // mm

  val referenceId: String = "paired_scapula_001_M_64_L"

  val targetIds: IndexedSeq[String] = IndexedSeq(
    "paired_scapula_002_M_56_L",
    "paired_scapula_004_F_67_L",
    "paired_scapula_007_M_26_L",
    "paired_scapula_010_F_43_L",
    "paired_scapula_012_M_68_L"
  )

  // Output folder (sibling of the data directory)
  val outFolderName: String = "Scapula_GP_Registered"

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    if (!dataDir.exists())
      sys.error(
        s"Data directory not found: ${dataDir.getAbsolutePath}\n" +
          "Set SCAPULA_DATA_DIR to the folder that contains the STL files and landmark CSV."
      )
    println(s"Data directory : ${dataDir.getAbsolutePath}")

    // ── Load actual per-specimen landmark coordinates from the CSV ─────────────
    val csv = ScapulaData.csvFile(dataDir)
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println(
        "  !! WARNING: landmark column names did not match header — " +
          "using fallback column offsets. Verify the CSV structure."
      )
    println(s"  ${landmarks.size} specimens in CSV, " +
      s"${ScapulaData.landmarkNames.mkString("/")} per specimen")

    // ── Reference ──────────────────────────────────────────────────────────────
    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(),
      s"Reference STL not found: ${refFile.getAbsolutePath}")
    require(landmarks.contains(referenceId),
      s"Reference '$referenceId' not in CSV. First 5 keys: ${landmarks.keys.take(5).mkString(", ")}")

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

    // ── GP prior ───────────────────────────────────────────────────────────────
    println(s"\nGP kernel      : Gaussian  sigma=$gpSigma mm  amplitude=$gpAmplitude mm")
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](gpSigma) * gpAmplitude, 3)
    )

    // ── Output folder ──────────────────────────────────────────────────────────
    val outDir = new File(dataDir.getParentFile, outFolderName)
    outDir.mkdirs()
    println(s"Output folder  : ${outDir.getAbsolutePath}\n")

    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None

    // ── Show reference in its own group ───────────────────────────────────────
    ui.foreach { scalismoUi =>
      val refGrp  = scalismoUi.createGroup("Reference")
      val refView = scalismoUi.show(refGrp, refMesh, referenceId)
      refView.color   = new Color(240, 190, 80)
      refView.opacity = 0.6f
    }

    // ── Register each target ───────────────────────────────────────────────────
    for ((tId, idx) <- targetIds.zipWithIndex) {
      println(s"═══ [${idx + 1}/5]  $tId ═══")

      val tFile = new File(dataDir, s"$tId.stl")
      if (!tFile.exists())
        println(s"  SKIP – STL not found: ${tFile.getAbsolutePath}")
      else if (!landmarks.contains(tId))
        println(s"  SKIP – '$tId' absent from landmark CSV")
      else {

        // Load + decimate target
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

        // Sanity: landmark spread < 5 mm almost certainly means a bad CSV row
        val spread = {
          val xs = targetLms.map(_.point.x)
          val ys = targetLms.map(_.point.y)
          val zs = targetLms.map(_.point.z)
          math.sqrt(
            math.pow(xs.max - xs.min, 2) +
              math.pow(ys.max - ys.min, 2) +
              math.pow(zs.max - zs.min, 2)
          )
        }
        if (spread < 5.0)
          println(f"  !! WARNING: landmark spread = $spread%.1f mm (< 5 mm). CSV row looks wrong.")

        // ── 1. Landmark Procrustes + trimmed rigid ICP ────────────────────────
        val (alignedRef: TriangleMesh[_3D], _) = RigidAlign.landmarkThenIcp(
          refMesh, refLms, targetMesh, targetLms,
          icpIterations = Config.icpIterations
        )
        println(s"  1. Procrustes + rigid ICP done")

        // ── 2. Low-rank GP approximation ─────────────────────────────────────
        val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
          alignedRef,
          gp,
          relativeTolerance = Config.gpRelativeTolerance,
          interpolator      = NearestNeighborInterpolator3D()
        )
        println(s"  2. GP approximated  (rank ${lowRankGP.rank})")

        // ── 3. GP non-rigid ICP ──────────────────────────────────────────────
        var model   = PointDistributionModel[_3D, TriangleMesh](alignedRef, lowRankGP)
        val ptStride = math.max(1, alignedRef.pointSet.numberOfPoints / 2500)
        val ptIds   = (0 until alignedRef.pointSet.numberOfPoints by ptStride).map(PointId(_))

        for (iter <- 0 until Config.icpIterations) {
          val meanMesh = model.mean
          val correspondences: IndexedSeq[(PointId, Point[_3D])] = ptIds.flatMap { pid =>
            val pt      = meanMesh.pointSet.point(pid)
            val nearest = targetMesh.operations.closestPointOnSurface(pt).point
            if ((nearest - pt).norm < 15.0) Some((pid, nearest)) else None
          }
          if (correspondences.nonEmpty)
            model = model.posterior(correspondences, sigma2 = 1.0)
          if (iter == 0 || (iter + 1) % 10 == 0 || iter == Config.icpIterations - 1)
            println(s"    GP iter ${iter + 1}/${Config.icpIterations}: ${correspondences.size} active correspondences")
        }

        val registeredMesh: TriangleMesh[_3D] = model.mean
        println(s"  3. GP non-rigid ICP done")

        // Quality report
        val distStats = Metrics.symmetric(registeredMesh, targetMesh)
        println(s"  Quality : ${distStats.render}")
        if (distStats.mean > 3.0)
          println(s"  !! mean surface distance > 3 mm — check landmark CSV row for $tId")

        // ── 4. Save registered surface and posterior model ────────────────────
        val vtkOut = new File(outDir, s"registered_$tId.vtk")
        val h5Out  = new File(outDir, s"model_$tId.h5")
        MeshIO.writeMesh(registeredMesh, vtkOut).get
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5Out).get
        println(s"  Saved .vtk  → ${vtkOut.getName}")
        println(s"  Saved .h5   → ${h5Out.getName}")

        // ── 5. Visualise: target + registered mesh + shape model modes ────────
        ui.foreach { scalismoUi =>
          val grp = scalismoUi.createGroup(s"[${idx + 1}] $tId")

          // Target surface (grey, semi-transparent)
          val tView = scalismoUi.show(grp, targetMesh, "Target in ref space")
          tView.color   = new Color(210, 210, 210)
          tView.opacity = 0.40f

          // Registered mean (blue, opaque)
          val rView = scalismoUi.show(grp, registeredMesh, "Registered mesh")
          rView.color   = new Color(50, 155, 220)
          rView.opacity = 0.90f

          // Posterior shape model — drag the sliders to explore PC modes
          // Mode 1 = direction of most shape variance for this target
          scalismoUi.show(grp, model, "Shape model (modes)")
        }
      }
      println()
    }

    println(s"All done.  Open Scalismo Viewer to explore mode sliders.")
    println(s"Results   : ${outDir.getAbsolutePath}")
  }
}
