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
 * Registers 5 target scapulae to the reference IN THE REFERENCE COORDINATE SPACE.
 *
 * KEY FIX (root cause of scattered / misaligned visualization):
 *   Previous scripts moved the REFERENCE into each target's space.  Every target
 *   lives at different world coordinates, so all meshes appeared scattered when
 *   shown together, and the saved surfaces were useless for SSM building.
 *
 *   Correct approach: move each TARGET into the reference space (rigid align
 *   target -> reference), then GP-deform the reference to match that rigidly-
 *   aligned target.  All registered meshes end up in ONE common reference frame —
 *   they can be shown together, and they are ready for SSM building directly.
 *
 * Pipeline per target:
 *   1. Rigid align TARGET -> reference space (landmark Procrustes + trimmed ICP)
 *   2. GP non-rigid ICP: deform reference mesh to match target-in-ref-space
 *   3. Registered mesh = model mean  (lives in reference space)
 *
 * Viewer shows per target (all in reference space — overlap correctly):
 *   "Target in ref space"  grey  40% — target after rigid alignment to ref
 *   "Registered mesh"      blue  90% — non-rigidly registered result
 *   "Shape model (modes)"        —— interactive PC sliders
 *
 * Output folder:  <data-parent>/Scapula_GP_Registered/
 *   registered_<id>.vtk   registered surface in reference space
 *   model_<id>.h5         posterior PDM in reference space
 */
object RegisterAllFiveScapulae {

  // ── GP kernel ───────────────────────────────────────────────────────────────
  // sigma = 65 mm = half the scapula diagonal.  Captures smooth whole-bone
  // deformations without letting the glenoid and blade move independently.
  // amplitude = 20 mm covers realistic inter-subject shape variation.
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

    // ── Data directory ────────────────────────────────────────────────────────
    val dataDir = Config.dataDir
    if (!dataDir.exists())
      sys.error(
        s"Data directory not found: ${dataDir.getAbsolutePath}\n" +
          "Set SCAPULA_DATA_DIR to the folder containing the STL files and landmark CSV."
      )
    println(s"Data directory : ${dataDir.getAbsolutePath}")

    // ── Landmark CSV (actual per-specimen 3D coordinates) ─────────────────────
    val csv = ScapulaData.csvFile(dataDir)
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(csv)
    if (!fromHeader)
      println("  !! WARNING: landmark columns resolved by fallback offsets — verify CSV structure.")
    println(s"  ${landmarks.size} specimens in CSV, landmarks: ${ScapulaData.landmarkNames.mkString(", ")}")

    // ── Reference mesh ────────────────────────────────────────────────────────
    val refFile = new File(dataDir, s"$referenceId.stl")
    require(refFile.exists(),    s"Reference STL not found: ${refFile.getAbsolutePath}")
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

    // ── GP prior — computed ONCE on the fixed reference mesh ──────────────────
    // The GP domain is refMesh; it stays fixed for all targets.
    println(s"\nGP kernel      : Gaussian  sigma=$gpSigma mm  amplitude=$gpAmplitude mm")
    val gp = GaussianProcess[_3D, EuclideanVector[_3D]](
      DiagonalKernel(GaussianKernel[_3D](gpSigma) * gpAmplitude, 3)
    )
    println("Approximating GP on reference mesh (done once for all targets)...")
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      refMesh,
      gp,
      relativeTolerance = Config.gpRelativeTolerance,
      interpolator      = NearestNeighborInterpolator3D()
    )
    println(s"  GP rank: ${lowRankGP.rank}")

    // Point ids sampled uniformly over the reference — reused every target
    val ptStride = math.max(1, refMesh.pointSet.numberOfPoints / 2500)
    val ptIds    = (0 until refMesh.pointSet.numberOfPoints by ptStride).map(PointId(_))

    // ── Output folder ─────────────────────────────────────────────────────────
    val outDir = new File(dataDir.getParentFile, outFolderName)
    outDir.mkdirs()
    println(s"\nOutput folder  : ${outDir.getAbsolutePath}\n")

    val ui: Option[ScalismoUI] = if (Config.showUi) Some(ScalismoUI()) else None

    // Show reference in its own group (gold)
    ui.foreach { scalismoUi =>
      val grp  = scalismoUi.createGroup("Reference")
      val view = scalismoUi.show(grp, refMesh, referenceId)
      view.color   = new Color(240, 190, 80)
      view.opacity = 0.70f
    }

    // ── Register each target ──────────────────────────────────────────────────
    for ((tId, idx) <- targetIds.zipWithIndex) {
      println(s"=== [${idx + 1}/5]  $tId ===")

      val tFile = new File(dataDir, s"$tId.stl")
      if (!tFile.exists())
        println(s"  SKIP - STL not found: ${tFile.getAbsolutePath}")
      else if (!landmarks.contains(tId))
        println(s"  SKIP - '$tId' absent from landmark CSV")
      else {

        // Load and decimate target
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

        // Landmark spread sanity check
        val xs = targetLms.map(_.point.x)
        val ys = targetLms.map(_.point.y)
        val zs = targetLms.map(_.point.z)
        val spread = math.sqrt(
          math.pow(xs.max - xs.min, 2) + math.pow(ys.max - ys.min, 2) + math.pow(zs.max - zs.min, 2)
        )
        if (spread < 5.0)
          println(f"  !! WARNING: landmark spread = $spread%.1f mm — CSV row looks wrong.")

        // ── Step 1: Rigid align TARGET -> reference space ─────────────────────
        // NOTE: arguments are (moving=target, fixedTarget=refMesh).
        // This is the OPPOSITE of before. The target is brought into the
        // reference coordinate frame.  All subsequent work is in ref space.
        val (targetInRefSpace: TriangleMesh[_3D], _) = RigidAlign.landmarkThenIcp(
          targetMesh, targetLms,   // moving mesh  (target brought to ref)
          refMesh,    refLms,      // fixed target (reference is fixed)
          icpIterations = Config.icpIterations
        )
        println(s"  1. Target rigidly aligned into reference space")

        // ── Step 2: GP non-rigid ICP (deform reference to match target-in-ref-space)
        // Start from a fresh model every target
        var model = PointDistributionModel[_3D, TriangleMesh](refMesh, lowRankGP)

        for (iter <- 0 until Config.icpIterations) {
          val meanMesh = model.mean
          // Correspondences: mean points -> closest on TARGET-IN-REF-SPACE
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

        // Registration quality: registered (in ref space) vs target (in ref space)
        val distStats = Metrics.symmetric(registeredMesh, targetInRefSpace)
        println(s"  Quality : ${distStats.render}")
        if (distStats.mean > 3.0)
          println(s"  !! mean > 3 mm - inspect landmark CSV row for $tId")

        // ── Step 3: Save (both in reference space) ────────────────────────────
        val vtkOut = new File(outDir, s"registered_$tId.vtk")
        val h5Out  = new File(outDir, s"model_$tId.h5")
        MeshIO.writeMesh(registeredMesh, vtkOut).get
        StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, h5Out).get
        println(s"  Saved .vtk -> ${vtkOut.getName}")
        println(s"  Saved .h5  -> ${h5Out.getName}")

        // ── Step 4: Visualise — all in reference space, overlap correctly ──────
        ui.foreach { scalismoUi =>
          val grp = scalismoUi.createGroup(s"[${idx + 1}] $tId")

          // Target rigidly brought into reference space (grey, semi-transparent)
          val tView = scalismoUi.show(grp, targetInRefSpace, "Target in ref space")
          tView.color   = new Color(210, 210, 210)
          tView.opacity = 0.40f

          // Non-rigidly registered mean (blue, opaque)
          val rView = scalismoUi.show(grp, registeredMesh, "Registered mesh")
          rView.color   = new Color(50, 155, 220)
          rView.opacity = 0.90f

          // Posterior shape model — drag PC sliders to see modes
          scalismoUi.show(grp, model, "Shape model (modes)")
        }
      }
      println()
    }

    println(s"All done.  All registered meshes are in the reference coordinate space.")
    println(s"Results -> ${outDir.getAbsolutePath}")
  }
}
