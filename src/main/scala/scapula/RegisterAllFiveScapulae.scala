package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.utils.Random
import scalismo.ui.api.*
import java.io.File

/**
 * STAGE 2 -- BASELINE REGISTRATION.
 *
 * Registers the first 5 left scapulae to a reference using GP-ICP with the default
 * kernel parameters (sigma=65 mm, scale=20 mm).  Output: one VTK per target, plus the
 * HDF5 GP prior, written to $SCAPULA_OUT_DIR/Scapula_GP_Registered/.
 *
 * Run with:
 *   sbt -DSCAPULA_DATA_DIR=/path/to/stls -DSCAPULA_OUT_DIR=/path/to/out \
 *       "runMain scapula.RegisterAllFiveScapulae"
 *
 * Or using environment variables:
 *   SCAPULA_DATA_DIR=/path/to/stls SCAPULA_OUT_DIR=/path/to/out sbt \
 *       "runMain scapula.RegisterAllFiveScapulae"
 */
object RegisterAllFiveScapulae {

  val DefaultSigma = 65.0   // kernel bandwidth (mm)
  val DefaultScale = 20.0   // kernel amplitude (mm)
  val NumTargets   = 5

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dataDir = Config.dataDir
    val outDir  = new File(Config.outDir, "Scapula_GP_Registered")
    outDir.mkdirs()

    println("=" * 80)
    println("STAGE 2 -- GP-ICP REGISTRATION (baseline parameters)")
    println("=" * 80)
    println(s"Data dir : ${dataDir.getAbsolutePath}")
    println(s"Out dir  : ${outDir.getAbsolutePath}")
    println(s"sigma    : $DefaultSigma mm   scale: $DefaultScale mm")
    println(s"ICP iters: ${Config.icpIterations}   GP tol: ${Config.gpRelativeTolerance}   max rank: ${Config.gpMaxRank}")

    // ---- load landmarks + specimens ----------------------------------------
    val (landmarks, fromHeader, _) = ScapulaData.readLandmarkCsv(ScapulaData.csvFile(dataDir))
    if (!fromHeader)
      println("  WARNING: landmark columns resolved by fallback offsets, not header names. Verify A1 output.")

    val leftSpecs = ScapulaData.specimens(dataDir)
      .filter(s => !s.isRight && landmarks.contains(s.modelId))
    require(leftSpecs.size > NumTargets,
      s"Need at least ${NumTargets + 1} left specimens with landmarks, found ${leftSpecs.size}")

    // ---- reference ---------------------------------------------------------
    val refSpec  = leftSpecs.head
    val rawRef   = ScapulaData.loadMesh(refSpec.file)
    val refMesh  = GPRegistration.decimateIfNeeded(rawRef, Config.modelResolution)
    val refLms   = landmarks(refSpec.modelId)

    println(s"\nReference: ${refSpec.modelId}  (${refMesh.pointSet.numberOfPoints} vertices)")

    // ---- build GP prior ----------------------------------------------------
    println(s"\nBuilding GP prior ...")
    val model     = GPRegistration.buildModel(refMesh, DefaultSigma, DefaultScale)
    val priorFile = new File(outDir, s"${refSpec.modelId}_prior.h5")
    GPRegistration.saveModel(model, priorFile)

    // Save first 3 mode shapes for sanity-checking the prior
    for (mode <- 0 until math.min(3, model.rank)) {
      val sign = if (mode == 0) "pos" else "pos"
      val f    = new File(outDir, s"${refSpec.modelId}_mode${mode + 1}_pos3std.vtk")
      GPRegistration.saveModeShape(model, mode, 3.0, f)
    }

    // ---- optional UI -------------------------------------------------------
    val ui = if (Config.showUi) {
      val u = ScalismoUI()
      val g = u.createGroup("reference")
      u.show(refMesh, g, refSpec.modelId)
      Some(u)
    } else None

    // ---- register targets --------------------------------------------------
    val targets = leftSpecs.tail.take(NumTargets)

    targets.zipWithIndex.foreach { case (spec, idx) =>
      println(s"\n[${idx + 1}/$NumTargets] Registering ${spec.modelId} ...")

      val rawMesh = ScapulaData.loadMesh(spec.file)
      val rawLms  = landmarks(spec.modelId)

      // Rigid landmark Procrustes + trimmed ICP to bring target into reference frame
      println("  Rigid alignment ...")
      val (rigidMesh, _) = RigidAlign.landmarkThenIcp(rawMesh, rawLms, refMesh, refLms)
      val dRigid = Metrics.symmetric(rigidMesh, refMesh)
      println(s"  After rigid: ${dRigid.render}")

      // GP-ICP non-rigid registration
      println("  GP-ICP ...")
      val registered = GPRegistration.register(rigidMesh, model)

      // Save
      val outVtk = new File(outDir, s"${refSpec.modelId}_${spec.modelId}_registered.vtk")
      GPRegistration.saveRegistered(registered, outVtk)

      // Final surface statistics
      val dFinal = Metrics.symmetric(registered, rigidMesh)
      println(s"  Final vs target: ${dFinal.render}")

      ui.foreach { u =>
        val g = u.createGroup(spec.modelId)
        u.show(rigidMesh,  g, "target (rigid)")
        u.show(registered, g, "registered")
      }
    }

    println("\n" + "=" * 80)
    println("Done.  Results in: " + outDir.getAbsolutePath)
    println("=" * 80)

    if (Config.showUi) {
      println("Scalismo UI is open.  Close the window to exit.")
    }
  }
}
