package scapula

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.ui.api.*
import scalismo.utils.Random

/**
 * Scapula SSM — Full Pipeline Viewer
 *
 * Launches a ScalismoUI window with five labelled groups that let you inspect every stage of the
 * pipeline at a glance:
 *
 *   S01_Landmarks                  — raw landmark positions for every specimen.
 *                                    All landmarks from the same subject should lie close together.
 *                                    A cloud of scattered points means the CSV is mis-parsed.
 *
 *   S02_Reference                  — the reference mesh (first left scapula) and its five landmarks.
 *
 *   S03_LandmarkAligned            — every mesh after landmark Procrustes only.
 *                                    If the cluster is still scattered here the landmark placement
 *                                    protocol is inconsistent — see Stage1Diagnostics [A3].
 *
 *   S04_RigidAligned               — after trimmed rigid ICP on top of the Procrustes init.
 *                                    ALL bones should overlap into one tight cloud. Outliers here
 *                                    point to orientation problems (mirror axis) or failed ICP.
 *
 *   S05_NonRigid_Pass1             — after non-rigid GP ICP, Pass 1. The overlap should be tighter
 *                                    than S04. Remaining surface distance is captured by the SSM.
 *
 * If Config.refinePasses > 1 the viewer also adds:
 *   S06_Reference_Pass2            — rebuilt mean reference from Pass 1.
 *   S07_NonRigid_Pass2             — Pass 2 registrations (reference bias removed).
 *   ... and so on up to Config.refinePasses.
 *
 * Set the environment variable SCAPULA_UI=false to skip this viewer entirely and run the pipeline
 * headlessly (useful on a cluster node).
 */
object FullPipelineViewer {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    if (!Config.showUi) {
      println("SCAPULA_UI=false — running pipeline without viewer.")
      runHeadless()
      return
    }

    val ui = ScalismoUI("Scapula SSM — Full Pipeline Viewer")

    // -------------------------------------------------------------------------
    // Load data
    // -------------------------------------------------------------------------
    val dir  = Config.dataDir
    val csv  = ScapulaData.csvFile(dir)
    println(s"Data directory : ${dir.getAbsolutePath}")
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")

    val (landmarkMap, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens           = ScapulaData.specimens(dir)
    val specimenWithLm      = specimens.filter(s => landmarkMap.contains(s.modelId))

    println(s"${specimens.size} STL files, ${specimenWithLm.size} with landmark data")

    // -------------------------------------------------------------------------
    // S01 — raw landmarks
    // -------------------------------------------------------------------------
    val g01 = ui.createGroup("S01_Landmarks (GC/TS/IA/PLA/AC)")
    specimenWithLm.foreach { s =>
      ui.show(g01, landmarkMap(s.modelId).toIndexedSeq, s.modelId)
    }
    println("[S01] Landmarks loaded.")

    // -------------------------------------------------------------------------
    // S02 — reference mesh + landmarks
    // -------------------------------------------------------------------------
    val refSpecimen = specimenWithLm
      .find(!_.isRight)
      .getOrElse(specimenWithLm.head)

    val refMeshRaw  = ScapulaData.loadMesh(refSpecimen.file)
    val refLmsRaw   = landmarkMap(refSpecimen.modelId)

    // Decimate to target resolution so the reference has exactly modelResolution vertices.
    val refMesh = refMeshRaw.operations.decimate(targetedNumberOfVertices = Config.modelResolution)
    val refLms  = refLmsRaw  // landmarks are independent of decimation

    val g02 = ui.createGroup("S02_Reference")
    ui.show(g02, refMesh, refSpecimen.modelId)
    ui.show(g02, refLms.toIndexedSeq, "Landmarks")
    println(s"[S02] Reference: ${refSpecimen.modelId}  (${refMesh.pointSet.numberOfPoints} vertices after decimation)")

    // -------------------------------------------------------------------------
    // S03 — landmark-aligned meshes (Procrustes only)
    // -------------------------------------------------------------------------
    val g03 = ui.createGroup("S03_LandmarkAligned (if still scattered — landmarks are WRONG)")

    val lmAligned: IndexedSeq[(String, TriangleMesh[_3D], IndexedSeq[Landmark[_3D]])] =
      specimenWithLm.map { s =>
        val (mesh, lms) = prepareMesh(s, landmarkMap(s.modelId))
        val lmTrans     = ScapulaData.rigidFromLandmarks(lms, refLms)
        val aligned     = mesh.transform(lmTrans)
        val alignedLms  = lms.map(lm => lm.copy(point = lmTrans(lm.point)))
        val label       = if (s.isRight) s.modelId + "_mirrored" else s.modelId
        (label, aligned, alignedLms)
      }

    lmAligned.foreach { case (label, mesh, _) => ui.show(g03, mesh, label) }
    println(s"[S03] ${lmAligned.size} landmark-aligned meshes.")

    // -------------------------------------------------------------------------
    // S04 — rigid ICP on top of landmark init
    // -------------------------------------------------------------------------
    val g04 = ui.createGroup("S04_RigidAligned (ALL bones should overlap — verify alignment)")

    val rigidAligned: IndexedSeq[(String, TriangleMesh[_3D], IndexedSeq[Landmark[_3D]])] =
      specimenWithLm.map { s =>
        val (mesh, lms) = prepareMesh(s, landmarkMap(s.modelId))
        val (aligned, alignedLms) = RigidAlign.landmarkThenIcp(mesh, lms, refMesh, refLms)
        val label = if (s.isRight) s.modelId + "_mirrored" else s.modelId
        (label, aligned, alignedLms)
      }

    rigidAligned.foreach { case (label, mesh, _) => ui.show(g04, mesh, label) }
    println(s"[S04] ${rigidAligned.size} rigid-aligned meshes.")

    // -------------------------------------------------------------------------
    // S05+ — non-rigid GP ICP, one or more refinement passes
    // -------------------------------------------------------------------------
    println("[S05] Building GP deformation model…")
    var currentRef  = refMesh
    var currentLms  = refLms

    for (pass <- 1 to Config.refinePasses) {
      println(s"[S05/$pass] Fitting model to all specimens (this takes a few minutes)…")
      val model = NonRigidRegistration.buildModel(currentRef)

      val passLabel = if (pass == 1) "Pass1" else s"Pass$pass"
      val tightness = if (pass == 1) "tighter overlap than S04" else s"reference bias removed (pass $pass)"
      val groupName = s"S0${4 + pass}_NonRigid_$passLabel ($tightness)"
      val g = ui.createGroup(groupName)

      var done = 0
      val registered = NonRigidRegistration.registerAll(
        currentRef, currentLms, specimenWithLm, landmarkMap, model,
        onProgress = (label, mesh) => {
          ui.show(g, mesh, label)
          done += 1
          println(s"  [$done/${specimenWithLm.size}] $label")
        }
      )

      println(s"[S05/$pass] Pass $pass complete. ${registered.size} meshes registered.")

      if (pass < Config.refinePasses) {
        println(s"[S05/$pass] Rebuilding reference for pass ${pass + 1}…")
        // Only use left (or first-encountered-per-subject) meshes for the mean to avoid
        // inflating the sample size with the near-duplicate mirrored specimens.
        val leftRegistered = registered.filterNot(_._1.endsWith("_mirrored"))
        currentRef  = NonRigidRegistration.rebuildReference(leftRegistered)
        currentLms  = refLms  // carry original 5 landmarks; they are still valid on the mean
        val gRef = ui.createGroup(s"S0${5 + pass}_Reference_Pass${pass + 1}")
        ui.show(gRef, currentRef, s"mean_reference_pass$pass")
        println(s"[S05/$pass] New reference has ${currentRef.pointSet.numberOfPoints} vertices.")
      }
    }

    println("Pipeline viewer ready.  Use the Scalismo UI to inspect each stage.")
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /** Load and optionally mirror a specimen, returning the mesh and its (possibly mirrored) landmarks. */
  private def prepareMesh(s: ScapulaData.Specimen,
                           lms: IndexedSeq[Landmark[_3D]]
  ): (TriangleMesh[_3D], IndexedSeq[Landmark[_3D]]) =
    if (s.isRight)
      (ScapulaData.mirrorMesh(ScapulaData.loadMesh(s.file)), ScapulaData.mirrorLandmarks(lms))
    else
      (ScapulaData.loadMesh(s.file), lms)

  /**
   * Headless pipeline run (no UI). Computes and prints summary statistics for each registration
   * pass.  Useful for batch jobs where opening a Swing window is not possible.
   */
  private def runHeadless()(implicit rng: Random): Unit = {
    val dir  = Config.dataDir
    val csv  = ScapulaData.csvFile(dir)
    val (landmarkMap, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens           = ScapulaData.specimens(dir)
    val specimenWithLm      = specimens.filter(s => landmarkMap.contains(s.modelId))

    val refSpecimen = specimenWithLm.find(!_.isRight).getOrElse(specimenWithLm.head)
    val refMeshRaw  = ScapulaData.loadMesh(refSpecimen.file)
    val refMesh     = refMeshRaw.operations.decimate(targetedNumberOfVertices = Config.modelResolution)
    val refLms      = landmarkMap(refSpecimen.modelId)

    var currentRef = refMesh
    var currentLms = refLms

    for (pass <- 1 to Config.refinePasses) {
      println(s"=== Non-rigid Pass $pass ===")
      val model = NonRigidRegistration.buildModel(currentRef)

      val registered = NonRigidRegistration.registerAll(
        currentRef, currentLms, specimenWithLm, landmarkMap, model,
        onProgress = (label, _) => println(s"  registered $label")
      )

      val stats = registered.map { case (_, reg) =>
        Metrics.symmetric(reg, currentRef)
      }
      val meanDist = stats.map(_.mean).sum / stats.length
      val hd95     = stats.map(_.hd95).sum / stats.length
      println(f"  mean surface distance to reference: $meanDist%.2f mm  |  mean HD95: $hd95%.2f mm")

      if (pass < Config.refinePasses) {
        val leftRegistered = registered.filterNot(_._1.endsWith("_mirrored"))
        currentRef  = NonRigidRegistration.rebuildReference(leftRegistered)
        currentLms  = refLms
      }
    }
  }
}
