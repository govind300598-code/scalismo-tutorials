package scapula

import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Shows every intermediate stage of the pipeline for ONE example specimen, in one scalismo-ui session, so the whole
 * process can be inspected visually instead of trusting printed numbers alone:
 *
 *   1 raw (unprocessed)          -- the STL file exactly as scanned
 *   2 mirrored + rigidly aligned -- after side-canonicalization + landmark Procrustes + ICP, next to the pivot
 *   3 reference topology         -- Stage 2's final (post-refinement) reference mesh, if it has been run
 *   4 registered vs. real target -- the reference warped onto this specimen, overlaid on its real rigidly-aligned
 *                                   target -- toggle between the two to see the registration fit by eye
 *
 * Stages 3 and 4 need Stage2GPNonRigidRegistration to have already been run (they read its saved output); stages 1
 * and 2 work standalone. Pick which specimen to inspect with SCAPULA_INSPECT_SPECIMEN (default: the pivot itself).
 */
object ViewPipelineStages {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    require(specimens.nonEmpty, s"No specimens with both a mesh and a landmark row found in ${dir.getPath}")

    // Same canonicalization + pivot selection as Stage 2, so "rigid aligned" here matches Stage 2's own logic.
    final case class Canonical(mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]])
    val canonicalById: Map[String, Canonical] = specimens.map { s =>
      val rawMesh = ScapulaData.loadMesh(s.file)
      val rawLms = landmarks(s.modelId)
      val c = if (s.isRight) Canonical(ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
              else Canonical(rawMesh, rawLms)
      s.modelId -> c
    }.toMap
    // GPA over landmarks (RigidAlign.landmarkGPA) instead of aligning everyone to one arbitrary specimen's own raw
    // pose -- see Stage2GPNonRigidRegistration's matching comment for why.
    val canonicalLandmarksById = specimens.map(s => s.modelId -> canonicalById(s.modelId).lms).toMap
    val gpaMeanLandmarks = RigidAlign.landmarkGPA(canonicalLandmarksById)
    val gpaMeanByName = gpaMeanLandmarks.map(lm => lm.id -> lm.point).toMap
    val pivotSpecimen = specimens.minBy { s =>
      canonicalById(s.modelId).lms.map { lm => val d = (lm.point - gpaMeanByName(lm.id)).norm; d * d }.sum
    }
    val pivotCanonical = canonicalById(pivotSpecimen.modelId)
    val pivotToGpaMean = ScapulaData.rigidFromLandmarks(pivotCanonical.lms, gpaMeanLandmarks)
    val pivotMesh = pivotCanonical.mesh.transform(pivotToGpaMean)
    val pivotLms = pivotCanonical.lms.map(lm => lm.copy(point = pivotToGpaMean(lm.point)))

    val inspectId = sys.env.getOrElse("SCAPULA_INSPECT_SPECIMEN", pivotSpecimen.modelId)
    val inspectSpecimen = specimens.find(_.modelId == inspectId).getOrElse(
      throw new RuntimeException(s"Unknown SCAPULA_INSPECT_SPECIMEN '$inspectId' -- must be one of: " +
        specimens.map(_.modelId).mkString(", "))
    )
    println(s"Inspecting specimen: ${inspectSpecimen.modelId}  (pivot/reference is ${pivotSpecimen.modelId})")

    // ---- stage 1: raw ----
    val rawMesh = ScapulaData.loadMesh(inspectSpecimen.file)

    // ---- stage 2: canonicalized (mirrored if right) + rigidly aligned onto the GPA mean frame ----
    val canon = canonicalById(inspectSpecimen.modelId)
    val rigidAligned =
      if (inspectSpecimen.modelId == pivotSpecimen.modelId) pivotMesh
      else RigidAlign.landmarkThenIcp(canon.mesh, canon.lms, pivotMesh, pivotLms, Config.icpIterations)._1

    // ---- stage 3 + 4: reference and registered mesh, read from Stage 2's saved output if it has been run ----
    val registeredFile = new File(new File(Config.outDir, "registered"), s"${inspectSpecimen.modelId}.vtk")
    val referenceFile = new File(Config.outDir, "reference.vtk")
    val registeredOpt = if (registeredFile.exists()) Some(ScapulaData.loadMesh(registeredFile)) else None
    val referenceOpt = if (referenceFile.exists()) Some(ScapulaData.loadMesh(referenceFile)) else None
    if (registeredOpt.isEmpty)
      println(s"  (no ${registeredFile.getPath} found -- run Stage2GPNonRigidRegistration first to see stages 3/4)")

    if (Config.showUi) {
      val ui = ScalismoUI()

      val g1 = ui.createGroup("1 raw (unprocessed)")
      ui.show(g1, rawMesh, inspectSpecimen.modelId)

      val g2 = ui.createGroup("2 mirrored + rigidly aligned")
      ui.show(g2, rigidAligned, s"${inspectSpecimen.modelId} (rigid)")
      ui.show(g2, pivotMesh, s"pivot ${pivotSpecimen.modelId}")

      referenceOpt.foreach { reference =>
        val g3 = ui.createGroup("3 reference topology")
        ui.show(g3, reference, "reference (post-refinement)")
      }

      registeredOpt.foreach { registered =>
        val g4 = ui.createGroup("4 registered vs. real target")
        ui.show(g4, registered, s"${inspectSpecimen.modelId} (registered)")
        ui.show(g4, rigidAligned, s"${inspectSpecimen.modelId} (real target)")
      }

      println("\nScalismo-UI window opened with the following groups in the scene tree:")
      println("  1 raw (unprocessed)          -- the STL file exactly as scanned")
      println("  2 mirrored + rigidly aligned -- after side-canonicalization + landmark Procrustes + ICP, next to the pivot")
      if (referenceOpt.isDefined)
        println("  3 reference topology         -- Stage 2's final (post-refinement) reference mesh")
      if (registeredOpt.isDefined)
        println("  4 registered vs. real target -- toggle the two meshes in this group on/off against each other: " +
          "if registration worked, they should look nearly identical")
      println("\nToggle groups/meshes on and off in the scene tree to compare stages. Close the window when done.")
    } else {
      println("SCAPULA_UI=false -- skipping the interactive viewer.")
    }
  }
}
