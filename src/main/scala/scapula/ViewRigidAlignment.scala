package scapula

import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

/**
 * Quick VISUAL sanity check of the rigid (landmark Procrustes + ICP) alignment step, before committing to the full
 * ~1-2 hour non-rigid registration in Stage2GPNonRigidRegistration. Opens a scalismo-ui window showing all 24
 * rigidly-aligned specimens together: they should all sit in roughly the same pose (position/orientation) while
 * still showing their own real anatomical differences. A specimen that looks wildly off relative to the others
 * means its landmarks or its rigid alignment has a problem, and that's worth catching here rather than after an
 * hour of non-rigid registration downstream of it.
 *
 * Duplicates Stage2's pivot-selection and rigid-alignment logic (rather than refactoring Stage2 to expose
 * intermediate state) because this is a small, standalone, throwaway debugging tool, not part of the pipeline.
 */
object ViewRigidAlignment {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val dir = Config.dataDir
    val csv = ScapulaData.csvFile(dir)
    println(s"Data directory : ${dir.getAbsolutePath}")
    println(s"Landmark CSV   : ${csv.getAbsolutePath}")

    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimens = ScapulaData.specimens(dir).filter(s => landmarks.contains(s.modelId))
    require(specimens.nonEmpty, s"No specimens with both a mesh and a landmark row found in ${dir.getPath}")
    println(s"${specimens.length} specimens with mesh + landmarks")

    // Left and right scapulae are mirror images, not related by any rotation -- averaging raw landmarks or rigidly
    // aligning a right specimen onto a left reference (or vice versa) with a pure rotation cannot work, which is
    // exactly the doubled/backwards look from overlaying un-mirrored L and R specimens together. Canonicalize every
    // specimen onto the left-side convention first.
    final case class Canonical(mesh: TriangleMesh[_3D], lms: IndexedSeq[Landmark[_3D]])
    val canonicalById: Map[String, Canonical] = specimens.map { s =>
      val rawMesh = ScapulaData.loadMesh(s.file)
      val rawLms = landmarks(s.modelId)
      val c = if (s.isRight) Canonical(ScapulaData.mirrorMesh(rawMesh), ScapulaData.mirrorLandmarks(rawLms))
              else Canonical(rawMesh, rawLms)
      s.modelId -> c
    }.toMap
    println(s"Mirrored ${specimens.count(_.isRight)} right-side specimens onto the left-side convention.")

    // GPA (Generalized Procrustes Analysis) over landmarks: iteratively re-estimate the population mean landmark
    // configuration instead of aligning everyone to one arbitrary real specimen's own raw pose -- see
    // RigidAlign.landmarkGPA and Stage2GPNonRigidRegistration's matching comment for why.
    val canonicalLandmarksById = specimens.map(s => s.modelId -> canonicalById(s.modelId).lms).toMap
    val gpaMeanLandmarks = RigidAlign.landmarkGPA(canonicalLandmarksById)
    val gpaMeanByName = gpaMeanLandmarks.map(lm => lm.id -> lm.point).toMap
    val referenceSpecimen = specimens.minBy { s =>
      canonicalById(s.modelId).lms.map { lm =>
        val d = (lm.point - gpaMeanByName(lm.id)).norm
        d * d
      }.sum
    }
    println(s"Reference specimen (closest to GPA mean landmark configuration, left-side convention): " +
      s"${referenceSpecimen.modelId}")

    val refCanonical = canonicalById(referenceSpecimen.modelId)
    val refToGpaMean = ScapulaData.rigidFromLandmarks(refCanonical.lms, gpaMeanLandmarks)
    val referenceRawMesh = refCanonical.mesh.transform(refToGpaMean)
    val referenceLms = refCanonical.lms.map(lm => lm.copy(point = refToGpaMean(lm.point)))

    val alignedTargets = specimens.map { s =>
      val c = canonicalById(s.modelId)
      if (s.modelId == referenceSpecimen.modelId) s.modelId -> referenceRawMesh
      else {
        val (aligned, _) = RigidAlign.landmarkThenIcp(c.mesh, c.lms, referenceRawMesh, referenceLms, Config.icpIterations)
        s.modelId -> aligned
      }
    }
    println(s"Rigidly aligned all ${alignedTargets.length} specimens into the GPA mean landmark frame.")

    if (Config.showUi) {
      val ui = ScalismoUI()
      val group = ui.createGroup("rigid alignment check")
      alignedTargets.foreach { case (id, mesh) => ui.show(group, mesh, id) }
      println(s"\nScalismo-UI window opened with all ${alignedTargets.length} specimens in the 'rigid alignment " +
        "check' group. They should all roughly overlap in position/orientation while still showing their own real " +
        "shape differences (some bigger/smaller, different glenoid angle, etc.) -- that's expected and is exactly " +
        "what Stage 2's non-rigid step is for. What should NOT happen: a specimen sitting rotated or offset well " +
        "away from the rest, or facing the wrong way -- toggle specimens on/off in the scene tree to spot one if " +
        "there is one. Close the window when done.")
    } else {
      println("SCAPULA_UI=false -- skipping the interactive viewer.")
    }
  }
}
