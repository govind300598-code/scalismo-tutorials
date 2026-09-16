package scapula

import scalismo.geometry.*
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

    val meanLandmarks = ScapulaData.landmarkNames.map { name =>
      val pts = specimens.map(s => landmarks(s.modelId).find(_.id == name).get.point)
      name -> Point3D(pts.map(_.x).sum / pts.length, pts.map(_.y).sum / pts.length, pts.map(_.z).sum / pts.length)
    }.toMap
    val referenceSpecimen = specimens.minBy { s =>
      landmarks(s.modelId).map { lm =>
        val d = (lm.point - meanLandmarks(lm.id)).norm
        d * d
      }.sum
    }
    println(s"Reference specimen (closest to mean landmark configuration): ${referenceSpecimen.modelId}")

    val referenceLms = landmarks(referenceSpecimen.modelId)
    val referenceRawMesh = ScapulaData.loadMesh(referenceSpecimen.file)

    val alignedTargets = specimens.map { s =>
      val mesh = ScapulaData.loadMesh(s.file)
      val lms = landmarks(s.modelId)
      if (s.modelId == referenceSpecimen.modelId) s.modelId -> mesh
      else {
        val (aligned, _) = RigidAlign.landmarkThenIcp(mesh, lms, referenceRawMesh, referenceLms, Config.icpIterations)
        s.modelId -> aligned
      }
    }
    println(s"Rigidly aligned all ${alignedTargets.length} specimens into the reference's landmark frame.")

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
