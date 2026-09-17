package scapula

import scalismo.common.ScalarMeshField3D
import scalismo.geometry.*
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Per-vertex registered-vs-real-target distance, rendered as a colour heatmap in scalismo-ui, so small localized
 * problem patches (a bit of glenoid rim, an acromion tip) are visible directly, instead of trying to spot them by
 * toggling two solid-coloured meshes on and off against each other.
 *
 * Reads registered/<id>.vtk and aligned_targets/<id>.vtk from Stage2GPNonRigidRegistration's saved output --
 * run that first (aligned_targets/ is only written by the updated Stage 2, so re-run it once if that directory
 * doesn't exist yet). Shows one specimen per id in SCAPULA_HEATMAP_SPECIMENS (comma-separated), falling back to
 * SCAPULA_WATCH_SPECIMENS, then to a single SCAPULA_INSPECT_SPECIMEN, if unset.
 */
object ViewRegistrationHeatmap {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val ids: IndexedSeq[String] = {
      val fromHeatmap = sys.env.get("SCAPULA_HEATMAP_SPECIMENS")
        .map(_.split(",").map(_.trim).filter(_.nonEmpty).toIndexedSeq).filter(_.nonEmpty)
      val fromWatch = if (Config.watchSpecimenIds.nonEmpty) Some(Config.watchSpecimenIds) else None
      val fromInspect = sys.env.get("SCAPULA_INSPECT_SPECIMEN").map(IndexedSeq(_))
      fromHeatmap.orElse(fromWatch).orElse(fromInspect).getOrElse(
        throw new RuntimeException(
          "Set SCAPULA_HEATMAP_SPECIMENS (or SCAPULA_WATCH_SPECIMENS / SCAPULA_INSPECT_SPECIMEN) to at least one specimen id."
        )
      )
    }

    val registeredDir = new File(Config.outDir, "registered")
    val targetsDir = new File(Config.outDir, "aligned_targets")

    val ui = if (Config.showUi) Some(ScalismoUI()) else None
    if (ui.isEmpty) println("SCAPULA_UI=false -- skipping the interactive viewer; only the printed numbers below apply.")

    ids.foreach { id =>
      val registeredFile = new File(registeredDir, s"$id.vtk")
      val targetFile = new File(targetsDir, s"$id.vtk")
      if (!registeredFile.exists() || !targetFile.exists()) {
        println(s"  $id  !! missing ${registeredFile.getPath} or ${targetFile.getPath} -- run " +
          "Stage2GPNonRigidRegistration first (aligned_targets/ requires the updated version)")
      } else {
        val registered = ScapulaData.loadMesh(registeredFile)
        val target = ScapulaData.loadMesh(targetFile)
        val ops = target.operations
        val distances: IndexedSeq[Double] =
          registered.pointSet.points.map(p => (p - ops.closestPointOnSurface(p).point).norm).toIndexedSeq
        val meanD = distances.sum / distances.length
        val maxD = distances.max
        println(f"  $id%-24s per-vertex distance to real target: mean=$meanD%.3f mm  max=$maxD%.3f mm")

        ui.foreach { u =>
          val group = u.createGroup(id)
          val field = ScalarMeshField3D(registered, distances)
          u.show(group, field, s"$id error (mm)")
        }
      }
    }

    if (ui.isDefined) {
      println("\nScalismo-UI window opened with one group per specimen, each a colour-coded per-vertex " +
        "registration-error heatmap (mm, registered mesh vs. its real rigidly-aligned target). Small isolated " +
        "bright patches ARE exactly the localized fit problems a solid on/off toggle hides -- rotate each mesh " +
        "and look around the glenoid rim, acromion tip and coracoid process specifically. Close the window when done.")
    }
  }
}
