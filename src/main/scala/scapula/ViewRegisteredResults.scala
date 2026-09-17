package scapula

import scalismo.geometry.*
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Shows the actual output of non-rigid registration (Stage 2): for each requested specimen, the reference warped
 * onto it (the "registered" mesh) next to the real rigidly-aligned target it was fit to, in one scalismo-ui
 * session, one group per specimen. This is the direct visual test of registration quality -- toggle the two
 * meshes in a group on/off against each other; if registration worked, they should look nearly identical.
 *
 * Reads registered/<id>.vtk and aligned_targets/<id>.vtk from Stage2GPNonRigidRegistration's saved output --
 * run that first (aligned_targets/ requires the current version of Stage 2). Specimen ids default to
 * SCAPULA_WATCH_SPECIMENS, overridable via SCAPULA_VIEW_SPECIMENS (comma-separated).
 */
object ViewRegisteredResults {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val ids: IndexedSeq[String] = sys.env.get("SCAPULA_VIEW_SPECIMENS")
      .map(_.split(",").map(_.trim).filter(_.nonEmpty).toIndexedSeq)
      .filter(_.nonEmpty)
      .getOrElse(Config.watchSpecimenIds.toIndexedSeq)
    require(ids.nonEmpty, "Set SCAPULA_VIEW_SPECIMENS or SCAPULA_WATCH_SPECIMENS to at least one specimen id.")

    val registeredDir = new File(Config.outDir, "registered")
    val targetsDir = new File(Config.outDir, "aligned_targets")

    val ui = if (Config.showUi) Some(ScalismoUI()) else None
    if (ui.isEmpty) println("SCAPULA_UI=false -- skipping the interactive viewer; only the printed numbers below apply.")

    ids.foreach { id =>
      val registeredFile = new File(registeredDir, s"$id.vtk")
      val targetFile = new File(targetsDir, s"$id.vtk")
      if (!registeredFile.exists() || !targetFile.exists()) {
        println(s"  $id  !! missing ${registeredFile.getPath} or ${targetFile.getPath} -- run " +
          "Stage2GPNonRigidRegistration first (aligned_targets/ requires the current version)")
      } else {
        val registered = ScapulaData.loadMesh(registeredFile)
        val target = ScapulaData.loadMesh(targetFile)
        val stats = Metrics.symmetric(registered, target)
        println(f"  $id%-24s ${stats.render}")

        ui.foreach { u =>
          val group = u.createGroup(id)
          u.show(group, registered, s"$id (registered)")
          u.show(group, target, s"$id (real target)")
        }
      }
    }

    if (ui.isDefined) {
      println("\nScalismo-UI window opened with one group per specimen, each containing the '(registered)' mesh " +
        "(reference warped by non-rigid GP registration) and the '(real target)' mesh (the actual rigidly-aligned " +
        "specimen it was fit to). Toggle the two within a group on/off against each other: the closer they look, " +
        "the better that specimen's registration. Close the window when done.")
    }
  }
}
