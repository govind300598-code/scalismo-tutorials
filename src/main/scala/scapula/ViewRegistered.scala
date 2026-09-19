package scapula

import scalismo.io.MeshIO
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Load and display all registered meshes for a given GP set in the Scalismo UI.
 *
 * Usage:
 *   sbt "runMain scapula.ViewRegistered"           # shows set3-alemneh (default)
 *   SCAPULA_GP_SET=0 sbt "runMain scapula.ViewRegistered"   # shows set1-baseline
 *
 * Groups in the UI:
 *   "registered-<setName>"  — all registered meshes (semi-transparent green)
 *   The first mesh (alphabetically) is shown in red as the reference.
 */
object ViewRegistered {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val gpSetIdx = sys.env.getOrElse("SCAPULA_GP_SET", "2").toInt
      .max(0).min(Stage2NonRigidReg.gpParamSets.length - 1)
    val gpParams = Stage2NonRigidReg.gpParamSets(gpSetIdx)

    val setDir = new File(Config.outDir, gpParams.label)
    require(
      setDir.exists && setDir.isDirectory,
      s"Output folder not found: ${setDir.getAbsolutePath}\n" +
      s"Run 'sbt runMain scapula.Set${gpSetIdx + 1}_*' first to generate the registered meshes."
    )

    val stlFiles = Option(setDir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.endsWith("_registered.stl"))
      .sortBy(_.getName)

    require(stlFiles.nonEmpty, s"No *_registered.stl files found in ${setDir.getAbsolutePath}")

    println(s"Loading ${stlFiles.length} registered meshes from ${setDir.getAbsolutePath}")
    println(s"GP set: σ=${gpParams.sigma}  s=${gpParams.scaleFactor}  (${gpParams.label})")

    val ui    = ScalismoUI()
    val group = ui.createGroup(s"registered-${gpParams.label}")

    // First file is the reference (shown in red); the rest in semi-transparent green.
    stlFiles.zipWithIndex.foreach { case (f, idx) =>
      val mesh = MeshIO.readMesh(f).getOrElse(
        throw new RuntimeException(s"Could not read ${f.getName}")
      )
      val name = f.getName.stripSuffix("_registered.stl")
      val view = ui.show(group, mesh, name)
      if (idx == 0) {
        view.color   = java.awt.Color.RED
        view.opacity = 1.0f
        println(s"  [reference] ${f.getName}")
      } else {
        view.color   = java.awt.Color.GREEN
        view.opacity = 0.35f
        println(s"  [${idx}] ${f.getName}")
      }
    }

    println(s"\nAll ${stlFiles.length} meshes loaded.")
    println("Red = reference  |  Green (semi-transparent) = registered specimens")
    println("Use the Scalismo UI scene tree to toggle individual meshes on/off.")
  }
}
