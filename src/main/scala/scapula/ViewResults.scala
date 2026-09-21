package scapula

import scalismo.io.MeshIO
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Opens the Statistical Density Model's mean HU field in an interactive Scalismo-UI
 * window. Reads reference_mesh_meanHU.vtk, which DensityPipeline's Stage 4 writes --
 * run the pipeline to completion first.
 */
object ViewResults {
  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val outDir = new File(sys.env.getOrElse("SCAPULA_DENSITY_OUT", DensityPipeline.densityOutDir.getAbsolutePath))
    val vtkFile = new File(outDir, "reference_mesh_meanHU.vtk")

    if (!vtkFile.exists()) {
      println(s"ERROR: ${vtkFile.getAbsolutePath} not found.")
      println("Run the DensityPipeline to Stage 4 completion first, or set SCAPULA_DENSITY_OUT")
      println("to the output directory of a completed run.")
      sys.exit(1)
    }

    val field = MeshIO.readScalarMeshField[Float](vtkFile)
      .getOrElse(throw new RuntimeException(s"Could not read ${vtkFile.getAbsolutePath}"))

    println(s"Loaded mean HU field: ${field.domain.pointSet.numberOfPoints} vertices")
    println(s"Opening Scalismo-UI -- close the window to exit.")

    val ui = ScalismoUI()
    val group = ui.createGroup("Statistical Density Model")
    ui.show(group, field, "mean HU")
  }
}
