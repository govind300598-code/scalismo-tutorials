// VisualizeSSMAndSurfaces.scala
// Root-level visualization script (no package) — fixed for scalismo 0.92.
//
// API changes from earlier scalismo versions:
//   scalismo.ui.api.ScalismoUI        (was scalismo.ui.ScalismoUI)
//   scalismo.statisticalmodel.*       (was scalismo.statisticalshape.*)
//   scalismo.io.MeshIO.readMesh(f)    (was TriangleMesh3D.read(f))
//   scala.util.{Failure,Success,Try}  (must be explicitly imported in Scala 3)

import scalismo.ui.api.ScalismoUI
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.{PointDistributionModel, StatisticalMeshModel}
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh

import java.io.File
import scala.util.{Failure, Success, Try}

object VisualizeSSMAndSurfaces {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val ui = ScalismoUI("Scapula SSM + Surfaces")

    val outDir  = new File(sys.env.getOrElse("SCAPULA_OUT_DIR",
      "/home/g25upadh/Documents/100 plus scapula data/scapula_atlas_out"))
    val dataDir = new File(sys.env.getOrElse("SCAPULA_DATA_DIR",
      "/home/g25upadh/Documents/100 plus scapula data/paired_scapulae_STLs_scapula"))

    // ── Load SSM models (SSM1–SSM4) ──────────────────────────────────────────
    for (iter <- 1 to 4) {
      val modelFile = new File(outDir, s"results/SSM$iter/model/SSM$iter.h5")
      if (modelFile.exists()) {
        StatisticalModelIO.readStatisticalMeshModel(modelFile) match {
          case Failure(ex) =>
            println(s"Failed to parse SSM file: ${ex.getMessage}")
          case Success(model) =>
            val group = ui.createGroup(s"SSM$iter")
            ui.show(group, model.mean, s"SSM${iter}_mean")
            println(s"Loaded SSM$iter  rank=${model.rank}")
        }
      }
    }

    // ── Load a subset of surface STL files ───────────────────────────────────
    if (dataDir.exists()) {
      val surfGroup = ui.createGroup("Surfaces")
      Option(dataDir.listFiles())
        .getOrElse(Array.empty)
        .filter(_.getName.endsWith(".stl"))
        .sortBy(_.getName)
        .take(6)
        .foreach { file =>
          MeshIO.readMesh(file) match {
            case Failure(ex)   => println(s"Failed to load ${file.getName}: ${ex.getMessage}")
            case Success(mesh) => ui.show(surfGroup, mesh, file.getName.stripSuffix(".stl"))
          }
        }
    } else {
      println(s"Data directory not found: ${dataDir.getPath}")
    }

    println("Scalismo UI ready. Close the window to exit.")
  }
}
