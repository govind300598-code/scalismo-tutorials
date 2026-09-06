package scapula

import scalismo.geometry._3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel

import java.io.File

object SSMBuilder {

  def buildSSM(meshes: IndexedSeq[TriangleMesh[_3D]]): PointDistributionModel[_3D, TriangleMesh] = {
    require(meshes.nonEmpty, "No registered meshes to build SSM from")
    PointDistributionModel.createUsingPCA(meshes)
  }

  // ── Caching helpers ──────────────────────────────────────────────────────────

  def cacheDir(tag: String): File = new File(Config.outDir, tag)

  def saveMeshes(meshes: IndexedSeq[TriangleMesh[_3D]], tag: String): Unit = {
    val dir = cacheDir(tag)
    dir.mkdirs()
    meshes.zipWithIndex.foreach { case (m, i) =>
      MeshIO.writeMesh(m, new File(dir, f"mesh_$i%04d.vtk"))
        .recover { case e => println(s"[warn] Could not save mesh $i: ${e.getMessage}") }
    }
  }

  def loadMeshes(tag: String): Option[IndexedSeq[TriangleMesh[_3D]]] = {
    val dir = cacheDir(tag)
    if (!dir.exists()) return None
    val files = Option(dir.listFiles()).getOrElse(Array.empty)
      .filter(_.getName.endsWith(".vtk"))
      .sortBy(_.getName)
    if (files.isEmpty) return None
    val meshes = files.flatMap(f => MeshIO.readMesh(f).toOption)
    if (meshes.length == files.length) Some(meshes.toIndexedSeq) else None
  }

  def saveSSM(model: PointDistributionModel[_3D, TriangleMesh], name: String): Unit = {
    Config.outDir.mkdirs()
    val f = new File(Config.outDir, s"$name.h5")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, f)
      .recover { case e => println(s"[warn] Could not save SSM: ${e.getMessage}") }
  }

  def loadSSM(name: String): Option[PointDistributionModel[_3D, TriangleMesh]] = {
    val f = new File(Config.outDir, s"$name.h5")
    if (!f.exists()) None
    else StatisticalModelIO.readStatisticalTriangleMeshModel3D(f).toOption
  }
}
