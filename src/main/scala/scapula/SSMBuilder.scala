package scapula

import scalismo.common.DiscreteField
import scalismo.geometry.{EuclideanVector, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection

import java.io.File

object SSMBuilder {

  /**
   * Build a PCA-based SSM from meshes that are already in correspondence with `reference`.
   * The DataCollection is built by computing the deformation from each registered mesh
   * back to the reference, then running PCA on those deformation fields.
   */
  def buildSSM(
    reference: TriangleMesh[_3D],
    meshes: IndexedSeq[TriangleMesh[_3D]]
  ): PointDistributionModel[_3D, TriangleMesh] = {
    require(meshes.nonEmpty, "No registered meshes to build SSM from")

    val defFields = meshes.map { mesh =>
      val defs = reference.pointSet.pointsWithId.map { case (refPt, id) =>
        mesh.pointSet.point(id) - refPt
      }.toIndexedSeq
      DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]](reference, defs)
    }

    val dc = DataCollection[_3D, TriangleMesh, EuclideanVector[_3D]](defFields)
    PointDistributionModel.createUsingPCA(dc)
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
    val files = Option(dir.listFiles()).getOrElse(Array.empty[File])
      .filter(_.getName.endsWith(".vtk"))
      .sortBy(_.getName)
    if (files.isEmpty) return None
    val meshes = files.flatMap(f => MeshIO.readMesh(f).toOption)
    if (meshes.length == files.length) Some(meshes.toIndexedSeq) else None
  }

  def saveSSM(model: PointDistributionModel[_3D, TriangleMesh], name: String): Unit = {
    Config.outDir.mkdirs()
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, new File(Config.outDir, s"$name.h5"))
      .recover { case e => println(s"[warn] Could not save SSM: ${e.getMessage}") }
  }

  def loadSSM(name: String): Option[PointDistributionModel[_3D, TriangleMesh]] = {
    val f = new File(Config.outDir, s"$name.h5")
    if (!f.exists()) None
    else StatisticalModelIO.readStatisticalTriangleMeshModel3D(f).toOption
  }
}
