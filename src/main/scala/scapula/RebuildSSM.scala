package scapula

import scalismo.geometry._3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.utils.Random

import java.io.File

/**
 * Rebuilds ssm_final.h5 from the existing pass2/reg_*.stl registered meshes.
 * Use this when ssm_final.h5 is missing or corrupt.
 * Run with:  sbt "runMain scapula.RebuildSSM"
 */
object RebuildSSM {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir = Config.outDir
    require(outDir.exists(), s"Output directory not found: ${outDir.getAbsolutePath}")

    val pass2Dir = new File(outDir, "pass2")
    require(pass2Dir.exists(), s"pass2/ directory not found — run FullPipeline first")

    val meanFile = new File(outDir, "mean_pass2.stl")
    require(meanFile.exists(), s"mean_pass2.stl not found — run FullPipeline first")

    println(s"Reading reference from: ${meanFile.getName}")
    val reference = MeshIO.readMesh(meanFile)
      .getOrElse(throw new RuntimeException("Cannot read mean_pass2.stl"))
    println(s"  ${reference.pointSet.numberOfPoints} vertices")

    val regFiles = Option(pass2Dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(f => f.getName.startsWith("reg_") && f.getName.endsWith(".stl"))
      .sortBy(_.getName)

    require(regFiles.nonEmpty, s"No reg_*.stl files found in ${pass2Dir.getAbsolutePath}")
    println(s"Reading ${regFiles.length} registered meshes from pass2/ ...")

    val meshes: IndexedSeq[TriangleMesh[_3D]] = regFiles.map { f =>
      MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Cannot read ${f.getName}"))
    }.toIndexedSeq

    println("Building SSM via PCA ...")
    val dc = DataCollection.fromTriangleMesh3DSequence(reference, meshes)
    val model = PointDistributionModel.createUsingPCA[_3D, TriangleMesh](dc)

    println(s"  ${model.rank} modes  |  ${model.reference.pointSet.numberOfPoints} vertices")

    val modelFile = new File(outDir, "ssm_final.h5")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, modelFile)
      .getOrElse(throw new RuntimeException("Failed to write ssm_final.h5"))

    println(s"Saved: ${modelFile.getAbsolutePath}")
    println("Now run:  sbt \"runMain scapula.ViewSSM\"")
  }
}
