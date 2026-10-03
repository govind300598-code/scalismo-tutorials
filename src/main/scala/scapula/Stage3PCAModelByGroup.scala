package scapula

import scalismo.common.DiscreteField
import scalismo.geometry.*
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.PivotedCholesky
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection

import java.io.File

/**
 * Splits Stage 2's registered meshes into two cohorts by specimen id prefix and builds one PCA model per cohort:
 *   - "hill_sachs"                                  -> scapula_pca_model_hillsachs.json
 *   - "paired_scapula" and "paired_shoulder" pooled  -> scapula_pca_model_paired.json
 * The 3 known registration outliers are dropped first. No validation sweep is run -- just the two PCA models.
 * Self-contained: reads reference.vtk/registered/*.vtk directly instead of going through a Stage3 validation
 * helper, so it has no dependency beyond Config.outDir and the scalismo library itself.
 *
 * Run after Stage2GPNonRigidRegistration.
 */
object Stage3PCAModelByGroup {

  private val outlierIds = Set("hill_sachs_032_M_50_R", "hill_sachs_033_F_53_R", "hill_sachs_042_M_29_L")

  private def groupOf(modelId: String): Option[String] =
    if (modelId.startsWith("hill_sachs")) Some("hillsachs")
    else if (modelId.startsWith("paired_scapula") || modelId.startsWith("paired_shoulder")) Some("paired")
    else None

  private def loadMesh(f: File): TriangleMesh[_3D] =
    MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Could not read mesh ${f.getPath}"))

  private def loadRegistered(outDir: File): (TriangleMesh[_3D], IndexedSeq[(String, TriangleMesh[_3D])]) = {
    val referenceFile = new File(outDir, "reference.vtk")
    require(referenceFile.exists(), s"$referenceFile not found -- run Stage2GPNonRigidRegistration first.")
    val reference = loadMesh(referenceFile)

    val registeredDir = new File(outDir, "registered")
    val files = Option(registeredDir.listFiles()).getOrElse(Array.empty[File])
      .filter(_.getName.toLowerCase.endsWith(".vtk")).sortBy(_.getName)
    require(files.nonEmpty, s"No registered meshes found in $registeredDir")

    val registered = files.toIndexedSeq.map { f => f.getName.stripSuffix(".vtk") -> loadMesh(f) }
    (reference, registered)
  }

  private def buildModel(
      name: String,
      reference: TriangleMesh[_3D],
      members: IndexedSeq[(String, TriangleMesh[_3D])],
      outDir: File
  ): Unit = {
    require(members.nonEmpty, s"Group '$name' has no specimens -- check groupOf's prefixes against your file names.")

    val referencePoints = reference.pointSet.points.toIndexedSeq
    val dataItems = members.map { case (_, mesh) =>
      val meshPoints = mesh.pointSet.points.toIndexedSeq
      val displacements = referencePoints.indices.map(i => meshPoints(i) - referencePoints(i))
      new DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]](reference, displacements)
    }
    val dc = DataCollection(dataItems)

    val maxModes = members.length - 1
    val model = PointDistributionModel.createUsingPCA(dc, PivotedCholesky.NumberOfEigenfunctions(maxModes))

    val variance = model.gp.variance
    val total = variance.toArray.sum
    val topShares = variance.toArray.take(3).map(v => f"${v / total * 100}%.1f%%").mkString(", ")
    println(f"[$name] N=${members.length}%3d  rank=${model.rank}%2d  PC1-3 variance share: $topShares")
    println(s"         specimens: ${members.map(_._1).sorted.mkString(", ")}")

    val outFile = new File(outDir, s"scapula_pca_model_$name.json")
    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, outFile).get
    println(s"         wrote $outFile")
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val outDir = Config.outDir
    val (reference, registered) = loadRegistered(outDir)
    println(s"Loaded reference (${reference.pointSet.numberOfPoints} vertices) and ${registered.length} registered meshes.")

    val clean = registered.filterNot { case (modelId, _) => outlierIds.contains(modelId) }
    println(s"Dropped ${registered.length - clean.length} outlier(s): ${outlierIds.mkString(", ")}")

    val byGroup = clean.groupBy { case (modelId, _) => groupOf(modelId) }
    byGroup.getOrElse(None, IndexedSeq.empty).foreach { case (modelId, _) =>
      println(s"!! WARNING: '$modelId' matched neither the hill_sachs nor paired_* prefix -- not included in either model.")
    }

    val hillSachs = byGroup.getOrElse(Some("hillsachs"), IndexedSeq.empty)
    val paired = byGroup.getOrElse(Some("paired"), IndexedSeq.empty)
    require(hillSachs.length + paired.length == clean.length,
      s"${clean.length - hillSachs.length - paired.length} specimen(s) unclassified -- see warnings above.")

    buildModel("hillsachs", reference, hillSachs, outDir)
    buildModel("paired", reference, paired, outDir)

    println("\nTwo separate PCA models written -- no validation sweep run here.")
  }
}
