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
 * Splits the registered "final_<modelId>_fit.stl" meshes (written flat into Config.outDir, no reference.vtk /
 * registered/ subfolder) into two cohorts by specimen id prefix and builds one PCA model per cohort:
 *   - "hill_sachs"                                  -> scapula_pca_model_hillsachs.json
 *   - "paired_scapula" and "paired_shoulder" pooled  -> scapula_pca_model_paired.json
 * The 3 known registration outliers are dropped first. No validation sweep is run -- just the two PCA models.
 *
 * Every "_fit.stl" already shares the same point correspondence (all are the same reference topology warped onto
 * a different target), so no separate reference mesh file is needed -- the first one loaded is used as the
 * PCA domain. "_target.stl" files (not in correspondence) are ignored.
 */
object Stage3PCAModelByGroup {

  private val outlierIds = Set("hill_sachs_032_M_50_R", "hill_sachs_033_F_53_R", "hill_sachs_042_M_29_L")
  private val fitSuffix = "_fit.stl"
  private val finalPrefix = "final_"

  private def groupOf(modelId: String): Option[String] =
    if (modelId.startsWith("hill_sachs")) Some("hillsachs")
    else if (modelId.startsWith("paired_scapula") || modelId.startsWith("paired_shoulder")) Some("paired")
    else None

  private def loadFitMeshes(outDir: File): IndexedSeq[(String, TriangleMesh[_3D])] = {
    val files = Option(outDir.listFiles()).getOrElse(Array.empty[File])
      .filter(f => f.getName.startsWith(finalPrefix) && f.getName.endsWith(fitSuffix))
      .sortBy(_.getName)
    require(files.nonEmpty, s"No '$finalPrefix*$fitSuffix' files found directly in ${outDir.getPath}")

    files.toIndexedSeq.map { f =>
      val modelId = f.getName.stripPrefix(finalPrefix).stripSuffix(fitSuffix)
      val mesh = MeshIO.readMesh(f).getOrElse(throw new RuntimeException(s"Could not read mesh ${f.getPath}"))
      modelId -> mesh
    }
  }

  private def buildModel(
      name: String,
      reference: TriangleMesh[_3D],
      members: IndexedSeq[(String, TriangleMesh[_3D])],
      outDir: File
  ): Unit = {
    require(members.nonEmpty, s"Group '$name' has no specimens -- check groupOf's prefixes against your file names.")

    val referencePoints = reference.pointSet.points.toIndexedSeq
    val dataItems = members.map { case (modelId, mesh) =>
      require(mesh.pointSet.numberOfPoints == referencePoints.length,
        s"$modelId has ${mesh.pointSet.numberOfPoints} points, reference has ${referencePoints.length} -- " +
          "these fit meshes are not all in the same correspondence.")
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
    val fits = loadFitMeshes(outDir)
    println(s"Loaded ${fits.length} registered '_fit.stl' meshes from ${outDir.getPath}")

    val referenceFile = new File(outDir, "reference_mean_shape.stl")
    val reference =
      if (referenceFile.exists()) {
        val mesh = MeshIO.readMesh(referenceFile).getOrElse(throw new RuntimeException(s"Could not read ${referenceFile.getPath}"))
        println(s"Using reference_mean_shape.stl as the PCA reference topology (${mesh.pointSet.numberOfPoints} points).")
        mesh
      } else {
        println(s"reference_mean_shape.stl not found in ${outDir.getPath} -- falling back to '${fits.head._1}' as the " +
          "reference topology (any one works, since all fit meshes already share correspondence).")
        fits.head._2
      }

    val clean = fits.filterNot { case (modelId, _) => outlierIds.contains(modelId) }
    println(s"Dropped ${fits.length - clean.length} outlier(s): ${outlierIds.mkString(", ")}")

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
