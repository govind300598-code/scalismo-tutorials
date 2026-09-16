package scapula

import scalismo.common.DiscreteField
import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.numerics.PivotedCholesky
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI

import java.io.File
import scala.io.Source
import scala.util.Using

/**
 * VISUAL results check: builds the actual statistical shape model -- via scalismo's own PCA
 * (PointDistributionModel.createUsingPCA), not the hand-rolled breeze PCA in SSMPCACore used for Stage 3's
 * numeric validation curves -- from Stage 2's registered meshes, and opens it in scalismo-ui with interactive
 * mode-of-variation sliders. This is the pipeline's actual result made tangible: drag a slider and watch the mean
 * scapula deform along a real direction of shape variation learned from your 24 specimens.
 *
 * Run after Stage2GPNonRigidRegistration.
 */
object ViewSSMResult {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val (reference, registered) = Stage3SSMModelValidation.loadRegistered(Config.outDir)
    println(s"Loaded reference (${reference.pointSet.numberOfPoints} vertices) and ${registered.length} " +
      "registered meshes.")

    // Each registered mesh is the reference warped into correspondence with one specimen (see Stage2's header
    // comment) -- so the per-point displacement from the reference IS the deformation field PCA needs, with no
    // further correspondence-finding required.
    val referencePoints = reference.pointSet.points.toIndexedSeq
    val dataItems = registered.map { case (_, mesh) =>
      val meshPoints = mesh.pointSet.points.toIndexedSeq
      val displacements = referencePoints.indices.map(i => meshPoints(i) - referencePoints(i))
      new DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]](reference, displacements)
    }
    val dc = DataCollection(dataItems)

    val maxModes = math.min(Config.maxValidationModes, registered.length - 1)
    val model = PointDistributionModel.createUsingPCA(dc, PivotedCholesky.NumberOfEigenfunctions(maxModes))
    println(s"Built a PointDistributionModel with rank ${model.rank} from ${registered.length} specimens " +
      s"(capped at SCAPULA_MAX_VALIDATION_MODES=${Config.maxValidationModes}; raise it for more modes).")

    val accCsv = new File(Config.outDir, "registration_accuracy.csv")
    if (accCsv.exists()) {
      val lines = Using.resource(Source.fromFile(accCsv))(_.getLines().toIndexedSeq)
      val means = lines.tail.filter(_.trim.nonEmpty).map(_.split(",")(1).toDouble)
      println(f"Registration accuracy (${accCsv.getName}): mean=${means.sum / means.length}%.3f mm across " +
        f"${means.length} specimens (best=${means.min}%.3f mm, worst=${means.max}%.3f mm).")
    }

    if (Config.showUi) {
      val ui = ScalismoUI()
      val group = ui.createGroup("SSM result")
      ui.show(group, model, "scapula SSM")
      println(s"\nScalismo-UI window opened with the fitted SSM (rank ${model.rank}) in the 'SSM result' group. " +
        "Select 'scapula SSM' in the scene tree to get mode-of-variation sliders -- drag one and watch the mean " +
        "shape deform along a real direction of variation learned from your 24 scapulae. There is also a random " +
        "sample option there, to draw a new synthetic (but plausible) scapula from the model. Close the window " +
        "when done.")
    } else {
      println("SCAPULA_UI=false -- skipping the interactive viewer.")
    }
  }
}
