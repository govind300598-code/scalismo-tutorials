package scapula

import breeze.linalg.DenseVector
import scalismo.common.DiscreteField
import scalismo.geometry.*
import scalismo.mesh.TriangleMesh
import scalismo.numerics.PivotedCholesky
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI

import java.awt.Color
import java.io.File

/**
 * JOR-2024-style transparent overlay visualization of Mode 1 – Mode 6.
 *
 * Loads the n=99 registered VTK meshes from:
 *   /home/g25upadh/Documents/100 plus scapula data/scapula_gp_registration_ssm_out/
 *     reference.vtk
 *     registered/<id>.vtk  (one per subject)
 *
 * Builds PCA as deformation fields (exactly as ViewSSMResult does in the
 * trusting-planck-6p05j8 branch), then for each mode creates ONE ScalismoUI
 * group with all three states overlaid:
 *   - −3σ  coral red   (opacity 0.88)
 *   - Mean  charcoal    (opacity 0.88)
 *   - +3σ  steel blue  (opacity 0.88)
 *
 * Matches Silvestros et al., J. Orthop. Res. 2024 figure convention exactly.
 * Mode 1 and Mode 6 visible by default; Mode 2–5 hidden (toggle in tree).
 *
 * Run: sbt "runMain scapula.ViewModeShapes"
 */
object ViewModeShapes {

  private val n99OutDir = new File(
    "/home/g25upadh/Documents/100 plus scapula data/scapula_gp_registration_ssm_out")

  private val numModes  = 6
  private val sdExtreme = 3.0

  private val colorNeg  = new Color(204,  60,  60)  // coral red   – −3σ
  private val colorMean = new Color( 45,  45,  45)  // charcoal    – mean
  private val colorPos  = new Color( 90, 150, 215)  // steel blue  – +3σ
  private val opacity   = 0.88f

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    // ---- 1. load reference and registered meshes (same as ViewSSMResult) ----
    val referenceFile = new File(n99OutDir, "reference.vtk")
    require(referenceFile.exists(),
      s"Not found: ${referenceFile.getAbsolutePath}\nRun Stage2GPNonRigidRegistration first.")

    val registeredDir = new File(n99OutDir, "registered")
    require(registeredDir.exists() && registeredDir.isDirectory,
      s"Not found: ${registeredDir.getAbsolutePath}\nRun Stage2GPNonRigidRegistration first.")

    val reference: TriangleMesh[_3D] = ScapulaData.loadMesh(referenceFile)
    println(s"Reference: ${referenceFile.getAbsolutePath}  (${reference.pointSet.numberOfPoints} vertices)")

    val regFiles = Option(registeredDir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.toLowerCase.endsWith(".vtk"))
      .sortBy(_.getName)
    require(regFiles.length >= 3,
      s"Need at least 3 registered meshes in ${registeredDir.getAbsolutePath}, found ${regFiles.length}.")

    val registered: IndexedSeq[(String, TriangleMesh[_3D])] =
      regFiles.toIndexedSeq.map(f => f.getName.stripSuffix(".vtk") -> ScapulaData.loadMesh(f))
    println(s"Loaded ${registered.length} registered meshes from ${registeredDir.getAbsolutePath}")

    // ---- 2. build PCA as deformation fields (same as ViewSSMResult) ---------
    val referencePoints = reference.pointSet.points.toIndexedSeq
    val dataItems = registered.map { case (_, mesh) =>
      val meshPoints    = mesh.pointSet.points.toIndexedSeq
      val displacements = referencePoints.indices.map(i => meshPoints(i) - referencePoints(i))
      new DiscreteField[_3D, TriangleMesh, EuclideanVector[_3D]](reference, displacements)
    }
    val dc    = DataCollection(dataItems)
    val model = PointDistributionModel.createUsingPCA(dc,
      PivotedCholesky.NumberOfEigenfunctions(math.min(numModes + 2, registered.length - 1)))

    println(s"PCA model: rank=${model.rank}  (built from ${registered.length} specimens)")

    val variance = model.gp.variance.toArray
    val totalVar = variance.sum
    println("Variance per mode:")
    variance.indices.take(numModes).foreach { k =>
      println(f"  Mode ${k+1}: ${variance(k)/totalVar*100}%.1f%%  " +
              f"(cumulative ${variance.take(k+1).sum/totalVar*100}%.1f%%)")
    }

    // ---- 3. JOR-2024-style overlay groups -----------------------------------
    val ui       = ScalismoUI()
    val meanMesh = model.mean

    val meanGroup = ui.createGroup("Mean shape")
    val mv = ui.show(meanGroup, meanMesh, "mean")
    mv.color   = colorMean
    mv.opacity = 1.0f

    (0 until math.min(numModes, model.rank)).foreach { k =>
      val pct    = variance(k) / totalVar * 100.0
      val cumPct = variance.take(k + 1).sum / totalVar * 100.0
      val name   = f"Mode ${k+1} (−${sdExtreme.toInt}σ / Mean / +${sdExtreme.toInt}σ)" +
                   f"  ${pct}%.1f%% var  cumul ${cumPct}%.1f%%"
      val group  = ui.createGroup(name)

      def instance(sigma: Double): TriangleMesh[_3D] = {
        val c = DenseVector.zeros[Double](model.rank)
        c(k) = sigma
        model.instance(c)
      }

      val negView = ui.show(group, instance(-sdExtreme), s"Mode${k+1}_neg${sdExtreme.toInt}sd")
      val mView   = ui.show(group, meanMesh,              s"Mode${k+1}_mean")
      val posView = ui.show(group, instance( sdExtreme), s"Mode${k+1}_pos${sdExtreme.toInt}sd")

      negView.color = colorNeg;  negView.opacity = opacity
      mView.color   = colorMean; mView.opacity   = opacity
      posView.color = colorPos;  posView.opacity = opacity

      val visibleByDefault = k == 0 || k == 5
      if (!visibleByDefault) {
        negView.opacity = 0.0f; mView.opacity = 0.0f; posView.opacity = 0.0f
        println(s"  Mode ${k+1}  [hidden — toggle in tree]")
      } else {
        println(s"  Mode ${k+1}  [VISIBLE]")
      }
    }

    println(
      s"""
         |Legend (JOR 2024):
         |  Coral red  = −${sdExtreme.toInt}σ
         |  Charcoal   = Mean
         |  Steel blue = +${sdExtreme.toInt}σ
         |  Opacity: ${(opacity * 100).toInt}%
         |""".stripMargin)
  }
}
