package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random as ScalismoRandom

import java.awt.Color
import java.io.File

/**
 * JOR-2024-style transparent overlay visualization of the first three PCA modes.
 *
 * Loads ALL final_*_fit.stl meshes from Config.outDir (the same shapes Stage3PCAModel uses),
 * builds the PCA in-memory, then for each of PC1–PC5 creates a SINGLE ScalismoUI group
 * containing all three states stacked in one view:
 *
 *   - −3σ  coral red    (opacity 0.88)
 *   - Mean  charcoal     (opacity 0.88)
 *   - +3σ  steel blue   (opacity 0.88)
 *
 * This matches the published convention from Silvestros et al., J. Orthop. Res. 2024.
 *
 * Published anatomical interpretations (cross-checked from literature):
 *   PC1 – overall size (Halloran JSES 2018, ~72% in N=110)
 *   PC2 – coracoacromial arch rotation / superior-inferior scaling
 *   PC3 – acromion shape (Bigliani type axis)
 *   PC4 – glenoid version / scapular body curvature
 *   PC5 – acromion spine WIDTH and elongation; +3σ = wider/longer spine,
 *          −3σ = narrower/shorter spine with less-curved, flatter acromion
 *          (Scapula SSM construction, ResearchGate 2014; Springer 2025)
 *
 * PC1 and PC5 start visible by default; PC2/3/4 start hidden — unhide in
 * the ScalismoUI tree to compare modes without visual clutter.
 *
 * Requires: Stage2ReferenceRefinement to have completed (final_*_fit.stl + reference_mean_shape.stl).
 * Run: sbt "runMain scapula.ViewModeShapes"
 */
object ViewModeShapes {

  private val numModes   = 5   // PC1 – PC5
  private val sdExtreme  = 3.0

  // Colours matching JOR 2024 exactly
  private val colorNeg  = new Color(204,  60,  60) // coral red   – −3σ
  private val colorMean = new Color( 45,  45,  45) // charcoal    – mean
  private val colorPos  = new Color( 90, 150, 215) // steel blue  – +3σ
  private val opacity   = 0.88f  // raised for maximum visibility

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: ScalismoRandom = ScalismoRandom(Config.seed)

    // ---- auto-detect the output directory with the MOST registered fits -----
    // Looks at Config.outDir and every sibling directory whose name starts with
    // "scapula_ssm_out", then picks the one with the most final_*_fit.stl files.
    // Override by setting SCAPULA_OUT_DIR explicitly if needed.
    val base = Config.outDir.getParentFile
    val candidates: Array[File] = {
      val siblings = Option(base.listFiles(f => f.isDirectory &&
        f.getName.toLowerCase.startsWith("scapula_ssm_out")))
        .getOrElse(Array.empty[File])
      (siblings :+ Config.outDir).distinct
    }

    println("Scanning candidate output directories:")
    val ranked = candidates.map { d =>
      val n = Option(d.listFiles((_, name) => name.startsWith("final_") && name.endsWith("_fit.stl")))
        .getOrElse(Array.empty).length
      println(f"  ${n}%3d fit(s)  ${d.getAbsolutePath}")
      (n, d)
    }.sortBy(-_._1)

    val (nFits, dir) = ranked.headOption.getOrElse(
      throw new RuntimeException(s"No scapula_ssm_out* directories found under ${base.getAbsolutePath}"))

    require(nFits >= 3,
      s"Best candidate ${dir.getAbsolutePath} has only $nFits fit(s) -- need at least 3.")
    println(s"\nUsing: ${dir.getAbsolutePath}  ($nFits registered fits)")

    // ---- 1. load reference --------------------------------------------------
    val referenceFile = new File(dir, "reference_mean_shape.stl")
    require(referenceFile.exists(),
      s"$referenceFile not found -- run Stage2ReferenceRefinement first.")
    val reference: TriangleMesh[_3D] = MeshIO.readMesh(referenceFile).get

    // ---- 2. load all registered fits ----------------------------------------
    val fitFiles = Option(dir.listFiles((_, name) => name.startsWith("final_") && name.endsWith("_fit.stl")))
      .getOrElse(Array.empty[File])
      .sortBy(_.getName)

    require(fitFiles.length >= 3,
      s"Need at least 3 registered fits to build a PCA; found ${fitFiles.length} in ${dir.getAbsolutePath}.")

    val subjectIds    = fitFiles.map(_.getName.stripPrefix("final_").stripSuffix("_fit.stl"))
    val fittedMeshes  = fitFiles.map(f => MeshIO.readMesh(f).get)

    println(s"Loaded ${fittedMeshes.length} registered fits from ${dir.getAbsolutePath}:")
    subjectIds.foreach(id => println(s"  $id"))

    // ---- 3. build PCA in-memory (same logic as Stage3PCAModel) ---------------
    val dc       = DataCollection.fromTriangleMesh3DSequence(reference, fittedMeshes.toIndexedSeq)
    val pcaModel = PointDistributionModel.createUsingPCA(dc)
    println(s"\nPCA model: rank=${pcaModel.rank} (max = N-1 = ${fittedMeshes.length - 1})")

    val variance  = pcaModel.gp.variance.toArray
    val totalVar  = variance.sum
    println("Variance per mode:")
    variance.zipWithIndex.take(math.min(numModes, pcaModel.rank)).foreach { case (v, k) =>
      val pct     = v / totalVar * 100.0
      val cumPct  = variance.take(k + 1).sum / totalVar * 100.0
      println(f"  PC${k+1}: ${pct}%.1f%%  (cumulative ${cumPct}%.1f%%)")
    }

    // ---- 4. open ScalismoUI and build overlay groups -------------------------
    val ui = ScalismoUI()

    val meanMesh = pcaModel.mean

    // Always-visible standalone mean (clean reference, no overlapping extremes)
    val meanGroup = ui.createGroup("Mean shape")
    val mv = ui.show(meanGroup, meanMesh, "mean")
    mv.color   = colorMean
    mv.opacity = 1.0f

    // One overlay group per mode
    (0 until math.min(numModes, pcaModel.rank)).foreach { k =>
      val pct    = variance(k) / totalVar * 100.0
      val cumPct = variance.take(k + 1).sum / totalVar * 100.0
      val name   = f"PC${k+1} overlay (−${sdExtreme.toInt}σ / Mean / +${sdExtreme.toInt}σ)  " +
        f"${pct}%.1f%% var  cumul ${cumPct}%.1f%%"

      val group = ui.createGroup(name)

      def instance(sigma: Double): TriangleMesh[_3D] = {
        val c = DenseVector.zeros[Double](pcaModel.rank)
        c(k) = sigma
        pcaModel.instance(c)
      }

      val negView  = ui.show(group, instance(-sdExtreme), s"PC${k+1}_neg${sdExtreme.toInt}sd")
      val mView    = ui.show(group, meanMesh,              s"PC${k+1}_mean")
      val posView  = ui.show(group, instance( sdExtreme), s"PC${k+1}_pos${sdExtreme.toInt}sd")

      negView.color  = colorNeg;  negView.opacity  = opacity
      mView.color    = colorMean; mView.opacity     = opacity
      posView.color  = colorPos;  posView.opacity   = opacity

      // PC1 and PC5 start visible; PC2/3/4 start hidden to keep the scene clean
      val visibleByDefault = k == 0 || k == 4
      if (!visibleByDefault) {
        negView.opacity  = 0.0f
        mView.opacity    = 0.0f
        posView.opacity  = 0.0f
        println(s"Group '$name'  [hidden — toggle in ScalismoUI tree]")
      } else {
        println(s"Group '$name'  [VISIBLE by default]")
      }
    }

    println(
      s"""
         |Legend (JOR 2024 colour convention):
         |  Coral red   = −${sdExtreme.toInt}σ
         |  Charcoal    = Mean
         |  Steel blue  = +${sdExtreme.toInt}σ
         |  All at ${(opacity * 100).toInt}% opacity — overlap regions show blended colour
         |
         |Dataset: ${dir.getAbsolutePath}
         |Subjects: ${fittedMeshes.length}
         |PCA rank: ${pcaModel.rank}
         |""".stripMargin)
  }
}
