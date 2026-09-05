package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Scalismo UI visualisation application.
 *
 * Groups shown (all optional, detected from output folders):
 *
 *   Original_AllLeft    – raw original scapulae, R ones mirrored to L frame
 *   SSMn_Rigid          – landmark + ICP aligned (before non-rigid step)
 *   SSMn_NonRigid       – GPMM registered (dense correspondence, ready for PCA)
 *   SSMn_Modes          – static mode meshes: mode 1-3 at −3σ / mean / +3σ
 *   SSMn_Interactive    – live SSM slider: drag mode sliders to explore shape space
 *   Means               – Mean1, Mean2, Mean3, Mean4 overlaid
 *
 * HOW TO USE THE UI
 * ─────────────────
 * Scene panel (left):  click the eye icon to show/hide any group or mesh.
 *                      Expand a group to reach individual shapes.
 * SSMn_Interactive:    select it → a "Shape Model" panel appears on the right.
 *                      Drag the slider for "Mode 1 (xx.x%)" left/right to see
 *                      the main axis of shape variation.  Mode 2, Mode 3, … do
 *                      the same for their axes.  Click "Reset" to return to mean.
 * SSMn_Modes:          shows fixed meshes at −3σ, mean, +3σ for modes 1-3 so
 *                      you can turn them on/off and compare side by side.
 * Original_AllLeft:    the raw (non-registered) scapulae – compare these with
 *                      SSMn_Rigid and SSMn_NonRigid to see alignment quality.
 *
 * Run with: sbt "runMain scapula.VisualizationApp"
 */
object VisualizationApp {

  private def loadOpt(f: File): Option[TriangleMesh[_3D]] =
    if (f.exists()) Some(ScapulaData.loadMesh(f)) else None

  private def stlsIn(d: File, limit: Int = 9999, prefix: String = ""): IndexedSeq[File] =
    if (!d.exists()) IndexedSeq.empty
    else Option(d.listFiles()).getOrElse(Array.empty[File])
      .filter(f => f.getName.endsWith(".stl") && f.getName.startsWith(prefix))
      .sortBy(_.getName).take(limit).toIndexedSeq

  private def loadModel(f: File): Option[StatisticalMeshModel] =
    if (!f.exists()) None
    else scalismo.io.StatisticalModelIO.readStatisticalMeshModel(f).toOption

  // Build SSM on-the-fly from registered meshes when no .h5 is available
  private def buildSSM(meshes: IndexedSeq[TriangleMesh[_3D]]): Option[StatisticalMeshModel] = {
    if (meshes.length < 3) return None
    val dc = scalismo.statisticalmodel.dataset.DataCollection
      .fromTriangleMesh3DSequence(meshes.head, meshes)
    StatisticalMeshModel.createUsingPCA(dc).toOption
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dataDir    = Config.dataDir
    val outDir     = Config.outDir
    val resultsDir = new File(outDir, "results")

    // Detect which folder layout the user has:
    //   New: outDir/results/SSM{n}/nonrigid_registered/*.stl
    //   Old: outDir/pass{n}/reg_*.stl  (produced by RebuildSSM)
    val useNewLayout = new File(resultsDir, "SSM1").isDirectory
    if (useNewLayout) println("Layout: new  (results/SSM{n}/)")
    else              println("Layout: old  (pass{n}/rigid_*.stl + reg_*.stl)")

    val ui = ScalismoUI("ScapulaAtlasRefinement – Iterative SSM Viewer")

    // ── Original meshes (all shown in LEFT-scapula frame) ────────────────────
    val origGroup     = ui.createGroup("Original_AllLeft")
    val origSpecimens = ScapulaData.specimens(dataDir)
    println(s"\nL/R classification for first specimens:")
    origSpecimens.take(12).foreach { s =>
      println(s"  ${s.modelId}  ->  ${if (s.isRight) "RIGHT (will mirror)" else "LEFT"}")
    }
    stlsIn(dataDir, limit = 6).foreach { f =>
      val id   = f.getName.stripSuffix(".stl")
      val spec = origSpecimens.find(_.modelId == id).get
      val raw  = ScapulaData.loadMesh(f)
      val mesh = if (spec.isRight) ScapulaData.mirrorMesh(raw) else raw
      val tag  = if (spec.isRight) s"${id}_MIRRORED" else id
      ui.show(origGroup, mesh, tag)
    }
    println(s"Loaded ${math.min(6, origSpecimens.length)} original meshes into 'Original_AllLeft' group (R ones mirrored).")

    // ── SSM iterations ───────────────────────────────────────────────────────
    val means = scala.collection.mutable.ArrayBuffer.empty[(String, TriangleMesh[_3D])]

    for (iter <- 1 to 4) {
      val label = s"SSM$iter"

      // Resolve directories for whichever layout is present
      val (rigidDir, nrDir, modelFile, meanFile) = if (useNewLayout) {
        val iterDir = new File(resultsDir, label)
        (new File(iterDir, "rigid_registered"),
         new File(iterDir, "nonrigid_registered"),
         new File(iterDir, s"model/$label.h5"),
         new File(iterDir, s"mean/${label}_mean.stl"))
      } else {
        val passDir = new File(outDir, s"pass$iter")
        (passDir,           // rigid_*.stl live here in old layout
         passDir,           // reg_*.stl live here in old layout
         new File(""),      // no .h5 in old layout
         new File(outDir, s"mean_pass$iter.stl"))
      }

      val (rigidPrefix, nrPrefix) =
        if (useNewLayout) ("", "")
        else              ("rigid_", "reg_")

      val rigidFiles = stlsIn(rigidDir, limit = 6, prefix = rigidPrefix)
      val nrFiles    = stlsIn(nrDir,    limit = 6, prefix = nrPrefix)

      if (rigidFiles.isEmpty && nrFiles.isEmpty) {
        println(s"$label output not found – skipping.")
      } else {
        // Rigid-registered
        if (rigidFiles.nonEmpty) {
          val rigGroup = ui.createGroup(s"${label}_Rigid")
          rigidFiles.foreach { f =>
            val name = f.getName.stripSuffix(".stl").stripPrefix("rigid_")
            ui.show(rigGroup, ScapulaData.loadMesh(f), name)
          }
          println(s"$label: ${rigidFiles.length} rigid-aligned meshes loaded")
        }

        // Non-rigid registered (ALL specimens)
        if (nrFiles.nonEmpty) {
          val nrGroup = ui.createGroup(s"${label}_NonRigid")
          nrFiles.foreach { f =>
            val name = f.getName.stripSuffix(".stl").stripPrefix("reg_")
            ui.show(nrGroup, ScapulaData.loadMesh(f), name)
          }
          println(s"$label: ${nrFiles.length} non-rigid registered meshes loaded")
        }

        // Population mean
        loadOpt(meanFile).foreach { mean => means += (s"Mean$iter" -> mean) }

        // SSM: load .h5 if available, else build from reg meshes
        val ssmOpt = loadModel(modelFile).orElse {
          val allNr = stlsIn(nrDir, prefix = nrPrefix)
          if (allNr.nonEmpty) buildSSM(allNr.map(ScapulaData.loadMesh))
          else None
        }

        ssmOpt.foreach { ssm =>
          val evs    = ssm.gp.klBasis.map(_.eigenvalue)
          val total  = evs.sum
          val nModes = math.min(3, ssm.rank)

          println(s"$label  rank=${ssm.rank}")
          evs.take(5).zipWithIndex.foreach { case (ev, i) =>
            println(f"  mode ${i+1}: ${ev / total * 100}%.1f%% variance  (std-dev=${math.sqrt(ev)}%.2f mm)")
          }

          // ── Interactive slider (drag to explore shape space live) ───────────
          val interactiveGroup = ui.createGroup(s"${label}_Interactive")
          ui.show(interactiveGroup, ssm, s"$label")
          println(s"  → ${label}_Interactive: select this in the scene panel, then use")
          println(s"    the 'Shape Model' sliders on the right to explore modes.")

          // ── Static ±3σ meshes for side-by-side comparison ──────────────────
          val modesGroup = ui.createGroup(s"${label}_Modes")
          for (modeIdx <- 0 until nModes) {
            val stdDev  = math.sqrt(evs(modeIdx))
            val modePct = evs(modeIdx) / total * 100
            for (alpha <- Seq(-3.0, 0.0, 3.0)) {
              val coeffs = DenseVector.zeros[Double](ssm.rank)
              coeffs(modeIdx) = alpha * stdDev
              val shape = ssm.instance(coeffs)
              val tag =
                if (alpha < 0) s"m${(-alpha).toInt}sd"
                else if (alpha == 0.0) "mean"
                else s"p${alpha.toInt}sd"
              ui.show(modesGroup, shape, f"${label}_mode${modeIdx+1}(${modePct}%.1f%%)_$tag")
            }
          }
        }
      }
    }

    // ── All means in one group ────────────────────────────────────────────────
    if (means.nonEmpty) {
      val meansGroup = ui.createGroup("Means")
      means.foreach { case (name, mesh) =>
        ui.show(meansGroup, mesh, name)
        println(s"  $name: ${mesh.pointSet.numberOfPoints} vertices")
      }
    }

    println("\nScalismo UI ready. Close the window to exit.")
  }
}
