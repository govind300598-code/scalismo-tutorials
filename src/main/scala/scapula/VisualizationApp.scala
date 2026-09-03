package scapula

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.PointDistributionModel
import scalismo.ui.api.ScalismoUI

import java.io.File

/**
 * Scalismo UI visualisation application.
 *
 * Shows (all optional, presence detected from output folders):
 *   Group "Original"    – a subset of original non-registered scapulae
 *   Group "Rigid"       – landmark-rigid registered scapulae (SSM1 pass)
 *   Group "NonRigid"    – GPMM registered scapulae (SSM1 pass)
 *   Group "Means"       – Mean1, Mean2, Mean3, Mean4
 *   Group "SSM1_Modes"  – modes 1-3 at -3σ / mean / +3σ
 *   …                   – repeated for SSM2, SSM3, SSM4
 *
 * Run with: sbt "runMain scapula.VisualizationApp"
 */
object VisualizationApp {

  private def loadOpt(f: File): Option[TriangleMesh[_3D]] =
    if (f.exists()) Some(ScapulaData.loadMesh(f)) else None

  private def stlsIn(d: File, limit: Int = 5): IndexedSeq[File] =
    if (!d.exists()) IndexedSeq.empty
    else Option(d.listFiles()).getOrElse(Array.empty).filter(_.getName.endsWith(".stl"))
      .sortBy(_.getName).take(limit).toIndexedSeq

  private def loadModel(f: File): Option[PointDistributionModel[_3D, TriangleMesh]] =
    if (!f.exists()) None
    else scalismo.io.StatisticalModelIO.readStatisticalMeshModel(f).toOption

  def main(args: Array[String]): Unit = {
    scalismo.initialize()

    val dataDir    = Config.dataDir
    val outDir     = Config.outDir
    val resultsDir = new File(outDir, "results")

    val ui = ScalismoUI("ScapulaAtlasRefinement – Iterative SSM Viewer")

    // ── Original meshes ──────────────────────────────────────────────────────
    val origGroup  = ui.createGroup("Original")
    val origSpecimens = ScapulaData.specimens(dataDir)
    stlsIn(dataDir, limit = 6).foreach { f =>
      val mesh = ScapulaData.loadMesh(f)
      ui.show(origGroup, mesh, f.getName.stripSuffix(".stl"))
    }
    println(s"Loaded ${math.min(6, origSpecimens.length)} original meshes into 'Original' group.")

    // ── SSM iterations ───────────────────────────────────────────────────────
    val means = scala.collection.mutable.ArrayBuffer.empty[(String, TriangleMesh[_3D])]

    for (iter <- 1 to 4) {
      val label     = s"SSM$iter"
      val iterDir   = new File(resultsDir, label)
      val rigidDir  = new File(iterDir, "rigid_registered")
      val nrDir     = new File(iterDir, "nonrigid_registered")
      val modelFile = new File(iterDir, s"model/$label.h5")
      val meanFile  = new File(iterDir, s"mean/${label}_mean.stl")

      if (!iterDir.exists()) {
        println(s"$label output not found – run the pipeline first.")
      } else {
        // Rigid-registered (first 4)
        if (rigidDir.exists()) {
          val rigGroup = ui.createGroup(s"${label}_Rigid")
          stlsIn(rigidDir, 4).foreach { f =>
            ui.show(rigGroup, ScapulaData.loadMesh(f), f.getName.stripSuffix(".stl"))
          }
        }

        // Non-rigid registered (first 4)
        if (nrDir.exists()) {
          val nrGroup = ui.createGroup(s"${label}_NonRigid")
          stlsIn(nrDir, 4).foreach { f =>
            ui.show(nrGroup, ScapulaData.loadMesh(f), f.getName.stripSuffix(".stl"))
          }
        }

        // Population mean
        loadOpt(meanFile).foreach { mean =>
          means += (s"Mean$iter" -> mean)
        }

        // SSM model + mode deformations
        loadModel(modelFile).foreach { ssm =>
          val modesGroup = ui.createGroup(s"${label}_Modes")
          val nModes     = math.min(3, ssm.rank)
          println(s"$label rank=${ssm.rank}, showing modes 1-$nModes (±3σ)")

          for (modeIdx <- 0 until nModes) {
            val stdDev = math.sqrt(ssm.variance(modeIdx))
            for (alpha <- Seq(-3.0, 0.0, 3.0)) {
              val coeffs = DenseVector.zeros[Double](ssm.rank)
              coeffs(modeIdx) = alpha * stdDev
              val shape = ssm.instance(coeffs)
              val tag   =
                if (alpha < 0) s"m${(-alpha).toInt}sd"
                else if (alpha == 0.0) "mean"
                else s"p${alpha.toInt}sd"
              ui.show(modesGroup, shape, s"${label}_mode${modeIdx + 1}_$tag")
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
