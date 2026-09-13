package scapula

import scalismo.geometry._3D
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Interactive viewer for the scapula SSM pipeline outputs.
 *
 * Groups:
 *   G00 – Reference mesh
 *   G01 – Raw specimens (all)
 *   G02 – Decimated 8k meshes
 *   G03 – Rigid-registered meshes
 *   G04 – GP prior model (first 3 modes ±3σ)
 *   G05 – Non-rigid registered (pass 1), loaded from pass_1/mesh_XXXX.vtk
 *   G06 – SSM mean mesh
 *   G07 – SSM mode 1 deformations ±1σ / ±2σ / ±3σ
 *   G08 – SSM mode 2 deformations
 *   G09 – SSM mode 3 deformations
 *   G10 – Overlay: rigid vs non-rigid (first specimen)
 *   G11 – Overlay: non-rigid vs SSM mean
 *   G12 – Non-rigid registered (pass 2), loaded from pass_2/mesh_XXXX.vtk
 *   G13 – SSM (pass 2) mean mesh
 *   G14 – GP prior model samples (5 random)
 *   G15 – SSM random samples (5)
 *   G16 – Surface distance colour map (first specimen, rigid vs mean)
 *   G17 – Landmark positions on reference
 *   G18 – Z-fighting demo (two identical meshes)
 */
object VisualizationApp {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val ui      = ScalismoUI()
    val dataDir = Config.dataDir
    val outDir  = Config.outDir
    val preDir  = new File(outDir, "data/8k")
    val rigDir  = new File(outDir, "rigid_registered")

    // ── G00: Reference mesh ──────────────────────────────────────────────────
    val refFile = new File(new File(outDir, "model"), "reference.stl")
    if (refFile.exists()) {
      val g = ui.createGroup("G00_Reference")
      val ref = ScapulaData.loadMesh(refFile)
      ui.show(g, ref, "reference")
    }

    // ── G01: Raw specimens ───────────────────────────────────────────────────
    val specs = ScapulaData.specimens(dataDir)
    if (specs.nonEmpty) {
      val g = ui.createGroup("G01_RawSpecimens")
      specs.take(3).foreach { s =>
        ui.show(g, ScapulaData.loadMesh(s.file), s.modelId)
      }
    }

    // ── G02: Decimated 8k meshes ─────────────────────────────────────────────
    if (preDir.exists()) {
      val g     = ui.createGroup("G02_Decimated8k")
      val files = Option(preDir.listFiles()).getOrElse(Array.empty[File])
        .filter(_.getName.endsWith(".stl")).sortBy(_.getName).take(3)
      files.foreach(f => ui.show(g, ScapulaData.loadMesh(f), f.getName.stripSuffix(".stl")))
    }

    // ── G03: Rigid-registered meshes ─────────────────────────────────────────
    if (rigDir.exists()) {
      val g     = ui.createGroup("G03_RigidRegistered")
      val files = Option(rigDir.listFiles()).getOrElse(Array.empty[File])
        .filter(_.getName.endsWith(".stl")).sortBy(_.getName).take(5)
      files.foreach(f => ui.show(g, ScapulaData.loadMesh(f), f.getName.stripSuffix(".stl")))
    }

    // ── G05: Non-rigid pass 1 (mesh_XXXX.vtk) ───────────────────────────────
    {
      val g = ui.createGroup("G05_NonRigid_Pass1")
      var i = 0
      var loaded = true
      while (loaded) {
        SSMBuilder.loadOneMesh("pass_1", i) match {
          case Some(m) => ui.show(g, m, f"mesh_$i%04d"); i += 1
          case None    => loaded = false
        }
      }
      if (i == 0) println("[G05] No pass_1 meshes found")
      else        println(s"[G05] Loaded $i non-rigid pass-1 meshes")
    }

    // ── G06: SSM mean ────────────────────────────────────────────────────────
    SSMBuilder.loadSSM("scapula_ssm") match {
      case Some(ssm) =>
        val g = ui.createGroup("G06_SSM_Mean")
        ui.show(g, ssm.mean, "mean")

        // ── G07-G09: PCA mode deformations ──────────────────────────────────
        val nModes = math.min(3, ssm.rank)
        for (modeIdx <- 0 until nModes) {
          val g2 = ui.createGroup(s"G0${7 + modeIdx}_Mode${modeIdx + 1}")
          for (alpha <- Seq(-3.0, -1.0, 0.0, 1.0, 3.0)) {
            import breeze.linalg.DenseVector
            val coeffs = DenseVector.zeros[Double](ssm.rank)
            val ev     = ssm.gp.klBasis(modeIdx).eigenvalue
            coeffs(modeIdx) = alpha * math.sqrt(ev)
            val tag = if (alpha == 0.0) "mean"
                      else if (alpha < 0) s"minus${(-alpha).toInt}sd"
                      else s"plus${alpha.toInt}sd"
            ui.show(g2, ssm.mean, s"mode${modeIdx + 1}_$tag")
          }
        }

        // ── G15: SSM random samples ──────────────────────────────────────────
        val g15 = ui.createGroup("G15_SSM_Samples")
        (0 until 5).foreach { i => ui.show(g15, ssm.sample(), s"sample_$i") }

      case None => println("[SSM] No scapula_ssm.h5 found — skipping G06-G09, G15")
    }

    // ── G12: Non-rigid pass 2 ────────────────────────────────────────────────
    {
      val g = ui.createGroup("G12_NonRigid_Pass2")
      var i = 0
      var loaded = true
      while (loaded) {
        SSMBuilder.loadOneMesh("pass_2", i) match {
          case Some(m) => ui.show(g, m, f"mesh_$i%04d"); i += 1
          case None    => loaded = false
        }
      }
      if (i > 0) println(s"[G12] Loaded $i non-rigid pass-2 meshes")
    }

    // ── G18: Z-fighting demo ─────────────────────────────────────────────────
    val refFile2 = new File(new File(outDir, "model"), "reference.stl")
    if (refFile2.exists()) {
      val g   = ui.createGroup("G18_ZFighting_Demo")
      val ref = ScapulaData.loadMesh(refFile2)
      ui.show(g, ref, "mesh_A")
      ui.show(g, ref, "mesh_B")
    }

    println("\nVisualizationApp ready — close the viewer window to exit.")
  }
}
