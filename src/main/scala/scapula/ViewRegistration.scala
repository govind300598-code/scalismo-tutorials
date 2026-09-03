package scapula

import scalismo.io.{MeshIO, StatismoIO}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

import java.io.File

/**
 * Registration quality viewer.
 *
 * Prints a per-specimen surface-distance table (mean, RMS, HD95, HD) comparing
 * each registered mesh in pass2/ to the original (rigidly pre-aligned) specimen.
 * When SCAPULA_UI=true (the default) it also opens a scalismo-ui window showing
 * the reference mean and the first few registered meshes.
 */
object ViewRegistration {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir  = if (args.nonEmpty)    new File(args(0)) else Config.outDir
    val dataDir = if (args.length > 1) new File(args(1)) else Config.dataDir
    val pass2Dir = new File(outDir, "pass2")

    require(pass2Dir.exists(),
      s"pass2/ not found: ${pass2Dir.getPath}\n  Set SCAPULA_OUT_DIR or pass the output directory as the first argument.")

    val registeredFiles = Option(pass2Dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.endsWith(".vtk"))
      .sortBy(_.getName)

    require(registeredFiles.nonEmpty,
      s"No registered meshes found in ${pass2Dir.getPath} — run RebuildSSM first")

    val csv = ScapulaData.csvFile(dataDir)
    val (landmarks, _, _) = ScapulaData.readLandmarkCsv(csv)
    val specimenMap = ScapulaData.specimens(dataDir)
      .filter(s => landmarks.contains(s.modelId))
      .map(s => s.modelId -> s)
      .toMap

    // ── metrics table ────────────────────────────────────────────────────────
    println(f"  ${"Specimen"}%-30s  ${"Mean(mm)"}%10s  ${"RMS(mm)"}%10s  ${"HD95(mm)"}%10s  ${"HD(mm)"}%10s")
    println("-" * 80)

    val allStats = registeredFiles.flatMap { f =>
      val modelId = f.getName.stripSuffix(".vtk")
      specimenMap.get(modelId).map { spec =>
        val registered = MeshIO.readMesh(f).get
        val rawMesh    = ScapulaData.loadMesh(spec.file)
        val original   = if (spec.isRight) ScapulaData.mirrorMesh(rawMesh) else rawMesh
        val st = Metrics.symmetric(registered, original)
        println(f"  $modelId%-30s  ${st.mean}%10.3f  ${st.rms}%10.3f  ${st.hd95}%10.3f  ${st.hd}%10.3f")
        st
      }
    }

    if (allStats.nonEmpty) {
      println("-" * 80)
      val n     = allStats.size.toDouble
      val mean  = allStats.map(_.mean).sum / n
      val rms   = allStats.map(_.rms).sum  / n
      val hd95  = allStats.map(_.hd95).sum / n
      val hd    = allStats.map(_.hd).sum   / n
      println(f"  ${"MEAN (all specimens)"}%-30s  $mean%10.3f  $rms%10.3f  $hd95%10.3f  $hd%10.3f")
    }

    // ── optional UI ──────────────────────────────────────────────────────────
    if (Config.showUi) {
      val ui    = ScalismoUI()
      val group = ui.createGroup("Registration quality")

      val ssmFile = new File(pass2Dir, "ssm.h5")
      if (ssmFile.exists()) {
        val ssm = StatismoIO.readStatismoMeshModel(ssmFile).get
        ui.show(group, ssm.mean, "reference (mean)")
      }

      val toShow = registeredFiles.take(5)
      toShow.foreach { f =>
        val mesh = MeshIO.readMesh(f).get
        ui.show(group, mesh, f.getName.stripSuffix(".vtk"))
      }
      println(s"\nShowing ${toShow.size} registered meshes in UI.")
    }
  }
}
