package scapula

import scalismo.geometry.*
import scalismo.io.MeshIO
import scalismo.mesh.TriangleMesh
import scalismo.utils.Random

import java.io.{File, PrintWriter}

/**
 * Full SSM pipeline for all three scapula datasets:
 *   1. hill_sachs_STLs_scapula         (56 meshes, naming: ..._R_Scapula / ..._L_Scapula)
 *   2. paired_scapulae_STLs_scapula    (24 meshes, naming: ..._R / ..._L)
 *   3. paired_shoulder_STLs_scapula    (many meshes, naming: ..._R_Scapula / ..._L_Scapula)
 *
 * Usage:
 *   sbt "runMain scapula.MultiDatasetPipeline"
 *
 * All paths are configurable through environment variables (see Config) or the
 * SCAPULA_MULTI_ROOT variable below; everything else is picked up automatically.
 *
 * Pipeline per dataset:
 *   Stage 2  — GP non-rigid registration → dense correspondence
 *   Stage 3  — PCA SSM
 *   Stage 4  — Compactness / Generalization / Specificity validation
 */
object MultiDatasetPipeline {

  private val multiRoot: File = {
    val r = sys.env.getOrElse("SCAPULA_MULTI_ROOT", "/home/g25upadh/Documents/100 plus scapula data")
    new File(r)
  }

  private val globalOut: File = {
    val o = sys.env.getOrElse("SCAPULA_MULTI_OUT",
      new File(multiRoot.getParentFile, "scapula_multi_ssm_out").getAbsolutePath)
    new File(o)
  }

  // Dataset subdirectory names and short tags used in output filenames.
  private val datasets: IndexedSeq[(String, String)] = IndexedSeq(
    ("hill_sachs_STLs_scapula",       "hill_sachs"),
    ("paired_scapulae_STLs_scapula",  "paired_scapulae"),
    ("paired_shoulder_STLs_scapula",  "paired_shoulder")
  )

  // ---------------------------------------------------------------------------
  // Alignment & metric report for the rigid stage (mirrors the original script).
  // ---------------------------------------------------------------------------
  private def rigidReport(
    registered: IndexedSeq[(String, TriangleMesh[_3D])],
    reference:  TriangleMesh[_3D],
    tag:        String,
    outDir:     File
  ): Unit = {
    val rows = registered.map { case (id, mesh) =>
      val m = Metrics.symmetric(mesh, reference)
      (id, m)
    }
    println(s"\n[Rigid + ICP alignment summary — $tag]")
    rows.foreach { case (id, m) =>
      println(f"  $id%-38s ${m.render}")
    }
    val means = rows.map(_._2.mean)
    val hds   = rows.map(_._2.hd)
    val hd95s = rows.map(_._2.hd95)
    println(f"  MEAN across ${rows.length} shapes: mean=${means.sum/means.length}%.2f mm  " +
            f"HD=${hds.sum/hds.length}%.2f mm  HD95=${hd95s.sum/hd95s.length}%.2f mm")

    val pw = new PrintWriter(new File(outDir, s"rigid_metrics_$tag.csv"))
    try {
      pw.println("model_id,mean_mm,rms_mm,hd95_mm,hd_mm")
      rows.foreach { case (id, m) =>
        pw.println(f"$id,${m.mean}%.4f,${m.rms}%.4f,${m.hd95}%.4f,${m.hd}%.4f")
      }
    } finally pw.close()
  }

  // ---------------------------------------------------------------------------
  // Compute and print post-nonrigid registration quality vs reference.
  // ---------------------------------------------------------------------------
  private def nonRigidReport(
    registered: IndexedSeq[(String, TriangleMesh[_3D])],
    reference:  TriangleMesh[_3D],
    tag:        String,
    outDir:     File
  ): Unit = {
    val rows = registered.map { case (id, mesh) =>
      val m = Metrics.symmetric(mesh, reference)
      (id, m)
    }
    println(s"\n[Non-rigid registration summary — $tag]")
    rows.foreach { case (id, m) =>
      println(f"  $id%-38s ${m.render}")
    }
    val means = rows.map(_._2.mean)
    println(f"  AGGREGATE: mean=${means.sum/means.length}%.3f mm")

    val pw = new PrintWriter(new File(outDir, s"nonrigid_metrics_$tag.csv"))
    try {
      pw.println("model_id,mean_mm,rms_mm,hd95_mm,hd_mm")
      rows.foreach { case (id, m) =>
        pw.println(f"$id,${m.mean}%.4f,${m.rms}%.4f,${m.hd95}%.4f,${m.hd}%.4f")
      }
    } finally pw.close()
  }

  // ---------------------------------------------------------------------------
  // Run all stages for one dataset directory.
  // ---------------------------------------------------------------------------
  private def runDataset(dataDir: File, tag: String)(implicit rng: Random): Unit = {
    println(s"\n${"=" * 80}")
    println(s"DATASET: $tag  [${dataDir.getAbsolutePath}]")
    println(s"${"=" * 80}")

    if (!dataDir.exists()) {
      println(s"  WARNING: directory does not exist — skipping.")
      return
    }

    val outDir = new File(globalOut, tag)
    outDir.mkdirs()

    // ---- Stage 2 -----------------------------------------------------------
    println(s"\n=== STAGE 2: NON-RIGID REGISTRATION [$tag] ===")
    val nr = Stage2NonRigid.run(dataDir, outDir)

    rigidReport(nr.registered, nr.reference, tag, outDir)
    nonRigidReport(nr.registered, nr.reference, tag, outDir)

    // ---- Stage 3 -----------------------------------------------------------
    println(s"\n=== STAGE 3: SSM [$tag] ===")
    val ssm = Stage3SSM.run(nr.reference, nr.registered, outDir, tag)

    // ---- Stage 4 -----------------------------------------------------------
    Stage4Validation.run(
      model      = ssm.model,
      reference  = nr.reference,
      registered = nr.registered,
      outDir     = outDir,
      tag        = tag,
      maxModes   = math.min(20, nr.registered.length - 2),
      specificitySamples = 50
    )

    // If paired (bilateral) dataset, also build a single-side independent model.
    val hasPairs = nr.registered.map(_._1).exists(ScapulaData.isRightSide)
    if (hasPairs && Config.buildIndependentModel) {
      println(s"\n=== STAGE 3b: INDEPENDENT (LEFT-ONLY) SSM [$tag] ===")
      val leftOnly = nr.registered.filterNot(r => ScapulaData.isRightSide(r._1))
      if (leftOnly.length >= 3) {
        val ssmL = Stage3SSM.run(nr.reference, leftOnly, outDir, s"${tag}_left_only")
        Stage4Validation.run(
          model      = ssmL.model,
          reference  = nr.reference,
          registered = leftOnly,
          outDir     = outDir,
          tag        = s"${tag}_left_only",
          maxModes   = math.min(20, leftOnly.length - 2),
          specificitySamples = 50
        )
      } else {
        println(s"  Only ${leftOnly.length} left-side shapes — skipping independent model.")
      }
    }

    println(s"\n  [$tag] Complete. Outputs in ${outDir.getAbsolutePath}")
  }

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    println(s"Root data dir : ${multiRoot.getAbsolutePath}")
    println(s"Output dir    : ${globalOut.getAbsolutePath}")
    globalOut.mkdirs()

    if (!multiRoot.exists())
      throw new RuntimeException(s"Root data directory not found: ${multiRoot.getAbsolutePath}\n" +
        "Set SCAPULA_MULTI_ROOT to the folder containing the three dataset subfolders.")

    datasets.foreach { case (subdir, tag) =>
      runDataset(new File(multiRoot, subdir), tag)
    }

    println(s"\n${"=" * 80}")
    println("ALL DATASETS COMPLETE")
    println(s"${"=" * 80}")
    println(s"Results in: ${globalOut.getAbsolutePath}")
  }
}
