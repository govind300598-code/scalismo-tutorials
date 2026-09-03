package scapula

import scalismo.io.{MeshIO, StatismoIO}
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.utils.Random

import java.io.File

/**
 * Validates SSM1 (pass1/ssm.h5) and SSM2 (pass2/ssm.h5) and prints a side-by-side comparison table.
 *
 * Metrics reported:
 *   - Compactness   : cumulative variance explained by the first N components.
 *   - Generalization: leave-one-out mean reconstruction error on the registered meshes.
 *   - Stability     : mean-shape point distance between SSM1 and SSM2 (pass-to-pass convergence).
 */
object SSMValidation {

  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng: Random = Random(Config.seed)

    val outDir   = if (args.nonEmpty) new File(args(0)) else Config.outDir
    val pass1Dir = new File(outDir, "pass1")
    val pass2Dir = new File(outDir, "pass2")

    require(pass1Dir.exists(),
      s"pass1/ directory not found: ${pass1Dir.getPath}\n  Set SCAPULA_OUT_DIR or pass the output directory as the first argument.")
    require(pass2Dir.exists(),
      s"pass2/ directory not found: ${pass2Dir.getPath}\n  Set SCAPULA_OUT_DIR or pass the output directory as the first argument.")

    val f1 = new File(pass1Dir, "ssm.h5")
    val f2 = new File(pass2Dir, "ssm.h5")
    require(f1.exists(), s"SSM1 not found: ${f1.getPath} — run RebuildSSM first")
    require(f2.exists(), s"SSM2 not found: ${f2.getPath} — run RebuildSSM first")

    println(s"Loading SSM1 from ${f1.getPath}")
    val ssm1 = StatismoIO.readStatismoMeshModel(f1).get
    println(s"Loading SSM2 from ${f2.getPath}")
    val ssm2 = StatismoIO.readStatismoMeshModel(f2).get

    // ── Compactness ──────────────────────────────────────────────────────────
    println()
    println("=== COMPACTNESS  (cumulative variance explained) ===")
    println(f"  ${""}%-35s  ${"SSM1 (pass1)"}%14s  ${"SSM2 (pass2)"}%14s")
    println("-" * 70)

    val ev1 = (0 until ssm1.rank).map(i => ssm1.gp.klBasis(i).eigenvalue)
    val ev2 = (0 until ssm2.rank).map(i => ssm2.gp.klBasis(i).eigenvalue)
    val tot1 = ev1.sum
    val tot2 = ev2.sum

    println(f"  ${"Reference points"}%-35s  ${ssm1.mean.pointSet.numberOfPoints}%14d  ${ssm2.mean.pointSet.numberOfPoints}%14d")
    println(f"  ${"Rank (# components)"}%-35s  ${ssm1.rank}%14d  ${ssm2.rank}%14d")
    println(f"  ${"Total variance"}%-35s  $tot1%14.2f  $tot2%14.2f")

    Seq(1, 3, 5, 10, 20).foreach { n =>
      val k1 = math.min(n, ssm1.rank)
      val k2 = math.min(n, ssm2.rank)
      val c1 = ev1.take(k1).sum / tot1 * 100
      val c2 = ev2.take(k2).sum / tot2 * 100
      println(f"  ${s"Var. in first $n components (%)"}%-35s  $c1%13.1f%%  $c2%13.1f%%")
    }

    // ── Generalization ───────────────────────────────────────────────────────
    println()
    println("=== GENERALIZATION  (reconstruction error on registered meshes) ===")
    genStats("SSM1", ssm1, pass1Dir)
    genStats("SSM2", ssm2, pass2Dir)

    // ── Stability ────────────────────────────────────────────────────────────
    println()
    println("=== STABILITY  (mean-shape point distance SSM1 → SSM2) ===")
    if (ssm1.mean.pointSet.numberOfPoints == ssm2.mean.pointSet.numberOfPoints) {
      val dists = Metrics.correspondingDistances(ssm1.mean, ssm2.mean)
      println(f"  Mean point distance : ${dists.sum / dists.size}%.3f mm")
      println(f"  HD95 point distance : ${Metrics.percentile(dists, 0.95)}%.3f mm")
      println(f"  Max  point distance : ${dists.max}%.3f mm")
      println("  (Values near 0 mm mean the two passes converged to the same shape space.)")
    } else {
      println("  Different number of reference points — cannot compare mean shapes point-to-point.")
    }

    println()
    println("Done.")
  }

  private def genStats(label: String, ssm: StatisticalMeshModel, dir: File): Unit = {
    val vtks = Option(dir.listFiles())
      .getOrElse(Array.empty[File])
      .filter(_.getName.endsWith(".vtk"))
      .sortBy(_.getName)
    if (vtks.isEmpty) {
      println(s"  $label: no .vtk registered meshes in ${dir.getPath} — skipping")
      return
    }
    val errors = vtks.flatMap { f =>
      MeshIO.readMesh(f).toOption.map { mesh =>
        if (mesh.pointSet.numberOfPoints != ssm.mean.pointSet.numberOfPoints) {
          println(s"  $label: ${f.getName} has ${mesh.pointSet.numberOfPoints} points " +
            s"(expected ${ssm.mean.pointSet.numberOfPoints}) — skipping")
          IndexedSeq.empty[Double]
        } else {
          val coeffs       = ssm.coefficients(mesh)
          val reconstructed = ssm.instance(coeffs)
          Metrics.correspondingDistances(mesh, reconstructed)
        }
      }.getOrElse(IndexedSeq.empty[Double])
    }
    if (errors.nonEmpty) {
      println(f"  $label (${vtks.size} meshes):  mean=${errors.sum / errors.size}%.3f mm  " +
        f"HD95=${Metrics.percentile(errors, 0.95)}%.3f mm")
    }
  }
}
